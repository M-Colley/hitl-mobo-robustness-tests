"""Finite-catalog optimizers that receive only purchased human feedback.

The common protocol is ``ask() -> indices``, ``tell(indices, feedback)`` and
``recommend() -> seen_index``. Rating feedback is real valued. Pair feedback is
the probability assigned to the first alternative: 1, 0, or 0.5 for a tie.
External initialization must also be supplied through ``tell``; no policy reads
the objective, its optimum, or an unobserved rating.

``reward_policy`` is a one-step, finite-bandit analogue of reward-model RLHF:
it fits a regularized Bradley--Terry reward model and samples the exact
KL-regularized policy, pi(a) proportional to exp(reward(a) / beta), with a
uniform reference. It is not a sequential RL or language-model implementation.
``dpo`` directly fits categorical policy logits using the DPO logistic loss
beta * (log pi(first) - log pi(second)); the uniform reference cancels.
An explicit L2 penalty on logits identifies a finite solution under separation.

``scores`` are in rating units for ridge/GP models and latent preference units
for preference methods. Ridge, Bradley--Terry, DPO and dueling uncertainty is a
confidence proxy, not a calibrated GP posterior SD. Comparing its numerical
scale between models would be inappropriate. ``random``, ``sobol`` and ``cem``
expose empirical rating summaries, not a surrogate model. Sobol consumes the
catalog in order; supply a Sobol catalog to obtain a Sobol baseline.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax


METHODS = (
    "random", "sobol", "cem", "ridge_ucb", "dueling_ucb",
    "reward_policy", "dpo", "bo_logei",
)
FEEDBACK = {name: ("pairwise" if name in {"dueling_ucb", "reward_policy", "dpo"}
                   else "rating") for name in METHODS}


def _positive(name, value, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class _Policy:
    feedback_type = "rating"

    def __init__(self, X, seed):
        self.X = np.array(X, dtype=float, copy=True)
        if (self.X.ndim != 2 or not self.X.shape[0] or not self.X.shape[1]
                or not np.isfinite(self.X).all()
                or (self.X < 0).any() or (self.X > 1).any()):
            raise ValueError("X must be a nonempty finite matrix normalized to [0, 1]")
        self.X.flags.writeable = False
        self.n = len(self.X)
        self.seed = _integer("seed", seed, 0)
        self.rng = np.random.default_rng(self.seed)
        self.seen: list[int] = []
        self._seen = set()
        self.indices: list[tuple[int, ...]] = []
        self.feedback: list[float] = []
        if self.feedback_type == "pairwise" and self.n < 2:
            raise ValueError("Pairwise policies require at least two catalog designs")

    def tell(self, indices, feedback):
        indices = tuple(indices)
        arity = 2 if self.feedback_type == "pairwise" else 1
        if len(indices) != arity:
            raise ValueError(f"Expected {arity} indices for {self.feedback_type} feedback")
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
               or not 0 <= i < self.n for i in indices):
            raise ValueError("Feedback indices must be valid integer catalog indices")
        indices = tuple(map(int, indices))
        if arity == 2 and indices[0] == indices[1]:
            raise ValueError("A comparison must expose two distinct designs")
        feedback = float(feedback)
        if not np.isfinite(feedback):
            raise ValueError("Feedback must be finite")
        if arity == 2 and feedback not in (0.0, 0.5, 1.0):
            raise ValueError("Pairwise feedback must be 0, 0.5, or 1")
        self.indices.append(indices)
        self.feedback.append(feedback)
        for i in indices:
            if i not in self._seen:
                self._seen.add(i)
                self.seen.append(i)
        self._update(indices, feedback)

    def _update(self, indices, feedback):
        pass

    def _unseen(self):
        return np.array([i for i in range(self.n) if i not in self._seen], dtype=int)

    def _available(self):
        unseen = self._unseen()
        return unseen if len(unseen) else np.arange(self.n)

    def _require_feedback(self):
        if not self.seen:
            raise RuntimeError("A recommendation requires at least one purchased feedback item")

    def _best_seen(self, scores):
        self._require_feedback()
        # Seen order, rather than sorted catalog IDs, fixes first-exposure ties.
        return self.seen[int(np.argmax(np.asarray(scores)[self.seen]))]

    def recommend(self):
        self._require_feedback()
        return self.indices[int(np.argmax(self.feedback))][0]

    def scores(self):
        """Empirical rating means and standard-error proxies across the catalog."""
        counts = np.zeros(self.n)
        sums = np.zeros(self.n)
        for (i,), y in zip(self.indices, self.feedback):
            counts[i] += 1
            sums[i] += y
        mean = np.full(self.n, np.mean(self.feedback) if self.feedback else 0.0)
        np.divide(sums, counts, out=mean, where=counts > 0)
        sd = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        return mean, sd


class _Random(_Policy):
    def ask(self):
        return (int(self.rng.choice(self._available())),)


class _Sequential(_Policy):
    def __init__(self, X, seed):
        super().__init__(X, seed)
        self._cursor = 0

    def ask(self):
        unseen = self._unseen()
        if len(unseen):
            return (int(unseen[0]),)
        index = self._cursor % self.n
        self._cursor += 1
        return (index,)


class _CEM(_Random):
    def __init__(self, X, seed, *, elite_fraction=0.25, min_std=0.05,
                 epsilon=0.1, n_initial=5):
        super().__init__(X, seed)
        self.elite_fraction = _positive("elite_fraction", elite_fraction)
        if self.elite_fraction > 1:
            raise ValueError("elite_fraction must be <= 1")
        self.min_std = _positive("min_std", min_std)
        self.epsilon = _probability("epsilon", epsilon)
        self.n_initial = _integer("n_initial", n_initial)

    def ask(self):
        available = self._available()
        if len(self.seen) < self.n_initial or self.rng.random() < self.epsilon:
            return (int(self.rng.choice(available)),)
        means, _ = self.scores()
        order = sorted(self.seen, key=lambda i: -means[i])
        elite = self.X[order[:max(2, int(np.ceil(len(order) * self.elite_fraction)))]]
        center = elite.mean(axis=0)
        sd = np.maximum(elite.std(axis=0), self.min_std)
        log_density = -0.5 * np.square((self.X[available] - center) / sd).sum(axis=1)
        return (int(available[np.argmax(log_density)]),)


def _probability(name, value):
    value = _positive(name, value, allow_zero=True)
    if value > 1:
        raise ValueError(f"{name} must be <= 1")
    return value


def _rbf_features(X, rng, n_features, lengthscale):
    """Fixed, bounded RBF basis; construction never depends on responses."""
    count = min(_integer("n_features", n_features), len(X))
    lengthscale = _positive("lengthscale", lengthscale)
    centers = X[rng.choice(len(X), count, replace=False)]
    squared = np.square(X[:, None, :] - centers[None, :, :]).sum(axis=2)
    return np.column_stack([np.ones(len(X)), np.exp(-squared / (2 * lengthscale**2))]) / np.sqrt(count + 1)


class _RidgeUCB(_Policy):
    def __init__(self, X, seed, *, regularization=1.0, n_features=64,
                 lengthscale=0.25, exploration=1.0):
        super().__init__(X, seed)
        self.regularization = _positive("regularization", regularization)
        self.exploration = _positive("exploration", exploration, allow_zero=True)
        self.features = _rbf_features(self.X, self.rng, n_features, lengthscale)
        self._inverse = np.eye(self.features.shape[1]) / self.regularization
        self._b = np.zeros(self.features.shape[1])

    def _update(self, indices, feedback):
        f = self.features[indices[0]]
        v = self._inverse @ f
        self._inverse -= np.outer(v, v) / (1.0 + f @ v)
        self._b += feedback * f

    def scores(self):
        mean = self.features @ (self._inverse @ self._b)
        variance = np.einsum("ij,jk,ik->i", self.features, self._inverse, self.features)
        return mean, np.sqrt(np.maximum(variance, 0.0))

    def ask(self):
        mean, sd = self.scores()
        available = self._available()
        return (int(available[np.argmax((mean + self.exploration * sd)[available])]),)

    def recommend(self):
        return self._best_seen(self.scores()[0])


class _DuelingUCB(_Policy):
    """Empirical Borda leader challenged by optimistic pairwise opponents.

    Unknown pairs have an upper win probability of one. Equal upper bounds
    favor pairs with fewer observations, so uncertainty reaches new designs
    while every query still compares against the current feedback-based leader.
    """
    feedback_type = "pairwise"

    def __init__(self, X, seed, *, exploration=1.0):
        super().__init__(X, seed)
        self.exploration = _positive("exploration", exploration, allow_zero=True)
        # Canonical (smaller ID, larger ID) -> [first wins, number of looks].
        self._pairs = {}

    def _update(self, indices, feedback):
        i, j = indices
        if i > j:
            i, j, feedback = j, i, 1.0 - feedback
        wins, count = self._pairs.get((i, j), (0.0, 0))
        self._pairs[i, j] = (wins + feedback, count + 1)

    def scores(self):
        mean = np.full(self.n, 0.5)
        variance = np.full(self.n, (self.n - 1) / 12.0)
        for (i, j), (wins, count) in self._pairs.items():
            p = (wins + 1) / (count + 2)
            var = p * (1.0 - p) / (count + 3)
            mean[i] += (p - 0.5) / (self.n - 1)
            mean[j] -= (p - 0.5) / (self.n - 1)
            variance[i] += var - 1.0 / 12.0
            variance[j] += var - 1.0 / 12.0
        return mean, np.sqrt(np.maximum(variance, 0.0)) / (self.n - 1)

    def ask(self):
        if not self.feedback:
            return tuple(map(int, self.rng.choice(self.n, 2, replace=False)))
        # Starting a disjoint duel for every new pair makes the whole search
        # feedback-blind whenever the catalog is larger than the query budget.
        # Reuse the observed leader immediately, without exhausting the catalog.
        leader = self._best_seen(self.scores()[0])
        factor = self.exploration * np.sqrt(np.log(2 + len(self.feedback)) / 2)
        challenge = np.ones(self.n)
        challenge[leader] = -np.inf
        counts = np.zeros(self.n, dtype=int)
        for (i, j), (wins, count) in self._pairs.items():
            if leader not in (i, j):
                continue
            other = j if leader == i else i
            p_other = (1 - wins / count) if leader == i else wins / count
            challenge[other] = min(1.0, p_other + factor / np.sqrt(count))
            counts[other] = count
        candidates = np.flatnonzero(challenge == challenge.max())
        challenger = int(candidates[np.argmin(counts[candidates])])
        return (leader, challenger)

    def recommend(self):
        return self._best_seen(self.scores()[0])


class _PreferencePolicy(_Policy):
    feedback_type = "pairwise"

    def __init__(self, X, seed, *, beta=0.2, epsilon=0.1, fit_maxiter=200):
        super().__init__(X, seed)
        self.beta = _positive("beta", beta)
        self.epsilon = _probability("epsilon", epsilon)
        self.fit_maxiter = _integer("fit_maxiter", fit_maxiter)
        self._fitted_at = -1

    def _distribution(self):
        raise NotImplementedError

    def ask(self):
        probability = (1.0 - self.epsilon) * self._distribution() + self.epsilon / self.n
        # Sampling without replacement can underflow with a nearly point-mass
        # policy. Condition the second draw explicitly, with a uniform limiting
        # distribution when no other machine-representable mass remains.
        first = int(self.rng.choice(self.n, p=probability))
        probability = probability.copy()
        probability[first] = 0.0
        mass = probability.sum()
        if mass == 0:
            probability[:] = 1.0
            probability[first] = 0.0
            mass = probability.sum()
        second = int(self.rng.choice(self.n, p=probability / mass))
        return (first, second)

    def recommend(self):
        return self._best_seen(self.scores()[0])

    def _optimize(self, objective, start):
        result = minimize(objective, start, jac=True, method="L-BFGS-B",
                          options={"maxiter": self.fit_maxiter, "ftol": 1e-11, "gtol": 1e-7})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Preference model fit failed: {result.message}")
        return result.x


class _RewardPolicy(_PreferencePolicy):
    def __init__(self, X, seed, *, regularization=0.1, n_features=64,
                 lengthscale=0.25, **kwargs):
        super().__init__(X, seed, **kwargs)
        self.regularization = _positive("regularization", regularization)
        self.features = _rbf_features(self.X, self.rng, n_features, lengthscale)
        self._weights = np.zeros(self.features.shape[1])
        self._inverse = np.eye(self.features.shape[1]) / self.regularization

    def _fit(self):
        if self._fitted_at == len(self.feedback):
            return
        if self.feedback:
            pairs = np.asarray(self.indices)
            differences = self.features[pairs[:, 0]] - self.features[pairs[:, 1]]
            targets = np.asarray(self.feedback)

            def objective(weights):
                logits = differences @ weights
                loss = np.sum(np.logaddexp(0.0, logits) - targets * logits)
                gradient = differences.T @ (expit(logits) - targets)
                return (loss + 0.5 * self.regularization * (weights @ weights),
                        gradient + self.regularization * weights)

            self._weights = self._optimize(objective, self._weights)
            probs = expit(differences @ self._weights)
            curvature = probs * (1.0 - probs)
            precision = self.regularization * np.eye(len(self._weights))
            precision += differences.T @ (differences * curvature[:, None])
            self._inverse = np.linalg.solve(precision, np.eye(len(self._weights)))
        self._fitted_at = len(self.feedback)

    def scores(self):
        self._fit()
        mean = self.features @ self._weights
        variance = np.einsum("ij,jk,ik->i", self.features, self._inverse, self.features)
        return mean, np.sqrt(np.maximum(variance, 0.0))

    def _distribution(self):
        reward = self.scores()[0]
        # Shift before scaling so a small positive beta cannot overflow the
        # maximum logit. Negative infinity is a valid zero-probability limit.
        with np.errstate(over="ignore"):
            return softmax((reward - reward.max()) / self.beta)


class _DPO(_PreferencePolicy):
    def __init__(self, X, seed, *, regularization=0.01, **kwargs):
        super().__init__(X, seed, **kwargs)
        self.regularization = _positive("regularization", regularization)
        self._logits = np.zeros(self.n)
        self._diagonal_precision = np.full(self.n, self.regularization)

    def _fit(self):
        if self._fitted_at == len(self.feedback):
            return
        if self.feedback:
            pairs = np.asarray(self.indices)
            targets = np.asarray(self.feedback)
            first, second = pairs[:, 0], pairs[:, 1]

            def objective(logits):
                log_ratios = self.beta * (logits[first] - logits[second])
                loss = np.sum(np.logaddexp(0.0, log_ratios) - targets * log_ratios)
                residual = self.beta * (expit(log_ratios) - targets)
                gradient = self.regularization * logits.copy()
                np.add.at(gradient, first, residual)
                np.add.at(gradient, second, -residual)
                return loss + 0.5 * self.regularization * (logits @ logits), gradient

            self._logits = self._optimize(objective, self._logits)
            probs = expit(self.beta * (self._logits[first] - self._logits[second]))
            # A diagonal Fisher confidence proxy avoids an O(catalog^2) matrix.
            curvature = self.beta**2 * probs * (1.0 - probs)
            self._diagonal_precision[:] = self.regularization
            np.add.at(self._diagonal_precision, first, curvature)
            np.add.at(self._diagonal_precision, second, curvature)
        self._fitted_at = len(self.feedback)

    def scores(self):
        self._fit()
        return self.beta * self._logits.copy(), self.beta / np.sqrt(self._diagonal_precision)

    def _distribution(self):
        self._fit()
        return softmax(self._logits)


class _BOLogEI(_Random):
    """Optional GP LogEI, scored on the common finite catalog.

    The incumbent is the highest latent posterior mean at an observed design.
    Fits and posterior arrays are cached until new feedback arrives. Fit errors
    stop the run rather than silently substituting a different algorithm.
    """
    def __init__(self, X, seed, *, n_initial=5):
        super().__init__(X, seed)
        self.n_initial = _integer("n_initial", n_initial, 2)
        self._fitted_at = -1
        self._gp = None
        self._mean = np.zeros(self.n)
        self._sd = np.ones(self.n)
        self._cached = np.zeros(self.n, dtype=bool)

    def _fit(self):
        if self._fitted_at == len(self.feedback):
            return
        if len(self.feedback) < 2:
            self._fitted_at = len(self.feedback)
            self._cached[:] = True
            return
        try:
            import torch
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms.outcome import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError as exc:
            raise RuntimeError("bo_logei requires the optional torch, gpytorch and botorch dependencies") from exc
        train_X = torch.tensor(self.X[[row[0] for row in self.indices]], dtype=torch.double)
        train_Y = torch.tensor(self.feedback, dtype=torch.double).unsqueeze(-1)
        try:
            # BoTorch retry initialization uses torch's global generator. Restore
            # it afterward so this policy cannot perturb a simulated rater.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((self.seed + len(self.feedback)) % (2**63 - 1))
                gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
                fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            gp.eval()
        except Exception as exc:
            raise RuntimeError(f"bo_logei fit failed after {len(self.feedback)} feedback items") from exc
        self._gp = gp
        self._cached[:] = False
        self._fitted_at = len(self.feedback)

    def _posterior(self, indices):
        """Cache only requested marginal moments, independently of GP fitting."""
        self._fit()
        indices = np.asarray(indices, dtype=int)
        missing = indices[~self._cached[indices]]
        if len(missing):
            import torch
            with torch.no_grad():
                for start in range(0, len(missing), 512):
                    block = missing[start:start + 512]
                    # Independent q=1 batches avoid an unused catalog-by-catalog
                    # joint covariance matrix during early recommendations.
                    query = torch.tensor(self.X[block], dtype=torch.double).unsqueeze(-2)
                    posterior = self._gp.posterior(query)
                    mean = posterior.mean.reshape(-1).cpu().numpy()
                    sd = posterior.variance.reshape(-1).clamp_min(0).sqrt().cpu().numpy()
                    if not np.isfinite(mean).all() or not np.isfinite(sd).all():
                        raise RuntimeError("bo_logei posterior contains non-finite values")
                    self._mean[block], self._sd[block] = mean, sd
                    self._cached[block] = True
        return self._mean[indices].copy(), self._sd[indices].copy()

    def scores(self):
        return self._posterior(np.arange(self.n))

    def ask(self):
        if len(self.feedback) < self.n_initial:
            return super().ask()
        available = self._available()
        incumbent = float(self._posterior(self.seen)[0].max())
        mean, sd = self._posterior(available)
        import torch
        from botorch.acquisition.analytic import LogExpectedImprovement

        class CachedLogEI(LogExpectedImprovement):
            # Retain BoTorch's own stable analytic formula and decorators while
            # reusing the exact marginal moments already needed to recommend.
            def _mean_and_sigma(acq_self, X, compute_sigma=True, min_var=1e-12):
                cached_mean = torch.tensor(mean, dtype=X.dtype, device=X.device).unsqueeze(-1)
                cached_sd = torch.tensor(sd, dtype=X.dtype, device=X.device).unsqueeze(-1)
                return cached_mean, cached_sd.clamp_min(np.sqrt(min_var)) if compute_sigma else None

        acq = CachedLogEI(self._gp, best_f=incumbent)
        with torch.no_grad():
            values = acq(torch.tensor(self.X[available], dtype=torch.double).unsqueeze(-2))
        values = values.cpu().numpy().reshape(-1)
        if not np.isfinite(values).all():
            raise RuntimeError("bo_logei acquisition contains non-finite values")
        return (int(available[np.argmax(values)]),)

    def recommend(self):
        self._require_feedback()
        mean, _ = self._posterior(self.seen)
        return self.seen[int(np.argmax(mean))]


_FACTORIES = dict(random=_Random, sobol=_Sequential, cem=_CEM,
                  ridge_ucb=_RidgeUCB, dueling_ucb=_DuelingUCB,
                  reward_policy=_RewardPolicy, dpo=_DPO, bo_logei=_BOLogEI)


_OPTION_VALIDATORS = {
    "regularization": _positive, "lengthscale": _positive, "min_std": _positive,
    "beta": _positive, "epsilon": _probability, "n_features": _integer,
    "fit_maxiter": _integer, "exploration": lambda name, v: _positive(name, v, allow_zero=True),
    "n_initial": _integer,
}
_SUPPORTED_OPTIONS = {
    "random": set(), "sobol": set(),
    "cem": {"elite_fraction", "min_std", "epsilon", "n_initial"},
    "ridge_ucb": {"regularization", "n_features", "lengthscale", "exploration"},
    "dueling_ucb": {"exploration"},
    "reward_policy": {"regularization", "n_features", "lengthscale", "beta", "epsilon", "fit_maxiter"},
    "dpo": {"regularization", "beta", "epsilon", "fit_maxiter"},
    "bo_logei": {"n_initial"},
}


def validate_policy_options(method, options):
    """Validate configuration without constructing a policy or importing torch."""
    if method not in _FACTORIES:
        raise ValueError(f"Unknown method {method!r}; choose one of {', '.join(METHODS)}")
    if not isinstance(options, dict):
        raise ValueError("Policy options must be a dictionary")
    unknown = set(options) - _SUPPORTED_OPTIONS[method]
    if unknown:
        raise ValueError(f"Unsupported options for {method}: {', '.join(sorted(map(str, unknown)))}")
    for name, value in options.items():
        if name == "n_initial" and method == "bo_logei":
            _integer(name, value, 2)
        elif name == "elite_fraction":
            value = _positive(name, value)
            if value > 1:
                raise ValueError("elite_fraction must be <= 1")
        else:
            _OPTION_VALIDATORS[name](name, value)


def make_policy(method, X, seed, **kwargs):
    """Construct a policy without objective access or benchmark execution.

    Unknown keyword arguments raise ``TypeError`` to prevent misspelled or
    accidentally ignored settings. ``regularization`` always means an explicit
    positive L2 precision; its defaults differ between ridge (1), reward (.1)
    and DPO (.01). Pair methods' ``beta`` defaults to .2 and sampling mixes 10%
    uniform exploration. All pseudorandom state belongs to the policy.
    """
    if method not in _FACTORIES:
        raise ValueError(f"Unknown method {method!r}; choose one of {', '.join(METHODS)}")
    return _FACTORIES[method](X, seed, **kwargs)
