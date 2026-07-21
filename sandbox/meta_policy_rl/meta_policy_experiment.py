"""Sandbox prototype: offline RL for an adaptive elicitation meta-policy in HITL BO.

Instantiates the approach of Bucinca et al. (2026), "Offline Reinforcement
Learning for Adaptive Support in AI-Assisted Decision-Making", on top of this
repo's oracle+noise simulator instead of human participants:

  1. Environment  = one simulated participant session: an extra_trees oracle
     fit on the eHMI archival data supplies "true" ratings; gaussian or AR(1)
     error is injected on every observation (the repo's jitter_iteration=0
     human-plausible condition).
  2. Behavior policy = uniform-random over the meta-actions per iteration
     (the paper's quasi-uniform exploratory policy).
  3. Offline learning = tabular Q-learning over the logged transitions,
     BOTH pooled across conditions and per noise regime (the analogue of the
     paper's per-NFC-group policies).
  4. Evaluation = greedy learned policies vs static baselines on FRESH seeds
     (training and evaluation seed blocks are disjoint, mirroring the repo's
     screening/confirmatory discipline), paired per seed, per condition, with
     BH correction over the primary test family.

Meta-actions (the analogue of the paper's assistance types):
  ei         - fit GP, query the Expected Improvement argmax over a pool
  exploit    - query the GP posterior-mean argmax over a pool (pure exploit)
  replicate  - re-query the current recommended design (validates/averages
               the incumbent's noisy ratings via the GP noise model)
  probe      - query a uniform-random design (ignores a possibly noise-led GP)
  drop_worst - distrust: permanently discard the observation with the largest
               raw |residual| vs the current GP, refit, then EI-query (skipped
               and executed as plain EI while fewer than 5 points exist)
  rest       - skip this query slot entirely; resets serial error correlation
               (a break resets fatigue/drift). Reward is exactly 0, so resting
               is rational wherever the alternatives' expected reward is
               negative - judge its usage against the empirical per-action
               rewards in the summary, not as a red flag per se.

State (observable quantities only, 3 x 2 x 2 = 12 discrete states):
  phase      - early / late / endgame by DECISION index (endgame = last 3
               decisions; separates terminal from bootstrapped targets; time-
               based rather than query-based because `rest` consumes a slot
               without adding an observation)
  surprise   - mean |observed - GP-predicted| of the last 3 observations,
               scaled by the observed-y std, above/below SURPRISE_THRESHOLD
  stall      - recommendation's posterior mean improved less than STALL_EPS
               (in observed-y std units) over the last 3 decisions

Reward: per-step change in the TRUE oracle value of the recommended design
(posterior-mean argmax over visited designs - the repo's inference incumbent).
With GAMMA = 1.0 over the finite horizon this telescopes exactly to the final
recommendation quality, so the trained objective matches the repo's
deployment-relevant metric: final inference simple regret (true).

Noise-stream pairing: episodes with the same seed entropy share the design and
noise streams across policies. Every decision slot consumes exactly one noise
draw (`rest` draws and discards), so common-random-numbers pairing by slot
index is preserved for all policies.

Caveats (documented, deliberate for a sandbox prototype):
  - sklearn GP (RBF + white kernel) + candidate-pool acquisition instead of
    BoTorch, for ~100x speed; pool screening mirrors the repo's own approach.
  - EI uses the LATENT posterior std (fitted white-noise level subtracted),
    per review: sklearn's return_std includes observation noise.
  - Oracle is extra_trees on raw eHMI rows (composite objective); policies
    learned here inherit oracle-fidelity limits (ehmi held-out R^2 ~0.55 with
    the per-design mean target) - hypothesis-generating, not confirmatory.
  - AR(1) noise starts from its stationary distribution (fixes the cold-start
    issue found in the main simulator during review).

Usage:
  python sandbox/meta_policy_rl/meta_policy_experiment.py            # full run
  python sandbox/meta_policy_rl/meta_policy_experiment.py --quick    # smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sps
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".dataset_cache" / "ehmi-optimization-chi25-data" / "eHMI-bo-participantdata"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

PARAM_COLUMNS = [
    "verticalPosition", "verticalWidth", "horizontalWidth",
    "r", "g", "b", "a", "blinkFrequency", "volume",
]
OBJECTIVE_COLUMNS = ["Trust", "Understanding", "PerceivedSafety", "Aesthetics", "Acceptance"]

ACTIONS = ["ei", "exploit", "replicate", "probe", "drop_worst", "rest"]
N_ACTIONS = len(ACTIONS)
N_STATES = 12  # phase(3) x surprise(2) x stall(2)

# Episode shape mirrors the eHMI study: ~20 evaluations per participant.
INITIAL_SAMPLES = 3
TOTAL_QUERIES = 20
N_DECISIONS = TOTAL_QUERIES - INITIAL_SAMPLES  # 17 meta-decision slots

PHASE_SPLIT_DECISIONS = 7  # t < 7 -> early (matches the old <10-queries split)
ENDGAME_DECISIONS = 3      # last 3 decisions -> endgame
SURPRISE_THRESHOLD = 0.5   # mean |resid| / std(y_obs) over last 3 observations
STALL_EPS = 0.02           # improvement in std(y_obs) units over last 3 decisions
CANDIDATE_POOL = 512
EI_XI = 0.01
MIN_POINTS_AFTER_DROP = INITIAL_SAMPLES + 1

DEFAULT_JITTER_STDS = [0.3, 1.0, 2.5]
DEFAULT_ERROR_MODELS = ["gaussian", "ar1"]
AR1_RHO = 0.8

TRAIN_SEED_BASE = 0        # training episodes: seeds 0..n_train-1 per condition
EVAL_SEED_BASE = 100_000   # evaluation episodes: disjoint block
Y_OPT_SEED = 424_242
Y_OPT_SAMPLES = 200_000

# GAMMA = 1.0 so the return of the telescoping delta reward equals final
# recommendation quality exactly (finite horizon; terminal transitions carry
# done=True). Any gamma < 1 would train on a different objective than the
# evaluation metric.
GAMMA = 1.0
Q_EPOCHS = 60
Q_LR0 = 0.2


# --------------------------------------------------------------------------
# Data and oracle (mirrors repo conventions: composite = mean of objective
# columns, bounds from data min/max, y_opt = anchored random search).
# --------------------------------------------------------------------------

def load_ehmi_frame() -> pd.DataFrame:
    frames = []
    for csv in sorted(DATA_DIR.glob("u_*/ObservationsPerEvaluation.csv")):
        frames.append(pd.read_csv(csv, sep=";"))
    if not frames:
        raise FileNotFoundError(f"No participant CSVs under {DATA_DIR}")
    df = pd.concat(frames, ignore_index=True)
    # Some archival files carry locale-mangled numerics (e.g. '19.999.999...');
    # coerce to NaN and drop those rows rather than crash.
    needed = PARAM_COLUMNS + OBJECTIVE_COLUMNS
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=needed)
    if n_before - len(df):
        print(f"  dropped {n_before - len(df)} rows with unparseable numerics")
    return df


@dataclasses.dataclass
class Oracle:
    model: ExtraTreesRegressor
    low: np.ndarray
    high: np.ndarray
    y_std: float
    y_opt: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.model.predict(pd.DataFrame(X, columns=PARAM_COLUMNS))


def build_oracle(df: pd.DataFrame, seed: int = 7, n_jobs: int = 1) -> Oracle:
    X = df[PARAM_COLUMNS].to_numpy(dtype=float)
    y = df[OBJECTIVE_COLUMNS].to_numpy(dtype=float).mean(axis=1)  # composite
    model = ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=n_jobs)
    model.fit(pd.DataFrame(X, columns=PARAM_COLUMNS), y)

    low = X.min(axis=0)
    high = X.max(axis=0)

    # Anchored random-search y_opt, as in the main simulator.
    rng = np.random.default_rng(Y_OPT_SEED)
    best = float(np.max(model.predict(pd.DataFrame(X, columns=PARAM_COLUMNS))))
    remaining = Y_OPT_SAMPLES
    while remaining > 0:
        m = min(20_000, remaining)
        Xs = rng.uniform(low, high, size=(m, len(low)))
        best = max(best, float(np.max(model.predict(pd.DataFrame(Xs, columns=PARAM_COLUMNS)))))
        remaining -= m
    return Oracle(model=model, low=low, high=high, y_std=float(np.std(y)), y_opt=best)


# Per-worker oracle cache: build_oracle is deterministic (fixed seeds), so
# every loky worker reconstructs the identical oracle once instead of
# pickling a 300-tree forest into every task.
_ORACLE: Oracle | None = None


def _get_oracle() -> Oracle:
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = build_oracle(load_ehmi_frame())
    return _ORACLE


# --------------------------------------------------------------------------
# Noise models (jitter from the very first observation; AR(1) starts from its
# stationary distribution).
# --------------------------------------------------------------------------

class NoiseProcess:
    def __init__(self, error_model: str, jitter_std: float, rng: np.random.Generator):
        self.error_model = error_model
        self.jitter_std = jitter_std
        self.rng = rng
        if error_model == "ar1":
            self.prev: float | None = float(rng.normal(0.0, jitter_std))  # stationary init
        elif error_model != "gaussian":
            raise ValueError(f"Unknown error model: {error_model}")

    def sample(self) -> float:
        if self.error_model == "gaussian":
            return float(self.rng.normal(0.0, self.jitter_std))
        if self.prev is None:  # fresh start after a rest: stationary marginal
            err = float(self.rng.normal(0.0, self.jitter_std))
        else:
            innovation = float(self.rng.normal(0.0, self.jitter_std * np.sqrt(1.0 - AR1_RHO**2)))
            err = AR1_RHO * self.prev + innovation
        self.prev = err
        return err

    def reset(self) -> None:
        """Break serial correlation (no-op for gaussian)."""
        if self.error_model == "ar1":
            self.prev = None


# --------------------------------------------------------------------------
# Environment: one simulated participant session.
# --------------------------------------------------------------------------

def make_gp() -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.full(len(PARAM_COLUMNS), 0.5), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e2))
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=0
    )


class Episode:
    """Runs one session under a given meta-policy; logs transitions."""

    def __init__(self, oracle: Oracle, condition: dict, seed_entropy: list[int]):
        self.oracle = oracle
        ss = np.random.SeedSequence(seed_entropy)
        design_ss, noise_ss = ss.spawn(2)
        self.rng = np.random.default_rng(design_ss)
        self.noise = NoiseProcess(condition["error_model"], condition["jitter_std"], np.random.default_rng(noise_ss))

        self.t = 0  # decision index (drives the phase feature; rest advances it too)
        self.X: list[np.ndarray] = []
        self.y_obs: list[float] = []
        self.y_true: list[float] = []
        # Parallel to X: |y_obs - gp prediction| at query time (NaN for the
        # initial samples, which precede any GP). Kept index-parallel so
        # drop_worst can prune the dropped observation's residual too.
        self.residuals: list[float] = []
        self.rec_mu_history: list[float] = []
        self.n_drops = 0

        for _ in range(INITIAL_SAMPLES):
            x = self.rng.uniform(oracle.low, oracle.high)
            self._query(x, gp=None)

        self.gp = self._fit_gp()
        self.rec_quality = self._recommendation_quality()
        self.rec_mu_history.append(self._recommendation_mu())

    # -- core mechanics ----------------------------------------------------

    def _query(self, x: np.ndarray, gp: GaussianProcessRegressor | None) -> None:
        true_val = float(self.oracle.predict(x)[0])
        obs = true_val + self.noise.sample()
        if gp is not None:
            pred = float(gp.predict(x.reshape(1, -1))[0])
            self.residuals.append(abs(obs - pred))
        else:
            self.residuals.append(float("nan"))
        self.X.append(np.asarray(x, dtype=float))
        self.y_true.append(true_val)
        self.y_obs.append(obs)

    def _fit_gp(self) -> GaussianProcessRegressor:
        gp = make_gp()
        gp.fit(np.vstack(self.X), np.asarray(self.y_obs))
        return gp

    def _visited_unique(self) -> np.ndarray:
        return np.unique(np.vstack(self.X), axis=0)

    def _recommendation(self) -> np.ndarray:
        """Posterior-mean argmax over visited designs (repo inference rule)."""
        Xv = self._visited_unique()
        mu = self.gp.predict(Xv)
        return Xv[int(np.argmax(mu))]

    def _recommendation_mu(self) -> float:
        Xv = self._visited_unique()
        return float(np.max(self.gp.predict(Xv)))

    def _recommendation_quality(self) -> float:
        return float(self.oracle.predict(self._recommendation())[0])

    # -- state features (observable only) ----------------------------------

    def state(self) -> int:
        if self.t >= N_DECISIONS - ENDGAME_DECISIONS:
            phase = 2
        elif self.t >= PHASE_SPLIT_DECISIONS:
            phase = 1
        else:
            phase = 0
        obs_std = max(float(np.std(self.y_obs)), 1e-8)
        recent = [r for r in self.residuals[-3:] if np.isfinite(r)]
        surprise = 1 if (recent and float(np.mean(recent)) / obs_std > SURPRISE_THRESHOLD) else 0
        if len(self.rec_mu_history) >= 4:
            gain = self.rec_mu_history[-1] - self.rec_mu_history[-4]
            stall = 1 if gain < STALL_EPS * obs_std else 0
        else:
            stall = 0
        return phase * 4 + surprise * 2 + stall

    # -- actions -------------------------------------------------------------

    def _candidate_pool(self) -> np.ndarray:
        return self.rng.uniform(self.oracle.low, self.oracle.high, size=(CANDIDATE_POOL, len(self.oracle.low)))

    def _latent_sigma(self, sigma: np.ndarray) -> np.ndarray:
        """Subtract the fitted white-noise level from the predictive std.

        sklearn's return_std includes observation noise; EI should use the
        latent-function std. With normalize_y=True the noise level lives in
        normalized units, so rescale by the stored training std.
        """
        try:
            noise_norm = float(self.gp.kernel_.k2.noise_level)
            y_scale = float(getattr(self.gp, "_y_train_std", 1.0))
            return np.sqrt(np.maximum(sigma**2 - noise_norm * y_scale**2, 1e-12))
        except Exception:
            return np.maximum(sigma, 1e-9)

    def _ei_candidate(self) -> np.ndarray:
        pool = self._candidate_pool()
        mu, sigma = self.gp.predict(pool, return_std=True)
        sigma = self._latent_sigma(sigma)
        best_f = self._recommendation_mu()  # noise-robust incumbent
        z = (mu - best_f - EI_XI) / sigma
        ei = (mu - best_f - EI_XI) * sps.norm.cdf(z) + sigma * sps.norm.pdf(z)
        return pool[int(np.argmax(ei))]

    def act(self, action: int) -> float:
        """Execute one meta-action; returns the reward (delta true rec quality)."""
        name = ACTIONS[action]
        x: np.ndarray | None
        if name == "ei":
            x = self._ei_candidate()
        elif name == "exploit":
            pool = self._candidate_pool()
            mu = self.gp.predict(pool)
            x = pool[int(np.argmax(mu))]
        elif name == "replicate":
            x = self._recommendation()
        elif name == "probe":
            x = self.rng.uniform(self.oracle.low, self.oracle.high)
        elif name == "drop_worst":
            if len(self.X) >= MIN_POINTS_AFTER_DROP + 1:
                Xm = np.vstack(self.X)
                mu = self.gp.predict(Xm)
                worst = int(np.argmax(np.abs(np.asarray(self.y_obs) - mu)))
                del self.X[worst], self.y_obs[worst], self.y_true[worst], self.residuals[worst]
                self.gp = self._fit_gp()
                self.n_drops += 1
            x = self._ei_candidate()
        elif name == "rest":
            # Consume this slot's noise draw so streams stay slot-aligned
            # across policies, then break the serial error correlation.
            self.noise.sample()
            self.noise.reset()
            x = None
        else:
            raise ValueError(name)

        if x is not None:
            self._query(x, gp=self.gp)
            self.gp = self._fit_gp()
        new_quality = self._recommendation_quality()
        reward = new_quality - self.rec_quality
        self.rec_quality = new_quality
        self.rec_mu_history.append(self._recommendation_mu())
        self.t += 1
        return reward

    def final_inference_regret(self) -> float:
        return self.oracle.y_opt - self.rec_quality


# --------------------------------------------------------------------------
# Policies (picklable specs so loky workers can reconstruct them).
# --------------------------------------------------------------------------

def uniform_policy(state: int, rng: np.random.Generator) -> int:
    return int(rng.integers(N_ACTIONS))


def make_greedy_policy(Q: np.ndarray, coverage: np.ndarray, fallback: int = 0):
    """Greedy argmax over OBSERVED actions; unseen states fall back to EI.

    Masking unvisited actions prevents an optimistic Q=0 initialization from
    beating a genuinely negative-but-observed Q value.
    """
    def policy(state: int, rng: np.random.Generator) -> int:
        if coverage[state].sum() == 0:
            return fallback
        q = np.where(coverage[state] > 0, Q[state], -np.inf)
        return int(np.argmax(q))
    return policy


def make_fixed_policy(action: int):
    def policy(state: int, rng: np.random.Generator) -> int:
        return action
    return policy


def policy_from_spec(spec: tuple):
    kind = spec[0]
    if kind == "uniform":
        return uniform_policy
    if kind == "fixed":
        return make_fixed_policy(int(spec[1]))
    if kind == "greedy":
        return make_greedy_policy(np.asarray(spec[1]), np.asarray(spec[2]))
    raise ValueError(spec)


def run_episode(oracle: Oracle, condition: dict, seed_entropy: list[int], policy) -> dict:
    """policy(state, decision_rng) -> action index. Returns episode record."""
    ep = Episode(oracle, condition, seed_entropy)
    decision_rng = np.random.default_rng(np.random.SeedSequence(seed_entropy + [777]))
    transitions = []
    action_counts = [0] * N_ACTIONS
    for t in range(N_DECISIONS):
        s = ep.state()
        a = int(policy(s, decision_rng))
        action_counts[a] += 1
        r = ep.act(a)
        s_next = ep.state()
        transitions.append((s, a, r, s_next, t == N_DECISIONS - 1))
    return {
        "transitions": transitions,
        "final_inference_regret": ep.final_inference_regret(),
        "final_rec_quality": ep.rec_quality,
        "n_observations": len(ep.X),
        "n_drops": ep.n_drops,
        "action_counts": action_counts,
    }


def _episode_batch(specs: list[tuple[dict, list[int], tuple]]) -> list[dict]:
    """Worker entry: run a batch of (condition, entropy, policy_spec) episodes."""
    # loky ships this function by value, bypassing module import — re-apply
    # the warnings filter inside the worker.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    oracle = _get_oracle()
    return [run_episode(oracle, cond, entropy, policy_from_spec(pspec)) for cond, entropy, pspec in specs]


def run_parallel(specs: list[tuple[dict, list[int], tuple]], n_jobs: int, chunk: int = 24) -> list[dict]:
    batches = [specs[i:i + chunk] for i in range(0, len(specs), chunk)]
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_episode_batch)(b) for b in batches)
    return [rec for batch in results for rec in batch]


# --------------------------------------------------------------------------
# Offline Q-learning on logged transitions.
# --------------------------------------------------------------------------

def q_learning(transitions: list[tuple], gamma: float = GAMMA) -> np.ndarray:
    Q = np.zeros((N_STATES, N_ACTIONS))
    visits = np.zeros((N_STATES, N_ACTIONS))

    # Bootstrap only over actions observed in the data: an unvisited cell's
    # optimistic Q=0 must not leak into upstream targets.
    seen = np.zeros((N_STATES, N_ACTIONS), dtype=bool)
    for s, a, _, _, _ in transitions:
        seen[s, a] = True

    def masked_max(s_next: int) -> float:
        if not seen[s_next].any():
            return 0.0
        return float(np.max(Q[s_next][seen[s_next]]))

    rng = np.random.default_rng(1)
    idx = np.arange(len(transitions))
    for _ in range(Q_EPOCHS):
        rng.shuffle(idx)
        for i in idx:
            s, a, r, s_next, done = transitions[i]
            visits[s, a] += 1
            lr = Q_LR0 / (1.0 + visits[s, a] / 100.0)
            target = r if done else r + gamma * masked_max(s_next)
            Q[s, a] += lr * (target - Q[s, a])
    return Q


def coverage_of(transitions: list[tuple]) -> np.ndarray:
    cov = np.zeros((N_STATES, N_ACTIONS), dtype=int)
    for s, a, _, _, _ in transitions:
        cov[s, a] += 1
    return cov


STATE_NAMES = [
    f"{['early', 'late', 'endgame'][s // 4]}/{'surprise' if (s // 2) % 2 else 'calm'}/{'stall' if s % 2 else 'progress'}"
    for s in range(N_STATES)
]


def policy_table(Q: np.ndarray, coverage: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in range(N_STATES):
        if coverage[s].sum():
            best = int(np.argmax(np.where(coverage[s] > 0, Q[s], -np.inf)))
        else:
            best = -1
        rows.append({
            "state": s, "state_name": STATE_NAMES[s],
            "chosen_action": ACTIONS[best] if best >= 0 else "UNSEEN->ei",
            **{f"Q_{a}": Q[s, i] for i, a in enumerate(ACTIONS)},
            **{f"n_{a}": int(coverage[s, i]) for i, a in enumerate(ACTIONS)},
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Statistics helpers.
# --------------------------------------------------------------------------

def paired_comparison(piv: pd.DataFrame, a: str, b: str) -> dict | None:
    """Paired stats for policy a - policy b (negative favors a)."""
    if a not in piv.columns or b not in piv.columns:
        return None
    diff = (piv[a] - piv[b]).dropna()
    if len(diff) < 5 or float(np.std(diff, ddof=1)) == 0:
        return None
    _, p_t = sps.ttest_rel(piv[a].loc[diff.index], piv[b].loc[diff.index])
    try:
        _, p_w = sps.wilcoxon(diff)
    except ValueError:
        p_w = float("nan")
    return {
        "comparison": f"{a} vs {b}",
        "n_pairs": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "dz": float(diff.mean() / diff.std(ddof=1)),
        "p_t": float(p_t),
        "p_wilcoxon": float(p_w),
    }


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (monotone step-up)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank_from_end, i in enumerate(order[::-1]):
        rank = n - rank_from_end
        running = min(running, p[i] * n / rank)
        adj[i] = running
    return adj.tolist()


# --------------------------------------------------------------------------
# Experiment driver.
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    parser.add_argument("--train-episodes", type=int, default=300, help="training episodes per condition")
    parser.add_argument("--eval-episodes", type=int, default=200, help="evaluation episodes per condition per policy")
    parser.add_argument("--jitter-stds", type=str, default=None, help="comma-separated noise levels")
    parser.add_argument("--error-models", type=str, default=None, help="comma-separated subset of gaussian,ar1")
    parser.add_argument("--output-tag", type=str, default="", help="suffix for the output directory")
    parser.add_argument("--n-jobs", type=int, default=0, help="parallel workers (0 = auto)")
    args = parser.parse_args()

    jitter_stds = (
        [float(v) for v in args.jitter_stds.split(",")] if args.jitter_stds else DEFAULT_JITTER_STDS
    )
    error_models = args.error_models.split(",") if args.error_models else DEFAULT_ERROR_MODELS
    conditions = [{"error_model": em, "jitter_std": std} for em in error_models for std in jitter_stds]

    n_train = 4 if args.quick else args.train_episodes
    n_eval = 5 if args.quick else args.eval_episodes
    if args.quick:
        conditions = conditions[:1]
    n_jobs = args.n_jobs if args.n_jobs > 0 else max(1, min(14, (os.cpu_count() or 4) - 2))

    global OUTPUT_DIR
    if args.output_tag:
        OUTPUT_DIR = OUTPUT_DIR.with_name(OUTPUT_DIR.name + "_" + args.output_tag)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Loading eHMI data and fitting oracle ... (episode workers: {n_jobs})", flush=True)
    df = load_ehmi_frame()
    oracle = build_oracle(df, n_jobs=-1)
    composite = df[OBJECTIVE_COLUMNS].to_numpy(dtype=float).mean(axis=1)
    print(
        f"  rows={len(df)}  composite range=[{composite.min():.2f}, {composite.max():.2f}] "
        f"std={oracle.y_std:.3f}  y_opt={oracle.y_opt:.3f}"
    )
    print(f"  jitter levels {sorted({c['jitter_std'] for c in conditions})} vs composite std {oracle.y_std:.3f}")

    def cond_key(cond: dict) -> str:
        return f"{cond['error_model']}_std{cond['jitter_std']:g}"

    # ---- Phase 1: exploratory data collection ----------------------------
    print(f"\nPhase 1: exploratory policy, {n_train} episodes x {len(conditions)} conditions ...", flush=True)
    transitions_by_cond: dict[int, list[tuple]] = {}
    for ci, cond in enumerate(conditions):
        specs = [(cond, [TRAIN_SEED_BASE, ci, e], ("uniform",)) for e in range(n_train)]
        recs = run_parallel(specs, n_jobs)
        transitions_by_cond[ci] = [tr for rec in recs for tr in rec["transitions"]]
        print(f"  condition {cond} done ({(time.time() - t0):.0f}s)", flush=True)
    all_transitions = [tr for trs in transitions_by_cond.values() for tr in trs]

    # ---- Phase 2: offline Q-learning (pooled + per regime) ----------------
    print(f"\nPhase 2: tabular Q-learning on {len(all_transitions)} transitions "
          f"(pooled + {len(conditions)} per-regime tables) ...", flush=True)
    cov_pooled = coverage_of(all_transitions)
    Q_pooled = q_learning(all_transitions)
    policy_table(Q_pooled, cov_pooled).to_csv(OUTPUT_DIR / "learned_policy_pooled.csv", index=False)

    Q_regime: dict[int, np.ndarray] = {}
    cov_regime: dict[int, np.ndarray] = {}
    for ci, cond in enumerate(conditions):
        Q_regime[ci] = q_learning(transitions_by_cond[ci])
        cov_regime[ci] = coverage_of(transitions_by_cond[ci])
        tbl = policy_table(Q_regime[ci], cov_regime[ci])
        tbl.to_csv(OUTPUT_DIR / f"learned_policy_{cond_key(cond)}.csv", index=False)
        print(f"  {cond_key(cond)}: chosen actions = "
              + ", ".join(f"{n.split('/', 1)[0][0]}{s % 4}:{a}" for s, (n, a) in
                          enumerate(zip(tbl['state_name'], tbl['chosen_action']))), flush=True)
    print(f"  pooled coverage min/median per (s,a) = {cov_pooled.min()}/{int(np.median(cov_pooled))}; "
          f"per-regime min = {min(int(cov_regime[ci].min()) for ci in cov_regime)}")

    # ---- Phase 3: evaluation on fresh seeds --------------------------------
    policy_specs_by_cond: dict[int, dict[str, tuple]] = {
        ci: {
            "q_regime": ("greedy", Q_regime[ci], cov_regime[ci]),
            "q_pooled": ("greedy", Q_pooled, cov_pooled),
            "always_ei": ("fixed", ACTIONS.index("ei")),
            "uniform_meta": ("uniform",),
            "always_probe": ("fixed", ACTIONS.index("probe")),
        }
        for ci in range(len(conditions))
    }
    policy_names = list(next(iter(policy_specs_by_cond.values())).keys())
    print(f"\nPhase 3: evaluating {policy_names} on {n_eval} fresh seeds per condition ...", flush=True)
    rows = []
    for ci, cond in enumerate(conditions):
        specs, keys = [], []
        for e in range(n_eval):
            for pname, pspec in policy_specs_by_cond[ci].items():
                specs.append((cond, [EVAL_SEED_BASE, ci, e], pspec))
                keys.append((e, pname))
        recs = run_parallel(specs, n_jobs)
        for (e, pname), rec in zip(keys, recs):
            rows.append({
                "error_model": cond["error_model"], "jitter_std": cond["jitter_std"],
                "policy": pname, "eval_seed": e,
                "final_inference_regret": rec["final_inference_regret"],
                "final_rec_quality": rec["final_rec_quality"],
                "n_observations": rec["n_observations"],
                "n_drops": rec["n_drops"],
                **{f"a_{a}": rec["action_counts"][i] for i, a in enumerate(ACTIONS)},
            })
        print(f"  condition {cond} done ({(time.time() - t0):.0f}s)", flush=True)

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(OUTPUT_DIR / "eval_results.csv", index=False)

    # ---- Summary stats -----------------------------------------------------
    lines = []
    lines.append("Adaptive meta-policy experiment v2 - summary")
    lines.append(f"actions: {ACTIONS}")
    lines.append(f"train episodes/condition={n_train}, eval episodes/condition/policy={n_eval}")
    lines.append(f"transitions={len(all_transitions)}, pooled coverage min/median per (s,a)="
                 f"{cov_pooled.min()}/{int(np.median(cov_pooled))}")
    lines.append("")

    comp_rows = []
    for ci, cond in enumerate(conditions):
        sub = eval_df[(eval_df["error_model"] == cond["error_model"]) & (eval_df["jitter_std"] == cond["jitter_std"])]
        lines.append(f"condition: {cond['error_model']} jitter_std={cond['jitter_std']}")
        for pname in policy_names:
            vals = sub[sub["policy"] == pname]["final_inference_regret"]
            lines.append(f"  {pname:13s} final inference regret: mean={vals.mean():.4f}  se={vals.sem():.4f}")
        piv = sub.pivot(index="eval_seed", columns="policy", values="final_inference_regret")
        for a, b, family in [
            ("q_regime", "always_ei", "primary"),
            ("q_pooled", "always_ei", "secondary"),
            ("q_regime", "q_pooled", "secondary"),
        ]:
            res = paired_comparison(piv, a, b)
            if res:
                res.update({"error_model": cond["error_model"], "jitter_std": cond["jitter_std"], "family": family})
                comp_rows.append(res)
        lines.append("")

    comp_df = pd.DataFrame(comp_rows)
    if not comp_df.empty:
        # BH within each test FAMILY across its member tests: primary =
        # q_regime vs always_ei (6 tests), secondary = the two other
        # comparisons pooled (12 tests).
        comp_df["p_t_bh"] = np.nan
        comp_df["p_wilcoxon_bh"] = np.nan
        for _, fam in comp_df.groupby("family"):
            comp_df.loc[fam.index, "p_t_bh"] = bh_adjust(fam["p_t"].tolist())
            comp_df.loc[fam.index, "p_wilcoxon_bh"] = bh_adjust(fam["p_wilcoxon"].tolist())
        comp_df.to_csv(OUTPUT_DIR / "paired_comparisons.csv", index=False)

        lines.append("Paired comparisons (negative mean diff favors the first policy; BH within test family):")
        for _, r in comp_df.iterrows():
            lines.append(
                f"  [{r['error_model']} std={r['jitter_std']:g}] {r['comparison']:24s} "
                f"mean diff={r['mean_diff']:+.4f}  dz={r['dz']:+.2f}  "
                f"t p={r['p_t']:.4f} (BH {r['p_t_bh']:.4f})  "
                f"wilcoxon p={r['p_wilcoxon']:.4f} (BH {r['p_wilcoxon_bh']:.4f})"
            )
        lines.append("")

    # Empirical mean reward per action from the exploratory log: the honest
    # yardstick for judging learned action choices (rest's reward is exactly
    # 0, so resting is rational wherever the alternatives average below 0).
    lines.append("Empirical mean reward per action in the exploratory log (per condition):")
    for ci, cond in enumerate(conditions):
        by_action: dict[int, list[float]] = {a: [] for a in range(N_ACTIONS)}
        for _, a, r, _, _ in transitions_by_cond[ci]:
            by_action[a].append(r)
        parts = "  ".join(
            f"{ACTIONS[a]}={np.mean(v):+.4f}" for a, v in by_action.items() if v
        )
        lines.append(f"  [{cond_key(cond)}] {parts}")
    lines.append("")

    rest_use = (
        eval_df[eval_df["policy"].isin(["q_regime", "q_pooled"])]
        .groupby(["policy", "error_model", "jitter_std"])[["a_rest", "a_drop_worst", "n_drops"]].mean()
    )
    lines.append("Mean rest / drop_worst actions (and realized drops) per episode, learned policies:")
    for (pname, em, std), row in rest_use.iterrows():
        lines.append(
            f"  {pname:9s} {em:8s} std={std:g}: rest={row['a_rest']:.2f} "
            f"drop_worst={row['a_drop_worst']:.2f} (realized drops={row['n_drops']:.2f})"
        )

    summary = "\n".join(lines)
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    meta = {
        "version": 2,
        "conditions": conditions, "n_train": n_train, "n_eval": n_eval, "n_jobs": n_jobs,
        "gamma": GAMMA, "actions": ACTIONS, "state_names": STATE_NAMES,
        "total_queries": TOTAL_QUERIES, "initial_samples": INITIAL_SAMPLES,
        "y_opt": oracle.y_opt, "composite_std": oracle.y_std,
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- Figure -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conds = [(c["error_model"], c["jitter_std"]) for c in conditions]
        colors = ["#4C72B0", "#8172B3", "#DD8452", "#55A868", "#C44E52"]
        fig, axes = plt.subplots(1, max(len(conds), 1), figsize=(3.6 * len(conds), 3.8), sharey=True, squeeze=False)
        for ax, (em, std) in zip(axes[0], conds):
            sub = eval_df[(eval_df["error_model"] == em) & (eval_df["jitter_std"] == std)]
            means = sub.groupby("policy")["final_inference_regret"].mean().reindex(policy_names)
            sems = sub.groupby("policy")["final_inference_regret"].sem().reindex(policy_names)
            ax.bar(range(len(means)), means.values, yerr=sems.values, capsize=3, color=colors)
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index, rotation=35, ha="right", fontsize=7)
            ax.set_title(f"{em}, std={std}", fontsize=10)
        axes[0][0].set_ylabel("final inference simple regret (true)")
        fig.suptitle(
            "Adaptive meta-policies (offline Q-learning, per-regime + pooled) vs static baselines\n"
            "(bars: between-seed mean +/- SEM; inference is paired - see paired_comparisons.csv)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "results.png", dpi=150)
        print(f"figure -> {OUTPUT_DIR / 'results.png'}")
    except Exception as exc:  # matplotlib optional
        print(f"figure skipped: {exc}", file=sys.stderr)

    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
