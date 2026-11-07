"""The rank rule's family-wise test in the scope of its recoveries, with tied ratings at the cap broken at random.

heldout_remedies.py (section 4b) tests the rank refit (ordinal_lcb1) on the held-out seeds 12-16 of the
gross-fault and capped-scale arms, with the standard process shipping the EARLIEST design tied at the top
rating. Under the cap that convention adds to the measured cost (Appendix app:shiprules), so the paper names
the missing test with random ties as a gap. This check runs it: the standard process's regret, for the noisy
run and its clean twin, becomes the exact expectation over a uniformly random choice among the tied designs
(checks/tie_break.py's remedies_with_uniform_reference, from its scan); the rank rule's own regret does not
change, since it chooses by a GP criterion with no exact ties. Each arm's per-landscape gains are tested with
heldout_remedies' own Wilcoxon test, and Holm is recomputed over its core family with the two main-sweep
rank-rule tests replaced by the two arms' ordinal_lcb1 tests (column p_holm_core_lcb1_replaced_test of
section 4b), once with the published first-index tests (the control, which must reproduce section 4b) and
once with the random-tie tests.

    python scripts/checks/rank_rule_random_ties.py > output-boba/analysis/review/register_checks/rank_rule_random_ties.txt

Needs review/tie_break/tie_runs.parquet (`tie_break.py scan`, git-ignored) and the arms' hitl_remedies_heldout
per-run tables; reads the published core family from review/heldout_remedies.csv.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import heldout_remedies as hr  # noqa: E402
import tie_break as tb  # noqa: E402

ARMS = ("output-boba-spike", "output-boba-ceiling")
PROC = hr.PAPER_RANK_RULE
HELDOUT = REPO / "output-boba" / "analysis" / "review" / "heldout_remedies.csv"


def arm_test(per_run: pd.DataFrame, opt_z: dict[str, float]) -> dict:
    """heldout_remedies' per-landscape gains, Wilcoxon p and recovery of the rank rule on one per-run table."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "per_run.csv.gz"
        per_run.to_csv(path, index=False)
        long = hr.load_remedies(path, opt_z)
    cube = hr.Cube(long)
    if tuple(cube.seeds) != hr.TEST_SEEDS:
        raise ValueError(f"seeds {cube.seeds}, not the test seeds {hr.TEST_SEEDS}")
    g = hr.per_landscape_gains(cube, PROC, hr.TEST_SEEDS)
    rec = hr.recovery_summary(long, PROC, hr.TEST_SEEDS)
    return {"p": hr._wilcoxon_p(g), "gaining": int((g > 0).sum()), "n": int(np.isfinite(g).sum()),
            "recovery": rec["value"], "lo": rec["lo"], "hi": rec["hi"]}


def main() -> None:
    opt_z = tb._opt_z()
    runs = tb.load_runs()
    published = pd.read_csv(HELDOUT, low_memory=False)
    core = published[(published["section"] == "4_holm") & (published["family"] == "core")]
    base_other = core.loc[core["group"] != "rank rule", "p_test"].to_numpy(float) if "group" in core else None
    if base_other is None:
        raise KeyError("heldout_remedies.csv lacks the 'group' column of section 4_holm")
    pub4b = published[(published["section"] == "4b_holm_rank_scope") & (published["candidate"] == PROC)].set_index("problem")
    results = {}
    for arm in ARMS:
        per = pd.read_csv(REPO / arm / "analysis" / "hitl_remedies_heldout" / "hitl_remedies_per_run.csv.gz",
                          low_memory=False)
        uni = tb.remedies_with_uniform_reference(per, runs, arm)
        results[arm] = {"first": arm_test(per, opt_z), "uniform": arm_test(uni, opt_z)}
    lines = ["The rank refit (ordinal_lcb1) on the held-out seeds 12-16 of the gross-fault and capped-scale arms, with the",
             "standard process shipping the earliest tied design (first) or a uniformly random one (uniform, the exact",
             "expectation); two-sided Wilcoxon over the twenty landscapes (heldout_remedies' test); Holm over section 4's",
             f"core family with its two main-sweep rank-rule tests replaced by the two arms' tests ({len(base_other) + 2} tests)."]
    for conv in ("first", "uniform"):
        ps = np.array([results[a][conv]["p"] for a in ARMS])
        adj = multipletests(np.concatenate([base_other, ps]), method="holm")[1][len(base_other):]
        for arm, a in zip(ARMS, adj):
            r = results[arm][conv]
            r["p_holm"] = float(a)
            lines.append(f"  {conv:7s} {arm:20s} recovers {100 * r['recovery']:.1f}% [{100 * r['lo']:.0f}, "
                         f"{100 * r['hi']:.0f}] of the cost, gains on {r['gaining']}/{r['n']} landscapes, "
                         f"Wilcoxon p {r['p']:.2g}, Holm p {r['p_holm']:.2g}")
    for arm in ARMS:
        f = results[arm]["first"]
        want = pub4b.loc[arm]
        gaps = (abs(f["p"] - float(want["p_test"])), abs(f["p_holm"] - float(want["p_holm_core_lcb1_replaced_test"])),
                abs(f["recovery"] - float(want["test_value"])))
        if max(gaps) > 1e-12:
            raise ValueError(f"{arm}: the first-index control does not reproduce section 4b (gaps {gaps})")
    lines.append("control: the first-index rows reproduce heldout_remedies.csv section 4b (p, Holm p, recovery) to 1e-12")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
