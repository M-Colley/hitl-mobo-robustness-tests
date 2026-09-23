# C1/D7: dominated designs, recall and shortfall of the multi-objective deployed front.
# Moved from the 2026-09-23 register triage; run all checks with
#   python scripts/run_review_checks.py
"""C1/D7: can '5.2 of 10.9 designs truly dominated', 'misses most truly
non-dominated evaluated designs' and 'hypervolume shortfall exceeds that of the
evaluated set in every cell' be reproduced from the MO logs? No GP refit: the
reported front is ND of the observed vectors, exactly the arm's own rule."""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import replay_mo_front as rf  # noqa: E402
import boba_multiobjective as mob  # noqa: E402

inp = REPO / "output-boba-mo"
stats = rf.load_run_stats(inp)
groups, orphans = rf.discover_groups(inp, acquisitions=set(rf.HYPERVOLUME_ACQUISITIONS), error_model="gaussian",
                                     suffix_for=lambda ch, s: "")
print(len(groups), "groups;", len(orphans), "orphans; problems", sorted({g.dataset for g in groups}))
rows = []
for g in groups:
    if g.dataset == "dh2":
        continue
    spec = mob.MO_BENCHMARKS[g.dataset]
    ref = np.asarray(stats[g.dataset]["ref_point"], float)
    max_hv = float(stats[g.dataset]["max_hv"])
    for std, onset, path in g.noisy:
        run = rf.load_run(path, spec.num_objectives)
        Yo, Yt = run["Y_observed"], run["Y_true"]
        front = rf.non_dominated(Yo)
        dom = rf.strictly_dominated(Yt)
        true_nd = np.flatnonzero(~dom)
        hv_front = rf.hypervolume(Yt[front], ref)
        hv_all = rf.hypervolume(Yt, ref)
        rows.append({"dataset": g.dataset, "acq": g.acquisition, "seed": g.seed, "std": std, "onset": onset,
                     "front": len(front), "dominated": int(dom[front].sum()),
                     "true_nd": len(true_nd), "recall": len(set(front) & set(true_nd)) / max(len(true_nd), 1),
                     "short_front": max_hv - hv_front, "short_all": max_hv - hv_all,
                     "check": abs(hv_front - run["logged_inference"])})
d = pd.DataFrame(rows)
print("max |replayed - logged| front hypervolume:", d.check.max())
cell = d.groupby(["std", "onset"]).agg(n=("front", "size"), front=("front", "mean"), dominated=("dominated", "mean"),
                                        true_nd=("true_nd", "mean"), recall=("recall", "mean"),
                                        short_front=("short_front", "mean"), short_all=("short_all", "mean"))
cell["front_short_exceeds_all"] = cell.short_front > cell.short_all
print(cell.round(3).to_string())
