"""Compact table from a harness RESULTS_JSON block on stdin."""
import json
import sys

text = sys.stdin.read()
block = text.split("RESULTS_JSON_BEGIN")[1].split("RESULTS_JSON_END")[0]
for r in json.loads(block)["rows"]:
    r2 = r.get("r2_mean_fold", r.get("r2_pooled"))
    rmse = r.get("rmse_pooled", float("nan"))
    ystd = r.get("y_std", r.get("y_test_std", float("nan")))
    model = r.get("model", "?")
    print(
        f"{r['dataset']:<11} {model:<22} {r['protocol']:<19} "
        f"r2={r2:7.3f}  rmse={rmse:.3f}  y_std={ystd:.3f}"
    )
