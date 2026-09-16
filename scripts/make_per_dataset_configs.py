"""Split datasets.json into one config per dataset, with matched error grids.

The fitted-oracle companion has to be run on the same error scale as the
synthetic arm, and that scale is per-dataset: an error of one landscape standard
deviation is 0.365 rating points on eHMI and 0.374 on OptiCarVis. The
data-driven driver takes a single global ``--jitter-stds``, so the companion
needs one invocation per dataset, each with its own grid. This writes those
configs and prints the grids.

  python scripts/make_per_dataset_configs.py --anchor output/noise_anchor.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# The synthetic arm's grid, in landscape standard deviations.
SIGMA_GRID = (0.05, 0.25, 1.0, 5.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=Path, default=Path("datasets.json"))
    parser.add_argument("--anchor", type=Path, default=Path("output/noise_anchor.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/per_dataset"))
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.datasets.read_text(encoding="utf-8"))
    anchor = pd.read_csv(args.anchor).set_index("dataset")

    manifest = {}
    for entry in payload:
        name = entry["name"]
        if name not in anchor.index:
            print(f"skipping {name}: no sigma_f in {args.anchor}")
            continue
        sigma_f = float(anchor.loc[name, "sigma_f"])
        grid = [round(s * sigma_f, 6) for s in SIGMA_GRID]
        path = args.out_dir / f"datasets-{name}.json"
        path.write_text(json.dumps([entry], indent=2), encoding="utf-8")
        manifest[name] = {
            "config": str(path),
            "sigma_f": sigma_f,
            "jitter_stds": ",".join(str(g) for g in grid),
            "sigma_grid": list(SIGMA_GRID),
        }
        print(f"{name:<12s} sigma_f={sigma_f:.4f}  --jitter-stds {manifest[name]['jitter_stds']}")

    out = args.out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
