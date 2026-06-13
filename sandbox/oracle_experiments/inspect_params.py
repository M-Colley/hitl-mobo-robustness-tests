"""Quick inspection of param columns / ranges to design features."""
import numpy as np
import harness

for name in harness.DATASET_NAMES:
    d = harness.load_dataset(name)
    print(f"=== {name}: {d.X.shape[0]} rows x {d.X.shape[1]} params, {len(np.unique(d.groups))} users")
    for i, c in enumerate(d.param_columns):
        col = d.X[:, i]
        u = np.unique(col)
        print(f"  {c:<45} min={col.min():.3f} max={col.max():.3f} nuniq={len(u)}"
              + (f" vals={u[:8]}" if len(u) <= 8 else ""))
    print(f"  y: min={d.y.min():.3f} max={d.y.max():.3f} std={d.y.std():.3f}")
