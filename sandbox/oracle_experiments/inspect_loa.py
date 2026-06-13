"""Check LevelOfAutonomy discreteness for provoice."""
import numpy as np
import harness

d = harness.load_dataset("provoice")
loa = d.X[:, d.param_columns.index("LevelOfAutonomy")]
u, c = np.unique(np.round(loa, 3), return_counts=True)
print("nuniq raw:", len(np.unique(loa)))
print("value:count for values with count>=5:")
for v, n in zip(u, c):
    if n >= 5:
        print(f"  {v:.3f}: {n}")
print("histogram (10 bins):", np.histogram(loa, bins=10)[0])
