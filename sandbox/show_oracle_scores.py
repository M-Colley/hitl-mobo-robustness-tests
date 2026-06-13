import json

d = json.load(open("output/best_oracle_models.json"))
for ds in d["datasets"]:
    for obj, e in ds["objectives"].items():
        ranked = sorted(e["scores"].items(), key=lambda kv: -kv[1])
        top = ", ".join(f"{k}={v:.3f}" for k, v in ranked[:3])
        best = e["scores"][e["best_model"]]
        print(f"{ds['name']}/{obj}: best={e['best_model']} R2={best:.3f} | top3: {top}")
