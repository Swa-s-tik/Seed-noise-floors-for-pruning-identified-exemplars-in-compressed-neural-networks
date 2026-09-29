# Per-model fidelity vs Hooker's tstat.csv -> results/fidelity.csv (both decoders). No population-level quantity.
import json, os, re
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
inv = json.load(open(os.path.join(ROOT, "INVENTORY.json")))["runs"]
y = pd.read_csv(os.path.join(ROOT, "results", "preds", "image_order.csv")).label.values
t = pd.read_csv(os.path.join(ROOT, "prior-gate", "g13", "tstat.csv"), usecols=["dir", "class", "recall"])
t["key"] = t.dir.str.replace(r".*/class_accuracy_/", "", regex=True).str.replace("/model.ckpt-32000", "")
t["cid"] = t["class"].str.rsplit("_", n=1).str[1].astype(int)
correct = {k: np.bincount(g.cid, weights=np.round(g.recall * 50), minlength=1000) for k, g in t.groupby("key")}  # 50 imgs/class

rows = []
for dec, d in (("ifast", "preds"), ("islow", "preds_islow")):
    for rid, r in sorted(inv.items()):
        f = os.path.join(ROOT, "results", d, rid + ".npz")
        if not (r["complete"] and os.path.exists(f)):
            continue
        z = np.load(f)
        top1, top5 = (z["top1"] == y).mean(), (z["top5"] == y[:, None]).any(1).mean()
        dc = np.bincount(y[z["top1"] == y], minlength=1000) - correct[r["key"]]
        rows.append({"decoder": dec, "run_id": rid, "role": r["role"], "sparsity": r["sparsity"], "cohort": r["cohort"],
                     "top1": top1, "tstat_top1": r["tstat_top1"], "delta_top1_pp": 100 * (top1 - r["tstat_top1"]),
                     "top5": top5, "tstat_top5": r["tstat_top5"], "delta_top5_pp": 100 * (top5 - r["tstat_top5"]),
                     "class_absdiff_sum": int(np.abs(dc).sum()), "classes_differing": int((dc != 0).sum())})
df = pd.DataFrame(rows)
df["outside_0.1pp"] = df.delta_top1_pp.abs() > 0.1
df.to_csv(os.path.join(ROOT, "results", "fidelity.csv"), index=False, float_format="%.6g")
for dec, g in df.groupby("decoder"):
    print(dec, "models", len(g), "within 0.1pp", int((~g["outside_0.1pp"]).sum()),
          "| delta pp mean %.4f sd %.4f min %.4f max %.4f" % (g.delta_top1_pp.mean(), g.delta_top1_pp.std(), g.delta_top1_pp.min(), g.delta_top1_pp.max()),
          "| class|d| median", g.class_absdiff_sum.median())
    print(g.reindex(g.delta_top1_pp.abs().sort_values(ascending=False).index).head(5)[["run_id", "role", "sparsity", "delta_top1_pp", "delta_top5_pp", "class_absdiff_sum"]].to_string(index=False))
