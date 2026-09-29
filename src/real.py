# Breadth (a), PREREG-ADDENDUM-1 §10: ImageNet-ReaL labels (Beyer et al. 2020; data/real/real.json, sha256 in logs).
# PIE counts are label-free; this reports ReaL top-1 per model, M4 with ReaL-based recall on the M1/M2 splits (n=13),
# and label-quality rates among PIEs vs non-PIEs (weighted by the per-image PIE frequency f over splits).
# usage: python src/real.py ifast|islow -> results/analysis/<dec>/real.json
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A  # noqa: E402


def main(dec):
    pop = A.Pop(dec)
    o = pd.read_csv(os.path.join(A.ROOT, "results", "preds", "image_order.csv"))
    real = json.load(open(os.path.join(A.ROOT, "data", "real", "real.json")))  # index = val id - 1
    R = [set(real[int(v) - 1]) for v in o.path.str.extract(r"val_(\d+)")[0]]
    has = np.array([len(r) > 0 for r in R]); multi = np.array([len(r) > 1 for r in R])
    wrong = np.array([len(r) > 0 and y not in r for r, y in zip(R, pop.y)])  # original label not in ReaL set
    real_ok = lambda t: np.array([t[i] in R[i] for i in range(pop.N)])
    out = {"n_images_with_real_labels": int(has.sum()), "per_model_real_top1": {}}
    den = np.bincount(pop.y[has], minlength=1000)
    for k, t in pop.P.items():
        ok = real_ok(t) & has
        out["per_model_real_top1"][k] = float(ok.sum() / has.sum())
        pop.R[k] = np.bincount(pop.y[ok], minlength=1000) / den  # ReaL-based per-class recall (original class, ReaL-correct)
    h13 = A.halves_list(A.RNG["split13"], 13, pop.base, pop.thr)
    p13 = {s: A.draws(A.RNG[f"pruned13_{s}"], pop.pr[s], 13) for s in A.SPARS}
    _, _, m4r, m4own = A.run_block(pop, h13, p13, m4=True)
    out["M4_real"] = {str(k): {"uncorrected": A.summ(m4r[k][:, 0]), "bh": A.summ(m4r[k][:, 1]),
                               "own_mean_uncorrected": A.summ(m4own[k][:, 0]), "own_mean_bh": A.summ(m4own[k][:, 1])} for k in m4r}
    f = np.load(os.path.join(A.ROOT, "results", "analysis", dec, "f.npz"))
    rates = {}
    for key in ("n13_floor", *[f"n13_{s}" for s in A.SPARS]):
        w = f[key].astype(np.float64)
        rate = lambda m, ww: float((ww * m).sum() / ww.sum())
        rates[key] = {"expected_pie_count": float(w.sum()),
                      **{f"{nm}_pie": rate(m, w) for nm, m in (("empty_real", ~has), ("multi_label", multi), ("orig_label_not_in_real", wrong))},
                      **{f"{nm}_nonpie": rate(m, 1 - w) for nm, m in (("empty_real", ~has), ("multi_label", multi), ("orig_label_not_in_real", wrong))}}
    out["label_quality_pie_vs_nonpie"] = rates
    json.dump(out, open(os.path.join(A.ROOT, "results", "analysis", dec, "real.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_model_real_top1"}, indent=1)[:3000])


if __name__ == "__main__":
    main(sys.argv[1])
