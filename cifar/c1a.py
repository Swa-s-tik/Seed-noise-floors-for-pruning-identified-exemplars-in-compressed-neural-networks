# PREREG-ADDENDUM-3 C1(a): measured CIFAR-10 30-vs-30 dense floor (60 dense seeds), validation of the n -> 30 extrapolation,
# and full 30-vs-30 pruned counts against it.  usage: python cifar/c1a.py  -> results/analysis/cifar/c1a.json
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import analysis as A  # noqa: E402

K, SP = 1000, (0.3, 0.7, 0.9)
COND = ["split30", "boot", "mcboot"]  # ADDENDUM-3: children of SeedSequence([20260928, 11]) in this order
RNG = {c: np.random.default_rng(s) for c, s in zip(COND, np.random.SeedSequence([20260928, 11]).spawn(len(COND)))}
PRED = {"a+b/sqrt(n), n>=3": 109.56, "a+b/sqrt(n), n>=5": 110.76, "a+b*n^-c, n>=3": 109.63, "a+b*n^-c, n>=5": 114.81}  # frozen in ADDENDUM-3
TOL = (0.95 * min(PRED.values()), 1.05 * max(PRED.values()))


def main():
    d = os.path.join(ROOT, "results", "cifar", "preds")
    dense = [np.load(os.path.join(d, f"s0.0_seed{i}.npz"))["top1"].astype(np.int16) for i in range(60)]
    y = np.load(os.path.join(d, "s0.0_seed0.npz"))["labels"].astype(np.int64)
    pr = {s: np.stack([np.load(os.path.join(d, f"s{s}_seed{1000 * k + i}.npz"))["top1"].astype(np.int16) for i in range(30)])
          for s, k in ((0.3, 3), (0.7, 7), (0.9, 9))}
    D = np.stack(dense); N = len(y)
    mp = {s: A.modal(pr[s]) for s in SP}
    c = {"floor": np.zeros(K, int), **{s: np.zeros(K, int) for s in SP}}
    f = {key: np.zeros(N) for key in c}
    for k in range(K):
        p = RNG["split30"].permutation(60)
        mA = A.modal(D[p[:30]])
        pies = {"floor": mA != A.modal(D[p[30:]]), **{s: mA != mp[s] for s in SP}}
        for key, v in pies.items():
            c[key][k] = v.sum(); f[key] += v
    f = {f"n30_{key}": v / K for key, v in f.items()}
    F = A.summ(c["floor"], RNG["mcboot"])
    m = F["mean"]
    out = {"measured_F30": F, "predictions": PRED, "tolerance": TOL,
           "extrapolation_validated": bool(TOL[0] <= m <= TOL[1]),
           "relative_error_per_fit": {k: (v - m) / m for k, v in PRED.items()},
           "pruned_30v30": {str(s): {**A.summ(c[s]), "z_vs_F30": (c[s].mean() - m) / F["sd"]} for s in SP},
           "image_bootstrap": A.img_boot(f, [("F30", "n30_floor", None), *[(f"D_{s}", f"n30_{s}", "n30_floor") for s in SP]], RNG["boot"], N),
           "dense_top1_seeds30_59": [float((dense[i] == y).mean()) for i in range(30, 60)]}
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "cifar", "c1a.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "dense_top1_seeds30_59"}, indent=1, default=float))


if __name__ == "__main__":
    main()
