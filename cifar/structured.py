# PREREG-ADDENDUM-4: CIFAR-10 structured (L1 filter) pruning arm against the 60-seed dense floor.
# usage: python cifar/structured.py  -> results/analysis/cifar/structured.json
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import analysis as A  # noqa: E402  (modal = Hooker's mode().max(), summ, img_boot)

K, SP = 1000, (0.3, 0.7, 0.9)
COND = ["split13", *[f"st13_{s}" for s in SP], "split30", "boot", "mcboot"]  # ADDENDUM-4: SeedSequence([20260928, 40]) children, this order
RNG = {c: np.random.default_rng(s) for c, s in zip(COND, np.random.SeedSequence([20260928, 40]).spawn(len(COND)))}


def main():
    d = os.path.join(ROOT, "results", "cifar", "preds")
    D = np.stack([np.load(os.path.join(d, f"s0.0_seed{i}.npz"))["top1"].astype(np.int16) for i in range(60)])
    y = np.load(os.path.join(d, "s0.0_seed0.npz"))["labels"].astype(np.int64)
    ST = {s: np.stack([np.load(os.path.join(d, f"st{s}_seed{1000 * k + i}.npz"))["top1"].astype(np.int16) for i in range(30)])
          for s, k in ((0.3, 13), (0.7, 17), (0.9, 19))}
    UN = {s: np.stack([np.load(os.path.join(d, f"s{s}_seed{1000 * k + i}.npz"))["top1"].astype(np.int16) for i in range(30)])
          for s, k in ((0.3, 3), (0.7, 7), (0.9, 9))}
    acc = {"dense60": float((D == y).mean()), **{f"structured_{s}": float((ST[s] == y).mean()) for s in SP}}
    per_run = {s: (ST[s] == y).mean(1) for s in SP}
    out = {"mean_top1": acc, "collapsed(mean<0.5)": {str(s): bool(per_run[s].mean() < 0.5) for s in SP},
           "runs_below_0.85": {str(s): int((per_run[s] < 0.85).sum()) for s in SP}}
    N = len(y)
    # 13 v 13: floor from the 60 dense seeds; structured draws 13 of 30
    c = {"floor": np.zeros(K, int), **{s: np.zeros(K, int) for s in SP}}
    f = {k: np.zeros(N) for k in c}
    for k in range(K):
        p = RNG["split13"].permutation(60)
        mA = A.modal(D[p[:13]])
        pies = {"floor": mA != A.modal(D[p[13:26]]), **{s: mA != A.modal(ST[s][RNG[f"st13_{s}"].permutation(30)[:13]]) for s in SP}}
        for key, v in pies.items():
            c[key][k] = v.sum(); f[key] += v
    F = A.summ(c["floor"], RNG["mcboot"])
    out["n13"] = {"floor": F, **{str(s): {**A.summ(c[s]), "z": (c[s].mean() - F["mean"]) / F["sd"]} for s in SP}}
    out["decisions_secondary"] = {"H1st_holds(|z0.3|<=3)": bool(abs(out["n13"]["0.3"]["z"]) <= 3),
                                  "H2st_0.7(z>3)": bool(out["n13"]["0.7"]["z"] > 3), "H2st_0.9(z>3)": bool(out["n13"]["0.9"]["z"] > 3)}
    out["image_bootstrap_n13"] = A.img_boot({f"n13_{k}": v / K for k, v in f.items()},
                                           [(f"D_{s}", f"n13_{s}", "n13_floor") for s in SP], RNG["boot"], N)
    # 30 v 30: dense half of 30 (of 60) against all 30 structured
    c30 = {"floor": np.zeros(K, int), **{s: np.zeros(K, int) for s in SP}}
    ms = {s: A.modal(ST[s]) for s in SP}
    for k in range(K):
        p = RNG["split30"].permutation(60)
        mA = A.modal(D[p[:30]])
        c30["floor"][k] = (mA != A.modal(D[p[30:]])).sum()
        for s in SP:
            c30[s][k] = (mA != ms[s]).sum()
    F30 = A.summ(c30["floor"])
    out["n30"] = {"floor": F30, **{str(s): {**A.summ(c30[s]), "z": (c30[s].mean() - F30["mean"]) / F30["sd"]} for s in SP}}
    # descriptive: unstructured arm (ADDENDUM-2 runs) against the same 60-seed 13 v 13 floor draws is not re-drawn here;
    # its full-population 30 v 30 count is given for side-by-side reading
    mD30 = A.modal(D[:30])
    out["full_population_30v30_first30dense"] = {"structured": {str(s): int((mD30 != ms[s]).sum()) for s in SP},
                                                 "unstructured": {str(s): int((mD30 != A.modal(UN[s])).sum()) for s in SP}}
    os.makedirs(os.path.join(ROOT, "results", "analysis", "cifar"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "cifar", "structured.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("mean_top1", "decisions_secondary", "runs_below_0.85")}, indent=1, default=float))
    print({k: (round(v["mean"], 1), round(v.get("z", 0), 2)) for k, v in out["n13"].items()})
    print({k: (round(v["mean"], 1), round(v.get("z", 0), 2)) for k, v in out["n30"].items()})


if __name__ == "__main__":
    main()
