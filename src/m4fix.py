# PREREG-ADDENDUM-3 C1(b): M4 with zero-variance classes excluded (both populations constant -> Welch undefined/infinite),
# on the same splits/draws as the literal M4. usage: python src/m4fix.py -> results/analysis/m4fix.json
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A  # noqa: E402

ROOT = A.ROOT


def welch_fixed(Ra, Rb, ca, cb):
    keep = ~((np.ptp(Ra, 0) == 0) & (np.ptp(Rb, 0) == 0))  # exact: all per-model recalls identical in both groups
    with np.errstate(all="ignore"):
        p = stats.ttest_ind(Ra[:, keep] - ca, Rb[:, keep] - cb, axis=0, equal_var=False).pvalue
    assert not np.isnan(p).any()
    return int((p < 0.05).sum()), int((stats.false_discovery_control(p, method="bh") <= 0.05).sum()), int(keep.sum())


def block(rec, halves, pruned, c0, cs):
    """rec(ids)->(n,classes) recalls. Returns per key an array (K, 3): uncorrected, bh, m."""
    out = {"floor": [], **{s: [] for s in pruned}}
    for k, (a, b) in enumerate(halves):
        RA = rec(a)
        out["floor"].append(welch_fixed(RA, rec(b), c0, c0))
        for s, pk in pruned.items():
            out[s].append(welch_fixed(RA, rec(pk[k]), c0, cs[s]))
    return {str(k): {"uncorrected": A.summ(np.array(v)[:, 0]), "bh": A.summ(np.array(v)[:, 1]), "classes_tested": A.summ(np.array(v)[:, 2])} for k, v in out.items()}


def imagenet(pop, with6=True):
    rng = {c: np.random.default_rng(s) for c, s in zip(A.CONDITIONS, np.random.SeedSequence(20260928).spawn(len(A.CONDITIONS)))}
    h13 = A.halves_list(rng["split13"], 13, pop.base, pop.thr)
    p13 = {s: A.draws(rng[f"pruned13_{s}"], pop.pr[s], 13) for s in A.SPARS}
    res = {"n13": block(pop.rec, h13, p13, A.HOOKER_ACC[0.0], A.HOOKER_ACC)}
    if with6:
        h6 = A.halves_list(rng["split6"], 6, pop.base, pop.thr)
        res["n6"] = block(pop.rec, h6, {0.5: [pop.pr[0.5]] * A.K}, A.HOOKER_ACC[0.0], A.HOOKER_ACC)
    return res


def real_recall(pop):
    o = pd.read_csv(os.path.join(ROOT, "results", "preds", "image_order.csv"))
    real = json.load(open(os.path.join(ROOT, "data", "real", "real.json")))
    R = [set(real[int(v) - 1]) for v in o.path.str.extract(r"val_(\d+)")[0]]
    has = np.array([len(r) > 0 for r in R]); den = np.bincount(pop.y[has], minlength=1000)
    for k, t in pop.P.items():
        ok = np.array([t[i] in R[i] for i in range(pop.N)]) & has
        pop.R[k] = np.bincount(pop.y[ok], minlength=1000) / den
    return pop


def cifar():
    sys.path.insert(0, os.path.join(ROOT, "cifar"))
    import analyze as C  # fresh RNG streams (SeedSequence([20260928, 10])) -> same n=13 draws as ADDENDUM-2 M4c
    P, y = C.load()
    rec = {k: np.bincount(y[v == y], minlength=10) / np.bincount(y, minlength=10) for k, v in P.items()}
    acc = {s: np.mean([(P[(s, i)] == y).mean() for i in range(30)]) for s in (0.0, *C.SP)}
    h = C.halves(C.RNG["split13"], 13)
    pr = {s: [[(s, i) for i in C.RNG[f"pruned13_{s}"].permutation(30)[:13]] for _ in range(C.K)] for s in C.SP}
    return {"n13": block(lambda ids: np.stack([rec[x] for x in ids]), h, pr, acc[0.0], acc)}


def main():
    out = {"rule": "exclude classes with zero per-model recall variance in BOTH populations; Welch + BH over the rest (ADDENDUM-3 C1b)"}
    out["val_ifast"] = imagenet(A.Pop("ifast"))
    out["val_islow"] = imagenet(A.Pop("islow"))
    out["v2_ifast"] = imagenet(A.Pop("ifast", "v2"), with6=True)
    out["real_ifast"] = imagenet(real_recall(A.Pop("ifast")))
    out["cifar"] = cifar()
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "m4fix.json"), "w"), indent=1)
    for ds, v in out.items():
        if ds != "rule":
            for n, r in v.items():
                print(ds, n, {k: "unc %.1f bh %.2f m %.0f" % (x["uncorrected"]["mean"], x["bh"]["mean"], x["classes_tested"]["mean"]) for k, x in r.items()})


if __name__ == "__main__":
    main()
