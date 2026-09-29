# EXPLORATORY (not pre-registered; labelled as such wherever reported).
# 1) pairwise top-1 disagreement by pair type, incl. same-experiment-id pairs (shared-seed check);
# 3) M4 artefact check: classes whose recall is identical across ALL models of both populations have zero variance;
#    Hooker's constant normalisation (recall - acc[s]) then gives t = +-inf, p = 0 whenever acc[s] != acc[0.0].
# 2) extrapolation of the M3 curves to Hooker's population size (n = 30) for comparison with the printed counts.
# usage: python src/exploratory.py ifast|islow -> results/analysis/<dec>/exploratory.json
import collections, itertools, json, os, sys
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A  # noqa: E402

FORMS = {"a+b/sqrt(n)": (lambda n, a, b: a + b / np.sqrt(n), (2000, 5000)),
         "a+b*n^-c": (lambda n, a, b, c: a + b * n ** (-c), (2000, 5000, 0.5))}


def main(dec):
    pop = A.Pop(dec); inv = pop.inv
    ks = sorted(pop.P); P = np.stack([pop.P[k] for k in ks])
    g = collections.defaultdict(list)
    for i, j in itertools.combinations(range(len(ks)), 2):
        ra, rb = inv[ks[i]], inv[ks[j]]
        types = " vs ".join(sorted(f"{r['role']}@{r['sparsity']}" for r in (ra, rb)))
        g[("same exp id" if ra["exp"] == rb["exp"] else "different exp id") + ": " + types].append(float((P[i] != P[j]).mean()))
    out = {"pairwise_disagreement": {k: {"pairs": len(v), "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in sorted(g.items())}}
    S = json.load(open(os.path.join(A.ROOT, "results", "analysis", dec, "summary.json")))
    ext = {}
    for key in ("floor", "0.3", "0.7", "0.9"):
        ns = np.array(sorted(int(n) for n in S["M3"])); y = np.array([S["M3"][str(n)][key]["mean"] for n in ns])
        ext[key] = {}
        for fn, (f, p0) in FORMS.items():
            for lo in (3, 5):
                m = ns >= lo
                p, _ = curve_fit(f, ns[m], y[m], p0=p0, maxfev=20000)
                ext[key][f"{fn}, n>={lo}"] = {"n30": float(f(30, *p)), "params": [float(x) for x in p],
                                              "max_abs_resid": float(np.abs(f(ns[m], *p) - y[m]).max())}
    out["extrapolation_to_n30"] = ext
    from scipy import stats
    h13 = A.halves_list(A.RNG["split13"], 13, pop.base, pop.thr)
    zv = {}
    for s in A.SPARS:
        p13 = A.draws(A.RNG[f"pruned13_{s}"], pop.pr[s], 13)
        r = []
        for (a, _), pk in zip(h13, p13):
            RA, RP = pop.rec(a), pop.rec(pk)
            zero = (RA.var(0) == 0) & (RP.var(0) == 0)
            with np.errstate(all="ignore"):
                pv = stats.ttest_ind(RA - A.HOOKER_ACC[0.0], RP - A.HOOKER_ACC[s], axis=0, equal_var=False).pvalue
            pv = np.where(np.isnan(pv), 1.0, pv); q = stats.false_discovery_control(pv, method="bh") <= 0.05
            r.append((zero.sum(), (q & zero).sum(), (q & ~zero).sum(), ((pv < 0.05) & ~zero).sum()))
        r = np.array(r, float)
        zv[str(s)] = {"zero_variance_classes": r[:, 0].mean(), "bh_sig_zero_variance": r[:, 1].mean(),
                      "bh_sig_excluding_zero_variance": r[:, 2].mean(), "uncorrected_sig_excluding_zero_variance": r[:, 3].mean()}
    out["m4_zero_variance_artefact_n13"] = zv
    print(json.dumps(zv, indent=1))
    out["printed_hooker_v3"] = {"0.3": 1819, "0.5": 2193, "0.7": 3073, "0.9": 5136, "float16": 2019, "dynamic int8": 2193, "fixed-point int8": 2093}
    json.dump(out, open(os.path.join(A.ROOT, "results", "analysis", dec, "exploratory.json"), "w"), indent=1)
    for k, v in ext.items():
        print(k, {kk: round(vv["n30"]) for kk, vv in v.items()})


if __name__ == "__main__":
    main(sys.argv[1])
