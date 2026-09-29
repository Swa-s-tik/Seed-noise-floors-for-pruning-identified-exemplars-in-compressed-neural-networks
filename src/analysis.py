# Pre-registered analysis (PREREG.md + PREREG-ADDENDUM-1.md): M1-M4, H1/H2, intervals, ablations, Hooker comparison.
# usage: python src/analysis.py ifast|islow [v2]  -> results/analysis/<decoder>[or v2_<decoder>]/{summary.json,counts.npz,f.npz}
import json, os, sys, time
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CLS, K, BOOT = 1000, 1000, 2000
SPARS = (0.3, 0.7, 0.9)
# Hooker, per_class_accuracy/gen_tstatistic.py: "mean accuracy across 30 models trained to different levels of sparsity."
HOOKER_ACC = {0.0: 0.76675, 0.1: 0.76662, 0.3: 0.76459, 0.5: 0.75874, 0.7: 0.75019, 0.9: 0.72569}
# RNG: one child of SeedSequence(20260928) per condition, in THIS order (PREREG-ADDENDUM-1 §3). Never reorder.
CONDITIONS = (["split13", *[f"pruned13_{s}" for s in SPARS], "split6", "boot", "mcboot"]
              + [f"split_n{n}" for n in range(3, 13) if n != 6]
              + [f"pruned_n{n}_{s}" for n in range(3, 13) for s in SPARS]
              + [f"pruned_n{n}_0.5" for n in range(3, 6)]
              + ["thr_split8", "base_split5", *[f"cross_n{n}" for n in range(3, 12)], *[f"thr_pruned8_{s}" for s in SPARS],
                 "cohort_split13", "uniform_split13", "tiebreak_random"])
RNG = {c: np.random.default_rng(s) for c, s in zip(CONDITIONS, np.random.SeedSequence(20260928).spawn(len(CONDITIONS)))}


def modal(P, tie="max", rng=None):
    """Per-image modal top-1 label over models (rows of P).
    Hooker, pie_dataset_gen/aggregate_predictions.py (verbatim):
        predicted_mode_variant = variant['predictions'].mode().max()
    i.e. most frequent label, ties -> LARGEST label (tie="max"). tie="min"/"random" are ablations."""
    S = np.sort(P, axis=0)
    n, N = S.shape
    run = np.ones(S.shape, np.int16)
    for j in range(1, n):
        run[j] = np.where(S[j] == S[j - 1], run[j - 1] + 1, 1)
    ends = run == run.max(0)  # ends of maximal runs; sorted ascending, so later end = larger label
    if tie == "max":
        j = n - 1 - np.argmax(ends[::-1], 0)
    elif tie == "min":
        j = np.argmax(ends, 0)
    else:
        r = (rng.random(N) * ends.sum(0)).astype(np.int64)
        j = np.argmax(ends & (np.cumsum(ends, 0) == r + 1), 0)
    return S[j, np.arange(N)]


def n_tied(P):
    S = np.sort(P, axis=0); run = np.ones(S.shape, np.int16)
    for j in range(1, S.shape[0]):
        run[j] = np.where(S[j] == S[j - 1], run[j - 1] + 1, 1)
    return int(((run == run.max(0)).sum(0) > 1).sum())


def load(decoder, dataset="val"):
    inv = json.load(open(os.path.join(ROOT, "INVENTORY.json")))["runs"]
    if dataset == "v2":  # ImageNet-V2 matched-frequency (PREREG-ADDENDUM-1 section 10), IFAST only
        d = os.path.join(ROOT, "results", "preds_v2")
        y = np.loadtxt(os.path.join(d, "image_order.csv"), delimiter=",", skiprows=1, usecols=2, dtype=np.int64)
    else:
        d = os.path.join(ROOT, "results", "preds" if decoder == "ifast" else "preds_islow")
        y = np.loadtxt(os.path.join(ROOT, "results", "preds", "image_order.csv"), delimiter=",", skiprows=1, usecols=4, dtype=np.int64)
    runs = sorted(k for k, r in inv.items() if r["complete"])
    top1 = {k: np.load(os.path.join(d, k + ".npz"))["top1"] for k in runs}
    return inv, runs, top1, y


def recall(top1, y):
    """Per-class recall (1000,) = correct / class size."""
    return np.bincount(y[top1 == y], minlength=N_CLS) / np.bincount(y, minlength=N_CLS)


def welch(Ra, Rb, ca, cb):
    """Hooker gen_tstatistic.py: ttest_ind(baseline_set, variant_set, equal_var=False), recall - mean accuracy."""
    with np.errstate(all="ignore"):
        p = stats.ttest_ind(Ra - ca, Rb - cb, axis=0, equal_var=False).pvalue
    p = np.where(np.isnan(p), 1.0, p)
    return int((p < 0.05).sum()), int((stats.false_discovery_control(p, method="bh") <= 0.05).sum())


class Pop:
    def __init__(self, decoder, dataset="val"):
        self.inv, runs, top1, self.y = load(decoder, dataset)
        self.N = len(self.y)
        role = lambda k: self.inv[k]["role"]
        self.dense = [k for k in runs if role(k) != "pruned"]
        self.base = [k for k in self.dense if role(k) == "dense-baseline"]
        self.thr = [k for k in self.dense if role(k) == "dense-threshold"]
        self.pr = {s: [k for k in runs if role(k) == "pruned" and self.inv[k]["sparsity"] == s] for s in (0.3, 0.5, 0.7, 0.9)}
        assert (len(self.base), len(self.thr), [len(v) for v in self.pr.values()]) == (11, 16, [30, 6, 30, 30])
        self.P = {k: v.astype(np.int16) for k, v in top1.items()}
        self.R = {k: recall(v, self.y) for k, v in top1.items()}

    def stack(self, ids):
        return np.stack([self.P[k] for k in ids])

    def rec(self, ids):
        return np.stack([self.R[k] for k in ids])


def strat_halves(rng, n, base, thr):
    """PREREG-ADDENDUM-1 §3: per half, #baseline by randomized rounding of n*11/27, rejection on availability."""
    x = n * len(base) / (len(base) + len(thr)); fl = int(np.floor(x))
    while True:
        bA = fl + int(rng.random() < x - fl); bB = fl + int(rng.random() < x - fl)
        if bA + bB <= len(base) and 2 * n - bA - bB <= len(thr) and bA <= n and bB <= n:
            break
    b, t = rng.permutation(len(base)), rng.permutation(len(thr))
    A = [base[i] for i in b[:bA]] + [thr[i] for i in t[:n - bA]]
    B = [base[i] for i in b[bA:bA + bB]] + [thr[i] for i in t[n - bA:2 * n - bA - bB]]
    return A, B


def halves_list(rng, n, base, thr):
    return [strat_halves(rng, n, base, thr) for _ in range(K)]


def draws(rng, ids, n):
    return [[ids[i] for i in rng.permutation(len(ids))[:n]] for _ in range(K)]


def run_block(pop, halves, pruned, tie="max", rng=None, keep_f=False, m4=False):
    """For each split k: floor = PIE(A_k,B_k); for each s: PIE(A_k, pruned[s][k]). Optional per-image f and M4."""
    out = {"floor": np.zeros(K, np.int32), **{s: np.zeros(K, np.int32) for s in pruned}}
    f = {key: np.zeros(pop.N, np.float64) for key in out} if keep_f else None
    m4r = {key: np.zeros((K, 2), np.int32) for key in out} if m4 else None
    m4own = {key: np.zeros((K, 2), np.int32) for key in out} if m4 else None
    for k, (A, B) in enumerate(halves):
        mA = modal(pop.stack(A), tie, rng)
        pies = {"floor": mA != modal(pop.stack(B), tie, rng)}
        for s, pk in pruned.items():
            pies[s] = mA != modal(pop.stack(pk[k]), tie, rng)
        for key, v in pies.items():
            out[key][k] = v.sum()
            if keep_f:
                f[key] += v
        if m4:
            RA = pop.rec(A); accA = RA.mean()  # own-mean sensitivity uses population mean top-1 (class-balanced val)
            for key in pies:
                Rv = pop.rec(B) if key == "floor" else pop.rec(pruned[key][k])
                cv = HOOKER_ACC[0.0] if key == "floor" else HOOKER_ACC[key]
                m4r[key][k] = welch(RA, Rv, HOOKER_ACC[0.0], cv)
                m4own[key][k] = welch(RA, Rv, accA, Rv.mean())
    if keep_f:
        f = {key: v / K for key, v in f.items()}
    return out, f, m4r, m4own


def summ(x, rng=None):
    x = np.asarray(x, np.float64)
    r = {"mean": x.mean(), "sd": x.std(ddof=1), "p2.5": np.percentile(x, 2.5), "p97.5": np.percentile(x, 97.5),
         "min": x.min(), "max": x.max(), "mc_se_mean": x.std(ddof=1) / np.sqrt(len(x))}
    if rng is not None:
        bs = x[rng.integers(0, len(x), (BOOT, len(x)))]
        r["mc_se_p2.5"], r["mc_se_p97.5"] = np.percentile(bs, 2.5, axis=1).std(), np.percentile(bs, 97.5, axis=1).std()
    return {k: float(v) for k, v in r.items()}


def img_boot(f, pairs, rng, N):
    """Image-level bootstrap (resample images): 95% CI of sum(f) and of paired differences sum(f_a - f_b)."""
    idx = [rng.integers(0, N, N) for _ in range(BOOT)]
    res = {}
    for name, a, b in pairs:
        d = f[a] - (f[b] if b else 0)
        v = np.array([d[i].sum() for i in idx])
        res[name] = {"point": float(d.sum()), "ci2.5": float(np.percentile(v, 2.5)), "ci97.5": float(np.percentile(v, 97.5)), "se": float(v.std())}
    return res


def main(decoder, dataset="val"):
    t0 = time.time()
    pop = Pop(decoder, dataset)
    out_dir = os.path.join(ROOT, "results", "analysis", decoder if dataset == "val" else f"{dataset}_{decoder}"); os.makedirs(out_dir, exist_ok=True)
    S, C, F = {}, {}, {}
    mc = RNG["mcboot"]

    # --- M1, M2 (n=13), M4, image f ---
    h13 = halves_list(RNG["split13"], 13, pop.base, pop.thr)
    p13 = {s: draws(RNG[f"pruned13_{s}"], pop.pr[s], 13) for s in SPARS}
    c, f, m4r, m4own = run_block(pop, h13, p13, keep_f=True, m4=True)
    for key, v in c.items():
        C[f"n13_{key}"] = v; F[f"n13_{key}"] = f[key]
        S[f"M1" if key == "floor" else f"M2_{key}"] = summ(v, mc)
        C[f"m4_n13_{key}"] = m4r[key]; C[f"m4own_n13_{key}"] = m4own[key]
        S[f"M4_{key}"] = {"uncorrected": summ(m4r[key][:, 0], mc), "bh": summ(m4r[key][:, 1], mc),
                          "own_mean_uncorrected": summ(m4own[key][:, 0]), "own_mean_bh": summ(m4own[key][:, 1])}
    # --- 0.5: 6 v 6 ---
    h6 = halves_list(RNG["split6"], 6, pop.base, pop.thr)
    c, f, m4r, m4own = run_block(pop, h6, {0.5: [pop.pr[0.5]] * K}, keep_f=True, m4=True)
    C["n6_floor"], C["n6_0.5"] = c["floor"], c[0.5]; F["n6_floor"], F["n6_0.5"] = f["floor"], f[0.5]
    S["M1_n6"], S["M2_0.5_n6"] = summ(c["floor"], mc), summ(c[0.5], mc)
    S["M4_n6_floor"] = {"uncorrected": summ(m4r["floor"][:, 0]), "bh": summ(m4r["floor"][:, 1])}
    S["M4_0.5_n6"] = {"uncorrected": summ(m4r[0.5][:, 0]), "bh": summ(m4r[0.5][:, 1])}

    # --- decisions (PREREG H1/H2; ADDENDUM-1 §5) ---
    sd, m1 = S["M1"]["sd"], S["M1"]["mean"]
    z = {s: (S[f"M2_{s}"]["mean"] - m1) / sd for s in SPARS}
    z05 = (S["M2_0.5_n6"]["mean"] - S["M1_n6"]["mean"]) / S["M1_n6"]["sd"]
    S["decisions"] = {"z": {str(s): z[s] for s in SPARS}, "z_0.5_n6": z05,
                      "H1_holds(|z0.3|<=3)": bool(abs(z[0.3]) <= 3), "H1_fails(z0.3>3)": bool(z[0.3] > 3),
                      "H2_0.7(z>3)": bool(z[0.7] > 3), "H2_0.9(z>3)": bool(z[0.9] > 3)}
    S["floor_subtracted_split"] = {str(s): {"diff": S[f"M2_{s}"]["mean"] - m1, "z": z[s]} for s in SPARS}
    S["floor_subtracted_split"]["0.5_n6"] = {"diff": S["M2_0.5_n6"]["mean"] - S["M1_n6"]["mean"], "z": z05}
    S["image_bootstrap"] = img_boot(F, [("M1", "n13_floor", None), *[(f"M2_{s}", f"n13_{s}", None) for s in SPARS],
                                        *[(f"D_{s}", f"n13_{s}", "n13_floor") for s in SPARS],
                                        ("M1_n6", "n6_floor", None), ("M2_0.5_n6", "n6_0.5", None), ("D_0.5_n6", "n6_0.5", "n6_floor")],
                                    RNG["boot"], pop.N)
    print("M1/M2/M4 done", f"{time.time() - t0:.0f}s", flush=True)
    if dataset != "val":  # replication sets: M1/M2/M4 + decisions only (ADDENDUM-1 section 10)
        S["meta"] = {"decoder": decoder, "dataset": dataset, "K": K, "BOOT": BOOT, "seed": 20260928, "n_images": pop.N}
        np.savez_compressed(os.path.join(out_dir, "counts.npz"), **C)
        json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
        print(json.dumps({"M1": S["M1"], "decisions": S["decisions"]}, indent=1)); return

    # --- M3: n = 3..13 ---
    m3 = {}
    for n in range(3, 14):
        h = h13 if n == 13 else h6 if n == 6 else halves_list(RNG[f"split_n{n}"], n, pop.base, pop.thr)
        pdraw = {s: p13[s] if n == 13 else draws(RNG[f"pruned_n{n}_{s}"], pop.pr[s], n) for s in SPARS}
        if n <= 6:
            pdraw[0.5] = [pop.pr[0.5]] * K if n == 6 else draws(RNG[f"pruned_n{n}_0.5"], pop.pr[0.5], n)
        c = run_block(pop, h, pdraw)[0]
        m3[n] = {str(key): summ(v) for key, v in c.items()}
        for key, v in c.items():
            C[f"m3_n{n}_{key}"] = v
    S["M3"] = m3
    print("M3 done", f"{time.time() - t0:.0f}s", flush=True)

    # --- ablations ---
    ab = {}
    c = run_block(pop, halves_list(RNG["thr_split8"], 8, [], pop.thr),
                  {s: draws(RNG[f"thr_pruned8_{s}"], pop.pr[s], 8) for s in SPARS})[0]
    ab["dense_threshold_only_n8"] = {str(k): summ(v) for k, v in c.items()}
    ab["dense_baseline_only_n5_floor"] = summ(run_block(pop, halves_list(RNG["base_split5"], 5, pop.base, []), {})[0]["floor"])
    cross = {}
    for n in range(3, 12):
        r = RNG[f"cross_n{n}"]
        hh = [([pop.base[i] for i in r.permutation(11)[:n]], [pop.thr[i] for i in r.permutation(16)[:n]]) for _ in range(K)]
        cross[n] = {"cross_type": summ(run_block(pop, hh, {})[0]["floor"]), "stratified_floor": m3[n]["floor"]}
    ab["cross_type_baseline_vs_threshold"] = cross
    coh = {k: pop.inv[k]["cohort"] for k in pop.dense}
    r = RNG["cohort_split13"]; hc = []
    for _ in range(K):
        order = [m for cc in r.permutation(sorted(set(coh.values()))) for m in r.permutation([x for x in pop.dense if coh[x] == cc])]
        A, B = order[0::2], order[1::2]
        A.pop(int(r.integers(len(A))))  # 14 -> 13
        hc.append((A, B))
    c = run_block(pop, hc, p13)[0]
    ab["cohort_stratified_n13"] = {str(k): summ(v) for k, v in c.items()}
    r = RNG["uniform_split13"]; hu = []
    for _ in range(K):
        p = r.permutation(27); hu.append(([pop.dense[i] for i in p[:13]], [pop.dense[i] for i in p[13:26]]))
    ab["uniform_n13"] = {str(k): summ(v) for k, v in run_block(pop, hu, p13)[0].items()}
    ab["tiebreak_min_n13"] = {str(k): summ(v) for k, v in run_block(pop, h13, p13, tie="min")[0].items()}
    ab["tiebreak_random_n13"] = {str(k): summ(v) for k, v in run_block(pop, h13, p13, tie="random", rng=RNG["tiebreak_random"])[0].items()}
    ab["tied_images_n13_split0"] = {"A": n_tied(pop.stack(h13[0][0])), "B": n_tied(pop.stack(h13[0][1]))}
    S["ablations"] = ab
    print("ablations done", f"{time.time() - t0:.0f}s", flush=True)

    # --- full-population recompute (Hooker's configuration: all pruned at s vs a dense pool; ADDENDUM-1 §8) ---
    full = {}
    pools = {"dense27": pop.dense, "threshold16": pop.thr, "baseline11": pop.base}
    for pn, ids in pools.items():
        md = modal(pop.stack(ids))
        full[pn] = {}
        for s in (0.3, 0.5, 0.7, 0.9):
            cnt = int((md != modal(pop.stack(pop.pr[s]))).sum())
            u, bh = welch(pop.rec(ids), pop.rec(pop.pr[s]), HOOKER_ACC[0.0], HOOKER_ACC[s])
            full[pn][str(s)] = {"pie": cnt, "pie_scaled_49187": cnt * 49187 / 50000, "m4_uncorrected": u, "m4_bh": bh}
    S["full_population"] = full
    S["meta"] = {"decoder": decoder, "K": K, "BOOT": BOOT, "seed": 20260928, "n_images": pop.N}  # no timing: outputs must be byte-reproducible
    np.savez_compressed(os.path.join(out_dir, "counts.npz"), **C)
    np.savez_compressed(os.path.join(out_dir, "f.npz"), **{k: v.astype(np.float32) for k, v in F.items()})
    json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps({"M1": S["M1"], **{f"M2_{s}": S[f"M2_{s}"] for s in SPARS}, "decisions": S["decisions"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "val")
