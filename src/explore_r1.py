# EXPLORATORY analyses added in the first review round (DEVIATIONS D12; stated before running; no pre-registered verdict changes).
#   python src/explore_r1.py int8true   E4 data: true fbgemm INT8 top-1 for the 27 dense models on the first 2,000 images (GPU calibration, CPU kernels)
#   python src/explore_r1.py analyze    E1-E6, E4 and E8-E10 (D15) -> results/analysis/explore_r1.json
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A  # noqa: E402

ROOT = A.ROOT; K = A.K
NAMES = ["j13v14", "cf_a", "cf_b", "cf_x"] + [f"pp_n{n}_{s}" for s in A.SPARS for n in range(3, 14)] + ["pp_n3_0.5"]
RNG = {c: np.random.default_rng(s) for c, s in zip(NAMES, np.random.SeedSequence([20260928, 30]).spawn(len(NAMES)))}
OUT4 = os.path.join(ROOT, "results", "preds_m5", "int8true_2000")
N4 = 2000


def fresh_prereg_rng():
    return {c: np.random.default_rng(s) for c, s in zip(A.CONDITIONS, np.random.SeedSequence(20260928).spawn(len(A.CONDITIONS)))}


def int8true():
    import torch
    import evalpreds as E, m5
    from torch.ao.quantization.quantize_fx import convert_fx
    torch.set_num_threads(8); torch.backends.quantized.engine = "fbgemm"
    os.makedirs(OUT4, exist_ok=True)
    X = torch.from_numpy(np.array(np.load(os.path.join(E.CACHE, "val_ifast.npy"), mmap_mode="r")[:N4]))
    inv = json.load(open(os.path.join(ROOT, "INVENTORY.json")))["runs"]
    dense = sorted(k for k, r in inv.items() if r["complete"] and r["role"] != "pruned")

    Q = {}

    def calib(rid):  # GPU part only (calibrated exactly as the simulated, stored predictions); the lock is released after it
        mp, _ = m5.int8_model(rid, "cuda"); torch.cuda.empty_cache()
        Q[rid] = convert_fx(mp)
    E.gpu_loop([k for k in dense if not os.path.exists(os.path.join(OUT4, k + ".npy"))], calib)
    for rid, q in Q.items():  # CPU integer kernels, no GPU lock held
        with torch.inference_mode():
            t1 = torch.cat([q(X[s:s + 100]).argmax(1) for s in range(0, N4, 100)]).numpy().astype(np.int16)
        np.save(os.path.join(OUT4, rid + ".npy"), t1); print(rid, flush=True)


def jac(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def analyze():
    out = {}
    pop = A.Pop("ifast"); R = fresh_prereg_rng()
    h13 = A.halves_list(R["split13"], 13, pop.base, pop.thr)
    p13 = {s: A.draws(R[f"pruned13_{s}"], pop.pr[s], 13) for s in A.SPARS}
    S = json.load(open(os.path.join(ROOT, "results", "analysis", "ifast", "summary.json")))
    # E1 pruned-vs-pruned floor
    e1 = {}
    for s in A.SPARS:
        e1[str(s)] = {}
        for n in range(3, 14):
            r = RNG[f"pp_n{n}_{s}"]; c = np.zeros(K, int)
            for k in range(K):
                p = r.permutation(30)
                c[k] = (A.modal(pop.stack([pop.pr[s][i] for i in p[:n]])) != A.modal(pop.stack([pop.pr[s][i] for i in p[n:2 * n]]))).sum()
            e1[str(s)][n] = {**A.summ(c), "dense_floor_mean": S["M3"][str(n)]["floor"]["mean"], "dense_floor_sd": S["M3"][str(n)]["floor"]["sd"]}
    r = RNG["pp_n3_0.5"]; c = np.zeros(K, int)
    for k in range(K):
        p = r.permutation(6)
        c[k] = (A.modal(pop.stack([pop.pr[0.5][i] for i in p[:3]])) != A.modal(pop.stack([pop.pr[0.5][i] for i in p[3:]]))).sum()
    e1["0.5"] = {3: {**A.summ(c), "dense_floor_mean": S["M3"]["3"]["floor"]["mean"]}}
    out["E1_pruned_vs_pruned"] = e1
    # E2 seed-variation Jaccard of PIE sets from two disjoint dense references (all 30 pruned as the compared population)
    mp = {s: A.modal(pop.stack(pop.pr[s])) for s in A.SPARS}
    e2 = {}
    for s in A.SPARS:
        ja = np.array([jac(A.modal(pop.stack(a)) != mp[s], A.modal(pop.stack(b)) != mp[s]) for a, b in h13])
        e2[str(s)] = {"13v13": A.summ(ja)}
    rr = RNG["j13v14"]; splits = [rr.permutation(27) for _ in range(K)]
    for s in A.SPARS:
        jb = np.array([jac(A.modal(pop.stack([pop.dense[i] for i in p[:13]])) != mp[s], A.modal(pop.stack([pop.dense[i] for i in p[13:]])) != mp[s]) for p in splits])
        e2[str(s)]["13v14"] = A.summ(jb)
    a2 = json.load(open(os.path.join(ROOT, "results", "a2.json")))
    e2["A2_label_only_bound_ifast"] = {k: v["jaccard_upper_bound"] for k, v in a2["ifast"].items()}
    out["E2_seed_variation_jaccard"] = e2
    # E3 ReaL rates on the compression-specific subset (PIE vs pruned and not PIE vs the other dense half, per split)
    o = pd.read_csv(os.path.join(ROOT, "results", "preds", "image_order.csv"))
    real = json.load(open(os.path.join(ROOT, "data", "real", "real.json")))
    Rs = [set(real[int(v) - 1]) for v in o.path.str.extract(r"val_(\d+)")[0]]
    has = np.array([len(x) > 0 for x in Rs]); multi = np.array([len(x) > 1 for x in Rs])
    wrong = np.array([len(x) > 0 and y not in x for x, y in zip(Rs, pop.y)])
    e3 = {}
    for s in A.SPARS:
        wc = np.zeros(pop.N); ws = np.zeros(pop.N); wf = np.zeros(pop.N)
        for k, (a, b) in enumerate(h13):
            mA = A.modal(pop.stack(a)); fl = mA != A.modal(pop.stack(b)); pr = mA != A.modal(pop.stack(p13[s][k]))
            wc += pr & ~fl; ws += pr & fl; wf += fl
        rate = lambda m, w: float((w * m).sum() / w.sum())
        e3[str(s)] = {grp: {"expected_count": float(w.sum() / K), "empty_real": rate(~has, w), "multi_label": rate(multi, w), "orig_not_in_real": rate(wrong, w)}
                      for grp, w in (("compression_specific", wc), ("shared_with_floor", ws), ("floor", wf))}
    e3["non_pie_reference_from_real_json"] = {k: v for k, v in json.load(open(os.path.join(ROOT, "results", "analysis", "ifast", "real.json")))["label_quality_pie_vs_nonpie"]["n13_floor"].items() if k.endswith("nonpie")}
    out["E3_real_compression_specific"] = e3
    # E5 paired per-split excess (n = 13)
    C = np.load(os.path.join(ROOT, "results", "analysis", "ifast", "counts.npz"))
    out["E5_paired_excess_n13"] = {s: {**A.summ(C[f"n13_{s}"] - C["n13_floor"]), "frac_positive": float(((C[f"n13_{s}"] - C["n13_floor"]) > 0).mean()),
                                       "z_mc_se": float(np.std([((C[f"n13_{s}"][i] - C["n13_floor"][i]).mean()) / C["n13_floor"][i].std(ddof=1)
                                                                for i in np.random.default_rng(1).integers(0, K, (2000, K))]))}
                                   for s in ("0.3", "0.7", "0.9")}
    # E6 CIFAR exchangeability of the 60 dense seeds
    d = os.path.join(ROOT, "results", "cifar", "preds")
    D = np.stack([np.load(os.path.join(d, f"s0.0_seed{i}.npz"))["top1"].astype(np.int16) for i in range(60)])
    e6 = {}
    for name, lo, hi in (("within_0_29", 0, 30), ("within_30_59", 30, 60)):
        r = RNG["cf_a" if lo == 0 else "cf_b"]; c = np.zeros(K, int)
        for k in range(K):
            p = lo + r.permutation(30)
            c[k] = (A.modal(D[p[:13]]) != A.modal(D[p[13:26]])).sum()
        e6[name] = A.summ(c)
    r = RNG["cf_x"]; c = np.zeros(K, int)
    for k in range(K):
        c[k] = (A.modal(D[r.permutation(30)[:13]]) != A.modal(D[30 + r.permutation(30)[:13]])).sum()
    e6["across"] = A.summ(c)
    e6["top1_seeds_0_29"] = float((D[:30] == np.load(os.path.join(d, "s0.0_seed0.npz"))["labels"]).mean())
    e6["top1_seeds_30_59"] = float((D[30:] == np.load(os.path.join(d, "s0.0_seed0.npz"))["labels"]).mean())
    out["E6_cifar_exchangeability"] = e6
    # E8-E10 (DEVIATIONS D15, second review round): structured-vs-structured floor, unstructured arms on the 60-seed floor, MC error of z
    R2 = {c: np.random.default_rng(s) for c, s in zip(["split", "st", "un", "boot"], np.random.SeedSequence([20260928, 50]).spawn(4))}
    y = np.load(os.path.join(d, "s0.0_seed0.npz"))["labels"]
    ST = {s: np.stack([np.load(os.path.join(d, f"st{s}_seed{1000 * k + i}.npz"))["top1"].astype(np.int16) for i in range(30)])
          for s, k in (("0.3", 13), ("0.7", 17), ("0.9", 19))}
    UN = {s: np.stack([np.load(os.path.join(d, f"s{s}_seed{1000 * k + i}.npz"))["top1"].astype(np.int16) for i in range(30)])
          for s, k in (("0.3", 3), ("0.7", 7), ("0.9", 9))}
    c = {k: np.zeros(K, int) for k in ["floor", *[f"st_{s}" for s in ST], *[f"un_{s}" for s in UN]]}
    f = {k: np.zeros(len(y)) for k in ["floor", *[f"un_{s}" for s in UN]]}
    for k in range(K):
        p = R2["split"].permutation(60); mA = A.modal(D[p[:13]])
        v = {"floor": mA != A.modal(D[p[13:26]]), **{f"un_{s}": mA != A.modal(UN[s][R2["un"].permutation(30)[:13]]) for s in UN}}
        for s in ST:
            q = R2["st"].permutation(30); c[f"st_{s}"][k] = (A.modal(ST[s][q[:13]]) != A.modal(ST[s][q[13:26]])).sum()
        for key, x in v.items():
            c[key][k] = x.sum(); f[key] += x
    F = A.summ(c["floor"])
    out["E8_structured_vs_structured_n13"] = {"dense_floor_60": F, **{s: A.summ(c[f"st_{s}"]) for s in ST}}
    ib = A.img_boot({k: v / K for k, v in f.items()}, [(f"D_un_{s}", f"un_{s}", "floor") for s in UN], R2["boot"], len(y))
    acc = lambda P: float((P == y).mean())
    out["E9_unstructured_on_60seed_floor_n13"] = {"dense_floor_60": F, "dense_top1_60": acc(D),
                                                  **{s: {**A.summ(c[f"un_{s}"]), "z": (c[f"un_{s}"].mean() - F["mean"]) / F["sd"],
                                                         "excess": ib[f"D_un_{s}"], "top1": acc(UN[s])} for s in UN},
                                                  "structured_top1": {s: acc(ST[s]) for s in ST}}
    se = lambda m, fl, z: float(np.sqrt((m.get("mc_se_mean", 0) ** 2 + fl["mc_se_mean"] ** 2) / fl["sd"] ** 2 + z ** 2 / (2 * (K - 1))))
    CS, C1 = [json.load(open(os.path.join(ROOT, "results", "analysis", "cifar", n))) for n in ("summary.json", "c1a.json")]
    SJ = json.load(open(os.path.join(ROOT, "results", "analysis", "cifar", "structured.json")))
    out["E10_cifar_z_mc_se"] = {
        "unstructured_n13": {s: se(CS[f"M2c_{s}"], CS["M1c"], CS["decisions_secondary"]["z"][s]) for s in UN},
        "unstructured_n30": {s: se(C1["pruned_30v30"][s], C1["measured_F30"], C1["pruned_30v30"][s]["z_vs_F30"]) for s in UN},
        "structured_n13": {s: se(SJ["n13"][s], SJ["n13"]["floor"], SJ["n13"][s]["z"]) for s in ST},
        "structured_n30": {s: se(SJ["n30"][s], SJ["n30"]["floor"], SJ["n30"][s]["z"]) for s in ST}}
    # E4 simulated vs true INT8 on the first 2,000 images (if the true-kernel predictions exist)
    if os.path.isdir(OUT4) and len(os.listdir(OUT4)) == 27:
        sim = {k: np.load(os.path.join(ROOT, "results", "preds_m5", "int8", k + ".npz"))["top1"][:N4].astype(np.int16) for k in pop.dense}
        tru = {k: np.load(os.path.join(OUT4, k + ".npy")) for k in pop.dense}
        fp = {k: pop.P[k][:N4] for k in pop.dense}
        st = lambda P, ids: np.stack([P[x] for x in ids])
        rows = {"paired_sim": [], "paired_true": [], "unpaired_sim": [], "unpaired_true": [], "floor": []}
        for a, b in h13:
            mA, mB = A.modal(st(fp, a)), A.modal(st(fp, b))
            ms, mt = A.modal(st(sim, a)), A.modal(st(tru, a))
            rows["paired_sim"].append((ms != mA).sum()); rows["paired_true"].append((mt != mA).sum())
            rows["unpaired_sim"].append((ms != mB).sum()); rows["unpaired_true"].append((mt != mB).sum()); rows["floor"].append((mA != mB).sum())
        e4 = {k: A.summ(np.array(v)) for k, v in rows.items()}
        e4["per_model_top1_disagreement_sim_vs_true"] = A.summ(np.array([(sim[k] != tru[k]).mean() for k in pop.dense]))
        e4["modal13_disagreement_sim_vs_true"] = A.summ(np.array([(A.modal(st(sim, a)) != A.modal(st(tru, a))).sum() for a, _ in h13]))
        e4["n_images"] = N4
        out["E4_int8_sim_vs_true_2000"] = e4
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "explore_r1.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "E1_pruned_vs_pruned"}, indent=1, default=float)[:6000])
    print({s: {n: round(v["mean"]) for n, v in e1[s].items() if n in (3, 13)} for s in e1})


if __name__ == "__main__":
    {"int8true": int8true, "analyze": analyze}[sys.argv[1]]()
