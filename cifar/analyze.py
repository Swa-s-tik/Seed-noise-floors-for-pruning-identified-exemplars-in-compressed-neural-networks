# PREREG-ADDENDUM-2 section 2: CIFAR-10 floor / pruned / population size / class tests. usage: python cifar/analyze.py
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import analysis as A  # noqa: E402  (modal = Hooker's mode().max(), summ, img_boot, welch)

K, SP = 1000, (0.3, 0.7, 0.9)
COND = ["split13", "split15", "boot", "mcboot", *[f"pruned13_{s}" for s in SP],
        *[f"split_n{n}" for n in range(3, 16) if n not in (13, 15)], *[f"pruned_n{n}_{s}" for n in range(3, 16) if n != 13 for s in SP]]
RNG = {c: np.random.default_rng(s) for c, s in zip(COND, np.random.SeedSequence([20260928, 10]).spawn(len(COND)))}
PRINTED = {"0.3": 114, "0.5": 144, "0.7": 137, "0.9": 216}  # Hooker v1-v3 CIFAR-10 # PIEs (second read)


def load():
    d = os.path.join(ROOT, "results", "cifar", "preds")
    P = {(s, i): np.load(os.path.join(d, f"s{s}_seed{1000 * k + i}.npz")) for s, k in ((0.0, 0), (0.3, 3), (0.7, 7), (0.9, 9)) for i in range(30)}
    y = P[(0.0, 0)]["labels"].astype(np.int64)
    return {k: v["top1"].astype(np.int16) for k, v in P.items()}, y


def halves(rng, n):
    return [(lambda p: ([(0.0, i) for i in p[:n]], [(0.0, i) for i in p[n:2 * n]]))(rng.permutation(30)) for _ in range(K)]


def main():
    P, y = load(); N = len(y)
    st = lambda ids: np.stack([P[k] for k in ids])
    rec = {k: np.bincount(y[v == y], minlength=10) / np.bincount(y, minlength=10) for k, v in P.items()}
    acc = {s: np.mean([(P[(s, i)] == y).mean() for i in range(30)]) for s in (0.0, *SP)}
    S = {"per_sparsity_mean_top1": acc, "per_run_top1": {f"{s}_{i}": float((P[(s, i)] == y).mean()) for (s, i) in P}}
    flags = [f"{s}_{i}" for (s, i) in P if (P[(s, i)] == y).mean() < 0.85]
    S["flagged_runs_top1_below_0.85"] = flags
    mc = RNG["mcboot"]
    res, f = {}, {}
    for n in range(3, 16):
        h = halves(RNG["split13" if n == 13 else "split15" if n == 15 else f"split_n{n}"], n)
        pr = {s: [[(s, i) for i in RNG[f"pruned13_{s}" if n == 13 else f"pruned_n{n}_{s}"].permutation(30)[:n]] for _ in range(K)] for s in SP}
        c = {"floor": np.zeros(K, int), **{s: np.zeros(K, int) for s in SP}}
        m4 = {key: np.zeros((K, 2), int) for key in c}
        fk = {key: np.zeros(N) for key in c}
        for k, (a, b) in enumerate(h):
            mA = A.modal(st(a))
            pies = {"floor": mA != A.modal(st(b)), **{s: mA != A.modal(st(pr[s][k])) for s in SP}}
            for key, v in pies.items():
                c[key][k] = v.sum(); fk[key] += v
            if n == 13:
                RA = np.stack([rec[x] for x in a])
                m4["floor"][k] = A.welch(RA, np.stack([rec[x] for x in b]), acc[0.0], acc[0.0])
                for s in SP:
                    m4[s][k] = A.welch(RA, np.stack([rec[x] for x in pr[s][k]]), acc[0.0], acc[s])
        res[n] = {str(key): A.summ(v, mc if n == 13 else None) for key, v in c.items()}
        if n == 13:
            S["M4c"] = {str(key): {"uncorrected": A.summ(v[:, 0]), "bh": A.summ(v[:, 1])} for key, v in m4.items()}
            f = {f"n13_{key}": v / K for key, v in fk.items()}
    S["M3c"] = res; S["M1c"], S["M1c_n15"] = res[13]["floor"], res[15]["floor"]
    for s in SP:
        S[f"M2c_{s}"] = res[13][str(s)]
    sd, m1 = S["M1c"]["sd"], S["M1c"]["mean"]
    z = {str(s): (S[f"M2c_{s}"]["mean"] - m1) / sd for s in SP}
    S["decisions_secondary"] = {"z": z, "H1c_holds(|z0.3|<=3)": abs(z["0.3"]) <= 3, "H2c_0.7": z["0.7"] > 3, "H2c_0.9": z["0.9"] > 3}
    S["image_bootstrap"] = A.img_boot(f, [("M1c", "n13_floor", None), *[(f"D_{s}", f"n13_{s}", "n13_floor") for s in SP]], RNG["boot"], N)
    full = A.modal(st([(0.0, i) for i in range(30)]))
    S["full_population_30v30"] = {str(s): int((full != A.modal(st([(s, i) for i in range(30)]))).sum()) for s in SP}
    S["printed_hooker_cifar10"] = PRINTED
    os.makedirs(os.path.join(ROOT, "results", "analysis", "cifar"), exist_ok=True)
    json.dump(S, open(os.path.join(ROOT, "results", "analysis", "cifar", "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: S[k] for k in ("per_sparsity_mean_top1", "M1c", "decisions_secondary", "full_population_30v30", "flagged_runs_top1_below_0.85")}, indent=1, default=float))


if __name__ == "__main__":
    main()
