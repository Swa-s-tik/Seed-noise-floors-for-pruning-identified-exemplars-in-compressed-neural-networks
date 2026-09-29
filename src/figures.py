# Figures from committed results (vector PDF + 300-dpi PNG), single-column width. usage: python src/figures.py
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures"); os.makedirs(OUT, exist_ok=True)
# categorical slots in fixed order (validated: dataviz validate_palette.js, light surface; low-contrast slots are direct-labelled)
COL = {"floor": "#2a78d6", "0.3": "#eb6834", "0.7": "#1baf7a", "0.9": "#eda100", "0.5": "#e87ba4"}
NAME = {"floor": "dense vs dense (floor)", "0.3": "30% pruned", "0.5": "50% pruned", "0.7": "70% pruned", "0.9": "90% pruned"}
INK, INK2 = "#0b0b0b", "#52514e"
MK = {"floor": "o", "0.3": "s", "0.5": "D", "0.7": "^", "0.9": "v"}      # shape per condition (colour-vision independent)
LS = {"floor": "-", "0.3": "--", "0.5": "-.", "0.7": ":", "0.9": (0, (5, 1, 1, 1))}
# Hooker et al. arXiv:1911.05248 v3 Table 2 (first read + independent second read, provenance/second_read/)
PRINTED = {"0.3": 1819, "0.5": 2193, "0.7": 3073, "0.9": 5136}
plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
W1 = 3.35  # single-column inches


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}); fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load(dec):
    d = os.path.join(RES, "analysis", dec)
    return json.load(open(os.path.join(d, "summary.json"))), np.load(os.path.join(d, "counts.npz"))


def fig_distributions(S, C, dec):
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    keys = [("floor", "n13_floor"), ("0.3", "n13_0.3"), ("0.7", "n13_0.7"), ("0.9", "n13_0.9")]
    lo, hi = min(C[k].min() for _, k in keys), max(C[k].max() for _, k in keys)
    bins = np.linspace(lo - 10, hi + 10, 90)
    for key, ck in keys:
        lab = NAME[key] + ("" if key == "floor" else f", z = {S['decisions']['z'][key]:.1f}")
        ax.hist(C[ck], bins=bins, histtype="stepfilled", color=COL[key], alpha=0.35, lw=0)
        ax.hist(C[ck], bins=bins, histtype="step", color=COL[key], lw=1.2, ls=LS[key], label=lab)
    ax.set_xlabel(f"PIE count, {'10,000 ImageNet-V2' if dec.startswith('v2') else '50,000 ImageNet val'} images (13 v 13 models)"); ax.set_ylabel("splits")
    ax.legend(loc="upper center", fontsize=6); ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    save(fig, f"fig1_pie_distributions_{dec}")


def fig_popsize(S, dec):
    ex_r1 = os.path.join(RES, "analysis", "explore_r1.json")
    pp = json.load(open(ex_r1))["E1_pruned_vs_pruned"] if (dec == "ifast" and os.path.exists(ex_r1)) else {}
    fig, axs = plt.subplots(1, 2 if pp else 1, figsize=(2 * W1 if pp else W1, 2.5), squeeze=False)
    ax = axs[0, 0]
    ns = sorted(int(n) for n in S["M3"])
    for key in ("floor", "0.3", "0.5", "0.7", "0.9"):
        xs = [n for n in ns if key in S["M3"][str(n)]]
        m = [S["M3"][str(n)][key]["mean"] for n in xs]
        ax.fill_between(xs, [S["M3"][str(n)][key]["p2.5"] for n in xs], [S["M3"][str(n)][key]["p97.5"] for n in xs], color=COL[key], alpha=0.2, lw=0)
        ax.plot(xs, m, color=COL[key], lw=1.6, ls=LS[key], marker=MK[key], ms=3, label=NAME[key] + ("" if key == "floor" else " vs dense"))
    for key, v in PRINTED.items():  # printed counts, 30 models per population
        ax.plot([30], [v], marker="x", ms=5, color=COL[key], ls="none")
    full = S["full_population"]["dense27"]
    for key in ("0.3", "0.7", "0.9"):  # 30 pruned vs 27 dense (0.5 omitted: only 6 pruned released)
        ax.plot([28.8], [full[key]["pie"]], marker=MK[key], ms=5, mfc="none", mew=1.1, color=COL[key], ls="none")
    ex = os.path.join(RES, "analysis", dec, "exploratory.json")
    if os.path.exists(ex):  # EXPLORATORY: floor extrapolated to n = 30, open circle with the range over the four fits
        v = [x["n30"] for x in json.load(open(ex))["extrapolation_to_n30"]["floor"].values()]
        ax.errorbar([31.2], [np.mean(v)], yerr=[[np.mean(v) - min(v)], [max(v) - np.mean(v)]], fmt="o", mfc="white", ms=7, mew=1.3, color=COL["floor"], capsize=3)  # range (~90 PIEs) is smaller than the marker; the caption gives it
    ax.set_xlabel("models per population (n)"); ax.set_ylabel("PIE count (mean, 95% of splits)")
    ax.set_xlim(2, 33); ax.set_ylim(0, 8200); ax.legend(fontsize=5, loc="upper right", frameon=False)
    ax.set_title("(a) against the dense population", fontsize=6.5, color=INK, loc="left")
    if pp:  # EXPLORATORY (DEVIATIONS D12, E1): pruned-vs-pruned floor at each sparsity against the dense floor
        bx = axs[0, 1]
        xs = [n for n in ns]
        bx.plot(xs, [S["M3"][str(n)]["floor"]["mean"] for n in xs], color=COL["floor"], lw=1.6, ls=LS["floor"], marker=MK["floor"], ms=3, label="dense vs dense")
        for key in ("0.3", "0.7", "0.9"):
            xq = sorted(int(n) for n in pp[key])
            bx.plot(xq, [pp[key][str(n)]["mean"] for n in xq], color=COL[key], lw=1.6, ls=LS[key], marker=MK[key], ms=3, label=f"{NAME[key].split(' ')[0]} pruned vs {NAME[key].split(' ')[0]} pruned")
        bx.set_xlabel("models per population (n)"); bx.set_ylabel("PIE count (mean over 1,000 splits)")
        bx.set_xlim(2, 14); bx.set_ylim(0, 8200); bx.legend(fontsize=5, loc="upper right", frameon=False)
        bx.set_title("(b) within-population floors (exploratory)", fontsize=6.5, color=INK, loc="left")
    save(fig, f"fig2_population_size_{dec}")


def fig_floor_subtracted(S, dec):
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ib = S["image_bootstrap"]
    rows = [("0.3", "D_0.3"), ("0.5", "D_0.5_n6"), ("0.7", "D_0.7"), ("0.9", "D_0.9")]
    for i, (key, bk) in enumerate(rows):
        b = ib[bk]
        ax.errorbar(b["point"], i, xerr=[[b["point"] - b["ci2.5"]], [b["ci97.5"] - b["point"]]], fmt="o", ms=5, color=COL[key], capsize=2, lw=1.5)
        ax.text(b["ci97.5"] + 40, i, f"{b['point']:.0f} [{b['ci2.5']:.0f}, {b['ci97.5']:.0f}]", va="center", fontsize=6, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAME[k] + (" (6 v 6)" if k == "0.5" else "") for k, _ in rows])
    ax.set_xlabel("PIEs above the floor (95% CI, image bootstrap)")
    ax.set_xlim(-150, max(ib[bk]["ci97.5"] for _, bk in rows) * 1.6)
    save(fig, f"fig3_floor_subtracted_{dec}")


def fig_m4(S, C, dec):
    """Class tests: uncorrected and BH as specified (literal), and BH with zero-variance classes excluded (ADDENDUM-3 C1b)."""
    fx = os.path.join(RES, "analysis", "m4fix.json")
    fix = json.load(open(fx)).get({"ifast": "val_ifast", "islow": "val_islow", "v2_ifast": "v2_ifast"}[dec], {}).get("n13") if os.path.exists(fx) else None
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    keys = [("floor", "m4_n13_floor"), ("0.3", "m4_n13_0.3"), ("0.7", "m4_n13_0.7"), ("0.9", "m4_n13_0.9")]
    marks = (("uncorrected p<0.05", "o", -0.22), ("BH q<=0.05, as specified", "s", 0.0), ("BH q<=0.05, degenerate classes excluded", "D", 0.22))
    for i, (key, ck) in enumerate(keys):
        for j, (lab, mk, dy) in enumerate(marks):
            if j < 2:
                v = C[ck][:, j]; m, lo, hi = v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)
            elif fix:
                r = fix[key]["bh"]; m, lo, hi = r["mean"], r["p2.5"], r["p97.5"]
            else:
                continue
            ax.errorbar(m, i + dy, xerr=[[m - lo], [hi - m]], fmt=mk, ms=3.5, color=COL[key], mfc=COL[key] if j == 0 else "none", capsize=2, lw=1.1,
                        label=lab if i == 0 else None)
    ax.set_yticks(range(4)); ax.set_yticklabels([NAME[k] for k, _ in keys])
    ax.set_xlabel("significant classes (Welch, 13 v 13; 95% of splits)")
    ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, frameon=False)
    save(fig, f"fig4_class_tests_{dec}")


def fig_fidelity():
    f = pd.read_csv(os.path.join(RES, "fidelity.csv"))
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    rng = np.random.default_rng(0)
    for i, (dec, lab) in enumerate((("ifast", "IFAST decode (official)"), ("islow", "ISLOW decode (frozen convert.py)"))):
        g = f[f.decoder == dec]
        ax.scatter(g.delta_top1_pp, i + rng.uniform(-0.18, 0.18, len(g)), s=8, color=COL["floor"] if i == 0 else COL["0.3"], lw=0.3, edgecolor="#fcfcfb")
        ax.text(0.125, i, f"{(~g['outside_0.1pp']).sum()}/{len(g)} within 0.1 pp", va="center", fontsize=6, color=INK)
    ax.axvspan(-0.1, 0.1, color="#f0efec", zorder=0)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["IFAST (official)", "ISLOW (frozen)"])
    ax.set_xlabel("our top-1 minus Hooker's released top-1 (pp), 123 models"); ax.set_xlim(-0.16, 0.22)
    save(fig, "fig5_fidelity")


def fig_m5():
    p = os.path.join(RES, "analysis", "ifast", "m5_counts.npz")
    if not os.path.exists(p):
        return
    C = np.load(p)
    fig, ax = plt.subplots(figsize=(W1, 0.35 + 0.28 * (1 + 2 * sum(f"{q}_paired" in C.files for q in ("fp16", "int8", "dynint8")))))
    qs = [(q, c) for q, c in (("fp16", COL["0.3"]), ("int8", COL["0.7"]), ("dynint8", COL["0.9"])) if f"{q}_paired" in C.files]
    rows = [("floor", C["floor"], COL["floor"])] + [(f"{q} {k}", C[f"{q}_{k}"], c) for q, c in qs for k in ("paired", "unpaired")]
    LAB = {"fp16": "FP16", "int8": "INT8 static", "dynint8": "INT8 dynamic"}
    for i, (lab, v, c) in enumerate(rows):
        ax.errorbar(v.mean(), i, xerr=[[v.mean() - np.percentile(v, 2.5)], [np.percentile(v, 97.5) - v.mean()]], fmt="o", ms=4, color=c, capsize=2)
        ax.text(np.percentile(v, 97.5) + 30, i, f"{v.mean():.0f}", va="center", fontsize=6, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(["dense vs dense (floor)"] + [f"{LAB[q]} vs {w}" for q, _ in qs for w in ("same models", "other half")])
    ax.set_xlabel("PIE count (13 v 13; mean, 95% of splits)"); ax.set_xlim(left=0)
    save(fig, "fig6_quantization_m5")


def fig_c1a():
    """CIFAR-10: floor for n = 3..15 (30 seeds), the four pre-registered n <= 13 fits, and the measured 30 v 30 floor (60 seeds)."""
    from scipy.optimize import curve_fit
    p = os.path.join(RES, "analysis", "cifar", "c1a.json")
    if not os.path.exists(p):
        return
    S, C1 = json.load(open(os.path.join(RES, "analysis", "cifar", "summary.json"))), json.load(open(p))
    ns = np.array(sorted(int(n) for n in S["M3c"])); y = np.array([S["M3c"][str(n)]["floor"]["mean"] for n in ns])
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.plot(ns, y, "o", ms=3, color=COL["floor"], label="measured floor, 30 dense seeds")
    xs = np.linspace(3, 31, 200)
    forms = {"a+b/sqrt(n)": (lambda n, a, b: a + b / np.sqrt(n), (100, 300)), "a+b*n^-c": (lambda n, a, b, c: a + b * n ** (-c), (100, 300, 0.5))}
    for i, (fn, (f, p0)) in enumerate(forms.items()):
        for lo in (3, 5):
            m = (ns >= lo) & (ns <= 13)
            prm, _ = curve_fit(f, ns[m], y[m], p0=p0, maxfev=20000)
            ax.plot(xs, f(xs, *prm), color=INK2, lw=0.8, ls="--" if i else ":", label="pre-registered fits (n <= 13)" if (i, lo) == (0, 3) else None)
    F = C1["measured_F30"]; lo, hi = C1["tolerance"]
    ax.axhspan(lo, hi, xmin=(28.5 - 2) / 30, xmax=(31.5 - 2) / 30, color="#f0efec")
    ax.errorbar([30], [F["mean"]], yerr=[[F["mean"] - F["p2.5"]], [F["p97.5"] - F["mean"]]], fmt="s", ms=4, color=COL["0.9"], capsize=2, label="measured 30 v 30 floor, 60 seeds")
    ax.set_xlim(2, 32); ax.set_xlabel("models per population (n)"); ax.set_ylabel("CIFAR-10 floor PIE count"); ax.legend(fontsize=5.5, loc="upper right")
    save(fig, "fig7_cifar_floor_extrapolation")


if __name__ == "__main__":
    for dec in ("ifast", "islow", "v2_ifast"):
        if os.path.exists(os.path.join(RES, "analysis", dec, "summary.json")):
            S, C = load(dec)
            fig_distributions(S, C, dec); fig_floor_subtracted(S, dec); fig_m4(S, C, dec)
            if "M3" in S:
                fig_popsize(S, dec)
    fig_fidelity(); fig_m5(); fig_c1a()
    print(sorted(os.listdir(OUT)))
