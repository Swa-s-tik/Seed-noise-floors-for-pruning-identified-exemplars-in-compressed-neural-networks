# Generates the supplementary LaTeX tables directly from the committed results (no hand transcription).
# usage: python scripts/make_tables.py  -> results/tables/supp_tables.tex (the supplementary tables of the article)
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
R = lambda *p: os.path.join(ROOT, "results", *p)
J = lambda *p: json.load(open(R(*p)))
f0 = lambda x: f"{x:,.0f}"; f1 = lambda x: f"{x:,.1f}"; f2 = lambda x: f"{x:.2f}"


def tab(caption, label, cols, rows, align=None):
    align = align or "l" + "r" * (len(cols) - 1)
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n" + "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n"  # shrink only tables wider than the text
            + f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"
            + " & ".join(cols) + r" \\" + "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}}" + "\n\\end{table}\n")


def main():
    out = []
    inv = json.load(open(os.path.join(ROOT, "INVENTORY.json")))["runs"]
    df = pd.DataFrame(inv.values())
    rows = []
    for (role, sp), g in df.groupby(["role", "sparsity"]):
        rows.append([role, f"{sp:.2f}", str(int(g.in_readme_list.sum())), str(int(g.in_tstat.sum())), str(int(g.complete.sum()))])
    out.append(tab("Checkpoints by role and sparsity in the authors' reference list, in their released class-level statistics, and complete in the public bucket.",
                   "tab:s_inventory", ["Role", "Sparsity", "Reference list", "Released statistics", "Complete in bucket"], rows))
    fid = pd.read_csv(R("fidelity.csv"))
    rows = []
    for dec, g in fid.groupby("decoder"):
        rows.append([{"ifast": "fast (adopted)", "islow": "accurate (sensitivity)"}[dec], str(len(g)), str(int((~g["outside_0.1pp"]).sum())),
                     f"{g.delta_top1_pp.mean():+.3f}", f"{g.delta_top1_pp.std():.3f}", f"{g.delta_top1_pp.abs().max():.3f}", f"{g.delta_top5_pp.abs().max():.3f}"])
    out.append(tab("Fidelity of the re-evaluated models against the released top-1 accuracy, by JPEG decoder. Differences are in percentage points.",
                   "tab:s_fidelity", ["JPEG decoder", "Models", "Within 0.1 pp", "Mean top-1 diff.", "SD", "Max abs. top-1", "Max abs. top-5"], rows))
    for dec, name in (("ifast", "fast JPEG decoder"), ("islow", "accurate JPEG decoder"), ("v2_ifast", "ImageNet-V2")):
        S = J("analysis", dec, "summary.json"); ib = S["image_bootstrap"]
        rows = [["Dense vs dense (floor)", f"{f1(S['M1']['mean'])} $\\pm$ {f1(S['M1']['sd'])}", f"[{f0(S['M1']['p2.5'])}, {f0(S['M1']['p97.5'])}]", f2(S['M1']['mc_se_mean']), "", ""]]
        for s in ("0.3", "0.7", "0.9"):
            m = S[f"M2_{s}"]; d = ib[f"D_{s}"]
            rows.append([f"{int(float(s) * 100)}\\% pruned vs dense", f"{f1(m['mean'])} $\\pm$ {f1(m['sd'])}", f"[{f0(m['p2.5'])}, {f0(m['p97.5'])}]", f2(m['mc_se_mean']),
                         f"{S['decisions']['z'][s]:.1f}", f"{f0(d['point'])} [{f0(d['ci2.5'])}, {f0(d['ci97.5'])}]"])
        m, fl, d = S["M2_0.5_n6"], S["M1_n6"], ib["D_0.5_n6"]
        rows.append(["Floor, 6 vs 6", f"{f1(fl['mean'])} $\\pm$ {f1(fl['sd'])}", f"[{f0(fl['p2.5'])}, {f0(fl['p97.5'])}]", f2(fl['mc_se_mean']), "", ""])
        rows.append(["50\\% pruned vs dense, 6 vs 6", f"{f1(m['mean'])} $\\pm$ {f1(m['sd'])}", f"[{f0(m['p2.5'])}, {f0(m['p97.5'])}]", f2(m['mc_se_mean']),
                     f"{S['decisions']['z_0.5_n6']:.1f}", f"{f0(d['point'])} [{f0(d['ci2.5'])}, {f0(d['ci97.5'])}]"])
        out.append(tab(f"PIE counts with the {name} (13 vs 13 unless stated; 1,000 splits). Columns give mean $\\pm$ SD over splits, the 2.5 and 97.5 percentiles, the Monte Carlo standard error of the mean, $z$, and the excess over the floor with its 95\\% image-bootstrap interval.",
                       f"tab:s_counts_{dec}", ["Comparison", "Mean $\\pm$ SD", "95\\% of splits", "MC SE", "$z$", "Excess [95\\% CI]"], rows, "lrrrrr"))
        if "M3" in S:
            rows = []
            for n in sorted(S["M3"], key=int):
                r = S["M3"][n]; sd = r["floor"]["sd"]
                rows.append([n, f"{f0(r['floor']['mean'])} $\\pm$ {f0(sd)}"] + [f"{f0(r[s]['mean'])} ({(r[s]['mean'] - r['floor']['mean']) / sd:.1f})" if s in r else "" for s in ("0.3", "0.5", "0.7", "0.9")])
            out.append(tab(f"PIE count against population size $n$ with the {name}. Pruned columns give the mean count and, in parentheses, $z$ against the floor at the same $n$.",
                           f"tab:s_popsize_{dec}", ["$n$", "Floor", "30\\%", "50\\%", "70\\%", "90\\%"], rows))
        if "ablations" in S:
            ab = S["ablations"]
            rows = [[lab, f"{f1(ab[k]['floor']['mean'])} $\\pm$ {f1(ab[k]['floor']['sd'])}", f1(ab[k]["0.3"]["mean"]), f"{(ab[k]['0.3']['mean'] - ab[k]['floor']['mean']) / ab[k]['floor']['sd']:.2f}"]
                    for lab, k in (("Balanced by kind of dense model (primary)", None), ("Balanced by training batch", "cohort_stratified_n13"), ("Unstratified", "uniform_n13"),
                                   ("Ties to smallest label", "tiebreak_min_n13"), ("Ties at random", "tiebreak_random_n13")) if k]
            rows.insert(0, ["Balanced by kind of dense model (primary)", f"{f1(S['M1']['mean'])} $\\pm$ {f1(S['M1']['sd'])}", f1(S["M2_0.3"]["mean"]), f"{S['decisions']['z']['0.3']:.2f}"])
            out.append(tab(f"Robustness of the 13 vs 13 floor and of the 30\\% comparison to split and tie-breaking choices ({name}).", f"tab:s_ablation_{dec}",
                           ["Variant", "Floor", "30\\% count", "$z_{0.3}$"], rows))
            rows = [[n, f1(v["cross_type"]["mean"]), f1(v["stratified_floor"]["mean"]), f"{(v['cross_type']['mean'] - v['stratified_floor']['mean']) / v['stratified_floor']['sd']:.2f}"]
                    for n, v in sorted(ab["cross_type_baseline_vs_threshold"].items(), key=lambda t: int(t[0]))]
            out.append(tab(f"Dense-baseline population against dense-threshold population of equal size $n$, compared with the balanced floor ({name}).", f"tab:s_cross_{dec}",
                           ["$n$", "Cross-kind count", "Balanced floor", "Difference in floor SD"], rows))
    fx = J("analysis", "m4fix.json")
    rows = []
    for ds, lab in (("val_ifast", "ImageNet val, fast"), ("val_islow", "ImageNet val, accurate"), ("v2_ifast", "ImageNet-V2"), ("real_ifast", "ImageNet-ReaL"), ("cifar", "CIFAR-10")):
        r = fx[ds]["n13"]
        for k in ("floor", "0.3", "0.7", "0.9"):
            rows.append([lab if k == "floor" else "", "floor" if k == "floor" else f"{int(float(k) * 100)}\\%", f1(r[k]["uncorrected"]["mean"]), f2(r[k]["bh"]["mean"]),
                         f"[{f0(r[k]['bh']['p2.5'])}, {f0(r[k]['bh']['p97.5'])}]", f1(r[k]["classes_tested"]["mean"])])
    out.append(tab("Per-class Welch test with degenerate classes excluded (13 vs 13). Columns give uncorrected and Benjamini--Hochberg counts, the 95\\% range of the latter over splits, and the mean number of classes tested.",
                   "tab:s_m4fix", ["Setting", "Comparison", "Uncorrected", "BH", "BH 95\\%", "Classes tested"], rows, "llrrrr"))
    a2 = J("a2.json"); rows = []
    for dec in ("ifast", "islow"):
        for k, v in a2[dec].items():
            pool, _, _, s = k.split("_")[0], *k.split("_")[1:3], k.split("_")[-1]
            rows.append([dec, pool, s, f0(v["our_pie_total_50000"]), f0(v["released_pie_total_49187"]), f"{v['jaccard_upper_bound']:.3f}", f"{v['per_class_pearson']:.2f}", v["A2_substitute_verdict"].split(" ")[0]])
    ms = a2["mapping_search"]
    out.append(tab(f"Gate A2 substitute evidence. The image mapping was not recoverable: the best of {len(ms['candidates'])} filename-hash rules gives a class-histogram distance of {min(c['L1'] for c in ms['candidates']):,}, against {ms['random_sharding_L1']:,} for random sharding. Columns give our PIE total, the released total, the label-only Jaccard upper bound, the per-class correlation and the verdict.".replace(":", ","),
                   "tab:s_a2", ["Decoder", "Dense pool", "Sparsity", "Ours", "Released", "Jaccard bound", "Pearson", "Verdict"], rows, "lllrrrrl"))
    m5 = J("analysis", "ifast", "m5.json"); rows = []
    PRINTED_TOP1 = {"fp16": "76.65", "int8": "76.46 (fixed-point)", "dynint8": "76.10"}  # arXiv:1911.05248 v3 Table 2 (three independent reads)
    for q, lab in (("fp16", "16-bit float"), ("int8", "8-bit static"), ("dynint8", "8-bit dynamic range")):
        if q in m5:
            v = m5[q]; t = np.mean(list(v["per_model_top1"].values())) * 100; fl = np.mean(list(v["per_model_flips_vs_fp32"].values()))
            rows.append([lab, PRINTED_TOP1[q], f"{t:.2f}", f0(fl), f"{f1(v['paired']['mean'])} $\\pm$ {f1(v['paired']['sd'])}", f"{f1(v['unpaired']['mean'])} $\\pm$ {f1(v['unpaired']['sd'])}",
                         f"{v['z_paired']:.1f}", f"{v['z_unpaired']:.1f}", "holds" if v["H3_holds"] else "fails"])
    out.append(tab(f"Quantization of the 27 dense models (M5). The floor is {f1(m5['floor_recomputed']['mean'])} $\\pm$ {f1(m5['floor_recomputed']['sd'])}. Columns give mean top-1, mean changed predictions per model, paired and unpaired counts, their $z$ against the floor, and the H3 decision.",
                   "tab:s_m5", ["Recipe", "Printed top-1 (\\%)", "Our top-1 (\\%)", "Changed", "Paired", "Unpaired", "$z$ paired", "$z$ unpaired", "H3"], rows))
    C = J("analysis", "cifar", "summary.json"); c1 = J("analysis", "cifar", "c1a.json")
    rows = [[f"{s}", f"{C['per_sparsity_mean_top1'][s] * 100:.2f}"] for s in ("0.0", "0.3", "0.7", "0.9")]
    out.append(tab("CIFAR-10 populations trained here, 30 seeds per sparsity. Mean single-view test top-1 accuracy.", "tab:s_cifar_acc", ["Sparsity", "Top-1 (\\%)"], rows))
    TEXN = {"a+b/sqrt(n), n>=3": "$a + b/\\sqrt{n}$, $n \\ge 3$", "a+b/sqrt(n), n>=5": "$a + b/\\sqrt{n}$, $n \\ge 5$",
            "a+b*n^-c, n>=3": "$a + b\\,n^{-c}$, $n \\ge 3$", "a+b*n^-c, n>=5": "$a + b\\,n^{-c}$, $n \\ge 5$"}
    rows = [[TEXN[k], f1(v), f"{c1['relative_error_per_fit'][k] * 100:+.1f}\\%"] for k, v in c1["predictions"].items()]
    rows.append(["Measured 30 vs 30 floor", f"{f1(c1['measured_F30']['mean'])} $\\pm$ {f1(c1['measured_F30']['sd'])}", ""])
    out.append(tab(f"Pre-registered extrapolation of the CIFAR-10 floor to 30 models per side, against its direct measurement. Tolerance [{c1['tolerance'][0]:.1f}, {c1['tolerance'][1]:.1f}]; validated = {c1['extrapolation_validated']}.",
                   "tab:s_c1a", ["Fit (on $n \\le 13$)", "Predicted / measured", "Relative error"], rows))
    # ---- round-1 additions: exploratory E1-E6 (DEVIATIONS D12) and the structured CIFAR-10 arm (ADDENDUM-4)
    X = J("analysis", "explore_r1.json")
    rows = [[s + " (paired split excess)", f"{f1(v['mean'])} $\\pm$ {f1(v['sd'])}", f"{v['frac_positive'] * 100:.1f}\\%", f"{v['z_mc_se']:.2f}"] for s, v in X["E5_paired_excess_n13"].items()]
    out.append(tab("Per-split excess of the pruned count over the floor at $n = 13$ (same split), the share of splits with a positive excess, and the Monte Carlo standard error of $z$ (bootstrap over splits). Exploratory.",
                   "tab:s_paired", ["Sparsity", "Excess, mean $\\pm$ SD", "Splits $> 0$", "MC SE of $z$"], rows))
    e1 = X["E1_pruned_vs_pruned"]
    rows = [[n, f"{f0(e1['0.3'][n]['dense_floor_mean'])}"] + [f"{f0(e1[s][n]['mean'])} $\\pm$ {f0(e1[s][n]['sd'])}" for s in ("0.3", "0.7", "0.9")] for n in sorted(e1["0.3"], key=int)]
    out.append(tab(f"Pruned-versus-pruned floor (two disjoint populations of $n$ from the 30 pruned models) against the dense floor. At 50\\% only 3 vs 3 is possible, giving {f0(e1['0.5']['3']['mean'])} against a dense floor of {f0(e1['0.5']['3']['dense_floor_mean'])}. Exploratory.",
                   "tab:s_pp", ["$n$", "Dense floor", "30\\%", "70\\%", "90\\%"], rows))
    e2 = X["E2_seed_variation_jaccard"]; b = e2["A2_label_only_bound_ifast"]
    rows = [[f"{int(float(s) * 100)}\\%", f"{e2[s]['13v13']['mean']:.3f} $\\pm$ {e2[s]['13v13']['sd']:.3f}", f"{e2[s]['13v14']['mean']:.3f} $\\pm$ {e2[s]['13v14']['sd']:.3f}",
             f"{min(v for k, v in b.items() if k.endswith(s)):.3f}--{max(v for k, v in b.items() if k.endswith(s)):.3f}" if any(k.endswith(s) for k in b) else "--"] for s in ("0.3", "0.7", "0.9")]
    out.append(tab("Jaccard overlap of PIE sets computed against two disjoint dense populations and the same 30 pruned models (seed variation only). Beside it is the label-only upper bound of gate A2 against the released sets (70\\% and 90\\% only, pre-registered threshold 0.8). Exploratory.",
                   "tab:s_jac", ["Sparsity", "13 vs 13", "13 vs 14", "A2 label-only bound"], rows))
    e3 = X["E3_real_compression_specific"]; npie = e3["non_pie_reference_from_real_json"]
    rows = [["Floor PIEs", f"{e3['0.3']['floor']['empty_real'] * 100:.1f}", f"{e3['0.3']['floor']['multi_label'] * 100:.1f}", f"{e3['0.3']['floor']['orig_not_in_real'] * 100:.1f}"]]
    for s in ("0.3", "0.7", "0.9"):
        for g, lab in (("compression_specific", "compression-specific"), ("shared_with_floor", "shared with the floor")):
            d = e3[s][g]; rows.append([f"{int(float(s) * 100)}\\% PIEs, {lab}", f"{d['empty_real'] * 100:.1f}", f"{d['multi_label'] * 100:.1f}", f"{d['orig_not_in_real'] * 100:.1f}"])
    rows.append(["Other images", f"{npie['empty_real_nonpie'] * 100:.1f}", f"{npie['multi_label_nonpie'] * 100:.1f}", f"{npie['orig_label_not_in_real_nonpie'] * 100:.1f}"])
    out.append(tab("ImageNet-ReaL label quality (\\%). Compression-specific PIEs are, in each split, PIEs against the pruned population that are not PIEs against the other dense population; the rest are shared with the floor. Exploratory.",
                   "tab:s_real", ["Image group", "No valid label", "Multiple labels", "Original label not in set"], rows, "lrrr"))
    e6 = X["E6_cifar_exchangeability"]
    rows = [[k.replace("_", " "), f"{f1(e6[k]['mean'])} $\\pm$ {f1(e6[k]['sd'])}"] for k in ("within_0_29", "within_30_59", "across")]
    out.append(tab(f"Seed-sampling variability of the CIFAR-10 dense floor, 13 vs 13 within each half of the 60-seed pool and across halves. Mean top-1 {e6['top1_seeds_0_29'] * 100:.2f}\\% (seeds 0--29) and {e6['top1_seeds_30_59'] * 100:.2f}\\% (seeds 30--59). Exploratory.",
                   "tab:s_seedvar", ["Seed pool", "Floor, mean $\\pm$ SD"], rows))
    if "E4_int8_sim_vs_true_2000" in X:
        e4 = X["E4_int8_sim_vs_true_2000"]
        rows = [[k.replace("_", " "), f"{f1(e4[k]['mean'])} $\\pm$ {f1(e4[k]['sd'])}"] for k in ("floor", "paired_sim", "paired_true", "unpaired_sim", "unpaired_true", "modal13_disagreement_sim_vs_true")]
        rows.append(["per-model top-1 disagreement, simulated vs integer", f"{e4['per_model_top1_disagreement_sim_vs_true']['mean'] * 100:.2f}\\%"])
        out.append(tab(f"Simulated against true integer 8-bit (static recipe) on the first {e4['n_images']:,} validation images, 13 vs 13 over 1,000 splits. Exploratory.",
                       "tab:s_e4", ["Quantity", "Value"], rows))
    st = R("analysis", "cifar", "structured.json")
    if os.path.exists(st):
        T = json.load(open(st)); rows = []
        for s in ("0.3", "0.7", "0.9"):
            rows.append([f"{int(float(s) * 100)}\\%", f"{T['mean_top1'][f'structured_{s}'] * 100:.2f}", str(T["runs_below_0.85"][s]),
                         f"{f1(T['n13'][s]['mean'])} $\\pm$ {f1(T['n13'][s]['sd'])}", f"{T['n13'][s]['z']:.1f}",
                         f"{f0(T['image_bootstrap_n13'][f'D_{s}']['point'])} [{f0(T['image_bootstrap_n13'][f'D_{s}']['ci2.5'])}, {f0(T['image_bootstrap_n13'][f'D_{s}']['ci97.5'])}]",
                         f"{T['n30'][s]['z']:.1f}", f"{X['E10_cifar_z_mc_se']['structured_n13'][s]:.2f}",
                         f"{f1(X['E8_structured_vs_structured_n13'][s]['mean'])} $\\pm$ {f1(X['E8_structured_vs_structured_n13'][s]['sd'])}"])
        out.append(tab(f"CIFAR-10 structured filter pruning (ADDENDUM-4), 30 seeds per sparsity. The 13 vs 13 floor is {f1(T['n13']['floor']['mean'])} $\\pm$ {f1(T['n13']['floor']['sd'])} and the 30 vs 30 floor is {f1(T['n30']['floor']['mean'])} $\\pm$ {f1(T['n30']['floor']['sd'])} from 60 dense seeds (dense top-1 {T['mean_top1']['dense60'] * 100:.2f}\\%). The last two columns are exploratory (D15). They give the Monte Carlo error of the 13 vs 13 $z$ and the structured-versus-structured floor (13 vs 13, against a dense floor of {f1(X['E8_structured_vs_structured_n13']['dense_floor_60']['mean'])} in the same draws).",
                       "tab:s_struct", ["Sparsity", "Top-1 (\\%)", "Runs $<$ 85\\%", "PIEs (13 vs 13)", "$z$", "Excess [95\\% CI]", "$z$, 30 vs 30", "MC error of $z$", "Structured vs structured"], rows))
        e9 = X["E9_unstructured_on_60seed_floor_n13"]; b = T["image_bootstrap_n13"]
        rows = [[m, f"{int(float(s) * 100)}\\%", f"{(e9['dense_top1_60'] - top) * 100:.2f}", f"{z:.1f}", f"{f0(d['point'])} [{f0(d['ci2.5'])}, {f0(d['ci97.5'])}]".replace("-", "$-$")]
                for m, s, top, z, d in sorted([("unstructured", s, e9[s]["top1"], e9[s]["z"], e9[s]["excess"]) for s in ("0.3", "0.7", "0.9")]
                                              + [("structured", s, T["mean_top1"][f"structured_{s}"], T["n13"][s]["z"], b[f"D_{s}"]) for s in ("0.3", "0.7", "0.9")],
                                              key=lambda r: e9["dense_top1_60"] - r[2])]
        out.append(tab("CIFAR-10 pruning arms against the same 60-seed dense floor (13 vs 13), ordered by accuracy cost. Structured rows are the pre-registered ADDENDUM-4 analysis; unstructured rows re-count the ADDENDUM-2 runs with the same procedure in separate draws (exploratory, D15).",
                       "tab:s_matched", ["Pruning", "Sparsity", "Accuracy cost (pp)", "$z$", "Excess [95\\% CI]"], rows))
    os.makedirs(R("tables"), exist_ok=True); open(R("tables", "supp_tables.tex"), "w").write("% generated by make_tables.py from results/ (do not edit by hand)\n" + "\n".join(out))
    print(len(out), "tables")


if __name__ == "__main__":
    main()
