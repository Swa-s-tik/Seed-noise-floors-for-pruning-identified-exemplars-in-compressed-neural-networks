# PRE-REGISTRATION ADDENDUM 1 — analysis specification, A2 route, breadth (a), M5

**Frozen:** 2026-09-29, before any population-level quantity has been computed. SHA-256 is in `logs/prereg_addendum1.sha256`. Read-only after freeze. It adds detail to PREREG.md (sha256 eba59b5f…) and to D3/D4 in DEVIATIONS.md. It changes no hypothesis, threshold or decision rule. Where it conflicts with PREREG.md, PREREG.md wins, and the conflict is logged in DEVIATIONS.md.

**Already seen when this was written:**
- Per-model top-1/top-5 and fidelity (`results/fidelity.csv`).
- The released per-image files' schema: 49,187 rows, 126 of 128 shards (6 and 111 missing), shard sizes 343–445, `area_bbox_mean` ≡ 0, `n_channels_out` ≡ 3.

No PIE count, modal label, split, Welch test or Jaccard from our predictions exists yet.

## 1. Inputs
- **Predictions:** `results/preds/<run_id>.npz` (IFAST, primary per D3) and `results/preds_islow/<run_id>.npz` (ISLOW, sensitivity), keyed by `results/preds/image_order.csv`. All 50,000 images. Only run_ids in INVENTORY.json with `complete=true` are used; tagged copies (`_rep2`, `_cachecheck*`) are ignored.
- **Populations:**
  - dense: 27, of which 11 dense-baseline and 16 dense-threshold;
  - pruned: 30 at 0.3, 6 at 0.5, 30 at 0.7, 30 at 0.9.

## 2. Modal label and PIE (Hooker's code, verbatim)
- **Modal label:** `pie_dataset_gen/aggregate_predictions.py` computes `predicted_mode_variant = variant['predictions'].mode().max()`, i.e. the most frequent top-1 label, with ties broken by the LARGEST label index. We compute this by per-image label counts and quote the line verbatim in the code.
- **PIE:** `generate_pie_dataset.py` sets `is_pie` when `variant_mode_label != baseline_mode_label`. PIE count = number of images with is_pie.

## 3. Randomness
- One master `numpy.random.SeedSequence(20260928)`. One child is spawned per named condition, in the fixed order listed in `src/analysis.py` (`CONDITIONS`). K = 1,000 draws per condition.
- **Stratified draw of dense halves of size n:** each half gets a number of dense-baseline models b drawn by randomized rounding of n·11/27: floor(x), or ceil(x) with probability x − floor(x). The draw is independent per half and rejected unless b_A + b_B ≤ 11 and (n − b_A) + (n − b_B) ≤ 16. The models within each role are then drawn uniformly without replacement, disjoint across halves.
- **Pruned draws:** uniform without replacement.

## 4. Measurements (IFAST primary; everything repeated under ISLOW)
- **M1:** PIE(dense A_k, dense B_k), n = 13, k = 1..1000.
- **M2:** PIE(dense A_k, pruned P_k) at s ∈ {0.3, 0.7, 0.9}, with P_k = 13 of 30 drawn per k, and A_k the M1 half of the same k. At s = 0.5: PIE(dense 6, all 6 pruned) against a separate 6-vs-6 dense floor, K = 1000 each.
- **M3:** as M1/M2 for n = 3..13 (0.5 for n ≤ 6), K = 1000 per n.
- **M4:**
  - Per split, per class c: recall_c of each model = correct/50 (every val class has exactly 50 images).
  - Test: scipy `ttest_ind(baseline − acc[0.0], variant − acc[s], equal_var=False)` with Hooker's printed constants `acc = {0.0: 0.76675, 0.3: 0.76459, 0.5: 0.75874, 0.7: 0.75019, 0.9: 0.72569}` (gen_tstatistic.py).
  - Count classes with p < 0.05 uncorrected, and with Benjamini–Hochberg q ≤ 0.05 over the 1000 classes. NaN p (zero variance in both groups) counts as not significant.
  - Floor = dense A_k vs dense B_k. Pruned = dense A_k vs P_k.
  - Sensitivity: normalise by each population's own mean top-1 instead of the constants.

## 5. Decisions (unchanged from PREREG)
- SD_floor = SD of the 1,000 M1 counts at n = 13. z_s = (mean M2_s − mean M1) / SD_floor.
- **H1** holds if |z_0.3| ≤ 3. It fails ("low-sparsity PIEs are real") if z_0.3 > 3.
- **H2** holds at s ∈ {0.7, 0.9} if z_s > 3.
- Decisions are taken on IFAST. ISLOW is reported beside them, and any disagreement goes in the main text.

## 6. Intervals
- **Split distribution:** mean, SD, 2.5 and 97.5 percentiles.
- **Monte-Carlo error:** SE(mean) = SD/√K, and bootstrap over the K split results (2,000 resamples) for the SE of each percentile.
- **Image-level bootstrap** (2,000 resamples of the 50,000 images, seed from the master sequence):
  - f_i = fraction of the K splits in which image i is a PIE. Count = Σ f_i. 95% percentile CI.
  - Floor-subtracted count D_s = Σ_i (f_i^{M2,s} − f_i^{M1}), paired resampling, 95% CI.
- The split-level floor-subtracted count mean(M2_s) − mean(M1) is also reported, with its z.

## 7. Ablations (descriptive; none changes H1/H2)
1. **Population size:** M3.
2. **Decoder:** ISLOW repeat.
3. **Dense pool:**
   - floor within dense-threshold only (8 v 8) and within dense-baseline only (5 v 5);
   - cross-type PIE(baseline n, threshold n) for n = 3..11, against the stratified floor at the same n;
   - M2 with dense-threshold-only dense halves (n = 8).
4. **Cohort:** M1 at n = 13 with splits stratified by cohort (D1; randomized-rounding allocation per cohort) instead of role, and with unstratified uniform splits.
5. **Tie-break:** M1 and M2 at n = 13 with `.mode().min()`, and with a uniformly random choice among tied labels.

## 8. Comparison with Hooker's counts (descriptive)
- **Printed:** PIE counts from arXiv:1911.05248 v1–v3, taken from a first read plus an independent second read (`provenance/second_read/`).
- **Released:** is_pie totals in `pie07.csv` and `pie09.csv` (49,187 images).
- **Ours:**
  - the full-population recompute, all 30 pruned vs each dense pool (27 / 16 / 11), on 50,000 images and scaled ×49,187/50,000;
  - the M2 n = 13 mean;
  - the floor.

## 9. A2 route
- The released per-image files carry no filename or other prediction-free image identity, only the shard name, `image_index`, stored class label and prediction-derived columns.
- **Primary:** find a prediction-free shard assignment, i.e. a candidate function of the image filename whose per-shard class-label multisets equal those of all 126 released shards. Then check the within-shard order against the `stored_class_label` sequence. If found, compute A2 exactly as pre-registered: Jaccard of is_pie on the 49,187 intersection, for each candidate reference population (dense pool 27/16/11 × all 30 pruned), at 0.7 and 0.9. PASS if the best Jaccard is ≥ 0.8.
- **If no prediction-free mapping is found:** A2 is reported as NOT COMPUTABLE AS SPECIFIED. Matching images on modal labels would be circular and is not used. Substitute evidence, computed per candidate population and sparsity:
  1. **Jaccard upper bound** over every image mapping consistent with class labels. Per class c, choose our count a'_c ∈ [max(0, a_c − m_c), a_c] closest to Hooker's b_c, where m_c = 50 − (Hooker's image count in c). Bound = Σ min(a'_c, b_c) / Σ max(a'_c, b_c). Bound < 0.8 ⇒ A2 FAIL. Bound ≥ 0.8 ⇒ A2 UNDETERMINED.
  2. Pearson and Spearman correlation of per-class PIE counts.
  3. The total-count difference.

## 10. Breadth (a): ImageNet-V2 and ImageNet-ReaL
- **ImageNet-V2 matched-frequency:**
  - 10,000 images; class index = folder name 0..999 in standard order; source is the public ImageNetV2 release. The download's checksum is recorded.
  - Same IFAST pipeline and evaluation settings; all 123 models, on the GPU under the P3 gate.
  - M1, M2 (n = 13) and M4 are recomputed on V2. M4 uses recall = correct/10; classes with fewer than 10 images use their own count. H1/H2 are re-evaluated on V2 as a replication (secondary, not decisive).
- **ImageNet-ReaL:**
  - `real.json` (Beyer et al. 2020, google-research/reassessed-imagenet), mapped by validation index; the file checksum is recorded.
  - PIE counts are label-free and unchanged.
  - Reported under ReaL: per-model top-1 (correct if the prediction is in the ReaL set; images with an empty set excluded); M4 with ReaL-based recall; and the rates of empty and multi-label ReaL sets among PIEs vs non-PIEs at n = 13 (descriptive).

## 11. M5 / H3 (PTQ of the 27 dense models)
- **FP16:** `model.half()` on the GPU, with inputs from the IFAST float32 cache cast to fp16. Settings as in A1 (deterministic cuDNN, no TF32, batch 100, the same batch order). One model per GPU unit.
- **INT8:** PyTorch FX graph-mode static PTQ, fbgemm backend, default fbgemm qconfig (per-channel symmetric weights, per-tensor affine activations, histogram observers). Calibration uses the first 1,024 images of one ImageNet-1k TRAIN parquet shard (IFAST preprocessing; never validation or V2). It runs on the CPU (≤ 8 threads) because PyTorch has no native GPU INT8 path.
- **Measurements** per split k (M1 halves, n = 13), for q ∈ {fp16, int8}:
  - paired = PIE(A_k in q, A_k in FP32);
  - unpaired = PIE(A_k in q, B_k in FP32);
  - floor = M1.
- **H3** holds if mean paired < mean floor − 3·SD_floor AND |mean unpaired − mean floor| ≤ 3·SD_floor. If paired ≈ unpaired (|difference| ≤ 3·SD_floor), the inference that Hooker's quantization rows reflect seed noise is withdrawn, as PREREG states.

## 12. Not in this addendum
The CIFAR-10 second population (breadth b) needs its own addendum, frozen before any training.
