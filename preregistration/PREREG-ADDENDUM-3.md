# PRE-REGISTRATION ADDENDUM 3 — G1 conditions C1(a), C1(b), C2

**Frozen:** 2026-09-29. SHA-256 is in `logs/prereg_addendum3.sha256`. Read-only after freeze. It is frozen before:
- any extra CIFAR-10 dense seed (C1a) is trained;
- any dynamic-range INT8 prediction (C2) exists;
- the fixed M4 (C1b) has been computed on ISLOW, ImageNet-V2, ReaL or CIFAR-10.

PREREG.md, ADDENDUM-1 and ADDENDUM-2 are unchanged. Nothing here alters H1/H2 or their n = 13 results, which stay primary.

**Already seen, disclosed:**
- all results in FINDINGS.md (2026-09-29);
- the CIFAR-10 M3c floor means for n = 3..15 (30 dense seeds);
- for ImageNet val IFAST at n = 13, the exploratory zero-variance check in `results/analysis/ifast/exploratory.json`. That check already computed BH counts excluding zero-variance classes for the pruned arms: 1.28 / 41.2 / 292.9 at 30/70/90%. It did not compute them for the floor.

## C1(a) — CIFAR-10: measured n = 30 dense floor, and validation of the n → 30 extrapolation
- **Training:** 30 extra dense runs, seeds 30..59. Recipe, data, evaluation and code exactly as ADDENDUM-2 (`cifar/run.py 0.0 30 … 59`). This gives 60 dense runs.
- **Measured floor F30:** K = 1,000 uniform random splits of the 60 dense runs into two disjoint halves of 30. Mean, SD, 2.5/97.5 percentiles and MC SE.
  - RNG: `numpy.random.SeedSequence([20260928, 11])`, children in the order listed in `cifar/c1a.py`.
- **Extrapolation under test:** the four fits used for ImageNet in `src/exploratory.py`: a + b/√n and a + b·n^(−c), each fitted to the floor means for n ≥ 3 and n ≥ 5 up to n = 13. They were fitted to the existing CIFAR M3c floor (seeds 0..29) BEFORE this addendum. Predictions for n = 30:

  | fit | predicted F30 |
  |---|---|
  | a + b/√n, n ≥ 3 | 109.56 |
  | a + b/√n, n ≥ 5 | 110.76 |
  | a + b·n^−c, n ≥ 3 | 109.63 |
  | a + b·n^−c, n ≥ 5 | 114.81 |

  The envelope is [109.56, 114.81].
- **Tolerance, stated now:** the extrapolation is VALIDATED if the measured mean F30 lies in [0.95 × 109.56, 1.05 × 114.81] = [104.08, 120.55]. Otherwise it is NOT VALIDATED, and the signed relative error of each fit is reported.
- **Framing consequence (C5):**
  - If validated, the ImageNet n = 30 comparison with Hooker's printed counts is called a "validated extrapolation", and is secondary to the n = 13 results.
  - If not validated, it stays "exploratory", with the CIFAR error reported as the extrapolation's known bias.
- **Full 30-vs-30 pruned counts against the measured floor:** per split k, PIE(dense half A_k of 30, all 30 pruned at s) for s ∈ {0.3, 0.7, 0.9}.
  - z_s = (mean − mean F30) / SD(F30).
  - Image-bootstrap 95% CI (2,000 resamples) of the floor-subtracted count.
  - Descriptive; secondary to ADDENDUM-2's n = 13 decisions.

## C1(b) — M4 fix (replaces nothing; reported beside the literal M4)
- **Rule, per comparison (split k, population pair):** a class is EXCLUDED when the per-model recall values have zero variance within BOTH populations. There the Welch statistic is undefined, or infinite after Hooker's constant normalisation.
- For the remaining m classes, apply Hooker's Welch test (`ttest_ind(..., equal_var=False)` on recall − acc constant, as in ADDENDUM-1 §4). Count p < 0.05 uncorrected, and BH q ≤ 0.05 over those m classes. Report m.
- **Rerun on the same splits and draws as the literal M4:**
  - ImageNet val IFAST and ISLOW (n = 13: floor, 30/70/90%; 6 v 6: floor and 50%);
  - ImageNet-V2 (n = 13);
  - ReaL-based recall (n = 13);
  - CIFAR-10 (n = 13, ADDENDUM-2 splits).
- **Constants:** Hooker's printed ImageNet constants for val, V2 and ReaL; CIFAR per-sparsity 30-run means as in ADDENDUM-2.
- **Code:** `src/m4fix.py`. Descriptive (M4 carries no decision rule in PREREG).

## C2 — Hooker-like dynamic-range INT8 (TFLite-style), H3 re-test
- **Simulation** (fake-quant on the GPU, float32 compute, FP32 settings as in A1), applied to each of the 27 dense models:
  1. Fold every BatchNorm into its preceding convolution (`torch.nn.utils.fusion.fuse_conv_bn_eval`), as the TFLite converter does.
  2. Weights: symmetric per-output-channel int8, scale = max|w| / 127 per channel, w_q = clamp(round(w / scale), −127, 127) · scale. This covers all convolutions and the final dense layer; the dense layer is per output unit.
  3. Activations: at the INPUT of every convolution and of the final dense layer, dynamic symmetric int8 per image and per tensor. Scale = max|x| / 127 computed on the fly for each image, x_q = clamp(round(x / scale), −127, 127) · scale. There is no calibration data.
  4. Biases stay float.
- **Measurements:** as ADDENDUM-1 §11 on the M1 splits (n = 13):
  - paired = PIE(A_k dyn-int8, A_k FP32);
  - unpaired = PIE(A_k dyn-int8, B_k FP32);
  - floor = M1.
- **H3 rule, unchanged:** H3 holds if mean paired < mean floor − 3·SD_floor AND |mean unpaired − mean floor| ≤ 3·SD_floor. The withdrawal clause applies if |paired − unpaired| ≤ 3·SD_floor.
- The FX static-PTQ INT8 result (H3 fails) stays reported. This is an additional recipe, not a replacement.
- **Code:** `src/m5.py eval dynint8` / `analyze`. Per-model top-1 and flips are reported against Hooker's printed dynamic-range INT8 top-1 (76.10).

## GPU
- C1(a) is about 30 × 25 s ≈ 0.2 GPU-h.
- C2 is about 27 × 125 s ≈ 1 GPU-h.
- Each batch holds `~/research/.gpu.p3` and releases it between batches, taking `~/research/.gpu.lock` per model or run.
