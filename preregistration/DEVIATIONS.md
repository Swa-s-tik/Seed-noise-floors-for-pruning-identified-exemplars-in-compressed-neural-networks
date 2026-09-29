# DEVIATIONS — append-only, timestamped

Release copy of the study's deviation log. Internal role names and review-item labels were reworded or removed; every timestamp, number, hash and decision is unchanged. In this repository the pre-registration files are in `preregistration/`, the reproduction scripts in `scripts/`, and only the checksum files and the environment record of `logs/` are included (the run logs named below are not).

## 2026-09-28T17:43:26+05:30 — D1 clarification: "cohort" definition
PREREG names cohort (INVENTORY field; M1 stratification) but does not define it. INVENTORY.json defines it as the launch batch:
experiment ids sorted, new cohort when the id gap > 100, named C<first id>. Gives 7 cohorts over the 123 released checkpoints
(C2677641, C2677938, C2688176, C2690405, C2799524, C2802319, C2806194). Reason: only batch structure recorded in the artefacts.
May be redefined before M1; if so, it is recorded here.

## 2026-09-28T17:43:26+05:30 — D2 FLAG (no action taken): M1 "dense(15) vs dense(15)" not feasible with released checkpoints
Bucket holds 27 dense checkpoints (11 dense-baseline + 16 dense-threshold), not 30. tstat.csv lists 47 dense (17 + 30);
the README reference list names 48 (18 + 30). Max balanced dense split from the bucket = 13 vs 13.
Pruned: 30 @0.3, 6 @0.5, 30 @0.7, 30 @0.9 (as PREREG assumed). Remedy (e.g. 13v13 floor, or re-define M1/M2 n) is a
protocol decision; nothing computed.

## 2026-09-28T18:42:18+05:30 — D3 PROPOSED, NOT ADOPTED (protocol decision pending): JPEG IDCT = JDCT_IFAST
A0 under the frozen protocol (convert.py preprocessing: PIL decode = libjpeg JDCT_ISLOW) FAILS: 2677641_b0.0 top-1
76.442 vs tstat 76.562 (-0.120 pp). The other two anchors pass (-0.086, -0.030 pp). All three sit below tstat.
Cause found: Hooker's pipeline decodes with tf.image.decode_jpeg(..., 3) (code/utils/data_input.py:57) and no
dct_method, and TF's default there is JDCT_IFAST. convert.py does not reproduce that. Re-running with a libjpeg-turbo
IFAST decode (simplejpeg fastdct=True, fancy upsampling; resize/crop/normalise unchanged from convert.py) gives:
2677641_b0.0 -0.014 pp, 2799528_t0.0 +0.020 pp, 2677646_t0.9 -0.006 pp, and cuts the per-class count mismatch against
tstat recall (sum |d| over 1000 classes) from 416/425/439 to 119/96/151. TF's 1024-entry bicubic coefficient table has
negligible effect (3 top-1 flips on 50k). Evidence is in results/diag/ and logs/diag_*.log. Threshold not relaxed. The
protocol has not been switched: results/preds/ holds convert.py-protocol predictions. Adopting D3 means changing
"as implemented in convert.py" to "convert.py + IFAST decode", which needs a separate protocol decision.

## 2026-09-28T18:44:08+05:30 — protocol decisions after A0/A1. No PIE count, population mode, dense-vs-dense or Welch computation exists yet (results/ holds only per-model predictions for the 3 anchors).

**D3 ADOPTED — JPEG decode = IFAST (TensorFlow's default), for all models.**
A0 as frozen FAILED: 2677641_b0.0 top-1 76.442 vs tstat 76.562 (-0.120 pp; limit 0.1). The other anchors passed
(-0.086, -0.030). Diagnosis: Hooker decodes with tf.image.decode_jpeg without dct_method, i.e. TF's default
INTEGER_FAST (IFAST); convert.py decoded with PIL's default ISLOW. With IFAST (resize/crop/normalise unchanged) all
three anchors pass: -0.014, +0.020, -0.006 pp, and the per-class count mismatch vs tstat drops ~4x (416/425/439 ->
119/96/151). PREREG specifies "Hooker's own evaluation preprocessing"; convert.py deviated from it in the decoder, so
this corrects an implementation defect relative to the stated protocol rather than relaxing the gate (threshold
unchanged). The decision is made blind: no hypothesis-relevant quantity has been computed.
Guard, pre-declared now: M1, M2 and H1 are ALSO reported under the original PIL-ISLOW decode as a sensitivity
analysis, and any disagreement between the two is reported in the main text. Official A0 = IFAST results in
results/diag/ (to be promoted to results/preds/); the ISLOW anchor predictions are retained.

**D4 ADOPTED — population size 13, not 15.** Only 27 dense checkpoints are released and complete (11 dense-baseline +
16 dense-threshold; INVENTORY.json, sha256 e22d9e71…). M1 = dense(13) vs dense(13), >= 1,000 random splits, each
split stratified by dense role (baseline/threshold) so both halves get ~equal shares of each; cohort (D1) reported as
a secondary stratification. M2 = pruned(13) vs dense(13) at 30/70/90% (13 pruned drawn at random per split from the
30 available) and 6 vs 6 at 50% against a 6-vs-6 dense floor. M3 = n from 3 to 13. H1/H2/H3 and their 3-split-SD
rule are unchanged. Fixed before any population-level number exists.

## 2026-09-28T19:09:52+05:30 — D5 implementation note: input cache is float32, not uint8
The plan called for uint8 pre-normalisation crops (~7.5 GB per decoder). The legacy-bicubic resize output
(convert.py and TF resize_bicubic) is fractional and unclipped, and it goes into the network as float32. A uint8
cache would quantise and clip it, so cached evals could not match the A0/A1 predictions bit for bit, which was the
requested check. data/cache/val_{ifast,islow}.npy therefore store the exact normalised float32 network inputs
(50000x3x224x224, 30.1 GB each) in image_order.csv order. batch_order_<dec>.npy records the A0/A1 streaming batch
composition, so cached evals use identical batches. Pipeline and numbers are unchanged.

## 2026-09-29T11:55:22+05:30 — D6 addendum frozen: PREREG-ADDENDUM-1.md (sha256 b57266fe…, logs/prereg_addendum1.sha256)
It gives the analysis specification (split algorithm, seeds, intervals, ablations), the A2 route (including the
NOT-COMPUTABLE branch and its substitute evidence), breadth (a) on ImageNet-V2/ReaL, and M5 details (INT8 = FX
static PTQ on CPU, calibrated on ImageNet-1k train). Frozen before any population-level quantity was computed.

## 2026-09-29T12:06:38+05:30 — D7 deviation from ADDENDUM-1 §11: INT8 is simulated on the GPU, not true fbgemm INT8 on the CPU
Measured true fbgemm INT8 throughput here: 15 img/s at the current shared-CPU load (fp32 CPU 6 img/s), i.e. ~25 CPU-h
for 27 models x 50k images. PyTorch has no native CUDA INT8 kernels. INT8 is therefore simulated on the GPU with the SAME FX
PTQ recipe: prepare_fx with get_default_qconfig_mapping("fbgemm") (per-channel symmetric int8 weights, per-tensor
affine uint8 activations, HistogramObserver), calibrated on the same 1,024 ImageNet-1k TRAIN images, then
convert_to_reference_fx (explicit quantize/dequantize, float compute). Pre-check on CPU (2677641, 200 val images,
100-image calibration): reference vs true fbgemm INT8 top-1 agreement 0.98. Before any H3 number exists, the final
run also reports reference vs true-INT8 agreement for 1 model on 2,000 images. Everything else in §11 is unchanged. Decided
before any M5 prediction exists.

## 2026-09-29T12:44:20+05:30 — D8 addendum frozen: PREREG-ADDENDUM-2.md (CIFAR-10 population, breadth b; sha256 in logs/prereg_addendum2.sha256)
Frozen before any population-member training run. Only a 3-run timing pilot (seeds 990-992, predictions unanalysed) preceded it.
Estimate: 120 runs x ~30 s = 1.0 GPU-h (<= 1.5 GPU-h with lock waits). Runs start after M5 and ImageNet-V2 finish, as planned.

## 2026-09-29T17:22:18+05:30 — D9 addendum frozen: PREREG-ADDENDUM-3.md (internal-review conditions C1a, C1b, C2; sha256 in logs/prereg_addendum3.sha256)
Frozen before: any extra CIFAR dense seed (30..59) was trained, any dynamic-range INT8 prediction existed, and the fixed M4
was computed on ISLOW, V2, ReaL or CIFAR. Pre-stated numeric CIFAR n=30 floor predictions: envelope [109.56, 114.81], tolerance [104.08, 120.55].
Disclosed: the val-IFAST exclusion counts for the pruned arms were already seen (exploratory.json).

## 2026-09-29T19:29:27+05:30 — D10 implementation note on ADDENDUM-3 C1(b) (no change of rule)
The rule excludes a class when per-model recall has zero variance within both populations. It is implemented as exact equality
of all per-model recalls (numpy ptp == 0). numpy var() of identical floats can return a tiny non-zero value, and the earlier
exploratory check used var()==0, which let some constant classes through (1.28 vs 0.05 BH classes at 30%). All reported fixed-M4 numbers
use the exact rule. Outcomes of ADDENDUM-3: C1(a) extrapolation NOT VALIDATED (measured F30 122.7 +- 6.7 vs tolerance [104.1, 120.6]);
C2 dynamic-range INT8 H3 HOLDS (paired 1026 +- 28, unpaired 2929 +- 34, z 1.17); the static FX INT8 H3 failure stays reported.

## 2026-09-29T20:03:56+05:30 — D11 disclosure correction (ADDENDUM-3 left untouched)
ADDENDUM-3 disclosed only the val-IFAST exploratory zero-variance counts as already seen. The same script had also been
run on the accurate (ISLOW) decoder before the freeze: results/analysis/islow/exploratory.json holds BH counts excluding
zero-variance classes for the pruned arms (1.20 / 41.18 / 291.76 at 30/70/90%). It had not been run on the floor, V2, ReaL or
CIFAR-10. It used the var()==0 screen, not the ptp==0 rule of C1(b). No rule or number changes.

## 2026-09-29T20:04:22+05:30 — D11a correction of D11 (append-only; D11 overstated the pre-freeze state)
The zero-variance block was added to src/exploratory.py at 17:16 on 09-29 (file mtime) and was run before the ADDENDUM-3
freeze (17:22) on the val-IFAST predictions only. The earlier ISLOW run of src/exploratory.py (about 13:00) predates that block,
so it produced only the n->30 extrapolation reported in an earlier internal summary. The ISLOW zero-variance counts now on disk
(results/analysis/islow/exploratory.json, written 18:22) were first computed during the analysis-stage re-run, after the freeze.
The original ADDENDUM-3 disclosure (val IFAST only) was therefore accurate. This rests on the file mtimes and the working record.
No file preserves the 13:00 ISLOW output.

## 2026-09-29T20:04:35+05:30 — D12 EXPLORATORY analyses for the first review round, stated before running (not pre-registered; no verdict changes)
All of these run on stored predictions; E4 also needs CPU integer inference. Code is src/explore_r1.py (E1-E6) and
src/fig_examples.py (E7). RNG: SeedSequence([20260928, 30]), children in the order listed in the code.
- E1 pruned-vs-pruned floor: for s in {0.3, 0.7, 0.9} and n = 3..13, K = 1000 uniform disjoint draws of two n-model
  populations from the 30 pruned models, giving the PIE count. At 0.5: 3 v 3 from the 6 models.
- E2 seed-variation Jaccard: the Jaccard of the PIE sets obtained against two disjoint dense references, both
  compared with the modal label of all 30 pruned models at s. Two variants: (a) the pre-registered 13 v 13 halves (1000 splits);
  (b) 13 v 14 splits of all 27 dense (1000 uniform splits). Reported next to the A2 label-only bound (A2 verdict unchanged).
- E3 ReaL rates of the compression-specific subset: per pre-registered split k, C_k = PIE(A_k, P_k) and not PIE(A_k, B_k).
  Image weight = fraction of splits with the image in C_k. Weighted empty / multi-label / original-label-not-in-ReaL rates.
- E4 simulated vs true integer INT8 (static recipe): true fbgemm INT8 top-1 for all 27 dense models on the first 2,000
  images of image_order.csv (calibration as in ADDENDUM-1 section 11). Paired and unpaired PIE counts on those 2,000 images under
  true vs simulated INT8, over the 1000 pre-registered splits. The dynamic-range recipe has no PyTorch integer conv kernel, so it
  gets no true-kernel check.
- E5 paired per-split excess at n = 13: the M2_s minus M1 difference per split (mean, SD, fraction > 0), from counts.npz.
- E6 CIFAR exchangeability: 13 v 13 dense floor within seeds 0-29, within seeds 30-59, and across (13 from each),
  K = 1000 each.
- E7 (Figure 1 re-selection, decided in review): the same ranking rule as before. An image is EXCLUDED if the torchvision
  COCO Faster R-CNN ResNet-50 FPN v2 detector finds a 'person' with score >= 0.3. The rule walks down each ranking until 6 images
  are kept per row.

## 2026-09-29T20:08:28+05:30 — D13 addendum frozen: PREREG-ADDENDUM-4.md (CIFAR-10 structured filter pruning; sha256 in logs/prereg_addendum4.sha256)
Frozen before any population-member structured run. Four smoke runs (seeds 99900-99903) preceded it; they are disclosed and discarded.

## 2026-09-29T20:09:05+05:30 — D14 operational note (ADDENDUM-4 runs)
The first launch of the structured chain had a seed-arithmetic error (it started seeds 16000+ instead of 13000+). It was
stopped by PID after about 25 s, before any prediction file was written, and relaunched with the pre-stated seeds 13000/17000/19000 + i.

## 2026-09-30T00:45:45+05:30 — D15 EXPLORATORY analyses for the second review round, stated before running (not pre-registered; no verdict changes)
All run on stored CIFAR-10 predictions (CPU). Code: src/explore_r1.py analyze, keys E8-E10. RNG: SeedSequence([20260928, 50]),
children in the order listed in the code. The E1-E7 streams are untouched.
- E8 structured-vs-structured floor: for s in {0.3, 0.7, 0.9}, K = 1000 uniform disjoint 13 v 13 draws from the 30
  structured runs, giving the PIE count. Compared with the 60-seed dense 13 v 13 floor of ADDENDUM-4.
- E9 matched-accuracy comparison on one floor: the unstructured CIFAR-10 arms (ADDENDUM-2 runs) re-counted with
  ADDENDUM-4's procedure against the 60-seed dense floor (13 of 60 dense vs 13 of 30 pruned, K = 1000, image bootstrap of the excess
  with 2000 resamples). Read side by side with the structured arm at equal accuracy cost (structured 30% and unstructured 70%).
- E10 Monte Carlo error of z for the CIFAR-10 rows: delta-method approximation
  SE(z) ~ sqrt((se_s^2 + se_0^2) / sd_0^2 + z^2 / (2 (K - 1))), from the stored per-condition summaries.
