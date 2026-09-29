# PRE-REGISTRATION — The PIE Floor (B13)

**Frozen:** 2026-09-28, before any per-model prediction from Hooker et al.'s checkpoints has been computed. SHA-256 recorded in `logs/prereg.sha256`. Read-only after freeze. Deviations go in `DEVIATIONS.md`, append-only and timestamped.

## What has already been seen (disclosed, not hidden)
The released class-level statistics (`tstat.csv`), the released per-image PIE files at 70% and 90% sparsity (`pie07.csv`, `pie09.csv`), the printed tables of arXiv:1911.05248 v1–v3, and class-level Welch/BH counts derived from them (gate of 2026-09-28). **Not yet seen by anyone:** any per-model prediction, any dense-vs-dense PIE count, any PIE count at population size < 30.

## Question
How many Pruning-Identified Exemplars (Hooker et al., arXiv:1911.05248) would two **independently seeded, equally sized populations of dense models** produce against each other — and how do the published PIE counts compare with that floor?

## Materials (fixed)
- **Models:** ResNet-50 checkpoints from `gs://gresearch/pruning_identified_exemplars`, converted to PyTorch with `prior-gate/g13/convert.py` (no TensorFlow). Before any population-level computation the builder writes `INVENTORY.json` (run id → role {dense-threshold, dense-baseline, pruned}, sparsity, cohort, source of that mapping, checksum) and records its SHA-256 in `logs/inventory.sha256`. The inventory is frozen from then on.
- **Data:** ImageNet-1k validation, all 50,000 images, from Hugging Face `ILSVRC/imagenet-1k` parquet. Where Hooker's released per-image files cover only 49,187 images, comparisons against them use the intersection; all new measurements use all 50,000.
- **Preprocessing:** Hooker's own evaluation preprocessing as implemented in `convert.py` (from `prior-gate/g13/code/`). FP32 inference.
- **PIE definition:** exactly Hooker's code — an image is a PIE when the modal top-1 label of population A differs from the modal top-1 label of population B; ties broken as Hooker's aggregation code does (to be quoted verbatim in the analysis code).

## Gates (in order; a FAIL stops that path)
| gate | test | PASS | FAIL → |
|---|---|---|---|
| **A0 anchor** | 3 converted checkpoints, including ≥1 at 90% sparsity, FP32 top-1 on 50k | within **0.1 pp** of the model's `tstat.csv` accuracy, all 3 | stop compute path; fall back to a comment from printed/released facts |
| **A1 determinism** | one model evaluated twice | bit-identical predictions | report and investigate before continuing |
| **A2 fidelity** | recompute is_pie at 70% and 90% against the released per-image files, trying each candidate reference population in `INVENTORY.json` | best Jaccard ≥ **0.8** on the 49,187-image intersection | continue, but the paper may not claim to reproduce Hooker's PIE sets; report the best Jaccard and which reference |

## Measurements (all pre-declared)
- **M1 floor:** dense(15) vs dense(15), ≥ **1,000** random splits stratified by cohort, fixed RNG seed 20260928. Report mean, SD, 95% interval of the PIE count.
- **M2 pruned:** pruned(15) vs dense(15) at 30%, 70%, 90% over the same splits; 50% as 6 vs 6 (only 6 exist), with the dense floor recomputed at 6 vs 6.
- **M3 population size:** PIE count vs population size n = 3…15 for the floor and for each sparsity.
- **M4 classes:** per-class Welch test as in Hooker's code, under the same splits, uncorrected and Benjamini–Hochberg at α = 0.05.
- **M5 quantization (later weeks):** fp16 and int8 PTQ of the dense models — **paired** (a quantized population against the same models in FP32) and **unpaired** (against an independent dense population).

## Pre-registered hypotheses and decision rule
- **H1 (headline):** at 30% sparsity, the pruned-vs-dense PIE count lies within **3 split-SD** of the dense-vs-dense floor at the same population size.
  - Holds → headline: *low-sparsity PIEs are indistinguishable from seed noise.*
  - Fails (exceeds by > 3 SD) → headline flips to *low-sparsity PIEs are real*. Still a reproducibility paper; title changes. Decided now, not after seeing the number.
- **H2:** at 70% and 90% the count exceeds the floor by > 3 SD. Report floor-subtracted counts with intervals; no fixed "attributable fraction" is claimed beyond that.
- **H3 (M5):** paired PTQ PIE counts are far below the floor; unpaired counts are near the floor. If paired ≈ unpaired, the inference that Hooker's quantization rows reflect seed noise is withdrawn.

## Not claimed, whatever the data say
Anything about CelebA/CIE fairness (not measured here); any claim that Hooker's qualitative conclusions at high sparsity are wrong; the "2,193 copy error" (dropped as unprovable).
