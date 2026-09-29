# PRE-REGISTRATION ADDENDUM 2 — breadth (b): a CIFAR-10 population trained here

**Frozen:** 2026-09-29, before any population-member training run. SHA-256 is in `logs/prereg_addendum2.sha256`. Read-only after freeze.

This addendum adds a second dataset and a second architecture to the study. The headline decisions (H1/H2 on ImageNet, PREREG.md) are unchanged. Everything here is a secondary replication.

**Already seen when this was written:**
- the ImageNet results under PREREG-ADDENDUM-1;
- a timing pilot of three runs (seeds 990, 991, 992), whose predictions were discarded unanalysed:
  - 27–29 s per run on the RTX 5060 Laptop;
  - dense top-1 0.933 and 0.9326 (single view);
  - 90% pruned top-1 0.9067, realised 3×3-conv sparsity 0.9000.

## 1. Training (code: `cifar/run.py`)
- **Base recipe:** airbench94, Jordan 2024 (`cifar/airbench94.py` = KellerJordan/cifar10-airbench `legacy/airbench94.py`, MIT; sha256 in `cifar/airbench94.sha256`). Hyperparameters unchanged: 9.9 epochs, batch 1024, 476 steps, Lookahead-EMA and whitening layer as in the original.
- **Data:** CIFAR-10 from Hugging Face `uoft-cs/cifar10` parquet, decoded losslessly (sha256 in `cifar/cifar10_hf.sha256`), 50,000 train and 10,000 test.
- **Two changes to the recipe:**
  1. `torch.manual_seed(seed)` per run.
  2. For pruned runs, gradual per-layer magnitude pruning of the six 3×3 convolutions. The whitening convolution and the linear head are not pruned. The schedule is Zhu & Gupta's cubic ramp at the same fractions of training as Hooker's 1000/9000/500 of 32,000 steps: start at step 15, end at step 134, update every 7 steps. Masks are recomputed per layer from |w| at each update, applied after every optimiser step and every Lookahead update, and fixed after the ramp.
- **Seeds:** seed = 1000·k + i, with i = 0..29 and k = 0 (dense), 3 (30%), 7 (70%), 9 (90%). That gives 120 runs.
- **Run handling:**
  - Training uses cuDNN autotune as in airbench, so it is not bit-reproducible; each seed is a population sample.
  - No run is filtered. Any run with test top-1 < 0.85, or a realised sparsity more than 0.5 pp from its target, is reported, kept, and flagged.
- **Evaluation:** CIFAR-10 test set (10,000), single view (tta_level 0), deterministic cuDNN inference. Top-1, top-5 and margin are saved per run in `results/cifar/preds/`.
- **Compute estimate:** 120 runs × ~30 s ≈ 1.0 GPU-h, ≤ 1.5 GPU-h including lock waits. Everything runs under the P3 GPU gate with a per-run lock.

## 2. Analysis (code: `cifar/analyze.py`, same definitions as PREREG-ADDENDUM-1)
- **Modal label and PIE:** Hooker's `mode().max()`, i.e. the modal label with ties to the largest label.
- **Randomness:** `numpy.random.SeedSequence([20260928, 10])`, one child per named condition in a fixed list in the code. K = 1,000 draws per condition.
- **Splits:** there are no dense subtypes, so dense halves are uniform draws without replacement from the 30 dense runs, disjoint.
- **Measurements:**
  - **M1c:** floor, 13 vs 13; also 15 vs 15.
  - **M2c:** dense half A vs 13 of 30 pruned at s ∈ {0.3, 0.7, 0.9}.
  - **M3c:** n = 3..15.
  - **M4c:** per-class Welch test with Hooker's normalisation. Here the constants are each sparsity's 30-run mean top-1, because Hooker's CIFAR constants are not released. Recall = correct/1000 per class. Counts are uncorrected p < 0.05 and BH q ≤ 0.05 over the 10 classes.
- **Intervals:** split distribution (mean, SD, 2.5/97.5 percentiles), Monte-Carlo SE, and the image bootstrap (2,000 resamples) of the count and the floor-subtracted count.
- **Replication decisions (secondary):** H1c holds if |z_0.3| ≤ 3; H2c at 0.7 and 0.9 if z > 3, with z = (mean M2c − mean M1c) / SD(M1c).
- **Comparison (descriptive):** Hooker's printed CIFAR-10 PIE counts (v1–v3: 114/144/137/216 at 30/50/70/90% with 30 WRN models, independent second read). The architectures differ, so this is context, not a test.
