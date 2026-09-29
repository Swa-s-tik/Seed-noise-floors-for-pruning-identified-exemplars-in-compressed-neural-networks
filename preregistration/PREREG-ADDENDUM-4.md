# PRE-REGISTRATION ADDENDUM 4 — breadth: structured (filter) pruning on CIFAR-10 (G3 round 1, DOMAIN-10)

**Frozen:** 2026-09-29, before any population-member structured-pruning run. The SHA-256 is recorded in `logs/prereg_addendum4.sha256`. The file is read-only after freezing. It adds a secondary replication arm. It changes no ImageNet hypothesis or result.

**Already seen, disclosed:**
- All results through G3 round 1, including the CIFAR-10 dense 60-seed pool and the unstructured arms.
- Four smoke runs of the structured code, discarded and not analysed:
  - seed 99900 (90%) diverged to NaN. Cause: masking the BatchNorm bias left a zero-variance channel whose fp16 backward overflowed. The code was fixed to mask the BatchNorm output instead.
  - seed 99901 was a debug run of that failure.
  - seed 99902 (90%): top-1 0.740, 3×3 weight zero-fraction 0.899.
  - seed 99903 (30%): top-1 0.926, zero-fraction 0.301.

## 1. Training (code: `cifar/run.py --structured`)
- The recipe, data, schedule, seeds handling and evaluation are exactly as in ADDENDUM-2, with one change: pruning is structured.
- **Pruning rule.** At each scheduled update, each of the six 3×3 convolutions zeroes the round(s × C_out) output filters with the smallest L1 norm. The same channel mask multiplies the output of the following BatchNorm, so a pruned channel is exactly zero in forward and backward. This is equivalent to removing the channel.
- **Excluded layers.** The whitening convolution and the linear head are not pruned.
- **Schedule.** The same cubic ramp over the same fractions of training as ADDENDUM-2.
- **Arms.** s ∈ {0.3, 0.7, 0.9} (fraction of filters per layer), 30 seeds each. Seeds are 13000 + i, 17000 + i and 19000 + i for i = 0..29, giving 90 runs.
- **Runs.** Training uses cuDNN autotune as in ADDENDUM-2. Evaluation uses a single view with deterministic cuDNN.
- **Environment.** Runs use the package's own `.venv`, which has the same pins as the earlier GPU stages.
- **Nearest achievable sparsity.** The realised filter fraction per layer is round(s × C)/C with C ∈ {64, 256}, that is 19/64, 45/64, 58/64 and 77/256, 179/256, 230/256. It is reported as achieved.
- **Accuracy flags.** No run is excluded. Runs with top-1 < 0.85 are counted and reported; the pilot suggests every 90% run will be flagged.
- **Collapse.** An arm whose mean top-1 is < 0.50 is declared collapsed, and its decision is reported as descriptive only.
- **Compute.** About 90 × 23 s ≈ 0.6 GPU-h. There is one P3-gated batch per sparsity, and the gate is released between batches.

## 2. Analysis (code: `cifar/structured.py`; definitions as ADDENDUM-1/2)
- **RNG.** `SeedSequence([20260928, 40])`, with children in the order listed in the code. K = 1,000 splits.
- **Floor.** 13 v 13 uniform disjoint draws from the 60 dense seeds (0–59).
- **M2st.** Dense half A_k (13) against 13 of the 30 structured runs at s, drawn uniformly per k.
- **Effect size.** z_s = (mean M2st_s − mean floor) / SD(floor). This is a descriptive effect size in split units, as in PREREG.
- **Decisions (secondary; the ImageNet H1/H2 stay primary).** H1st holds if |z_0.3| ≤ 3. H2st holds at 0.7 and 0.9 if z > 3.
- **Also reported, descriptive:**
  - image-bootstrap CIs of the floor-subtracted counts;
  - 30 v 30 (dense half of 30 from 60 against all 30 structured), z against the 30 v 30 floor from the same pool;
  - full-population counts beside the unstructured arm.
