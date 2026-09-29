#!/usr/bin/env bash
# One-command reproduction of "Seed-noise floors for pruning-identified exemplars in compressed neural networks".
# Every stage is idempotent: it skips outputs that already exist and resumes downloads.
# GPU sharing is OPT-IN: set PIE_GPU_LOCK_DIR=<dir> to wrap each GPU batch in `flock <dir>/gpu.gate` and take
# <dir>/gpu.lock per model (the scripts read PIE_GPU_LOCK_DIR). Unset = no locking.
# Wall time on an RTX 5060 Laptop (8 GB): ~1 h downloads, ~9 GPU-h FP32 inference (2 decoders x 123 models),
# ~1 h FP16/INT8 + ImageNet-V2, ~1 h CPU analysis. Disk: ~35 GB checkpoints + 30 GB (+30 GB ISLOW) float32 caches.
set -euo pipefail
cd "$(dirname "$0")/.."
# environment: python3.13 -m venv .venv && .venv/bin/pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu130
PY=${PY:-$PWD/.venv/bin/python}   # pinned in requirements.lock (top level: requirements.in), hardware/driver in env.txt
if [ -n "${PIE_GPU_LOCK_DIR:-}" ]; then GATE=(flock "$PIE_GPU_LOCK_DIR/gpu.gate"); else GATE=(); fi

# 0. frozen artefacts are unchanged (PREREG, inventory, addendum)
head -1 logs/inventory.sha256 | sha256sum -c -
for f in prereg prereg_addendum1 prereg_addendum2 prereg_addendum3 prereg_addendum4; do head -1 "logs/$f.sha256" | (cd preregistration && sha256sum -c -); done

# 1. data: ImageNet-1k validation parquet (Hugging Face ILSVRC/imagenet-1k is gated: accept its terms, set HF_TOKEN),
#    one ImageNet-1k train shard (INT8 calibration only), ImageNet-V2 matched-frequency, ImageNet-ReaL labels
$PY -c "from huggingface_hub import snapshot_download as s; s('ILSVRC/imagenet-1k', repo_type='dataset', allow_patterns='data/validation-*', local_dir='data/imagenet-1k')"
$PY -c "from huggingface_hub import hf_hub_download as h; h('ILSVRC/imagenet-1k', 'data/train-00000-of-00294.parquet', repo_type='dataset', local_dir='data/imagenet-1k-train')"
[ -d data/imagenetv2/imagenetv2-matched-frequency-format-val ] || {
  $PY -c "from huggingface_hub import hf_hub_download as h; h('vaishaal/ImageNetV2', 'imagenetv2-matched-frequency.tar.gz', repo_type='dataset', local_dir='data/imagenetv2')"
  tar xf data/imagenetv2/imagenetv2-matched-frequency.tar.gz -C data/imagenetv2; }
[ -f data/real/real.json ] || { mkdir -p data/real; curl -sSfL -o data/real/real.json https://raw.githubusercontent.com/google-research/reassessed-imagenet/master/real.json; }
head -1 logs/imagenetv2.sha256 | (cd data/imagenetv2 && sha256sum -c -)

# 2. Hooker et al.'s bucket files (keyless GCS, md5-verified, resumable; non-commercial terms, see NOTICE.md):
#    the released class statistics (tstat.csv), the released per-image PIE files and the bucket README (sha256 in BUCKET_META_SHA256),
#    then all 123 checkpoints. INVENTORY.json and the listing prior-gate/g13/all.json are frozen and never rebuilt.
$PY src/bucket.py fetch_meta
$PY src/bucket.py fetch all

# 3. exact network inputs (float32 caches), then per-image predictions for all 123 models under both decoders
[ -f data/cache/val_ifast.npy ] || $PY src/evalpreds.py cache
"${GATE[@]}" $PY src/evalpreds.py eval all
[ -f data/cache/val_islow.npy ] || $PY src/evalpreds.py cache --islow   # sensitivity decoder (D3); 30 GB, may be deleted after
"${GATE[@]}" $PY src/evalpreds.py eval all --islow
$PY src/fidelity.py                                                      # results/fidelity.csv (A0 for every model)

# 4. breadth (a): ImageNet-V2; M5 (FP16 on GPU, INT8 simulated PTQ per D7)
[ -f data/cache/v2_ifast.npy ] || $PY src/evalv2.py cache
"${GATE[@]}" $PY src/evalv2.py eval
[ -f data/cache/calib_ifast.npy ] || $PY src/m5.py calib
"${GATE[@]}" $PY src/m5.py eval fp16
"${GATE[@]}" $PY src/m5.py eval int8
"${GATE[@]}" $PY src/m5.py eval dynint8                               # ADDENDUM-3 C2
$PY src/m5.py int8check
"${GATE[@]}" $PY src/explore_r1.py int8true                              # exploratory E4 (D12): true integer INT8 on 2,000 images

# 4b. breadth (b), PREREG-ADDENDUM-2: CIFAR-10 populations (airbench94; 120 runs x ~22 s). Data: HF uoft-cs/cifar10 parquet.
[ -f cifar/cifar10/test.pt ] || $PY -c "
from huggingface_hub import hf_hub_download; import pyarrow.parquet as pq, numpy as np, io, torch; from PIL import Image
for sp in ('train','test'):
    t=pq.read_table(hf_hub_download('uoft-cs/cifar10',f'plain_text/{sp}-00000-of-00001.parquet',repo_type='dataset',local_dir='cifar/hf'))
    x=np.stack([np.asarray(Image.open(io.BytesIO(b)).convert('RGB')) for b in t.column('img').combine_chunks().field('bytes').to_pylist()])
    torch.save({'images':torch.tensor(x),'labels':torch.tensor(t.column('label').to_pylist()),'classes':['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']},f'cifar/cifar10/{sp}.pt')"
(cd cifar && sha256sum -c cifar10_hf.sha256 && sha256sum -c airbench94.sha256)
for k in 0 3 7 9; do "${GATE[@]}" $PY cifar/run.py "0.$k" $(seq $((1000*k)) $((1000*k+29))); done
"${GATE[@]}" $PY cifar/run.py 0.0 $(seq 30 59)                          # ADDENDUM-3 C1(a): 30 extra dense seeds
for k in 3 7 9; do "${GATE[@]}" $PY cifar/run.py "0.$k" $(seq $((10000+1000*k)) $((10029+1000*k))) --structured; done   # ADDENDUM-4

# 5. analysis stage: identical to the CPU-only script (pre-registered analyses, A2, ReaL, CIFAR-10 incl. ADDENDUM-4,
#    exploratory E1-E10, M5 decisions, figures)
PY="$PY" scripts/reproduce_analysis.sh all
$PY src/fig_examples.py                                                  # Figure 1 (needs the ImageNet images; person filter, D12 E7)

# 6. checks: image order, labels, preprocessing, A0 anchors, A1 bit-identity, Hooker tie-break; expected analysis outputs
$PY tests/test_pipeline.py
$PY tests/check_outputs.py        # compares analysis outputs with EXPECTED_SHA256 (npz compared per array payload)
