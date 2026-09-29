#!/usr/bin/env bash
# CPU-only analysis stage: every table/figure number from the cached per-image predictions (results/preds*, results/cifar/preds).
# usage: scripts/reproduce_analysis.sh [all|core|post]   (core = everything except the M5 analysis and figures; post = those two)
# Then `python tests/check_outputs.py` compares the outputs with EXPECTED_SHA256 (npz compared per array payload).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$PWD/.venv/bin/python}
STAGE=${1:-all}
if [ "$STAGE" != post ]; then
  $PY src/fidelity.py                 # A0 for every model
  $PY src/analysis.py ifast           # M1-M4, H1/H2, intervals, ablations (PREREG + ADDENDUM-1)
  $PY src/analysis.py islow           # D3 sensitivity decoder
  $PY src/analysis.py ifast v2        # ImageNet-V2 replication
  $PY src/a2.py                       # gate A2 (mapping search + label-only Jaccard bound)
  $PY src/real.py ifast               # ImageNet-ReaL
  $PY src/exploratory.py ifast        # exploratory: pairwise disagreement, n->30 extrapolation, M4 zero-variance check
  $PY src/exploratory.py islow
  $PY cifar/analyze.py                # CIFAR-10 (ADDENDUM-2)
  $PY cifar/c1a.py                    # CIFAR-10 measured 30 v 30 floor, extrapolation test (ADDENDUM-3 C1a)
  $PY src/m4fix.py                    # class test with zero-variance classes excluded (ADDENDUM-3 C1b)
  $PY cifar/structured.py             # CIFAR-10 structured filter pruning (ADDENDUM-4)
  $PY src/explore_r1.py analyze       # exploratory E1-E6, E8-E10 (+E4 when the integer-INT8 predictions exist), DEVIATIONS D12, D15
fi
if [ "$STAGE" != core ]; then
  $PY src/m5.py analyze               # M5 / H3: fp16, static int8, dynamic-range int8 (ADDENDUM-3 C2)
  $PY src/figures.py
  $PY scripts/make_tables.py         # supplementary tables -> results/tables/supp_tables.tex
fi
