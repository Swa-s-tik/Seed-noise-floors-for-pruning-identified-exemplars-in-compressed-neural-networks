"""Quick start: one dense-vs-dense floor and one pruned-vs-dense PIE count from the stored per-image predictions.

An image is a pruning-identified exemplar (PIE) when the modal top-1 label of one population of models differs from
that of the other (Hooker et al.'s definition, ties to the largest label). This draws one split of the 27 dense
ImageNet models into two populations of 13, as the pre-registered analysis does, and one population of 13 models
pruned to 90% sparsity, then counts PIEs on the 50,000 validation images. Needs only results/preds/ (no ImageNet, no GPU).
Usage: .venv/bin/python scripts/quickstart.py [seed]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import analysis as A  # noqa: E402

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
pop = A.Pop("ifast")  # stored IFAST-decoder predictions of all 123 released checkpoints
dense_a, dense_b = A.strat_halves(rng, 13, pop.base, pop.thr)  # 13 v 13, stratified by kind of dense model
pruned = [pop.pr[0.9][i] for i in rng.permutation(30)[:13]]
ref = A.modal(pop.stack(dense_a))
floor = int((ref != A.modal(pop.stack(dense_b))).sum())
pie = int((ref != A.modal(pop.stack(pruned))).sum())
print(f"dense 13 v dense 13 (floor): {floor} PIEs of {pop.N} images")
print(f"pruned 90% 13 v dense 13:    {pie} PIEs of {pop.N} images")
print("over 1,000 splits the paper reports means of 2,891 (floor) and 5,664 (90% sparsity); "
      "results/analysis/ifast/summary.json")
