# Runnable check: python tests/test_pipeline.py  (also collected by pytest)
# Fails if the image order, the label mapping, the preprocessing, or an anchor's A0 number breaks.
import csv, json, os, sys
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import evalpreds  # noqa: E402
convert = evalpreds.convert
ANCHORS = ("2677641_b0.0", "2799528_t0.0", "2677646_t0.9")  # A0 anchors (official decoder: IFAST, D3)

PREDS = os.path.join(ROOT, "results", "preds")


def test_order_and_labels():
    rows = list(csv.DictReader(open(os.path.join(PREDS, "image_order.csv"))))
    assert len(rows) == 50000 and len({r["path"] for r in rows}) == 50000
    syn = [r["path"].rsplit("_", 1)[1].split(".")[0] for r in rows]  # ILSVRC2012_val_XXXXXXXX_<synset>.JPEG
    rank = {s: i for i, s in enumerate(sorted(set(syn)))}
    assert len(rank) == 1000
    # HF label == index of the synset in sorted-wnid order == Hooker's (TFRecord label - 1) convention
    assert all(int(r["label"]) == rank[s] for r, s in zip(rows, syn))


def test_preprocess():
    assert torch.allclose(convert._legacy_bicubic_matrix(500, 256).sum(1), torch.ones(256), atol=1e-5)
    from PIL import Image
    import io
    b = io.BytesIO(); Image.new("RGB", (500, 375), (124, 116, 104)).save(b, "PNG"); b.seek(0)
    x = convert.preprocess(b)  # constant image at ~ImageNet mean -> ~0 after normalisation, 224x224
    assert x.shape == (3, 224, 224) and x.abs().max() < 0.01
    j = io.BytesIO(); Image.new("RGB", (500, 375), (124, 116, 104)).save(j, "JPEG", quality=95)
    for dec in ("ifast", "islow"):
        x = evalpreds.preprocess(j.getvalue(), dec)
        assert x.shape == (3, 224, 224) and x.abs().max() < 0.05, dec


def test_anchor_A0():
    inv = json.load(open(os.path.join(ROOT, "INVENTORY.json")))["runs"]
    labels = np.loadtxt(os.path.join(PREDS, "image_order.csv"), delimiter=",", skiprows=1, usecols=4, dtype=np.int64)
    for rid in ANCHORS:
        top1 = np.load(os.path.join(PREDS, rid + ".npz"))["top1"]
        assert abs(100 * ((top1 == labels).mean() - inv[rid]["tstat_top1"])) <= 0.1, rid
    a, b = (np.load(os.path.join(PREDS, f"2677641_b0.0{t}.npz")) for t in ("", "_rep2"))  # A1
    assert all(a[k].tobytes() == b[k].tobytes() for k in ("top1", "top5", "margin"))


def test_modal_is_hooker_mode_max():
    import pandas as pd
    import analysis  # src/ is on sys.path
    P = np.random.default_rng(0).integers(0, 4, (13, 2000)).astype(np.int16)
    assert (analysis.modal(P) == [pd.Series(P[:, i]).mode().max() for i in range(P.shape[1])]).all()


if __name__ == "__main__":
    for t in (test_order_and_labels, test_preprocess, test_anchor_A0, test_modal_is_hooker_mode_max):
        t(); print("PASS", t.__name__)
