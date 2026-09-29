# A0 diagnosis only (D3 evidence; the adopted IFAST pipeline is src/evalpreds.py): re-run an anchor with TF1 decode/resize details that convert.py does not emulate.
# usage: python src/diag_preproc.py <run_id> <ifast|ifast_tftable|tftable>
#   ifast   : JPEG IDCT = JDCT_IFAST (TF decode_jpeg default dct_method ""), fancy upsampling; else convert.py
#   tftable : TF ResizeBicubic legacy kernel weights from its 1024-entry coefficient table (lrintf(delta*1024))
import io, json, os, sys
import numpy as np, simplejpeg, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalpreds as E  # noqa: E402
C = E.convert


def tf_table_matrix(n_in, n_out, A=-0.75, K=1024):
    x = np.arange(K + 1, dtype=np.float32) / np.float32(K)
    t0 = ((A + 2) * x - (A + 3)) * x * x + 1          # coeffs_table[i*2]
    x1 = x + np.float32(1)
    t1 = ((A * x1 - 5 * A) * x1 + 8 * A) * x1 - 4 * A  # coeffs_table[i*2+1]
    scale = np.float32(n_in) / np.float32(n_out)
    W = np.zeros((n_out, n_in), np.float64)
    for o in range(n_out):
        f = np.float32(o) * scale; i0 = int(np.floor(f)); off = int(np.rint(np.float32(f - np.float32(i0)) * np.float32(K)))
        for k, w in zip(range(i0 - 1, i0 + 3), (t1[off], t0[off], t0[K - off], t1[K - off])):
            W[o, min(max(k, 0), n_in - 1)] += w
    return torch.from_numpy(W).float()


def make_preprocess(ifast, table):
    def pre(f, size=224):
        b = f.getvalue()
        try:
            a = simplejpeg.decode_jpeg(b, colorspace="RGB", fastdct=True, fastupsample=False) if ifast else None
        except ValueError:
            a = None  # non-JPEG/CMYK: fall back to PIL
        if a is None:
            a = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
        x = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)
        h, w = x.shape[1:]
        nh, nw = (int(h / w * (size + 32)), size + 32) if h > w else (size + 32, int(w / h * (size + 32)))
        M = tf_table_matrix if table else C._legacy_bicubic_matrix
        x = torch.einsum("oh,chw,pw->cop", M(h, nh), x, M(w, nw))
        oh, ow = (nh - size + 1) // 2, (nw - size + 1) // 2
        return (x[:, oh:oh + size, ow:ow + size] - C.MEAN) / C.STD
    return pre


if __name__ == "__main__":
    rid, var = sys.argv[1], sys.argv[2]
    assert torch.allclose(tf_table_matrix(500, 256).sum(1), torch.ones(256), atol=1e-4)
    pre = make_preprocess("ifast" in var, "tftable" in var)
    labels = E.labels()
    m, zf = E.load(rid)
    top5, margin = E.predict(m, E.stream(lambda b: pre(io.BytesIO(b))))
    out = os.path.join(E.ROOT, "results", "diag"); os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, f"{rid}_{var}.npz"), top1=top5[:, 0].copy(), top5=top5, margin=margin)
    inv = json.load(open(os.path.join(E.ROOT, "INVENTORY.json")))["runs"][rid]
    base = np.load(os.path.join(E.outdir("islow"), rid + ".npz"))["top1"]
    res = {"run_id": rid, "variant": var, "top1": float((top5[:, 0] == labels).mean()),
           "top5": float((top5 == labels[:, None]).any(1).mean()), "tstat_top1": inv["tstat_top1"], "tstat_top5": inv["tstat_top5"],
           "top1_flips_vs_convert_py": int((top5[:, 0] != base).sum())}
    res["delta_top1_pp"] = 100 * (res["top1"] - res["tstat_top1"])
    json.dump(res, open(os.path.join(out, f"{rid}_{var}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
