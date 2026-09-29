# Breadth (a), PREREG-ADDENDUM-1 §10: ImageNet-V2 matched-frequency, same IFAST pipeline and settings as evalpreds.
#   python src/evalv2.py cache   -> results/preds_v2/image_order.csv + data/cache/v2_ifast.npy (exact float32 inputs)
#   python src/evalv2.py eval    -> results/preds_v2/<run_id>.npz (GPU; optional sharing via PIE_GPU_LOCK_DIR, see scripts/reproduce.sh)
import glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalpreds as E  # noqa: E402

ROOT = E.ROOT
V2 = os.path.join(ROOT, "data", "imagenetv2", "imagenetv2-matched-frequency-format-val")
OUT = os.path.join(ROOT, "results", "preds_v2")
ORDER = os.path.join(OUT, "image_order.csv")
CACHE = os.path.join(E.CACHE, "v2_ifast.npy")


def files():
    """Order: class folder 0..999 (standard ImageNet index), then file name."""
    return sorted(((int(p.split("/")[-2]), p) for p in glob.glob(os.path.join(V2, "*", "*"))), key=lambda t: (t[0], os.path.basename(t[1])))


def pre(p):
    with open(p, "rb") as f:
        return E.preprocess(f.read(), "ifast").numpy()


def cache():
    fs = files(); assert len(fs) == 10000
    os.makedirs(OUT, exist_ok=True)
    open(ORDER, "w").write("idx,path,label\n" + "".join(f"{i},{os.path.relpath(p, V2)},{y}\n" for i, (y, p) in enumerate(fs)))
    X = np.lib.format.open_memmap(CACHE + ".part", "w+", np.float32, (len(fs), 3, 224, 224))
    torch.set_num_threads(1)
    with Pool(8) as pool:
        for i, x in enumerate(pool.imap(pre, [p for _, p in fs], chunksize=16)):
            X[i] = x
    X.flush(); del X; os.replace(CACHE + ".part", CACHE)


def batches():
    X = np.load(CACHE, mmap_mode="r")
    for s in range(0, len(X), E.BS):
        yield np.arange(s, s + E.BS), None, torch.from_numpy(np.ascontiguousarray(X[s:s + E.BS])).pin_memory()


def evaluate(rid, y):
    t0 = time.time()
    m, _ = E.load(rid)
    top5, margin = E.predict(m, batches(), n=len(y))
    np.savez(os.path.join(OUT, rid + ".npz"), top1=top5[:, 0].copy(), top5=top5, margin=margin)
    res = {"run_id": rid, "top1": float((top5[:, 0] == y).mean()), "top5": float((top5 == y[:, None]).any(1).mean()), "seconds": time.time() - t0}
    open(os.path.join(OUT, rid + ".json"), "w").write(json.dumps(res)); print(json.dumps(res), flush=True)
    del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        cache()
    else:
        y = np.loadtxt(ORDER, delimiter=",", skiprows=1, usecols=2, dtype=np.int64)
        inv = json.load(open(os.path.join(ROOT, "INVENTORY.json")))["runs"]
        todo = sorted(k for k, r in inv.items() if r["complete"] and not os.path.exists(os.path.join(OUT, k + ".npz")))
        E.gpu_loop(todo, lambda r: evaluate(r, y))
        print("V2 EVAL DONE", flush=True)
