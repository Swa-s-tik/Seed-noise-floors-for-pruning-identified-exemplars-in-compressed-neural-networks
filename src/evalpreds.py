# FP32 ImageNet-val predictions for converted Hooker checkpoints.
#   python src/evalpreds.py cache [--islow]            decode+preprocess 50k images once -> data/cache/val_<dec>.npy
#   python src/evalpreds.py eval <run_id|all> [--islow] [--tag T]
#       -> results/preds[_islow]/<run_id>[_T].npz ; 'all' = every complete run in INVENTORY, waiting for downloads
# Decoder: IFAST (TF decode_jpeg default; official since D3) unless --islow (PIL, convert.py as frozen; sensitivity).
import glob, io, json, os, sys, time
import numpy as np, pyarrow.parquet as pq, simplejpeg, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "prior-gate", "g13"))
import convert  # noqa: E402  (Hooker preprocessing + TF1->torch conversion, reused)

FILES = sorted(glob.glob(os.path.join(ROOT, "data", "imagenet-1k", "data", "validation-*-of-00014.parquet")))
ORDER = os.path.join(ROOT, "results", "preds", "image_order.csv")
CACHE = os.path.join(ROOT, "data", "cache")
N, BS, N_LOADERS = 50000, 100, 12

# FP32 protocol: no TF32 anywhere; deterministic cuDNN (A1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def outdir(decoder):
    return os.path.join(ROOT, "results", "preds" if decoder == "ifast" else "preds_islow")


def preprocess(b, decoder):
    """JPEG bytes -> normalised float32 [3,224,224]."""
    if decoder == "islow":
        return convert.preprocess(io.BytesIO(b))
    # IFAST IDCT + fancy upsampling = tf.image.decode_jpeg(x, 3) defaults; the rest mirrors convert.preprocess
    x = torch.from_numpy(simplejpeg.decode_jpeg(b, colorspace="RGB", fastdct=True, fastupsample=False).astype(np.float32)).permute(2, 0, 1)
    h, w, size = x.shape[1], x.shape[2], 224
    nh, nw = (int(h / w * (size + 32)), size + 32) if h > w else (size + 32, int(w / h * (size + 32)))
    x = torch.einsum("oh,chw,pw->cop", convert._legacy_bicubic_matrix(h, nh), x, convert._legacy_bicubic_matrix(w, nw))
    oh, ow = (nh - size + 1) // 2, (nw - size + 1) // 2
    return (x[:, oh:oh + size, ow:ow + size] - convert.MEAN) / convert.STD


def row_groups():
    """Global image order = parquet files sorted by name, then row order within file (= image_order.csv)."""
    out, start = [], 0
    for fi, f in enumerate(FILES):
        md = pq.ParquetFile(f).metadata
        for g in range(md.num_row_groups):
            out.append((fi, g, start)); start += md.row_group(g).num_rows
    assert start == N
    return out


def write_order():
    rows = []
    for f in FILES:
        t = pq.read_table(f, columns=["image", "label"])
        for r, (p, y) in enumerate(zip(t.column("image").combine_chunks().field("path").to_pylist(), t.column("label").to_pylist())):
            rows.append(f"{len(rows)},{os.path.basename(f)},{r},{p},{y}")
    assert len(rows) == N
    os.makedirs(os.path.dirname(ORDER), exist_ok=True)
    open(ORDER, "w").write("idx,file,row,path,label\n" + "\n".join(rows) + "\n")


def labels():
    return np.loadtxt(ORDER, delimiter=",", skiprows=1, usecols=4, dtype=np.int64)


class Val(torch.utils.data.IterableDataset):
    def __init__(self, fn):
        self.fn = fn

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        k, W = (wi.id, wi.num_workers) if wi else (0, 1)
        for fi, g, start in row_groups()[k::W]:
            t = pq.ParquetFile(FILES[fi]).read_row_group(g, columns=["image", "label"])
            for r, (b, y) in enumerate(zip(t.column("image").combine_chunks().field("bytes").to_pylist(), t.column("label").to_pylist())):
                yield start + r, y, self.fn(b)


def stream(fn):
    """Batches (idx, labels, x) of BS, in the deterministic data-loader round-robin order used for A0/A1."""
    buf = []
    for item in torch.utils.data.DataLoader(Val(fn), batch_size=None, num_workers=N_LOADERS, prefetch_factor=16):
        buf.append(item)
        if len(buf) == BS:
            yield [b[0] for b in buf], [b[1] for b in buf], torch.stack([b[2] for b in buf]); buf = []
    if buf:
        yield [b[0] for b in buf], [b[1] for b in buf], torch.stack([b[2] for b in buf])


def build_cache(decoder):
    """Exact network inputs (float32, normalised; the bicubic output is fractional and unclipped, so uint8 would
    change them) in image_order.csv order, plus the A0/A1 batch order so cached evals see identical batches."""
    os.makedirs(CACHE, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(CACHE, f"val_{decoder}.npy.part"), "w+", np.float32, (N, 3, 224, 224))
    order, y, t0 = [], labels(), time.time()
    for idx, lab, x in stream(lambda b: preprocess(b, decoder)):
        assert (y[idx] == lab).all()
        X[idx] = x.numpy(); order += idx
        if len(order) % 10000 == 0:
            print(decoder, len(order), f"{time.time() - t0:.0f}s", flush=True)
    assert sorted(order) == list(range(N))
    X.flush(); del X
    np.save(os.path.join(CACHE, f"batch_order_{decoder}.npy"), np.array(order, np.int32))
    os.replace(os.path.join(CACHE, f"val_{decoder}.npy.part"), os.path.join(CACHE, f"val_{decoder}.npy"))


class Cached(torch.utils.data.Dataset):
    """Batch b of the cache in A0/A1 batch order: contiguous runs read with preadv, rows back in in-batch order."""
    def __init__(self, decoder):
        self.path = os.path.join(CACHE, f"val_{decoder}.npy")
        self.off, self.row = np.load(self.path, mmap_mode="r").offset, 3 * 224 * 224 * 4
        self.order = np.load(os.path.join(CACHE, f"batch_order_{decoder}.npy"))

    def __len__(self):
        return N // BS

    def __getitem__(self, b):
        o = self.order[b * BS:(b + 1) * BS]; srt = np.sort(o)
        buf = np.empty((len(o), 3, 224, 224), np.float32); mv, i = memoryview(buf).cast("B"), 0
        with open(self.path, "rb", buffering=0) as f:
            for run in np.split(srt, np.flatnonzero(np.diff(srt) != 1) + 1):
                n = os.preadv(f.fileno(), [mv[i * self.row:(i + len(run)) * self.row]], self.off + int(run[0]) * self.row)
                assert n == len(run) * self.row; i += len(run)
        return o, torch.from_numpy(buf[np.argsort(np.argsort(o))])


def cached(decoder):
    for o, x in torch.utils.data.DataLoader(Cached(decoder), batch_size=None, num_workers=3, pin_memory=True, prefetch_factor=4):
        yield np.asarray(o), None, x


def predict(m, batches, n=N):
    top5 = np.full((n, 5), -1, np.int16); margin = np.zeros(n, np.float16)

    def store(idx, v, k):
        top5[idx] = k.cpu().numpy().astype(np.int16)
        margin[idx] = (v[:, 0] - v[:, 1]).cpu().numpy().astype(np.float16)
    prev = None
    for idx, _, x in batches:  # reading batch i overlaps the GPU on batch i-1 (results fetched one step late)
        with torch.inference_mode():
            v, k = m(x.cuda()).float().topk(5, 1)
        if prev:
            store(*prev)
        prev = (idx, v, k)
    store(*prev)
    assert (top5 >= 0).all()
    return top5, margin


def load(rid):
    m = convert.to_torch(convert.load_np(os.path.join(ROOT, "data", "ckpt", rid))).cuda()
    convs = [p for p in m.parameters() if p.dim() == 4]
    return m, sum((p == 0).sum().item() for p in convs) / sum(p.numel() for p in convs)


def gpu_loop(todo, fn, ready=lambda r: True):
    """Run fn(r) for each r in todo once ready(r). GPU sharing is OPT-IN: if PIE_GPU_LOCK_DIR is set, take
    flock(<dir>/gpu.lock) PER MODEL (outputs are written before release) and wait while free GPU memory < 2.5 GB;
    wrap the whole batch in `flock <dir>/gpu.gate` from the caller for priority (see scripts/reproduce.sh)."""
    import fcntl
    d = os.environ.get("PIE_GPU_LOCK_DIR")
    lock = open(os.path.join(d, "gpu.lock"), "a") if d else None
    todo = list(todo)
    while todo:
        rd = [r for r in todo if ready(r)]
        if not rd:
            time.sleep(30); continue
        if lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if torch.cuda.mem_get_info()[0] < 2.5e9:  # need ~1.3 GB (+ headroom); let other jobs drain
                fcntl.flock(lock, fcntl.LOCK_UN); time.sleep(30); continue
        fn(rd[0]); todo.remove(rd[0])
        if lock:
            fcntl.flock(lock, fcntl.LOCK_UN); time.sleep(3)  # flock is not FIFO: give queued jobs a chance to take it


def evaluate(rid, decoder, tag, inv, y):
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    m, zf = load(rid)
    top5, margin = predict(m, cached(decoder))
    name = rid + (f"_{tag}" if tag else "")
    np.savez(os.path.join(outdir(decoder), name + ".npz"), top1=top5[:, 0].copy(), top5=top5, margin=margin)
    r = inv[rid]
    res = {"run_id": rid, "decoder": decoder, "tag": tag, "top1": float((top5[:, 0] == y).mean()), "top5": float((top5 == y[:, None]).any(1).mean()),
           "tstat_top1": r.get("tstat_top1"), "tstat_top5": r.get("tstat_top5"), "conv_zero_frac": zf,
           "gpu_peak_alloc_GB": torch.cuda.max_memory_allocated() / 1e9, "seconds": time.time() - t0}
    open(os.path.join(outdir(decoder), name + ".json"), "w").write(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    a = sys.argv[1:]
    decoder = "islow" if "--islow" in a else "ifast"
    tag = a[a.index("--tag") + 1] if "--tag" in a else ""
    os.makedirs(outdir(decoder), exist_ok=True)
    if not os.path.exists(ORDER):
        write_order()
    if a[0] == "cache":
        build_cache(decoder)
    else:
        inv = json.load(open(os.path.join(ROOT, "INVENTORY.json")))["runs"]
        y = labels()
        rids = [k for k, r in inv.items() if r["complete"]] if a[1] == "all" else [a[1]]
        ck = lambda rid: all(os.path.exists(os.path.join(ROOT, "data", "ckpt", f"{rid}.{f}")) for f in ("data", "index"))
        todo = [r for r in rids if not os.path.exists(os.path.join(outdir(decoder), r + (f"_{tag}" if tag else "") + ".npz"))]
        gpu_loop(todo, lambda r: evaluate(r, decoder, tag, inv, y), ck)
        print("EVAL DONE", decoder, flush=True)
