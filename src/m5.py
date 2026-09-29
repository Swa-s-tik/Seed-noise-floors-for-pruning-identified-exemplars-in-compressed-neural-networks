# M5 / H3 (PREREG + PREREG-ADDENDUM-1 §11 + D7): FP16 and INT8 PTQ of the 27 dense models.
#   python src/m5.py calib                 first 1,024 ImageNet-1k TRAIN images (one parquet shard), IFAST -> data/cache/calib_ifast.npy
#   python src/m5.py eval fp16|int8|dynint8 (dynint8 = ADDENDUM-3 C2)       GPU (optional sharing via PIE_GPU_LOCK_DIR, see scripts/reproduce.sh) -> results/preds_m5/<q>/<rid>.npz
#   python src/m5.py int8check             simulated (reference) vs true fbgemm INT8 on CPU, 1 model x 2,000 val images
#   python src/m5.py analyze               paired / unpaired / floor over the M1 splits (n=13) -> results/analysis/ifast/m5.json
import copy, glob, json, os, sys, time, warnings
import numpy as np, pyarrow.parquet as pq, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalpreds as E  # noqa: E402

warnings.filterwarnings("ignore", module="torch.ao")
ROOT = E.ROOT
CALIB = os.path.join(E.CACHE, "calib_ifast.npy")
OUT = os.path.join(ROOT, "results", "preds_m5")


def calib():
    f = sorted(glob.glob(os.path.join(ROOT, "data", "imagenet-1k-train", "data", "train-*.parquet")))[0]
    t = pq.ParquetFile(f).read_row_groups([0, 1, 2], columns=["image"])  # >= 1,024 rows, file order
    b = t.column("image").combine_chunks().field("bytes").to_pylist()[:1024]
    assert len(b) == 1024
    np.save(CALIB, torch.stack([E.preprocess(x, "ifast") for x in b]).numpy())
    print("calib", f, np.load(CALIB, mmap_mode="r").shape)


def int8_model(rid, device):
    """FX static PTQ, fbgemm default qconfig, calibrated on CALIB; returned as the reference (simulated) model (D7)."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx
    m = E.convert.to_torch(E.convert.load_np(os.path.join(ROOT, "data", "ckpt", rid))).eval()
    C = torch.from_numpy(np.load(CALIB))
    mp = prepare_fx(m, get_default_qconfig_mapping("fbgemm"), example_inputs=(C[:2],)).to(device)
    with torch.inference_mode():
        for s in range(0, len(C), 64):
            mp(C[s:s + 64].to(device))
    return mp.cpu(), convert_to_reference_fx(copy.deepcopy(mp).cpu()).to(device).eval()


def dynint8_model(rid):
    """ADDENDUM-3 C2: TFLite-style dynamic-range INT8, simulated. BN folded into convs; symmetric per-output-channel int8
    weights; dynamic symmetric per-image per-tensor int8 activations at every conv / final-dense input; float bias."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    m = E.load(rid)[0].eval()
    pairs = [(m, "conv1", "bn1")] + [(blk, f"conv{i}", f"bn{i}") for L in (m.layer1, m.layer2, m.layer3, m.layer4) for blk in L for i in (1, 2, 3)]
    pairs += [(blk.downsample, "0", "1") for L in (m.layer1, m.layer2, m.layer3, m.layer4) for blk in L if blk.downsample is not None]
    for parent, c, b in pairs:
        setattr(parent, c, fuse_conv_bn_eval(getattr(parent, c), getattr(parent, b))); setattr(parent, b, torch.nn.Identity())

    def q(x, dims):
        s = x.abs().amax(dim=dims, keepdim=True) / 127
        return torch.where(s > 0, torch.clamp(torch.round(x / torch.where(s > 0, s, 1)), -127, 127) * s, torch.zeros_like(x))
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                mod.weight.copy_(q(mod.weight, tuple(range(1, mod.weight.dim()))))
                mod.register_forward_pre_hook(lambda _m, inp: (q(inp[0], tuple(range(1, inp[0].dim()))),))
    return m


class Half(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m.half()

    def forward(self, x):
        return self.m(x.half()).float()


def evaluate(q, rid, y):
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    m = Half(E.load(rid)[0]) if q == "fp16" else dynint8_model(rid) if q == "dynint8" else int8_model(rid, "cuda")[1]
    top5, margin = E.predict(m, E.cached("ifast"))
    np.savez(os.path.join(OUT, q, rid + ".npz"), top1=top5[:, 0].copy(), top5=top5, margin=margin)
    res = {"run_id": rid, "q": q, "top1": float((top5[:, 0] == y).mean()), "gpu_peak_alloc_GB": torch.cuda.max_memory_allocated() / 1e9, "seconds": time.time() - t0}
    open(os.path.join(OUT, q, rid + ".json"), "w").write(json.dumps(res)); print(json.dumps(res), flush=True)
    del m; torch.cuda.empty_cache()


def int8check(rid="2677641_b0.0", n=2000):
    from torch.ao.quantization.quantize_fx import convert_fx
    torch.set_num_threads(8); torch.backends.quantized.engine = "fbgemm"
    mp, ref = int8_model(rid, "cpu")
    true8 = convert_fx(mp)
    X = torch.from_numpy(np.array(np.load(os.path.join(E.CACHE, "val_ifast.npy"), mmap_mode="r")[:n]))
    with torch.inference_mode():
        a = torch.cat([true8(X[s:s + 100]).argmax(1) for s in range(0, n, 100)])
        b = torch.cat([ref(X[s:s + 100]).argmax(1) for s in range(0, n, 100)])
    res = {"run_id": rid, "n_images": n, "top1_agreement_reference_vs_true_int8": float((a == b).float().mean())}
    json.dump(res, open(os.path.join(OUT, "int8check.json"), "w")); print(res)


def analyze():
    import analysis as A
    pop = A.Pop("ifast")
    h13 = A.halves_list(A.RNG["split13"], 13, pop.base, pop.thr)  # same draws as the M1 splits (fresh stream)
    Q = {q: {k: np.load(os.path.join(OUT, q, k + ".npz"))["top1"].astype(np.int16) for k in pop.dense}
         for q in ("fp16", "int8", "dynint8") if os.path.isdir(os.path.join(OUT, q))}
    out, counts = {}, {}
    fl = np.array([(A.modal(pop.stack(a)) != A.modal(pop.stack(b))).sum() for a, b in h13])
    sd, m1 = fl.std(ddof=1), fl.mean()
    for q, P in Q.items():
        st = lambda ids: np.stack([P[k] for k in ids])
        paired = np.array([(A.modal(st(a)) != A.modal(pop.stack(a))).sum() for a, b in h13])
        unpaired = np.array([(A.modal(st(a)) != A.modal(pop.stack(b))).sum() for a, b in h13])
        counts[q] = {"paired": paired, "unpaired": unpaired}
        mp, mu = paired.mean(), unpaired.mean()
        out[q] = {"paired": A.summ(paired), "unpaired": A.summ(unpaired),
                  "per_model_top1": {k: float((v == pop.y).mean()) for k, v in P.items()},
                  "per_model_flips_vs_fp32": {k: int((v != pop.P[k]).sum()) for k, v in P.items()},
                  "H3_paired_below_floor(mean_p < m1 - 3sd)": bool(mp < m1 - 3 * sd),
                  "H3_unpaired_near_floor(|mu - m1| <= 3sd)": bool(abs(mu - m1) <= 3 * sd),
                  "withdraw_inference(|mp - mu| <= 3sd)": bool(abs(mp - mu) <= 3 * sd),
                  "z_paired": float((mp - m1) / sd), "z_unpaired": float((mu - m1) / sd)}
        out[q]["H3_holds"] = out[q]["H3_paired_below_floor(mean_p < m1 - 3sd)"] and out[q]["H3_unpaired_near_floor(|mu - m1| <= 3sd)"]
    out["floor_recomputed"] = A.summ(fl)
    os.makedirs(os.path.join(ROOT, "results", "analysis", "ifast"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "ifast", "m5.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(ROOT, "results", "analysis", "ifast", "m5_counts.npz"), floor=fl,
                        **{f"{q}_{k}": v for q, d in counts.items() for k, v in d.items()})
    print(json.dumps({q: {k: v for k, v in out[q].items() if not k.startswith("per_model")} for q in Q}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calib":
        calib()
    elif cmd == "eval":
        q = sys.argv[2]; os.makedirs(os.path.join(OUT, q), exist_ok=True)
        inv = json.load(open(os.path.join(ROOT, "INVENTORY.json")))["runs"]
        dense = sorted(k for k, r in inv.items() if r["complete"] and r["role"] != "pruned")
        E.gpu_loop([k for k in dense if not os.path.exists(os.path.join(OUT, q, k + ".npz"))], lambda r: evaluate(q, r, E.labels()))
        print("M5 EVAL DONE", q, flush=True)
    elif cmd == "int8check":
        os.makedirs(OUT, exist_ok=True); int8check()
    else:
        analyze()
