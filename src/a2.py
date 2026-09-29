# Gate A2 (PREREG + PREREG-ADDENDUM-1 §9): released PIE files vs our recompute.
# 1) search for a prediction-free image mapping (shard assignment = hash(filename) mod 128), 2) if none, substitute evidence:
#    Jaccard UPPER BOUND over all class-consistent mappings, per-class PIE-count correlation, total-count difference.
# usage: python src/a2.py  -> results/a2.json
import hashlib, json, os, sys, zlib
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A  # noqa: E402

ROOT = A.ROOT
G13 = os.path.join(ROOT, "prior-gate", "g13")


def released(s):
    d = pd.read_csv(os.path.join(G13, f"pie0{int(s * 10)}.csv"), sep="\t")
    d["shard"] = d.file.str[-14:-9].astype(int)
    return d


def mapping_search(order, d):
    """Candidate shard functions h(key) % 128; a match must reproduce every released shard's class-label multiset."""
    H = np.zeros((128, 1000), int)
    for s, g in d.groupby("shard"):
        H[s] = np.bincount(g.stored_class_label, minlength=1000)
    keep = np.array([s not in (6, 111) for s in range(128)])
    keys = {"ILSVRC2012_val_%08d.JPEG": lambda r: f"ILSVRC2012_val_{r.vid:08d}.JPEG", "ILSVRC2012_val_%08d": lambda r: f"ILSVRC2012_val_{r.vid:08d}",
            "<wnid>/ILSVRC2012_val_%08d.JPEG": lambda r: f"{r.syn}/ILSVRC2012_val_{r.vid:08d}.JPEG", "HF path": lambda r: r.path,
            "index 1-based": lambda r: str(r.vid), "index 0-based": lambda r: str(r.vid - 1)}
    hashes = {"md5": lambda b: int(hashlib.md5(b).hexdigest(), 16), "md5[:8] le": lambda b: int.from_bytes(hashlib.md5(b).digest()[:8], "little"),
              "sha1": lambda b: int(hashlib.sha1(b).hexdigest(), 16), "sha256": lambda b: int(hashlib.sha256(b).hexdigest(), 16),
              "crc32": zlib.crc32, "adler32": zlib.adler32}
    try:
        import farmhash  # TF tf.strings.to_hash_bucket_fast = Fingerprint64 % n
        hashes["farmhash Fingerprint64"] = lambda b: farmhash.fingerprint64(b.decode())
    except ImportError:
        pass
    rows, res = list(order.itertuples()), []
    for kn, kf in keys.items():
        ks = [kf(r).encode() for r in rows]
        for hn, hf in hashes.items():
            M = np.zeros((128, 1000), int)
            np.add.at(M, (np.array([hf(k) % 128 for k in ks]), order.label.values), 1)
            res.append({"key": kn, "hash": hn, "shards_exact": int((M[keep] == H[keep]).all(1).sum()), "L1": int(np.abs(M[keep] - H[keep]).sum())})
    # reference: L1 of an independent random multinomial sharding (what 'no signal' looks like)
    rng = np.random.default_rng(20260928)
    M = np.zeros((128, 1000), int); np.add.at(M, (rng.integers(0, 128, len(order)), order.label.values), 1)
    sizes = d.groupby("shard").size().values
    return {"candidates": res, "random_sharding_L1": int(np.abs(M[keep] - H[keep]).sum()),
            "shard_size": {"min": int(sizes.min()), "max": int(sizes.max()), "sd": float(sizes.std(ddof=1)),
                           "multinomial_sd": float(np.sqrt(50000 / 128 * (1 - 1 / 128)))},
            "found": any(r["shards_exact"] == 126 for r in res)}


def jaccard_upper_bound(a, b, m):
    """max over class-consistent mappings: per class choose our count a' in [max(0,a-m), a] closest to b."""
    ap = np.clip(b, np.maximum(0, a - m), a)
    return float(np.minimum(ap, b).sum() / np.maximum(ap, b).sum())


def main():
    out = {}
    order = pd.read_csv(os.path.join(ROOT, "results", "preds", "image_order.csv"))
    order["vid"] = order.path.str.extract(r"val_(\d+)").astype(int)[0]
    order["syn"] = order.path.str.extract(r"_(n\d+)\.")[0]
    d7 = released(0.7)
    out["mapping_search"] = mapping_search(order.sort_values("vid").reset_index(drop=True), d7)
    out["A2_as_specified"] = "computed" if out["mapping_search"]["found"] else "NOT COMPUTABLE AS SPECIFIED (no prediction-free image mapping)"
    for dec in ("ifast", "islow"):
        pop = A.Pop(dec)
        pools = {"dense27": pop.dense, "threshold16": pop.thr, "baseline11": pop.base}
        md = {pn: A.modal(pop.stack(ids)) for pn, ids in pools.items()}
        res = {}
        for s in (0.7, 0.9):
            d = released(s)
            b = np.bincount(d.stored_class_label[d.is_pie == "PIE"], minlength=1000)
            m = 50 - np.bincount(d.stored_class_label, minlength=1000)
            mp = A.modal(pop.stack(pop.pr[s]))
            for pn in pools:
                pie = md[pn] != mp
                a = np.bincount(pop.y[pie], minlength=1000)
                ub = jaccard_upper_bound(a, b, m)
                res[f"{pn}_vs_pruned30_{s}"] = {
                    "our_pie_total_50000": int(a.sum()), "released_pie_total_49187": int(b.sum()),
                    "jaccard_upper_bound": ub, "A2_substitute_verdict": "FAIL (bound < 0.8)" if ub < 0.8 else "UNDETERMINED (bound >= 0.8)",
                    "per_class_pearson": float(stats.pearsonr(a, b)[0]), "per_class_spearman": float(stats.spearmanr(a, b)[0])}
        out[dec] = res
    json.dump(out, open(os.path.join(ROOT, "results", "a2.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "mapping_search"}, indent=1))
    ms = out["mapping_search"]
    print("mapping found:", ms["found"], "| best L1", min(r["L1"] for r in ms["candidates"]), "| random L1", ms["random_sharding_L1"], "| shard sizes", ms["shard_size"])


if __name__ == "__main__":
    main()
