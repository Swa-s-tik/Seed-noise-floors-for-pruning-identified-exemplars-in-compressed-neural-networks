# Hooker PIE bucket: build INVENTORY.json (`inventory`) and fetch checkpoints (`fetch <run_id>...`).
import base64, collections, datetime, hashlib, json, os, re, sys, urllib.error, urllib.parse, urllib.request
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G13 = os.path.join(ROOT, "prior-gate", "g13")
sys.path.insert(0, G13)
import ckidx, convert  # noqa: E402  (reuse prior-gate reader + converter)

P = "pruning_identified_exemplars/"
CK = "/model.ckpt-32000"
EXT = {"data": ".data-00000-of-00001", "index": ".index", "meta": ".meta"}
README = os.path.join(ROOT, "data", "bucket_meta", "README.md")
IDX_DIR = os.path.join(ROOT, "data", "bucket_meta", "index")
CKPT_DIR = os.path.join(ROOT, "data", "ckpt")


def md5b64(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return base64.b64encode(h.digest()).decode()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


NBYTES = [0]  # bytes received this process (throughput log)


def gcs_get(name, dest, md5, tries=5):
    """Keyless GCS media download, resumable from dest.part via HTTP Range; verifies the listing md5."""
    if os.path.exists(dest) and md5b64(dest) == md5:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = "https://storage.googleapis.com/storage/v1/b/gresearch/o/" + urllib.parse.quote(P + name, safe="") + "?alt=media"
    part, got = dest + ".part", 0
    for i in range(tries):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if r.status == 206 else "wb") as f:
                while b := r.read(1 << 20):
                    f.write(b); got += len(b); NBYTES[0] += len(b)
        except urllib.error.HTTPError as e:
            if e.code != 416:  # 416 = .part already complete
                print("retry", name, repr(e), flush=True); continue
        except OSError as e:
            print("retry", name, repr(e), flush=True); continue
        if md5b64(part) == md5:
            os.replace(part, dest)
            return got
        print("md5 mismatch, restarting", name, flush=True); os.remove(part)
    raise IOError(f"failed after {tries} tries: {name}")


def listing():
    return {x["name"][len(P):]: x for x in json.load(open(os.path.join(G13, "all.json")))}


def run_id(key):  # key = '<exp>/train/_k/<method>/<frac>/...'
    p = key.split("/")
    return f"{p[0]}_{p[3][0]}{p[4]}"


def role(method, frac):
    return "dense-baseline" if method == "baseline" else ("dense-threshold" if frac == 0 else "pruned")


def cohorts(exps, gap=100):
    # ponytail: launch batch = run of experiment ids with no gap > 100 (Borg/XM ids are allocated sequentially)
    out, start, prev = {}, None, None
    for e in sorted(exps):
        if prev is None or e - prev > gap:
            start = e
        out[e], prev = f"C{start}", e
    return out


def check_index(path, data_size):
    """Completeness: index parses, data size matches, and convert.to_torch accepts the tensor set
    (names + shapes, weights and masks) — run on zero tensors so no data download is needed."""
    ents = ckidx.read(path)
    end = max(s + o for _, _, _, s, o in ents)
    T = {k: np.zeros(sh, np.float32) for k, dt, sh, s, o in ents if dt == 1 and not k.endswith("Momentum")}
    convert.to_torch(T)
    return {"n_tensors": len(ents), "n_mask": sum(k.endswith("/mask") for k in T),
            "data_bytes_expected": end, "data_size_ok": end == data_size}


def inventory():
    L = listing()
    readme_txt = open(README).read().split("Reference list of checkpoints")[1]
    readme = re.findall(r"pruning_identified_exemplars/(\d+/train/_\d/\w+/[\d.]+/[\d./]+?)" + CK, readme_txt)
    t = pd.read_csv(os.path.join(G13, "tstat.csv"), usecols=["dir", "top_1_accuracy", "top_5_eval_accuracy", "num_eval_images", "global_step"]).drop_duplicates("dir")
    t["key"] = t.dir.str.replace(r".*/class_accuracy_/", "", regex=True).str.replace(CK, "")
    tst = t.set_index("key")
    bucket = {n[:-len(CK + EXT["index"])] for n in L if re.match(r"^\d+/train/.*" + re.escape(CK + EXT["index"]) + "$", n)}
    keys = sorted(set(readme) | bucket | set(tst.index))
    coh = cohorts({int(k.split("/")[0]) for k in keys})
    runs = {}
    for k in keys:
        p = k.split("/")
        method, frac = p[3], float(p[4])
        files = {f: {"size": int(L[k + CK + e]["size"]), "md5": L[k + CK + e]["md5Hash"]} for f, e in EXT.items() if k + CK + e in L}
        r = {"key": k, "exp": p[0], "variant": p[2], "method": method, "sparsity": frac, "role": role(method, frac),
             "cohort": coh[int(p[0])], "in_readme_list": k in readme, "in_bucket": bool(files), "in_tstat": k in tst.index,
             "files": files}
        if k in tst.index:
            r["tstat_top1"], r["tstat_top5"] = float(tst.top_1_accuracy[k]), float(tst.top_5_eval_accuracy[k])
        if "index" in files:
            dest = os.path.join(IDX_DIR, run_id(k) + ".index")
            gcs_get(k + CK + EXT["index"], dest, files["index"]["md5"])
            r["index_check"] = check_index(dest, files.get("data", {}).get("size", -1))
        r["complete"] = bool({"data", "index"} <= set(files) and r.get("index_check", {}).get("data_size_ok")
                             and (r["role"] == "dense-baseline" or r["index_check"]["n_mask"] == 53))
        r["bytes_needed"] = sum(files[f]["size"] for f in ("data", "index") if f in files)
        runs[run_id(k)] = r
    assert len(runs) == len(keys)
    # objects in the bucket that are not population members
    dup = [n[:-len(EXT["data"])] for n in L if n.endswith(CK[1:] + EXT["data"]) and not re.match(r"^\d+/train/", n)]
    other = {}
    for n in dup:
        md5s = [L[n + e]["md5Hash"] for e in EXT.values() if n + e in L]
        same = [rid for rid, r in runs.items() if [r["files"].get(f, {}).get("md5") for f in EXT] == md5s]
        other[n] = {"note": ("byte-identical copy (md5 of data/index/meta) of " + ",".join(same)) if same else
                    "not a PIE population member (vog_checkpoints = VoG study, arXiv:2008.11600; no role/sparsity path)",
                    "data_size": int(L[n + EXT["data"]]["size"])}
    summ = collections.Counter()
    for r in runs.values():
        g = f'{r["role"]}@{r["sparsity"]}'
        summ[g + "|tstat"] += r["in_tstat"]; summ[g + "|readme"] += r["in_readme_list"]
        summ[g + "|bucket"] += r["in_bucket"]; summ[g + "|complete"] += r["complete"]
    rel = [r for r in runs.values() if r["complete"]]
    inv = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "definitions": {
            "run_id": "<exp>_<b|t><end_sparsity> from the checkpoint path",
            "role": "path method 'baseline' -> dense-baseline; 'threshold' with end sparsity 0.0 -> dense-threshold; 'threshold' > 0 -> pruned",
            "sparsity": "end sparsity path segment",
            "cohort": "launch batch: experiment ids sorted, new cohort when the id gap > 100; named C<first id>. Builder-defined (PREREG names cohort but does not define it).",
            "complete": "data+index in bucket, index md5 ok, data size == max(offset+size) in index, convert.to_torch accepts the tensor set (53 conv weights+53 masks for threshold; 53 kernels for baseline; 53 BN; final_dense)",
            "bytes_needed": "data + index bytes (the .meta graph is not needed by convert.py)",
        },
        "sources": {
            "role_sparsity": "checkpoint path segments, as documented in bucket README.md section 'Interpreting the Checkpoint Path' and parsed by Hooker code pie_dataset_gen/imagenet_predictions.py (split[11]=method, split[12]=fraction)",
            "readme_list": f"bucket README.md 'Reference list of checkpoints' (sha256 {sha256(README)})",
            "bucket": f"prior-gate/g13/all.json GCS listing (sha256 {sha256(os.path.join(G13, 'all.json'))})",
            "tstat": f"prior-gate/g13/tstat.csv == bucket wide_all_aggregated1569137618.33_recall_precision_t_statistic.csv (md5 {md5b64(os.path.join(G13, 'tstat.csv'))} matches listing); one row per model x class, model fields repeated",
            "index_files": "bucket .index files (downloaded to data/bucket_meta/index/, md5-verified, parsed with prior-gate/g13/ckidx.py)",
        },
        "summary": {
            "counts_by_role_sparsity": dict(sorted(summ.items())),
            "n_runs_union": len(runs), "n_readme_list": sum(r["in_readme_list"] for r in runs.values()),
            "n_tstat": sum(r["in_tstat"] for r in runs.values()), "n_in_bucket": sum(r["in_bucket"] for r in runs.values()),
            "n_complete": len(rel), "n_readme_listed_missing_from_bucket": sum(r["in_readme_list"] and not r["in_bucket"] for r in runs.values()),
            "bytes_needed_all_complete": sum(r["bytes_needed"] for r in rel),
            "bytes_all_complete_incl_meta": sum(sum(f["size"] for f in r["files"].values()) for r in rel),
        },
        "runs": runs,
        "non_population_objects": other,
    }
    out = os.path.join(ROOT, "INVENTORY.json")
    json.dump(inv, open(out, "w"), indent=1)
    print(json.dumps(inv["summary"], indent=1))


def fetch(rids, threads=12):
    """Parallel (one object per thread), resumable, md5-verified. `fetch all` = every complete run."""
    import threading, time
    from concurrent.futures import ThreadPoolExecutor
    runs = json.load(open(os.path.join(ROOT, "INVENTORY.json")))["runs"]
    rids = [k for k, r in runs.items() if r["complete"]] if rids == ["all"] else rids
    jobs = [(rid, f) for rid in sorted(rids, key=lambda k: -runs[k]["sparsity"]) for f in ("index", "data")]
    t0, done = time.time(), threading.Event()

    def meter():
        while not done.wait(60):
            print(f"[{time.time() - t0:.0f}s] {NBYTES[0] / 1e6:.0f} MB, {NBYTES[0] / 1e6 / (time.time() - t0):.2f} MB/s aggregate", flush=True)
    threading.Thread(target=meter, daemon=True).start()

    def one(job):
        rid, f = job
        r = runs[rid]
        assert r["complete"], rid
        n = gcs_get(r["key"] + CK + EXT[f], os.path.join(CKPT_DIR, f"{rid}.{f}"), r["files"][f]["md5"])
        print(f"[{time.time() - t0:.0f}s] OK {rid}.{f} {n} bytes", flush=True)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, jobs))
    done.set()
    print(f"ALL DONE {len(rids)} runs, {NBYTES[0]} bytes in {time.time() - t0:.0f}s", flush=True)


META = {  # bucket object -> local path used by the analysis (non-commercial research terms; NOT redistributed, see NOTICE.md)
    "wide_all_aggregated1569137618.33_recall_precision_t_statistic.csv": os.path.join(G13, "tstat.csv"),
    "pie_stats_eval/area_bbox_eval_imagenet-with-bbox-validation_0.7.csv": os.path.join(G13, "pie07.csv"),
    "pie_stats_eval/area_bbox_eval_imagenet-with-bbox-validation_0.9.csv": os.path.join(G13, "pie09.csv"),
    "README.md": README,
}


def fetch_meta():
    """Download the bucket files the analysis reads (md5 from the listing), then check BUCKET_META_SHA256."""
    L = listing()
    for name, dest in META.items():
        gcs_get(name, dest, L[name]["md5Hash"])
    exp = dict(line.split()[::-1] for line in open(os.path.join(ROOT, "BUCKET_META_SHA256")))
    for rel, h in exp.items():
        assert sha256(os.path.join(ROOT, rel)) == h, rel
    print("bucket metadata files present and verified:", len(exp))


if __name__ == "__main__":
    {"inventory": lambda: inventory(), "fetch": lambda: fetch(sys.argv[2:]), "fetch_meta": fetch_meta}[sys.argv[1]]()
