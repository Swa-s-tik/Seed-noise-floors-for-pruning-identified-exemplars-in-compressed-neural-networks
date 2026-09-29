# Expected-checksum check of the analysis-stage outputs.
#   python tests/check_outputs.py          verify against EXPECTED_SHA256 (exit 1 on any mismatch)
#   python tests/check_outputs.py --write  (re)write EXPECTED_SHA256 from the current outputs
# .npz files are zip archives with write-time stamps, so each array member's raw bytes are hashed instead (byte-for-byte on payloads).
import glob, hashlib, os, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "EXPECTED_SHA256")
PATTERNS = ["results/fidelity.csv", "results/a2.json", "results/analysis/*.json", "results/analysis/*/*.json",
            "results/analysis/*/*.npz", "results/figures/*.pdf", "results/figures/*.png"]


def entries(root=ROOT):
    out = {}
    for pat in PATTERNS:
        for p in sorted(glob.glob(os.path.join(root, pat))):
            rel = os.path.relpath(p, root)
            if os.path.basename(p).startswith("fig8_"):  # needs the ImageNet images (not part of the CPU analysis stage)
                continue
            if p.endswith(".npz"):
                with zipfile.ZipFile(p) as z:
                    for n in sorted(z.namelist()):
                        out[f"{rel}::{n}"] = hashlib.sha256(z.read(n)).hexdigest()
            else:
                out[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


if __name__ == "__main__":
    root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else ROOT
    cur = entries(root)
    if "--write" in sys.argv:
        open(MANIFEST, "w").write("".join(f"{h}  {k}\n" for k, h in sorted(cur.items())))
        print(len(cur), "entries written to", MANIFEST); sys.exit(0)
    exp = dict(reversed(line.rstrip("\n").split("  ", 1)) for line in open(MANIFEST))
    bad = sorted(k for k in exp if cur.get(k) != exp[k])
    extra = sorted(set(cur) - set(exp))
    print(f"{len(exp) - len(bad)}/{len(exp)} match" + (f"; MISMATCH: {bad}" if bad else "") + (f"; not in manifest: {extra}" if extra else ""))
    sys.exit(1 if bad else 0)
