# Qualitative figure from real data, chosen by a stated rule (no hand-picking; DEVIATIONS D12 E7):
#   row 1: validation images ranked by their floor PIE rate (dense vs dense, n = 13), highest first;
#   row 2: images ranked by the rise in PIE rate at 90% sparsity over the floor (f_0.9 - f_floor), highest first.
# Ties are broken by image index. Walking down each ranking, an image is EXCLUDED when the torchvision COCO Faster R-CNN
# ResNet-50 FPN v2 detector finds a 'person' with score >= 0.3; the first 6 kept images form each row.
# Titles give the two most-voted labels per population. Needs the ImageNet validation parquet. usage: python src/fig_examples.py
import io, json, os, sys, textwrap
import numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A  # noqa: E402
import figures as F  # noqa: E402

ROOT = A.ROOT
PERSON_SCORE = 0.3


def main():
    import torch, torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    torch.set_num_threads(8)
    names = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    wts = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    det = fasterrcnn_resnet50_fpn_v2(weights=wts).eval(); person = wts.meta["categories"].index("person")
    f = np.load(os.path.join(ROOT, "results", "analysis", "ifast", "f.npz"))
    o = pd.read_csv(os.path.join(ROOT, "results", "preds", "image_order.csv"))
    pop = A.Pop("ifast")
    PD, P9 = pop.stack(pop.dense), pop.stack(pop.pr[0.9])
    real = json.load(open(os.path.join(ROOT, "data", "real", "real.json")))
    tables = {}

    def image(i):
        row = o.iloc[i]
        if row.file not in tables:
            tables[row.file] = pq.read_table(os.path.join(ROOT, "data", "imagenet-1k", "data", row.file), columns=["image"])
        return Image.open(io.BytesIO(tables[row.file].column("image")[int(row.row)].as_py()["bytes"])).convert("RGB")

    def has_person(im):
        with torch.inference_mode():
            r = det([torchvision.transforms.functional.to_tensor(im)])[0]
        s = r["scores"][r["labels"] == person]
        return float(s.max()) if len(s) else 0.0

    rankings = {"Seed-noise PIEs (dense vs dense)": np.lexsort((np.arange(len(o)), -f["n13_floor"])),
                "PIEs added by 90% pruning": np.lexsort((np.arange(len(o)), -(f["n13_0.9"] - f["n13_floor"])))}
    sel, log = {}, {}
    for title, order in rankings.items():
        kept, skipped = [], []
        for i in order:
            ps = has_person(image(i))
            (skipped if ps >= PERSON_SCORE else kept).append((int(i), ps))
            if len(kept) == 6:
                break
        sel[title], log[title] = [k for k, _ in kept], {"kept": kept, "excluded_person": skipped}

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 4.4))
    info = {}
    for r, (title, idx) in enumerate(sel.items()):
        for c, i in enumerate(idx):
            row = o.iloc[i]
            im = image(i); im.thumbnail((256, 256))
            ax = axes[r, c]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])

            def votes(P):
                v, cnt = np.unique(P[:, i], return_counts=True); top = np.argsort(-cnt, kind="stable")[:2]
                return "; ".join(f"{names[v[j]]} ({cnt[j]})" for j in top)
            wrap = lambda s: "\n".join(textwrap.wrap(s, 24))
            rl = [names[k] for k in real[int(row.path.split("_")[2]) - 1]]
            ax.set_title(wrap(f"label: {names[int(row.label)]}") + "\n" + wrap(f"27 dense: {votes(PD)}") + "\n" + wrap(f"30 at 90%: {votes(P9)}"), fontsize=3.9, color=F.INK)
            ax.set_xlabel(wrap(f"PIE rate {f['n13_floor'][i]:.2f} (floor), {f['n13_0.9'][i]:.2f} (90%)") + "\n" + wrap(f"ReaL: {', '.join(rl) or 'none'}"), fontsize=3.8, color=F.INK2)
            info.setdefault(title, []).append({"idx": int(i), "path": row.path, "label": names[int(row.label)], "dense_votes": votes(PD), "pruned90_votes": votes(P9),
                                               "f_floor": float(f["n13_floor"][i]), "f_0.9": float(f["n13_0.9"][i]), "real": rl})
        axes[r, 0].set_ylabel(title, fontsize=5.5, color=F.INK)
    F.save(fig, "fig8_examples")
    json.dump({"selection": info, "person_filter": log, "rule": f"exclude if COCO person score >= {PERSON_SCORE}"},
              open(os.path.join(ROOT, "results", "figures", "fig8_examples_selection.json"), "w"), indent=1)
    print({t: [(x["label"], round(x["f_floor"], 2), round(x["f_0.9"], 2)) for x in v] for t, v in info.items()})
    print({t: len(v["excluded_person"]) for t, v in log.items()})


if __name__ == "__main__":
    main()
