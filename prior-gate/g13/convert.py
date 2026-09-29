# Hooker PIE TF1 checkpoint -> torchvision resnet50, no TF needed.
import numpy as np, torch, torch.nn.functional as F, torchvision, ckidx
from PIL import Image

def load_np(prefix):
    b = open(prefix + ".data", "rb").read()
    return {k: np.frombuffer(b[o:o+s], "<f4").reshape(sh)
            for k, dt, sh, s, o in ckidx.read(prefix + ".index") if dt == 1 and not k.endswith("Momentum")}

def to_torch(T):
    pruned = any(k.endswith("/mask") for k in T)
    def conv(name):
        p = "resnet_model/" + name
        w = T[p + "/weights"] * T[p + "/mask"] if pruned else T[p + "/kernel"]
        return torch.from_numpy(np.ascontiguousarray(w.transpose(3, 2, 0, 1)))
    sd, bn = {}, iter(range(53))
    def putbn(tp):
        i = next(bn); p = "resnet_model/batch_normalization" + (f"_{i}" if i else "")
        for a, b in [("weight", "gamma"), ("bias", "beta"), ("running_mean", "moving_mean"), ("running_var", "moving_variance")]:
            sd[f"{tp}.{a}"] = torch.from_numpy(T[f"{p}/{b}"].copy())
    sd["conv1.weight"] = conv("initial_conv"); putbn("bn1")
    for g, nb in zip(range(1, 5), [3, 4, 6, 3]):
        for j in range(nb):
            name = f"block_group_projection_block_group{g}" if j == 0 else f"block_group{g}_{j}_1"
            L = f"layer{g}.{j}"
            if j == 0:
                sd[f"{L}.downsample.0.weight"] = conv(f"bottleneck_projection_{name}"); putbn(f"{L}.downsample.1")
            for c in (1, 2, 3):
                sd[f"{L}.conv{c}.weight"] = conv(f"bottleneck_{c}_{name}"); putbn(f"{L}.bn{c}")
    assert next(bn, None) is None
    sd["fc.weight"] = torch.from_numpy(T["resnet_model/final_dense/kernel"].T.copy())
    sd["fc.bias"] = torch.from_numpy(T["resnet_model/final_dense/bias"].copy())
    m = torchvision.models.resnet50()
    missing = m.load_state_dict(sd, strict=False)
    assert all(k.endswith("num_batches_tracked") for k in missing.missing_keys) and not missing.unexpected_keys, missing
    # TF 'SAME' 3x3/s2 max-pool pads (0,1) not (1,1)
    m.maxpool = _TFMaxPool()
    return m.eval()

class _TFMaxPool(torch.nn.Module):
    def forward(self, x):
        return F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=float("-inf")), 3, 2)

def _legacy_bicubic_matrix(n_in, n_out):
    # TF1 resize_bicubic, align_corners=False, half_pixel_centers=False, A=-0.75
    A = -0.75; scale = n_in / n_out; W = np.zeros((n_out, n_in), np.float64)
    for o in range(n_out):
        x = o * scale; i0 = int(np.floor(x)); t = x - i0
        ws = [((A*(t+1)-5*A)*(t+1)+8*A)*(t+1)-4*A, ((A+2)*t-(A+3))*t*t+1,
              ((A+2)*(1-t)-(A+3))*(1-t)*(1-t)+1, ((A*(2-t)-5*A)*(2-t)+8*A)*(2-t)-4*A]
        for k, w in zip(range(i0-1, i0+3), ws): W[o, min(max(k, 0), n_in-1)] += w
    return torch.from_numpy(W).float()

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) * 255

def preprocess(path, size=224):
    x = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), np.float32)).permute(2, 0, 1)
    h, w = x.shape[1:]
    nh, nw = (int(h / w * (size+32)), size+32) if h > w else (size+32, int(w / h * (size+32)))
    x = torch.einsum("oh,chw,pw->cop", _legacy_bicubic_matrix(h, nh), x, _legacy_bicubic_matrix(w, nw))
    oh, ow = (nh - size + 1) // 2, (nw - size + 1) // 2
    return ((x[:, oh:oh+size, ow:ow+size] - MEAN) / STD)

if __name__ == "__main__":
    labels = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    # self-check: legacy bicubic of a constant row stays constant
    assert torch.allclose(_legacy_bicubic_matrix(500, 256).sum(1), torch.ones(256), atol=1e-5)
    xs = torch.stack([preprocess(p) for p in ("dog.jpg", "hopper.jpg")])
    for ck in ("base", "thr9"):
        m = to_torch(load_np(ck))
        if ck == "thr9":
            convs = [p for n, p in m.named_parameters() if p.dim() == 4]
            print("thr9 conv zero frac", sum((p == 0).sum().item() for p in convs) / sum(p.numel() for p in convs))
        with torch.inference_mode():
            p = m(xs).softmax(1)
        for i in range(len(xs)):
            v, k = p[i].topk(3)
            print(ck, i, [(labels[j], round(float(q), 3)) for q, j in zip(v, k)])
