# Breadth (b), PREREG-ADDENDUM-2: CIFAR-10 populations trained here with airbench94 (Jordan 2024, legacy/airbench94.py,
# MIT, sha256 in airbench94.sha256), unmodified except (i) torch.manual_seed(seed) per run and (ii) gradual per-layer
# magnitude pruning of the six 3x3 convs on Hooker's schedule (Zhu & Gupta cubic; start/end/frequency at the same fractions
# of training as Hooker's 1000/9000/500 of 32000 steps). Evaluation: single view (tta_level 0), deterministic cuDNN.
# usage: python cifar/run.py <sparsity> <seed>... [--structured]  -> results/cifar/preds/{s|st}<sparsity>_seed<seed>.npz
# --structured (ADDENDUM-4): L1 filter pruning of the six 3x3 convs, same schedule; pruned channels removed (BN output masked).
#        (GPU sharing is optional: PIE_GPU_LOCK_DIR, see scripts/reproduce.sh)
import json, os, sys, time
from math import ceil
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); os.chdir(HERE)  # airbench's CifarLoader caches the dataset under ./cifar10
import airbench94 as AB  # noqa: E402
from torch import nn  # noqa: E402

OUT = os.path.join(ROOT, "results", "cifar", "preds")
T0, T1, DT, TOTAL = 1000, 9000, 500, 32000  # Hooker's pruning schedule (steps), mapped to fractions of training


def target(step, T, s_final):
    t0, t1, dt = round(T * T0 / TOTAL), round(T * T1 / TOTAL), max(1, round(T * DT / TOTAL))
    if s_final == 0 or step < t0 or (step - t0) % dt:
        return None
    frac = min(1.0, (step - t0) / (t1 - t0))
    return s_final * (1 - (1 - frac) ** 3)


def prune_structured(convs, masks, s):
    """ADDENDUM-4: per layer, zero the round(s * out_channels) filters with the smallest L1 norm (masks are (out,1,1,1))."""
    for c, m in zip(convs, masks):
        l1 = c.weight.data.float().abs().sum(dim=(1, 2, 3))
        k = int(round(s * l1.numel()))
        keep = torch.ones_like(l1)
        if k:
            keep[torch.argsort(l1, stable=True)[:k]] = 0
        m.copy_(keep.view(-1, 1, 1, 1).to(m.dtype))


def prune(convs, masks, s):
    for c, m in zip(convs, masks):
        w = c.weight.data.float().abs().flatten()
        k = int(round(s * w.numel()))
        if k:
            thr = w.kthvalue(k).values
            m.copy_((c.weight.data.float().abs() > thr).to(m.dtype))


def train(seed, s_final, structured=False):
    """airbench94 main() with seed + pruning; returns the trained model, realized sparsity, test loader."""
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = False  # airbench default for training
    torch.manual_seed(seed)
    hyp = AB.hyp; bs, epochs, mom = hyp["opt"]["batch_size"], hyp["opt"]["train_epochs"], hyp["opt"]["momentum"]
    ks = 1024 * (1 + 1 / (1 - mom)); lr = hyp["opt"]["lr"] / ks; wd = hyp["opt"]["weight_decay"] * bs / ks; lrb = lr * hyp["opt"]["bias_scaler"]
    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp["opt"]["label_smoothing"], reduction="none")
    test_loader = AB.CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = AB.CifarLoader("cifar10", train=True, batch_size=bs, aug=hyp["aug"])
    T = ceil(len(train_loader) * epochs)
    model = AB.make_net()
    if structured:  # conv -> following BatchNorm, so a pruned filter's channel is removed exactly (BN output masked)
        pairs = [(g.conv1, g.norm1) for g in model.modules() if isinstance(g, AB.ConvGroup)] + [(g.conv2, g.norm2) for g in model.modules() if isinstance(g, AB.ConvGroup)]
        convs, bns = [c for c, _ in pairs], [b for _, b in pairs]
        masks = [torch.ones(c.weight.shape[0], 1, 1, 1, device=c.weight.device, dtype=c.weight.dtype) for c in convs]
    else:
        convs = [m for m in model.modules() if isinstance(m, AB.Conv) and m.kernel_size == (3, 3)]
        masks = [torch.ones_like(c.weight) for c in convs]
    nb = [p for k, p in model.named_parameters() if "norm" in k and p.requires_grad]
    op = [p for k, p in model.named_parameters() if "norm" not in k and p.requires_grad]
    opt = torch.optim.SGD([dict(params=nb, lr=lrb, weight_decay=wd / lrb), dict(params=op, lr=lr, weight_decay=wd / lr)], momentum=mom, nesterov=True)

    def get_lr(step):
        w = int(T * 0.23)
        return 0.2 * (1 - step / w) + step / w if step < w else 1.0 * (1 - (step - w) / (T - w)) + 0.07 * (step - w) / (T - w)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)
    alpha = 0.95 ** 5 * (torch.arange(T + 1) / T) ** 3
    la = AB.LookaheadState(model)
    AB.init_whitening_conv(model[0], train_loader.normalize(train_loader.images[:5000]))
    if structured:  # the channel mask also multiplies the BN output, so a pruned channel is exactly zero in forward AND backward
        for b, m in zip(bns, masks):  # (masking the BN bias instead makes the zero-variance BN backward overflow in fp16)
            b.register_forward_hook(lambda mod, inp, out, m=m: out * m.view(1, -1, 1, 1).to(out.dtype))
    apply = lambda: [c.weight.data.mul_(m) for c, m in zip(convs, masks)]
    step = 0
    for epoch in range(ceil(epochs)):
        model[0].bias.requires_grad = epoch < hyp["opt"]["whiten_bias_epochs"]
        model.train()
        for x, y in train_loader:
            loss = loss_fn(model(x), y).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            step += 1
            s = target(step, T, s_final)
            if s is not None:
                (prune_structured if structured else prune)(convs, masks, s)
            apply()
            if step % 5 == 0:
                la.update(model, decay=alpha[step].item()); apply()
            if step >= T:
                la.update(model, decay=1.0); apply()
                break
    zero = sum(int((c.weight == 0).sum()) for c in convs) / sum(c.weight.numel() for c in convs)
    return model, zero, test_loader, T


def run(seed, s_final, structured=False):
    t0 = time.time()
    model, zero, tl, T = train(seed, s_final, structured)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    logits = AB.infer(model, tl, tta_level=0).float()
    v, k = logits.topk(5, 1)
    y = tl.labels.cpu().numpy()
    np.savez(os.path.join(OUT, f"{'st' if structured else 's'}{s_final}_seed{seed}.npz"), top1=k[:, 0].cpu().numpy().astype(np.int16), top5=k.cpu().numpy().astype(np.int16),
             margin=(v[:, 0] - v[:, 1]).cpu().numpy().astype(np.float16), labels=y.astype(np.int16))
    res = {"seed": seed, "sparsity": s_final, "structured": structured, "top1": float((k[:, 0].cpu().numpy() == y).mean()), "conv3x3_zero_frac": zero, "steps": T, "seconds": time.time() - t0}
    print(json.dumps(res), flush=True)
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import evalpreds as E
    os.makedirs(OUT, exist_ok=True)
    st = "--structured" in sys.argv; args = [a for a in sys.argv[1:] if a != "--structured"]
    s = float(args[0]); seeds = [int(x) for x in args[1:]]
    E.gpu_loop([x for x in seeds if not os.path.exists(os.path.join(OUT, f"{'st' if st else 's'}{s}_seed{x}.npz"))], lambda x: run(x, s, st))
