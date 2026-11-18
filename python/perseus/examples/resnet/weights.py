"""ResNet-20 on CIFAR-10 (chenyaofo/pytorch-cifar-models, option-B shortcuts: 1x1 conv + BN): the checkpoint, its
convs with BatchNorm folded in, the float64 torch reference and the test images."""
from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np

URL = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt"
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2023, 0.1994, 0.2010])
WIDTHS = (16, 32, 64)
DEPTH = 3                  # blocks per stage


def checkpoint():
    from perseus.hub import cache_dir
    path = os.path.join(cache_dir("resnet20"), os.path.basename(URL))
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(URL, tmp)
        os.replace(tmp, path)
    with open(path, "rb") as f:     # the release names the file after its sha256 prefix
        assert hashlib.sha256(f.read()).hexdigest().startswith(URL.rsplit("-", 1)[1][:8]), path
    return path


def state_dict(path=None):
    import torch
    sd = torch.load(path or checkpoint(), map_location="cpu", weights_only=True)
    return {k: v.double().numpy() for k, v in sd.items() if not k.endswith("num_batches_tracked")}


def _folded(sd, conv, bn):
    from perseus.impl.conv import fold_bn
    return fold_bn(sd[conv + ".weight"], None, sd[bn + ".weight"], sd[bn + ".bias"],
                   sd[bn + ".running_mean"], sd[bn + ".running_var"])


def blocks(sd):
    """(stem, [(name, stride, conv1, conv2, shortcut | None)], fc): every conv as (W, b) with its BN folded."""
    stem = _folded(sd, "conv1", "bn1")
    out = []
    for s in range(len(WIDTHS)):
        for i in range(DEPTH):
            p = f"layer{s + 1}.{i}"
            short = _folded(sd, p + ".downsample.0", p + ".downsample.1") if p + ".downsample.0.weight" in sd else None
            out.append((p, 2 if s and not i else 1, _folded(sd, p + ".conv1", p + ".bn1"),
                        _folded(sd, p + ".conv2", p + ".bn2"), short))
    return stem, out, (sd["fc.weight"], sd["fc.bias"])


def reference(sd, x, relu=None):
    """float64 forward of normalized images x (n, 3, 32, 32) on the folded convs -> (n, 10) logits; `relu(t, site)`
    replaces the activation (default: the exact max(t, 0)); sites "stem", "<block>.a1", "<block>.a2"."""
    import torch
    import torch.nn.functional as F
    act = relu or (lambda t, site: torch.clamp_min(t, 0))
    stem, blks, (fw, fb) = blocks(sd)
    T = torch.from_numpy

    def conv(t, Wb, stride):
        return F.conv2d(t, T(Wb[0]), T(Wb[1]), stride=stride, padding=Wb[0].shape[-1] // 2)

    with torch.no_grad():
        h = act(conv(T(np.asarray(x, dtype=np.float64)), stem, 1), "stem")
        for p, stride, c1, c2, short in blks:
            y = conv(act(conv(h, c1, stride), p + ".a1"), c2, 1)
            h = act(y + (conv(h, short, stride) if short is not None else h), p + ".a2")
        return (h.mean(dim=(2, 3)) @ T(fw).T + T(fb)).numpy()


def normalize(img):
    """uint8 (32, 32, 3) -> (3, 32, 32) as the checkpoint was trained on."""
    return ((np.asarray(img, dtype=np.float64) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def cifar10(n, start=0, split="test"):
    """(n, 3, 32, 32) normalized images and their labels."""
    import datasets
    ds = datasets.load_dataset("uoft-cs/cifar10", split=split).select(range(start, start + n))
    return np.stack([normalize(r["img"].convert("RGB")) for r in ds]), np.array(ds["label"])
