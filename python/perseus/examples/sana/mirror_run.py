"""The sampler's transformer passes on the plaintext mirror (the encrypted dataflow on NumpyOps): every block's
kernels with the configs' polynomial maps, the dense linears; per block the error against the float64 dumps, then the
latent and the decoded image against the exact ones. The text side (Gemma, caption projection, the per-block keys /
values) is exact here.

    .venv/bin/python -m perseus.examples.sana.mirror_run [--configs <c>] --prompt 0 --image
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
import types

import numpy as np

from perseus.impl.ops import NumpyOps

from . import hf
from . import layout as L
from . import kernels as KN
from . import model as M
from . import ref as R


class MirrorOps(NumpyOps):
    """NumpyOps plus the members the blocked kernels and Rt's step scopes read."""
    ln_cheb = True
    inf = types.SimpleNamespace(step=lambda label: contextlib.nullcontext())

    def tag_reduce(self, a, stride):
        return a

    def decrypt_slots_complex(self, a):
        return a


def mirror_rt():
    from perseus.impl.rt import Rt
    from .layout import geometry
    return Rt(MirrorOps(), geometry()[0])


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=hf.CONFIGS)
    ap.add_argument("--dumps", default="ref")
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--layers", type=int, default=28)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    from perseus.impl.blocked import norm_blocked
    W = R.SanaWeights()
    from .nl import PolyNL
    nl = PolyNL(json.load(open(hf.data(a.configs))))
    k = hf.scm_constants()
    z = np.load(hf.scratch(a.dumps, f"p{a.prompt}.npz"))
    sel = np.array([0] + list(range(len(z["ids"]) - hf.MAX_SEQ + 1, len(z["ids"]))))
    mask = torch.as_tensor(z["mask"][sel]).bool()
    enc = mask.clone()
    enc[0] = False
    txt = R.text_states(W, torch.as_tensor(z["gemma_out_sel"]).double(), enc, R.Approx())
    n_key = int(mask.sum())
    x_in = z["noise"].astype(np.float64) * k["c_in"]
    rt = mirror_rt()
    r = nl.resid_scale
    _d, B, ch = L.geometry()
    t0 = time.time()
    for i, t in enumerate(hf.STEPS):
        Wi, s = W.step(i), hf.step_tag(i)
        x0 = L.to_slots(x_in) @ W["patch_embed.proj.weight"][:, :, 0, 0].numpy().T \
            + W["patch_embed.proj.bias"].numpy()
        X = L.encode(x0 / r, 3)
        for b in range(a.layers):
            Kb, Vb = M.text_keys(W, b, txt[:n_key].numpy(), enc[:n_key].numpy(), msc=KN.xa_scale(nl, f"b{b}.xa"))
            X = M.block(rt, X, M.fold_block(Wi, b, r, nl, s), Kb, Vb, nl, M.dense_lin, b)
            line = f"[mirror] p{a.prompt} {s}block {b:2d} {time.time() - t0:6.0f}s"
            if f"{s}block{b}" in z.files:
                ref = z[f"{s}block{b}"].astype(np.float64)
                got = L.tokens(L.decode(X, L.D)) * r
                line += f" rel L2 err {np.linalg.norm(got - ref) / np.linalg.norm(ref):.2e}" \
                        f" max {np.abs(got - ref).max() / np.abs(ref).max():.2e}"
            print(line, flush=True)
        if a.layers < W.n_layers:
            return
        N2 = norm_blocked(rt, M._unscale(rt, X, r, "out.ln"), nl.ln("out.ln"), ch, B)
        F = L.from_slots(L.decode(M.dense_lin(rt, N2, M.out_lin(Wi)), 32))
        if i + 1 < len(hf.STEPS):                    # the next model input (renoise: the client's noise z2)
            rn = hf.renoise(hf.STEPS[i + 1], t)
            x_in = rn["p"] * x_in + rn["q"] * F + rn["r"] * z["z2"].astype(np.float64)
            ref = z["x_in2"].astype(np.float64)
            print(f"[mirror] p{a.prompt} x_in2 rel L2 err {np.linalg.norm(x_in - ref) / np.linalg.norm(ref):.3e}",
                  flush=True)
    kl = hf.scm_constants(hf.STEPS[-1])
    lat = kl["a"] * x_in + kl["b"] * F
    ref_lat = z["latent"].astype(np.float64)
    e = np.linalg.norm(lat - ref_lat) / np.linalg.norm(ref_lat)
    line = f"[mirror] p{a.prompt} latent rel L2 err {e:.3e}"
    if a.image:
        from diffusers import AutoencoderDC
        from PIL import Image
        vae = AutoencoderDC.from_pretrained(hf.snapshot(), subfolder="vae").eval()

        def image(lt):
            im = vae.decode(torch.as_tensor(lt[None], dtype=torch.float32) / vae.config.scaling_factor,
                            return_dict=False)[0]
            return ((im[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy()
        im, ref_im = image(lat), image(ref_lat)
        psnr = 10 * np.log10(255.0 ** 2 / max(float(((im - ref_im) ** 2).mean()), 1e-12))
        Image.fromarray(im.round().astype(np.uint8)).save(hf.scratch("mirror", f"p{a.prompt}.png"))
        line += f" image PSNR {psnr:.1f} dB"
    np.save(hf.scratch("mirror", f"p{a.prompt}_latent.npy"), lat)
    print(line + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
