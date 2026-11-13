"""The SANA-Sprint transformer as float64 torch functions of its state dict, in the form the encrypted port computes:
the adaLN modulation folded (the timestep and the guidance are public constants of each sampler step), the linear
attention's query RMSNorm dropped (its per-token scale cancels between the numerator and the denominator). Every
nonlinearity goes through an `Approx` (exact here; the port's polynomials in the mirror), which also sees each site's
input (the calibration statistics).

    .venv/bin/python -m perseus.examples.sana.ref --check ref      # vs diffusers' F on the float dumps
"""
from __future__ import annotations

import argparse
import copy
import math

import numpy as np

from . import hf

D, H_SELF, DH_SELF, H_X, DH_X, MLP = 1152, 36, 32, 16, 72, 2880
GRID = hf.RES // 32                 # the latent image's side (DC-AE's 32x)


class Approx:
    """The exact nonlinearities. `see(site, x)`: the input of every site (no-op here)."""
    ln_eps = 1e-6
    qk_eps = 1e-5

    def see(self, site, x):
        pass

    def see_range(self, site, x):
        """A site recorded by its range only (no samples)."""

    def inv_sqrt(self, z, site):
        """1 / sqrt(z), z > 0 (per-token statistics, eps included by the caller)."""
        self.see(site, z)
        return z.rsqrt()

    def relu(self, x, site):
        self.see(site, x)
        return x.clamp_min(0.0)

    def silu(self, x, site):
        self.see(site, x)
        return x * x.sigmoid()

    def tanh(self, x, site):
        self.see(site, x)
        return x.tanh()

    def gelu_tanh(self, x, site):
        import torch
        self.see(site, x)
        return torch.nn.functional.gelu(x, approximate="tanh")

    def softmax(self, s, valid, site):
        """Softmax over the last axis restricted to `valid` (a bool mask broadcast against s)."""
        self.see(site, s[..., valid])
        s = s.masked_fill(~valid, float("-inf"))
        return s.softmax(-1)

    def recip(self, d, site):
        self.see(site, d)
        return 1.0 / d


class SanaWeights:
    """The transformer's float64 tensors and the public conditioning (time, guidance) of the first sampler step; `step`
    the same tensors at another step's."""

    def __init__(self, path=None, guidance=None):
        import torch
        from diffusers import SanaTransformer2DModel
        tr = SanaTransformer2DModel.from_pretrained(path or hf.snapshot(), subfolder="transformer").double().eval()
        g = hf.GUIDANCE * tr.config.guidance_embeds_scale if guidance is None else guidance
        self.mods = []
        with torch.no_grad():
            for t in hf.STEPS:
                t_mod, t_emb = tr.time_embed(torch.tensor([hf.scm_constants(t)["s"]], dtype=torch.float64),
                                             guidance=torch.tensor([g], dtype=torch.float64),
                                             hidden_dtype=torch.float64)
                self.mods.append((t_mod[0].detach(), t_emb[0].detach()))
        self.sd = {n: p.detach() for n, p in tr.state_dict().items()}
        self.t_mod, self.t_emb = self.mods[0]
        self.n_layers = tr.config.num_layers

    def step(self, i):
        """These weights at step i's time (hf.STEPS[i]): the state dict shared."""
        w = copy.copy(self)
        w.t_mod, w.t_emb = self.mods[i]
        return w

    def __getitem__(self, name):
        return self.sd[name]

    def modulation(self, b):
        """(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp), each (D,)."""
        return tuple((self[f"transformer_blocks.{b}.scale_shift_table"] + self.t_mod.reshape(6, D)).unbind(0))

    def out_modulation(self):
        shift, scale = (self["scale_shift_table"] + self.t_emb[None]).unbind(0)
        return shift, scale


def layer_norm(x, ap: Approx, site):
    """LayerNorm without affine: the centred row times inv_sqrt(variance + eps)."""
    c = x - x.mean(-1, keepdim=True)
    return c * ap.inv_sqrt((c * c).mean(-1, keepdim=True) + ap.ln_eps, site)


def rms_scale(x, ap: Approx, site, eps):
    return ap.inv_sqrt((x * x).mean(-1, keepdim=True) + eps, site)


def text_states(W: SanaWeights, emb, enc, ap: Approx):
    """caption_projection (linear, tanh-GELU, linear) and caption_norm on the (n, 2304) Gemma rows; the approximations
    run on the encrypted rows `enc` (the prompt's), the others (BOS: public; padding: masked) exactly."""
    ex = Approx()
    h = emb @ W["caption_projection.linear_1.weight"].T + W["caption_projection.linear_1.bias"]
    h = _rows(h, enc, lambda r: ap.gelu_tanh(r, "caption.act"), lambda r: ex.gelu_tanh(r, None))
    h = h @ W["caption_projection.linear_2.weight"].T + W["caption_projection.linear_2.bias"]
    ap.see("caption.norm.in", h[enc])
    r = _rows(h, enc, lambda r: rms_scale(r, ap, "caption.norm", 1e-5), lambda r: rms_scale(r, ex, None, 1e-5))
    return h * r * W["caption_norm.weight"]


def _rows(x, sel, f_sel, f_rest):
    out = f_rest(x)
    out[sel] = f_sel(x[sel])
    return out


def linear_attention(W: SanaWeights, b, h, ap: Approx):
    """attn1 on the modulated LN rows h (T, D): ReLU linear attention per head, q's RMSNorm dropped."""
    p = f"transformer_blocks.{b}.attn1."
    q = (h @ W[p + "to_q.weight"].T) * W[p + "norm_q.weight"]
    k = h @ W[p + "to_k.weight"].T
    v = h @ W[p + "to_v.weight"].T
    ap.see(f"b{b}.la.k.in", k)
    rk = rms_scale(k, ap, f"b{b}.la.knorm", ap.qk_eps)               # (T, 1)
    fq = ap.relu(q, f"b{b}.la.q")
    fk = ap.relu(k * W[p + "norm_k.weight"], f"b{b}.la.k") * rk     # rk > 0: ReLU before the RMS scale
    T = h.shape[0]
    fq, fk = fq.reshape(T, H_SELF, DH_SELF), fk.reshape(T, H_SELF, DH_SELF)
    vr = v.reshape(T, H_SELF, DH_SELF)
    S = (fk.unsqueeze(-1) * vr.unsqueeze(-2)).sum(0)                  # (H, dk, dv)
    z = fk.sum(0)                                                     # (H, dk)
    num = (fq.unsqueeze(-1) * S).sum(-2)                              # (T, H, dv)
    den = (fq * z).sum(-1, keepdim=True)                              # (T, H, 1)
    # SANA_LA_QNORM's constants (kernels._la_qnorm_map): the head's sum of relu(q), and den over it (a weighted mean
    # of z) on the rows where it is not ~0; ranges only (no samples: the other sites' draws stay as without them)
    hq = fq.sum(-1)
    ap.see_range(f"b{b}.la.hq", hq)
    ap.see_range(f"b{b}.la.zbar", (den[..., 0] / hq.clamp_min(1e-30))[hq > 1e-6 * hq.max()])
    o = (num * ap.recip(den + 1e-15, f"b{b}.la.den")).reshape(T, D)     # den = 0 (no ReLU overlap): num = 0 too
    return o @ W[p + "to_out.0.weight"].T + W[p + "to_out.0.bias"]


def cross_attention(W: SanaWeights, b, x, txt, valid, enc, ap: Approx):
    """attn2: the residual rows x (no norm) attend the text rows txt (n, D); `valid` (n,) the unpadded keys, `enc`
    the encrypted ones (the prompt's)."""
    p = f"transformer_blocks.{b}.attn2."
    q = x @ W[p + "to_q.weight"].T + W[p + "to_q.bias"]
    k = txt @ W[p + "to_k.weight"].T + W[p + "to_k.bias"]
    v = txt @ W[p + "to_v.weight"].T + W[p + "to_v.bias"]
    ap.see(f"b{b}.xa.q.in", q)
    q = q * rms_scale(q, ap, f"b{b}.xa.qnorm", ap.qk_eps) * W[p + "norm_q.weight"]
    ap.see(f"b{b}.xa.k.in", k[enc])
    rk = _rows(k, enc, lambda r: rms_scale(r, ap, f"b{b}.xa.knorm", ap.qk_eps),
               lambda r: rms_scale(r, Approx(), None, ap.qk_eps))
    k = k * rk * W[p + "norm_k.weight"]
    T, n = x.shape[0], txt.shape[0]
    qh = q.reshape(T, H_X, DH_X).transpose(0, 1)                      # (H, T, dh)
    kh = k.reshape(n, H_X, DH_X).transpose(0, 1)
    vh = v.reshape(n, H_X, DH_X).transpose(0, 1)
    s = qh @ kh.transpose(1, 2) / math.sqrt(DH_X)                     # (H, T, n)
    sv = s[..., valid]
    ap.see_range(f"b{b}.xa.sm.c", sv - sv.mean(-1, keepdim=True))     # each row centred (kernels.center_keys)
    P = ap.softmax(s, valid, f"b{b}.xa.sm")
    o = (P @ vh).transpose(0, 1).reshape(T, D)
    return o @ W[p + "to_out.0.weight"].T + W[p + "to_out.0.bias"]


def depthwise3x3(x, w, bias):
    """(T, C) token-major rows on the GRID x GRID image, a 3 x 3 per-channel filter w (C, 1, 3, 3), zero padding."""
    import torch
    C = x.shape[1]
    img = x.T.reshape(1, C, GRID, GRID)
    out = torch.nn.functional.conv2d(img, w, bias, padding=1, groups=C)
    return out.reshape(C, GRID * GRID).T


def glumbconv(W: SanaWeights, b, h, ap: Approx):
    p = f"transformer_blocks.{b}.ff."
    u = h @ W[p + "conv_inverted.weight"][:, :, 0, 0].T + W[p + "conv_inverted.bias"]
    u = ap.silu(u, f"b{b}.ff.act1")
    u = depthwise3x3(u, W[p + "conv_depth.weight"], W[p + "conv_depth.bias"])
    a, g = u[:, :MLP], u[:, MLP:]
    u = a * ap.silu(g, f"b{b}.ff.act2")
    ap.see(f"b{b}.ff.prod", u)
    return u @ W[p + "conv_point.weight"][:, :, 0, 0].T


def block(W: SanaWeights, b, x, txt, valid, enc, ap: Approx):
    sh1, sc1, g1, sh2, sc2, g2 = W.modulation(b)
    ap.see(f"b{b}.ln1.in", x)
    h = layer_norm(x, ap, f"b{b}.ln1") * (1 + sc1) + sh1
    x = x + g1 * linear_attention(W, b, h, ap)
    x = x + cross_attention(W, b, x, txt, valid, enc, ap)
    ap.see(f"b{b}.ln2.in", x)
    h = layer_norm(x, ap, f"b{b}.ln2") * (1 + sc2) + sh2
    return x + g2 * glumbconv(W, b, h, ap)


def forward(W: SanaWeights, x_in, emb, mask, ap: Approx = None, on_block=None):
    """F for the model input x_in (32, GRID, GRID), the selected Gemma rows emb (MAX_SEQ, 2304) and their mask."""
    import torch
    ap = ap or Approx()
    x_in, emb = torch.as_tensor(x_in, dtype=torch.float64), torch.as_tensor(emb, dtype=torch.float64)
    valid = torch.as_tensor(mask).bool()
    enc = valid.clone()
    enc[0] = False                                                    # BOS: public
    x = x_in.reshape(32, GRID * GRID).T @ W["patch_embed.proj.weight"][:, :, 0, 0].T + W["patch_embed.proj.bias"]
    txt = text_states(W, emb, enc, ap)
    for b in range(W.n_layers):
        x = block(W, b, x, txt, valid, enc, ap)
        if on_block is not None:
            on_block(b, x)
    shift, scale = W.out_modulation()
    ap.see("out.ln.in", x)
    h = layer_norm(x, ap, "out.ln") * (1 + scale) + shift
    y = h @ W["proj_out.weight"].T + W["proj_out.bias"]               # (T, 32)
    return y.T.reshape(32, GRID, GRID)


def main():
    import torch
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--check", default="ref", help="the ref_float dump directory under the scratch root")
    ap_.add_argument("--n", type=int, default=2)
    a = ap_.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(24)
    W = SanaWeights()
    k = hf.scm_constants()
    for i in range(a.n):
        z = np.load(hf.scratch(a.check, f"p{i}.npz"))
        sel = np.array([0] + list(range(len(z["ids"]) - hf.MAX_SEQ + 1, len(z["ids"]))))
        blocks = {}
        F = forward(W, z["noise"].astype(np.float64) * k["c_in"], z["gemma_out_sel"].astype(np.float64),
                    z["mask"][sel], on_block=lambda b, x: blocks.__setitem__(b, x))
        ref = torch.as_tensor(z["F"])
        e = float((F - ref).abs().max() / ref.abs().max())
        line = f"p{i}: F rel err {e:.2e}"
        if "block0" in z.files:
            eb = max((blocks[b] - torch.as_tensor(z[f"block{b}"])).abs().max().item()
                     / torch.as_tensor(z[f"block{b}"]).abs().max().item() for b in range(W.n_layers))
            eb = float("nan") if any(blocks[b].isnan().any() for b in blocks) else eb
            line += f", worst block rel err {eb:.2e}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
