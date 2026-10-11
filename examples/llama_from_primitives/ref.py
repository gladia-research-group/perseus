"""The plaintext mirror: HF Llama with the port's approximations patched in (the RMSNorm
Chebyshev seed + Newton, the THOR composite SiLU, the THOR softmax), evaluated in float64 on the
same polynomials and schedules as the ciphertext path. Position 0 (the BOS prefix, computed in
plaintext by the port) stays exact, so ``mirror - exact`` is the approximation error the
encrypted decode starts from.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m examples.llama_from_primitives.ref \
        --configs configs/model/approximation/llama31_8b_n32/configs.json --tokens pool.npy
"""
from __future__ import annotations

import argparse
import contextlib
import math

import numpy as np

from perseus.impl.ops import NumpyOps
from perseus.impl.poly import eval_chebyshev, eval_polynomial_ps, goldschmidt_inv_ndf, inv_sqrt_newton_d2

from .config import LlamaConfigs, load_configs

_ops = NumpyOps()       # its arithmetic is plain operators: it runs on torch tensors as well


def rms_norm_ref(x, cfg, gamma):
    """rms_norm (perseus/impl/norm.py) on the rows of x: c x * inv(c^2 (mean x^2 + eps)), times
    the descaled gamma the next linear carries."""
    c = cfg.center_scale
    xs = c * x
    z = (xs * xs).mean(-1, keepdim=True) + cfg.epsilon * c * c
    init = eval_chebyshev(_ops, z, cfg.cheb_coeffs, cfg.cheb_lo, cfg.cheb_hi)
    inv = inv_sqrt_newton_d2(_ops, z, init, cfg.cheb_nr_iters, 1.0 / cfg.inv_out_scale ** 2)
    return xs * inv * (gamma * cfg.descale)


def silu_ref(x, cfg):
    """activation.gelu's THOR composite with the SiLU coefficients: x (p2(p1(x/xmax)) + 1/2)."""
    t = x / cfg.xmax
    if cfg.thor_p1_cheb:            # the Chebyshev basis the kernel evaluates when present
        p1 = eval_chebyshev(_ops, t, cfg.thor_p1_cheb, cfg.thor_p1_a, cfg.thor_p1_b)
        g = eval_chebyshev(_ops, p1, cfg.thor_p2_cheb, cfg.thor_p2_a, cfg.thor_p2_b)
    else:
        g = eval_polynomial_ps(_ops, eval_polynomial_ps(_ops, t, cfg.thor_p1), cfg.thor_p2)
    return x * (g + 0.5)


def softmax_ref(s, valid, cfg):
    """softmax_thor (perseus/impl/attention.py) over the last axis: s (..., T) scores, valid the
    causal mask; kc (the cache count) is the row's number of valid entries."""
    import torch
    kc = valid.sum(-1, keepdim=True).to(s.dtype)
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    z = eval_chebyshev(_ops, s - mean, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = z * z
    z = torch.where(valid, z / kc, torch.zeros_like(z))
    tot = z.sum(-1, keepdim=True)
    y = goldschmidt_inv_ndf(_ops, z, tot, cfg.init_alpha - cfg.init_beta * tot,
                            cfg.gs_iters_first or cfg.gs_iters_scaled)
    for i in range(cfg.log2delta2):
        y = 2.0 * y
        z = (y * y) * (0.5 * torch.sqrt(kc) * 0.25)
        kcl = kc.long().flatten().tolist()
        r = torch.tensor([cfg.kc_r(i, k) for k in kcl], dtype=s.dtype, device=s.device).view(kc.shape)
        tot = z.sum(-1, keepdim=True)
        F = cfg.refine_alpha[i] * torch.sqrt(r) - cfg.refine_beta[i] * r * tot
        y = goldschmidt_inv_ndf(_ops, z, tot, F, int(cfg.per_step_refine_iters[i]))
    return torch.where(valid, y, torch.zeros_like(y))


@contextlib.contextmanager
def mirrored(model, cfgs: LlamaConfigs, norm=True, silu=True, softmax=True, stats=None):
    """Patch the approximations into `model` (rows >= 1 only; row 0 exact)."""
    import torch
    import transformers.models.llama.modeling_llama as M
    undo = []
    L = model.config.num_hidden_layers

    def keep0(exact, approx):
        out = approx.clone()
        out[..., 0, :] = exact[..., 0, :]
        return out

    if norm:
        from .hf import norm_module, norm_sites
        for site in norm_sites(L):
            mod, cfg = norm_module(model, site), cfgs.norm[site]
            f0 = mod.forward

            def fwd(x, mod=mod, cfg=cfg, f0=f0, site=site):
                xd = x.double()
                y = rms_norm_ref(xd, cfg, mod.weight.double())
                if stats is not None:
                    z = cfg.center_scale ** 2 * ((xd[..., 1:, :] ** 2).mean(-1) + cfg.epsilon)
                    stats.setdefault("norm_oob", 0)
                    stats["norm_oob"] += int(((z < cfg.cheb_lo) | (z > cfg.cheb_hi)).sum())
                return keep0(f0(x).double(), y).to(x.dtype)
            mod.forward = fwd
            undo.append(lambda mod=mod, f0=f0: setattr(mod, "forward", f0))
    if silu:
        for b, layer in enumerate(model.model.layers):
            mlp, cfg = layer.mlp, cfgs.act[b]
            f0 = mlp.forward

            def fwd(x, mlp=mlp, cfg=cfg, f0=f0):
                g = mlp.gate_proj(x).double()
                if stats is not None:
                    stats.setdefault("silu_oob", 0)
                    stats["silu_oob"] += int((g[..., 1:, :].abs() > cfg.xmax).sum())
                a = keep0(torch.nn.functional.silu(g), silu_ref(g, cfg)).to(x.dtype)
                return mlp.down_proj(a * mlp.up_proj(x))
            mlp.forward = fwd
            undo.append(lambda mlp=mlp, f0=f0: setattr(mlp, "forward", f0))
    if softmax:
        orig = M.eager_attention_forward

        def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            k = M.repeat_kv(key, module.num_key_value_groups)
            v = M.repeat_kv(value, module.num_key_value_groups)
            s = torch.matmul(query.double(), k.double().transpose(2, 3)) * scaling
            T = s.shape[-1]
            valid = torch.tril(torch.ones(T, T, dtype=torch.bool, device=s.device))
            p = softmax_ref(s, valid, cfgs.softmax[module.layer_idx])
            p[..., 0, :] = torch.softmax(s[..., 0, :].masked_fill(~valid[0], -1e30), -1)
            out = torch.matmul(p, v.double()).to(query.dtype)
            return out.transpose(1, 2).contiguous(), p
        M.eager_attention_forward = eager
        undo.append(lambda: setattr(M, "eager_attention_forward", orig))
    try:
        yield model
    finally:
        for u in reversed(undo):
            u()


def kl_rows(p_logits, q_logits):
    """KL(p || q) per row of two (T, V) logit arrays."""
    import torch
    lp = torch.log_softmax(p_logits.double(), -1)
    lq = torch.log_softmax(q_logits.double(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def evaluate(model, cfgs, toks, **which):
    """Per position >= 1 of every window: KL(exact || mirror) and top-1 agreement."""
    import torch
    kls, hits, stats = [], [], {}
    with torch.no_grad():
        for s in toks:
            ids = torch.as_tensor(s[None]).to(model.device)
            ex = model(ids).logits[0, 1:]
            with mirrored(model, cfgs, stats=stats, **which):
                ap = model(ids).logits[0, 1:]
            kls.append(kl_rows(ex, ap).cpu().numpy())
            hits.append((ex.argmax(-1) == ap.argmax(-1)).cpu().numpy())
    kls, hits = np.concatenate(kls), np.concatenate(hits)
    return {"median_kl": float(np.median(kls)), "mean_kl": float(kls.mean()), "p99_kl": float(np.quantile(kls, 0.99)),
            "max_kl": float(kls.max()), "top1": float(hits.mean()), "n": int(len(kls)), **stats}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--first", type=int, default=40, help="evaluate windows [first, first+n)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ablate", action="store_true", help="also each approximation alone")
    a = ap.parse_args(argv)
    import os
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    from .hf import load_model
    cfgs = load_configs(a.configs)
    toks = np.load(a.tokens)[a.first:a.first + a.n]
    model = load_model()
    runs = {"all": {}}
    if a.ablate:
        runs.update(norm_only=dict(silu=False, softmax=False), silu_only=dict(norm=False, softmax=False),
                    softmax_only=dict(norm=False, silu=False))
    for k, w in runs.items():
        print(f"[mirror] {k:13s} {evaluate(model, cfgs, toks, **w)}", flush=True)


if __name__ == "__main__":
    main()
