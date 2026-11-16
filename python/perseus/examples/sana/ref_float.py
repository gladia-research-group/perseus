"""Float64 reference dumps of the SANA-Sprint pipeline (CPU; hf.STEPS, the default 2 steps): per prompt, the Gemma-2
hidden states at the encrypted prompt positions, per step the transformer's input, per-block outputs (step 2's keyed
"s2.") and output F, the final denoised latent and the decoded image. The fp32 pipeline (output_type latent, the same
noises) checks the reproduction of the sampling arithmetic.

    .venv/bin/python -m perseus.examples.sana.ref_float --out ref --blocks 2
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import hf

PROMPTS = [
    "a photo of a cat",
    "a red apple and a green bowl on a wooden table",
    "a photo of three dogs",
    "an astronaut riding a horse on the moon, digital art",
    "a blue bicycle leaning against a yellow wall",
    "a bowl of ramen with chopsticks, studio lighting",
    "a lighthouse on a cliff at sunset",
    "a photo of a toaster and a teddy bear",
]


def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def gemma_states(te, lay, torch):
    """(27, n_tok, 2304) float64: the scaled embeddings and each layer's output at the prompt positions; plus the
    final-norm output at every position."""
    keep = lay.prompt_pos
    caps = []
    hooks = [te.layers[0].register_forward_pre_hook(
        lambda m, a, kw: caps.append(_first(kw.get("hidden_states", a[0] if a else None))[0, keep].clone()),
        with_kwargs=True)]
    hooks += [l.register_forward_hook(lambda m, a, o: caps.append(_first(o)[0, keep].clone())) for l in te.layers]
    ids = torch.from_numpy(lay.ids[None]).long()
    mask = torch.from_numpy(lay.mask[None]).long()
    out = te(ids, attention_mask=mask).last_hidden_state[0]
    for h in hooks:
        h.remove()
    return torch.stack(caps).numpy(), out.numpy()


def transformer_states(tr, x_in, emb, emask, g, s, torch, blocks: bool):
    caps = {}
    hooks = []
    if blocks:
        def keep(name):
            return lambda m, a, o: caps.__setitem__(name, _first(o)[0].clone().numpy())
        hooks.append(tr.patch_embed.register_forward_hook(keep("patch_embed")))
        hooks.append(tr.caption_projection.register_forward_hook(keep("caption_proj")))
        hooks.append(tr.caption_norm.register_forward_hook(keep("caption_norm")))
        def temb(m, a, o):
            caps["t_mod"], caps["t_emb"] = o[0][0].clone().numpy(), o[1][0].clone().numpy()
        hooks.append(tr.time_embed.register_forward_hook(temb))
        for i, b in enumerate(tr.transformer_blocks):
            hooks.append(b.register_forward_hook(keep(f"block{i}")))
        hooks.append(tr.norm_out.register_forward_hook(keep("norm_out")))
    F = tr(x_in, encoder_hidden_states=emb, encoder_attention_mask=emask, guidance=g, timestep=s,
           return_dict=False)[0]
    for h in hooks:
        h.remove()
    return F, caps


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ref")
    ap.add_argument("--n", type=int, default=len(PROMPTS))
    ap.add_argument("--blocks", type=int, default=2, help="per-block dumps for the first N prompts")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--steps", type=int, default=len(hf.STEPS), choices=(1, 2))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)

    pipe = hf.load_pipeline("float32")
    te64 = pipe.text_encoder.to(torch.float64)
    tr64 = pipe.transformer.to(torch.float64)
    k = hf.scm_constants()
    gs = hf.GUIDANCE * tr64.config.guidance_embeds_scale
    steps = hf.STEPS[:a.steps]
    meta = dict(name=hf.NAME, revision=hf.REVISION, scm=k, guidance=gs, prompts=PROMPTS[:a.n], steps=list(steps))
    with open(hf.scratch(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    for i, prompt in enumerate(PROMPTS[:a.n]):
        t0 = time.time()
        lay = hf.prompt_layout(pipe.tokenizer, prompt)
        g_states, g_out = gemma_states(te64, lay, torch)
        sel = lay.select
        emb = torch.from_numpy(g_out[sel][None])
        emask = torch.from_numpy(lay.mask[sel][None]).long()
        gen = torch.Generator("cpu").manual_seed(i)
        g = hf.RES // 32
        noise = torch.randn((1, 32, g, g), generator=gen, dtype=torch.float32)
        x_in = noise.double() * k["c_in"]
        F, caps = transformer_states(tr64, x_in, emb, emask, torch.tensor([gs], dtype=torch.float64),
                                     torch.tensor([k["s"]], dtype=torch.float64), torch, i < a.blocks)
        lat = k["a"] * x_in + k["b"] * F
        more = {}
        if len(steps) == 2:          # step 2 from the renoised step-1 output (the pipeline's generator draws z2)
            gen2 = torch.Generator("cpu").manual_seed(100 + i)
            z2 = torch.randn((1, 32, g, g), generator=gen2, dtype=torch.float32)
            rn, k2 = hf.renoise(steps[1], steps[0]), hf.scm_constants(steps[1])
            x_in2 = rn["p"] * x_in + rn["q"] * F + rn["r"] * z2.double()
            F2, caps2 = transformer_states(tr64, x_in2, emb, emask, torch.tensor([gs], dtype=torch.float64),
                                           torch.tensor([k2["s"]], dtype=torch.float64), torch, i < a.blocks)
            lat = k2["a"] * x_in2 + k2["b"] * F2
            more = dict(z2=z2[0].numpy(), x_in2=x_in2[0].numpy(), F2=F2[0].numpy(),
                        **{f"s2.{k_}": v for k_, v in caps2.items()})
        t1 = time.time()

        # the fp32 pipeline on the same prompt and noises
        te64.float(); tr64.float()
        p32 = pipe(prompt, num_inference_steps=len(steps), latents=noise, output_type="latent", height=hf.RES,
                   width=hf.RES,
                   generator=torch.Generator("cpu").manual_seed(100 + i), use_resolution_binning=False,
                   **({} if len(steps) == 2 else {"intermediate_timesteps": None})).images
        te64.double(); tr64.double()
        err = float((p32.double() - lat).abs().max() / lat.abs().max())

        np.savez(hf.scratch(a.out, f"p{i}.npz"), ids=lay.ids, mask=lay.mask, n_chi=lay.n_chi, n_pub=lay.n_pub,
                 n_tok=lay.n_tok,
                 gemma=g_states.astype(np.float32), gemma_out_sel=g_out[sel].astype(np.float32),
                 noise=noise[0].numpy(), F=F[0].numpy(), latent=lat[0].numpy().astype(np.float32),
                 **{k_: v.astype(np.float32) for k_, v in {**caps, **more}.items()})
        if not a.no_image:
            img = pipe.vae.decode(lat.float() / pipe.vae.config.scaling_factor, return_dict=False)[0]
            pipe.image_processor.postprocess(img, output_type="pil")[0].save(hf.scratch(a.out, f"p{i}.png"))
        print(f"p{i} n_pub={lay.n_pub} n_tok={lay.n_tok} "
              f"|F|max={float(F.abs().max()):.3f} fp32-vs-fp64 rel={err:.2e} "
              f"ref {t1 - t0:.0f}s total {time.time() - t0:.0f}s  {prompt!r}", flush=True)


if __name__ == "__main__":
    main()
