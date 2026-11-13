"""SANA-Sprint's text-to-image sampler encrypted (hf.STEPS): the SANA transformer alone (the client's ciphertexts: the
model input, the carry and the caption rows of the prompt's tokens, exact) or with --text the whole pipeline
(impl_text.SanaTextImpl: the prompt's Gemma-2 embedding rows -> Gemma-2 -> caption -> SANA). On the CPU twin
(perseus.impl.fakecap: CKKS levels, reactive refreshes, the PCMM linears' exact products; --capture writes the
planner's graphs) or planned on the device (planned.SanaPlannedPrefill, examples/llama/planned.py's pipeline; the
plan carries its configs and PCMM calibration). Every stage is gated against the float64 dumps (ref_float.py); the
client decrypts the latent and (--image) decodes the image with DC-AE.

    .venv/bin/python -m perseus.examples.sana.run [--text] --fake --layers 1 [--configs <c>] [--pcmm-calib calib.json]
    .venv/bin/python -m perseus.examples.sana.run [--text] --fake --capture <graph dir>
    JOBS=10 bash python/perseus/examples/sana/make_plan.sh <graph dir> <plan name>
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m perseus.examples.sana.run [--text] --plan bootstrap_placements/<plan> \\
        [--layers <the capture's>] [--image]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import hf
from . import layout as L


def rot_steps(text=False):
    """Every rotation the kernels take (layout.LANES lanes, NCH channels; with the text side, Gemma's layout's)."""
    t = L.LANES
    from . import kernels as KN
    s = {k for k in range(-33, 34)} | KN.dw_rotations()             # lanes: depthwise rows, lane sums
    s |= {m * t for m in range(-127, 128)}                            # channels: in-head, packs, windows
    s |= {t << j for j in range(15)} | {-(t << j) for j in range(15)}
    s |= {1, (1 << 15) - 1}                                           # the PCMM's
    from . import fold as FD
    if FD.FOLD:
        s |= FD.rotations()                                           # the hidden fold's sub-blocks
    if text:
        from . import gemma_fhe as GF
        s |= {m * t for m in range(-255, 256)} | GF.rot_steps()
    return {x for x in s if x % L.N}


def twin(a, W, nl, z, txt):
    """The CPU twin's session and model (SanaImpl, or SanaTextImpl with --text) on the dumps' prompt; with --capture,
    the run's levers and inputs written beside the graphs (make_plan.sh's plan carries them)."""
    from perseus.examples.gpt2 import env
    from perseus.impl import fakecap
    from . import levers
    over = levers.runtime(a.plan, [f"AUTO_BTS_LEVEL={a.twin_limit}", *a.set])
    if a.capture:
        os.makedirs(a.capture, exist_ok=True)
        os.environ["PF_PCMM_STAMPS_OUT"] = os.path.join(a.capture, "pcmm_stamps.json")
        with open(os.path.join(a.capture, "sana_runtime.json"), "w") as f:
            json.dump({k: v for k, v in over.items() if k != "AUTO_BTS_LEVEL"}, f, indent=1, sort_keys=True)
        levers.save_capture_inputs(a.capture, a.configs, a.pcmm_calib_in)
    env.export_env(None, chain="n32", CKKS_COMPLEX="1", **over)
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    _, inf = fakecap.make_session(L.geometry()[0], rot_steps(a.text), limit=a.twin_limit, bts_out=a.twin_bts_out)
    x_in = z["noise"].astype(np.float64) * hf.scm_constants()["c_in"]
    if a.text:
        from . import gemma as GM
        from .impl_text import SanaTextImpl
        n_pub, n_tok = int(z["n_pub"]), int(z["n_tok"])
        ids = z["ids"][:n_pub + n_tok]
        model = SanaTextImpl(inf, W, nl, GM.GemmaWeights(), n_pub=n_pub, prefix_ids=ids[:n_pub], core=fakecap.core,
                             n_layers=a.layers, text_layers=a.text_layers)
        model.set_prompt_ids(x_in, ids[n_pub:], txt[0], z.get("z2"))
    else:
        from .impl import SanaImpl
        model = SanaImpl(inf, W, nl, core=fakecap.core, n_layers=a.layers)
        model.set_prompt(x_in, txt[1:1 + int(z["n_tok"])], txt[0], z.get("z2"))
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
    return model, inf


def stage_check(a, model, z, txt):
    """check(i, slot vectors) -> a line: stage i's outputs against the float model (the caption rows after the text
    stages; per step the patch embedding after embed, the residual after a block, the next model input or the latent
    after the head), "" elsewhere."""
    from . import gemma_fhe as GF
    W = model.W
    n_tok = int(z["n_tok"])
    labels = [lab for lab, _ in model.stages()]
    x_ins = {"": model.x_in, "s2.": z.get("x_in2")}                   # each step's model input (the float one's)
    heads = {"": z.get("x_in2"), "s2.": z["latent"]} if len(hf.STEPS) == 2 else {"": z["latent"]}

    def err(got, want):
        return (f"rel L2 err {np.linalg.norm(got - want) / np.linalg.norm(want):.3e} "
                f"(|got| {np.abs(got).max():.3g}, |want| {np.abs(want).max():.3g})")

    def rows(vals, n, d):
        from . import fold as FD
        if FD.FOLD and n == 3:                                        # the hidden width: folded
            return L.tokens(L.decode(FD.unfold_vals(vals[:FD.FH.n_cts()]), d))
        return L.tokens(L.decode([[np.asarray(c) for c in vals[g * n:(g + 1) * n]] for g in range(L.G)], d))

    def check(i, vals):
        lab = labels[i]
        if lab == "caption":
            return "T rows " + err(GF.decode_sana(vals[2 * L.G:], n_tok, L.D), txt[1:1 + n_tok])
        s = lab[:lab.index(".") + 1] if "." in lab else ""
        base = lab[len(s):]
        if base == "embed" and x_ins.get(s) is not None:
            want = L.tokens(L.to_slots(x_ins[s])) @ W["patch_embed.proj.weight"][:, :, 0, 0].numpy().T \
                + W["patch_embed.proj.bias"].numpy()
            return f"{s}embed " + err(rows(vals, 3, L.D) * model.r, want)
        if base.startswith("blk") and f"{s}block{base[3:]}" in z.files:
            return f"{s}block {base[3:]} " + err(rows(vals, 3, L.D) * model.r, z[f"{s}block{base[3:]}"])
        if base == "head" and heads.get(s) is not None:
            return f"{s}head " + err(rows(vals, 1, 32), L.tokens(L.to_slots(heads[s])))
        return ""
    return check


def finish(a, model, z, out, tag):
    """The client's side: the latent decrypted, against the float one (and the image's PSNR)."""
    import torch
    lat = model.decrypt_latent(out)
    ref = z["latent"].astype(np.float64)
    line = f"latent rel L2 err {np.linalg.norm(lat - ref) / np.linalg.norm(ref):.3e}"
    np.save(hf.scratch("out", f"{tag}_p{a.prompt}_{'gpu' if a.plan else 'twin'}_latent.npy"), lat)
    if a.image:
        from diffusers import AutoencoderDC
        from PIL import Image
        vae = AutoencoderDC.from_pretrained(hf.snapshot(), subfolder="vae").eval()

        def image(lt):
            im = vae.decode(torch.as_tensor(lt[None], dtype=torch.float32) / vae.config.scaling_factor,
                            return_dict=False)[0]
            return ((im[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy()
        im, ref_im = image(lat), image(ref)
        psnr = 10 * np.log10(255.0 ** 2 / max(float(((im - ref_im) ** 2).mean()), 1e-12))
        Image.fromarray(im.round().astype(np.uint8)).save(hf.scratch("out", f"{tag}_p{a.prompt}.png"))
        line += f" image PSNR {psnr:.1f} dB"
    print(f"[{tag}] {line}", flush=True)
    return line


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true", help="the text side encrypted too (Gemma-2 + caption)")
    ap.add_argument("--fake", action="store_true", help="the CPU twin")
    ap.add_argument("--capture", default=None, help="twin: write the planner's graphs here")
    ap.add_argument("--plan", default=None, help="GPU, planned: the bootstrap_placements dir")
    ap.add_argument("--configs", default=None, help="a configs.json (a repo spelling, or fit.py's output under the "
                    "scratch root; default: the plan's, else hf.CONFIGS)")
    ap.add_argument("--dumps", default="ref")
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--layers", type=int, default=None, help="SANA blocks (default all)")
    ap.add_argument("--text-layers", type=int, default=None, help="Gemma layers (default all; fewer: a debug cut)")
    ap.add_argument("--twin-limit", type=int, default=50)
    ap.add_argument("--twin-bts-out", type=int, default=32)
    ap.add_argument("--pcmm-calib", default=None, help="twin: record each linear's max |y| here (the fake is exact)")
    ap.add_argument("--pcmm-calib-in", default=None, help="the band shifts from a recorded calibration (default: the "
                    "plan's, else hf.PCMM_CALIB)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--image", action="store_true", help="decode the image (DC-AE, CPU) and its PSNR vs the float one")
    ap.add_argument("--device", type=int, default=None, help="planned: GPU index (the first of CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--force", action="store_true", help="planned: run on a GPU with other processes")
    ap.add_argument("--wait-free", action="store_true", help="planned: record and compile now, then wait for the GPU "
                    "to have no other process before the setup (back-to-back runs on one GPU)")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--graphs", action="store_true", help="planned: replay each stage as CUDA graphs")
    ap.add_argument("--repeat", type=int, default=2, help="planned: timed forwards (wall = median)")
    a = ap.parse_args(argv)
    if not (a.fake or a.plan):
        print("--fake or --plan", file=sys.stderr)
        return 2
    if a.fake:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the twin alone; planned, the device stays visible
    from . import levers
    levers.plan_inputs(a)                                # the plan's configs and PCMM calibration unless given
    levers.runtime(a.plan, a.set)                        # the PF_* levers before impl.poly / impl.pcmm import
    if a.pcmm_calib:
        os.environ["PF_PCMM_CALIB"] = hf.scratch(a.pcmm_calib)
    if a.pcmm_calib_in:
        os.environ["PF_PCMM_CALIB_IN"] = hf.data(a.pcmm_calib_in)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    if a.plan and not a.compile_only:
        from perseus.examples.gpt2.run_decode import gpu_busy
        busy = gpu_busy(a.device)
        if busy and not (a.force or a.wait_free):
            print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
            return 2
    from . import ref as R
    from .nl import PolyNL
    tag = "t2i" if a.text else "sana"
    W = R.SanaWeights()
    nl = PolyNL(json.load(open(hf.data(a.configs))))
    z = np.load(hf.scratch(a.dumps, f"p{a.prompt}.npz"))
    sel = np.array([0] + list(range(len(z["ids"]) - hf.MAX_SEQ + 1, len(z["ids"]))))
    enc = torch.as_tensor(z["mask"][sel]).bool().clone()
    enc[0] = False
    txt = R.text_states(W, torch.as_tensor(z["gemma_out_sel"]).double(), enc, R.Approx()).numpy()
    model, inf = twin(a, W, nl, z, txt)
    check = stage_check(a, model, z, txt)
    full = model.n_layers == W.n_layers
    if a.plan:
        from .planned import device_run, summary
        res = device_run(a, model, check, tag)
        if res is None:
            return 0
        line = finish(a, model, z, res["out"], tag) if full else res["lines"].get(res["run"].n_stages - 1, "")
        summary(tag, res, line)
        res["run"].close()
        return 0
    ops = model.rt.ops
    t0 = [time.time(), int(inf.fhe.total_bootstraps)]

    def on_stage(i, state):
        bts, dt = int(inf.fhe.total_bootstraps), time.time() - t0[0]
        line = check(i, [ops.decrypt_slots_complex(x) for x in state])
        print(f"[{tag}] stage {i} {dt:.0f}s bts {bts - t0[1]}" + (f" {line}" if line else ""), flush=True)
        t0[0], t0[1] = time.time(), bts
        return state
    t_all = time.time()
    xs, _ch, _rec = model.prefill(on_block=on_stage)
    print(f"[{tag}] forward {time.time() - t_all:.0f}s, {int(inf.fhe.total_bootstraps)} bootstraps", flush=True)
    if a.capture:                          # the linears' stamps for the planner (PLAN_PCMM_STAMPS)
        from perseus.impl.pcmm import session_specs
        session_specs(model.lin.linears().values(), L.LANES)
    if full:
        finish(a, model, z, xs, tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
