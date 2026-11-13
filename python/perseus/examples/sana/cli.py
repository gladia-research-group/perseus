"""perseus-t2i: encrypted text-to-image from the command line.

The client tokenizes and embeds the prompt, draws the noise from the seed and encrypts both; the server runs the whole
pipeline under CKKS on the GPU (Gemma-2-2B text encoder -> caption -> SANA-Sprint 0.6B, 2 steps, 768 x 768: a planned
run, examples/sana/planned.py); the client decrypts the latent and decodes the image with DC-AE. --float runs the
plaintext pipeline instead: a preview in seconds, any step count.

    python -m perseus.t2i "a photo of a dog" --seed 3                 # one image (PNG under ./t2i_out)
    python -m perseus.t2i -i                                          # interactive: warm up once, then prompts
    python -m perseus.t2i "a photo of a dog" --compare                # + the plaintext image, side by side, PSNR
    python -m perseus.t2i "a red fox in the snow at dawn" --float --steps 4
    python -m perseus.t2i --list                                      # the plans

What a plan fixes. An encrypted run executes a compiled plan, cut for one prompt length (the encrypted Gemma attention
and SANA's cross-attention loop over the prompt's tokens) and one step count. A prompt of another length needs its
own plan, built once (~2 h of CPU). The warm-up happens once per process (-i serves many prompts from one warm-up,
~5 min each): the op stream recorded on the CPU twin (the PCMM weights encoded on worker threads beside it) and saved
under $SANA_SCRATCH/cli_record/<plan>, which later launches load while the code, switches and inputs match
(--no-record-cache records afresh); the programs, loaded from the plan or compiled, then kept in it; the session.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REGISTRY = "t2i_plans.json"         # in the plans dir: {"default": name, "plans": {name: {s_per_image, psnr_db, ...}}}


# ── plans ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def plans_dir() -> Path:
    from perseus._paths import ASSETS
    return Path(os.environ.get("PERSEUS_T2I_PLANS") or ASSETS / "bootstrap_placements")


def registry() -> dict:
    p = plans_dir() / REGISTRY
    return json.loads(p.read_text()) if p.exists() else {}


def plan_info(d: Path, reg: dict) -> dict | None:
    """A plan dir's run settings: the SANA_* switches its NOTE.txt names (read at import: exported before any sana
    module loads), the dumps its capture ran on (prompt 0: the recorded prompt), steps and resolution; the registry's
    measurements and prompt length on top."""
    note = d / "NOTE.txt"
    if not note.exists() or not (d / "sana_runtime.json").exists():
        return None
    txt = note.read_text()
    env = dict(re.findall(r"\b(SANA_[A-Z0-9_]+)=([^\s()]+)", txt))
    m = re.search(r"dumps (\w+)", txt)
    info = dict(name=d.name, path=str(d), env=env, dumps=m.group(1) if m else "ref768", prompt=0,
                steps=int(env.get("SANA_STEPS", "2")), res=int(env.get("SANA_RES", "768")),
                programs=(d / "programs").is_dir())
    info.update(reg.get("plans", {}).get(d.name, {}))
    return info


def all_plans() -> list[dict]:
    reg = registry()
    out = [plan_info(d, reg) for d in sorted(plans_dir().glob("sana_t2i*")) if d.is_dir()]
    return [p for p in out if p is not None]


def pick_plan(name, steps, res) -> dict:
    plans = all_plans()
    if name:
        for p in plans:
            if p["name"] == name or p["path"] == name or Path(name).name == p["name"]:
                return p
        raise SystemExit(f"perseus-t2i: no plan {name!r} (--list shows them)")
    ok = [p for p in plans if p.get("validated") and p["steps"] == steps and p["res"] == res]
    if not ok:
        have = sorted({(p["steps"], p["res"]) for p in plans if p.get("validated")})
        raise SystemExit(f"perseus-t2i: no validated plan for {steps} step(s) at {res} px (have: "
                         f"{', '.join(f'{s} step(s) at {r} px' for s, r in have) or 'none'}). The model runs 1 or 2 "
                         f"steps encrypted; a new plan takes ~2 h of CPU once. --float previews any step count.")
    default = registry().get("default")
    for p in ok:
        if p["name"] == default:
            return p
    return ok[-1]


def list_plans():
    reg = registry()
    default = reg.get("default")
    plans = [p for p in all_plans() if p["name"] in reg.get("plans", {})]
    rows = [("", "plan", "steps", "px", "prompt tok", "s/image", "PSNR dB", "programs", "note")]
    for p in plans:
        rows.append(("*" if p["name"] == default else ("" if p.get("validated") else "?"), p["name"], str(p["steps"]),
                     str(p["res"]), str(p.get("n_tok", "?")),
                     f"{p['s_per_image']:.1f}" if "s_per_image" in p else "-",
                     f"{p['psnr_db']:.1f}" if "psnr_db" in p else "-",
                     "saved" if p["programs"] else "compile", p.get("note", "")))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip())
    print("\n* the default   ? not validated on the device (--plan NAME runs it anyway)\n"
          "s/image and PSNR: the plan's timed run on its own prompt (PSNR of the image against the float64 pipeline's)")


# ── terminal ──────────────────────────────────────────────────────────────────────────────────────────────────────

class Term:
    """Progress bars on the terminal while everything else the run prints (Python, Rust, CUDA: fds 1 and 2) goes
    to the log."""

    def __init__(self, log_path: Path):
        self.tty = os.fdopen(os.dup(2), "w", buffering=1)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a", buffering=1)
        self.log_path = log_path

    def say(self, msg=""):
        print(msg, file=self.tty, flush=True)

    def bar(self, desc, total, unit="stage"):
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit=unit, file=self.tty, dynamic_ncols=True, leave=True,
                    bar_format="{desc:<24} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}s "
                               "[{elapsed}<{remaining}{postfix}]")

    @contextlib.contextmanager
    def quiet(self):
        sys.stdout.flush()
        sys.stderr.flush()
        s1, s2 = os.dup(1), os.dup(2)
        os.dup2(self.log.fileno(), 1)
        os.dup2(self.log.fileno(), 2)
        try:
            with contextlib.redirect_stdout(self.log), contextlib.redirect_stderr(self.log):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(s1, 1)
            os.dup2(s2, 2)
            os.close(s1)
            os.close(s2)


# ── the client's side ─────────────────────────────────────────────────────────────────────────────────────────────

def noises(seed: int, res: int):
    """The model input's noise (seed) and step 2's renoise draw (seed + 100): ref_float's convention (prompt i of the
    dumps uses seed i), so `--seed i` on dump prompt i reproduces its reference."""
    import torch
    g = res // 32
    n1 = torch.randn((1, 32, g, g), generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)
    n2 = torch.randn((1, 32, g, g), generator=torch.Generator("cpu").manual_seed(100 + seed), dtype=torch.float32)
    return n1, n2


def to_png(arr, path: Path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.round().clip(0, 255).astype("uint8")).save(path)
    return path


def psnr(a, b) -> float:
    import numpy as np
    return 10 * math.log10(255.0 ** 2 / max(float(((a - b) ** 2).mean()), 1e-12))


class Decoder:
    """DC-AE on the CPU: latent (32, g, g) -> (H, W, 3) floats in [0, 255]."""

    def __init__(self):
        from diffusers import AutoencoderDC
        from . import hf
        self.vae = AutoencoderDC.from_pretrained(hf.snapshot(), subfolder="vae").eval()

    def __call__(self, lat):
        import torch
        with torch.no_grad():
            im = self.vae.decode(torch.as_tensor(lat[None], dtype=torch.float32) / self.vae.config.scaling_factor,
                                 return_dict=False)[0]
        return ((im[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy()


class FloatPipe:
    """The plaintext SANA-Sprint pipeline (diffusers, fp32), on the same noises as the encrypted run."""

    def __init__(self, device="cpu"):
        from . import hf
        import torch
        torch.set_num_threads(int(os.environ.get("PERSEUS_T2I_THREADS", "8")))
        self.pipe = hf.load_pipeline("float32").to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device

    def __call__(self, prompt, seed, steps, res):
        import numpy as np
        import torch
        n1, _ = noises(seed, res)
        kw = {} if steps == 2 else {"intermediate_timesteps": None}
        with torch.no_grad():
            img = self.pipe(prompt, num_inference_steps=steps, latents=n1.to(self.device), height=res, width=res,
                            generator=torch.Generator("cpu").manual_seed(100 + seed), use_resolution_binning=False,
                            output_type="np", **kw).images[0]
        return np.asarray(img, dtype=np.float64) * 255.0


# ── the encrypted run ─────────────────────────────────────────────────────────────────────────────────────────────

class PromptLengthError(ValueError):
    pass


class Encrypted:
    """One plan's planned session: warm up once (record on the CPU twin, programs, session), then `generate` any
    prompt of the plan's length."""

    def __init__(self, plan: dict, device: int, work_dir: Path, term: Term, wait: bool, save_programs: bool,
                 record_cache: Path | None = None):
        self.plan, self.device, self.work, self.term = plan, device, work_dir, term
        self.wait, self.save_programs, self.record_cache = wait, save_programs, record_cache

    def _args(self):
        from types import SimpleNamespace
        return SimpleNamespace(text=True, fake=False, capture=None, plan=self.plan["path"], configs=None,
                               dumps=self.plan["dumps"], prompt=self.plan.get("prompt", 0), layers=None,
                               text_layers=None, twin_limit=50, twin_bts_out=32, pcmm_calib=None, pcmm_calib_in=None,
                               set=[], image=False, device=self.device, force=False, wait_free=False,
                               work_dir=str(self.work), compile_only=False, key_layout="packed", graphs=False,
                               repeat=1)

    def warm_up(self):
        """prepare_cpu + open_gpu."""
        return self.prepare_cpu().open_gpu()

    def prepare_cpu(self):
        """The plan's model on the CPU twin, the op stream recorded (the PCMM weights encoded on worker threads
        beside it), or loaded from the record cache when the code, switches and inputs match; the programs loaded or
        compiled."""
        import numpy as np
        import torch
        T = self.term
        a = self._args()
        with T.quiet():
            from . import hf, levers
            from . import run as RUN
            levers.plan_inputs(a)                       # the plan's configs and PCMM calibration
            levers.runtime(a.plan, a.set)               # its PF_* levers, before impl.poly / impl.pcmm import
            if a.pcmm_calib_in:
                os.environ["PF_PCMM_CALIB_IN"] = hf.data(a.pcmm_calib_in)
            from perseus import _pc
            from perseus.examples.bert.planned import plan_config
            from . import ref as R
            from .nl import PolyNL
            from .planned import SanaPlannedPrefill
            torch.set_grad_enabled(False)
            torch.set_num_threads(4)
            t0 = time.time()
            W = R.SanaWeights()
            nl = PolyNL(json.load(open(hf.data(a.configs))))
            z = np.load(hf.scratch(a.dumps, f"p{a.prompt}.npz"))
            sel = np.array([0] + list(range(len(z["ids"]) - hf.MAX_SEQ + 1, len(z["ids"]))))
            enc = torch.as_tensor(z["mask"][sel]).bool().clone()
            enc[0] = False
            txt = R.text_states(W, torch.as_tensor(z["gemma_out_sel"]).double(), enc, R.Approx()).numpy()
            model, _inf = RUN.twin(a, W, nl, z, txt)
            self.hf, self.model = hf, model
            self.n_tok, self.n_pub = int(z["n_tok"]), int(z["n_pub"])
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(hf.snapshot(), subfolder="tokenizer")
            keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
            run = SanaPlannedPrefill(model, a.plan, plan_config(a.plan, False), keys, work_dir=str(self.work),
                                     log=lambda m: print(m, flush=True))
            run.save_programs = self.save_programs
        T.say(f"  model loaded ({time.time() - t0:.0f} s)")
        bars = {}

        def tick(phase, i, n):
            if phase not in bars:
                bars[phase] = T.bar({"record": "recording (CPU twin)", "compile": "programs"}[phase], n)
            b = bars[phase]
            b.update(i - b.n)
            if i == n:
                b.close()
        run.progress = tick
        t0 = time.time()
        if self.record_cache is None:
            with T.quiet():
                run.record()
        else:
            files = [hf.data(a.configs), hf.scratch(a.dumps, f"p{a.prompt}.npz")]
            if os.environ.get("PF_PCMM_CALIB_IN"):
                files.append(os.environ["PF_PCMM_CALIB_IN"])
            self.record_files = files
            with T.quiet():
                loaded = run.record_cached(str(self.record_cache), files)
            if loaded:
                T.say(f"  recording loaded from {self.record_cache} ({time.time() - t0:.0f} s; recorded in "
                      f"{run.recorded_s:.0f} s)")
            else:
                T.say(f"  recorded in {run.record_s:.0f} s; saved to {self.record_cache} ({run.save_s:.0f} s) for "
                      f"the next launch")
        with T.quiet():
            run.prepare()
        loaded = getattr(run, "n_loaded", 0)
        T.say(f"  {run.n_stages} programs: {loaded} loaded from the plan, {run.n_stages - loaded} compiled"
              + (" (saved to the plan for the next run)" if self.save_programs and loaded < run.n_stages else ""))
        self.run = run
        return self

    def open_gpu(self):
        """The planned session on the GPU (keys, slab, registers, PCMM service; ~1 min)."""
        T, run = self.term, self.run
        from perseus.examples.gpt2.run_decode import gpu_busy
        busy = gpu_busy(self.device)
        if busy:
            if not self.wait:
                raise SystemExit(f"perseus-t2i: GPU {self.device} has other processes ({'; '.join(busy)}); "
                                 f"--wait waits for it, --device picks another")
            T.say(f"  GPU {self.device} busy: waiting for it to be free ...")
            while gpu_busy(self.device):
                time.sleep(30)
        t0 = time.time()
        with T.quiet():
            run.setup()
            run.input_values()                          # the packing reproduces the recorded inputs (once)
        T.say(f"  session ready on GPU {self.device} ({time.time() - t0:.0f} s; "
              f"{run.fhe.mem_info()[0] / 2 ** 30:.1f} GB free)")
        self.ready = True
        return self

    def check(self, prompt: str):
        lay = self.hf.prompt_layout(self.tok, prompt)
        if lay.n_tok != self.n_tok or lay.n_pub != self.n_pub:
            raise PromptLengthError(
                f"this plan is cut for {self.n_tok}-token prompts; {prompt!r} has {lay.n_tok} "
                f"({self.tokens(lay)}). Rephrase it to {self.n_tok} tokens, or build a plan for {lay.n_tok}.")
        return lay

    def tokens(self, lay):
        ids = lay.ids[lay.n_pub:lay.n_pub + lay.n_tok]
        return " | ".join(self.tok.decode([int(t)]) for t in ids)

    def generate(self, prompt: str, seed: int):
        """The encrypted image's latent (32, g, g) and the server's seconds."""
        import numpy as np
        import torch
        from . import gemma as GM
        lay = self.check(prompt)
        m, run, T = self.model, self.run, self.term
        ids = torch.as_tensor(lay.ids[lay.n_pub:lay.n_pub + lay.n_tok])
        rows = (m.Wg["embed_tokens.weight"][ids] * math.sqrt(GM.D)).numpy() / m.r_text
        n1, n2 = noises(seed, self.hf.RES)
        x_in = n1[0].numpy().astype(np.float64) * self.hf.scm_constants()["c_in"]
        own = m.emb_rows
        m.emb_rows = rows
        try:
            vals = m.input_values(x_in, None, n2[0].numpy())
        finally:
            m.emb_rows = own
        bar = T.bar("encrypted forward", run.n_stages)
        t0 = time.time()
        with T.quiet():
            regs = run.run(vals=vals, on_block=lambda b, _i: bar.update(1))
            bar.set_postfix_str("draining the GPU")
            run.fhe.device_sync()
        bar.set_postfix_str("")
        bar.close()
        secs = time.time() - t0
        with T.quiet():
            lat = m.decrypt_latent(run.decrypt(regs))
        return lat, secs

    def close(self):
        if getattr(self, "ready", False):
            with contextlib.suppress(Exception):
                self.run.close()


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def slug(prompt: str, seed: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:48] or "image"
    return f"{s}_s{seed}"


def out_path(out: str, prompt: str, seed: int, suffix="") -> Path:
    p = Path(out)
    if p.suffix.lower() == ".png":
        return p.with_name(p.stem + suffix + p.suffix)
    return p / f"{slug(prompt, seed)}{suffix}.png"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="perseus-t2i", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    ap.add_argument("prompt", nargs="*", help="the prompt (quote it); several words are joined")
    ap.add_argument("-i", "--interactive", action="store_true", help="warm up once, then read prompts (':seed N', "
                    "':quit')")
    ap.add_argument("--seed", type=int, default=0, help="the noise seed (default 0)")
    ap.add_argument("--steps", type=int, default=2, help="sampling steps: 1 or 2 encrypted (a plan for each), any "
                    "with --float (default 2)")
    ap.add_argument("--res", type=int, default=768, help="image side in pixels (default 768; the plans' own)")
    ap.add_argument("--out", default="t2i_out", help="output dir, or a .png path (default ./t2i_out)")
    ap.add_argument("--float", dest="float_only", action="store_true", help="the plaintext pipeline only")
    ap.add_argument("--compare", action="store_true", help="also the plaintext image on the same noise: saved, "
                    "side by side, with the PSNR")
    ap.add_argument("--float-device", default="cpu", help="where --float / --compare run (cpu, or cuda)")
    ap.add_argument("--plan", default=None, help="a plan name (--list); default: the validated default")
    ap.add_argument("--device", type=int, default=None, help="GPU index (default: the first of "
                    "CUDA_VISIBLE_DEVICES, else 0)")
    ap.add_argument("--wait", action="store_true", help="wait for the GPU to be free instead of refusing")
    ap.add_argument("--no-save-programs", action="store_true", help="do not keep compiled programs in the plan")
    ap.add_argument("--no-record-cache", action="store_true", help="record on the CPU twin even when a saved record "
                    "matches, and do not save one")
    ap.add_argument("--list", action="store_true", help="the plans and their measurements")
    a = ap.parse_args(argv)

    if a.list:
        list_plans()
        return 0
    prompt = " ".join(a.prompt).strip()
    if not prompt and not a.interactive:
        ap.error("a prompt, or -i")

    # before numpy / torch / any sana module: THP off (the twin's record), quiet hubs, the plan's switches
    with contextlib.suppress(Exception):
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)     # PR_SET_THP_DISABLE
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ["SANA_RES"] = str(a.res)
    plan = None if a.float_only else pick_plan(a.plan, a.steps, a.res)
    if plan is not None:
        os.environ.update(plan["env"])
        os.environ.update(plan.get("runtime_env", {}))      # the registry's run-time switches (pinned staging, finish groups)
        if a.device is None:
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            a.device = int(vis.split(",")[0]) if vis.strip() else 0
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.device))
    import warnings
    warnings.filterwarnings("ignore")

    out_dir = Path(a.out) if Path(a.out).suffix.lower() != ".png" else Path(a.out).parent
    term = Term(out_dir / "perseus-t2i.log")
    T = term
    from . import hf                                     # after SANA_RES

    float_pipe = None

    def get_float():
        nonlocal float_pipe
        if float_pipe is None:
            T.say(f"  loading the plaintext pipeline ({a.float_device}) ...")
            with T.quiet():
                float_pipe = FloatPipe(a.float_device)
        return float_pipe

    if a.float_only:
        def one(p, seed):
            t0 = time.time()
            with T.quiet():
                im = get_float()(p, seed, a.steps, a.res)
            path = to_png(im, out_path(a.out, p, seed, "_float"))
            T.say(f"  plaintext image ({a.steps} step(s), {time.time() - t0:.0f} s): {path}")
    else:
        T.say(f"perseus-t2i: plan {plan['name']} ({plan['steps']} steps, {plan['res']} px"
              + (f", {plan['s_per_image']:.0f} s/image, {plan['psnr_db']:.1f} dB on its prompt" if "s_per_image" in plan
                 else "") + f"); log {term.log_path}")
        if prompt and "n_tok" in plan:                    # a prompt of another length: refused before the warm-up
            from transformers import AutoTokenizer
            with T.quiet():
                tok = AutoTokenizer.from_pretrained(hf.snapshot(), subfolder="tokenizer")
            lay = hf.prompt_layout(tok, prompt)
            if lay.n_tok != plan["n_tok"]:
                ids = lay.ids[lay.n_pub:lay.n_pub + lay.n_tok]
                T.say(f"  ! this plan is cut for {plan['n_tok']}-token prompts; {prompt!r} has {lay.n_tok} ("
                      + " | ".join(tok.decode([int(t)]) for t in ids) + f"). Rephrase it to {plan['n_tok']} tokens.")
                return 2
        work = Path(hf.scratch("cli_work", plan["name"]))
        cache = None if a.no_record_cache else Path(hf.scratch("cli_record", plan["name"]))
        enc = Encrypted(plan, a.device, work, T, a.wait, not a.no_save_programs, cache)
        T.say("warming up (once per process): ...")
        enc.warm_up()
        T.say(f"  prompts of {enc.n_tok} tokens; decoder loading ...")
        with T.quiet():
            dec = Decoder()

        def one(p, seed):
            try:
                lay = enc.check(p)
            except PromptLengthError as e:
                T.say(f"  ! {e}")
                return
            T.say(f"\n{p!r}  seed {seed}  [{enc.tokens(lay)}]")
            lat, secs = enc.generate(p, seed)
            t0 = time.time()
            with T.quiet():
                im = dec(lat)
            path = to_png(im, out_path(a.out, p, seed))
            T.say(f"  encrypted image: {path}  (server {secs:.0f} s, decode {time.time() - t0:.0f} s)")
            if a.compare:
                t0 = time.time()
                with T.quiet():
                    ref = get_float()(p, seed, plan["steps"], a.res)
                import numpy as np
                fp = to_png(ref, out_path(a.out, p, seed, "_float"))
                cp = to_png(np.concatenate([im, np.full((im.shape[0], 16, 3), 255.0), ref], axis=1),
                            out_path(a.out, p, seed, "_compare"))
                T.say(f"  plaintext image: {fp} ({time.time() - t0:.0f} s); side by side: {cp}; "
                      f"PSNR {psnr(im, ref):.1f} dB")

    try:
        if prompt:
            one(prompt, a.seed)
        if a.interactive:
            seed = a.seed
            T.say("\ninteractive: a prompt per line; ':seed N' sets the seed, ':quit' (or Ctrl-D) ends")
            while True:
                try:
                    line = input("prompt> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                if line in (":q", ":quit", ":exit"):
                    break
                m = re.fullmatch(r":seed\s+(-?\d+)", line)
                if m:
                    seed = int(m.group(1))
                    T.say(f"  seed {seed}")
                    continue
                one(line, seed)
    except KeyboardInterrupt:
        T.say("\ninterrupted")
        return 130
    finally:
        if not a.float_only:
            enc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
