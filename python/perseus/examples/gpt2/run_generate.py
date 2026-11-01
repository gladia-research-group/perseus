"""GPT-2 generation with encrypted feedback, planned on the device: the prompt rows run teacher-forced through the
decode programs, then per generated token the encrypted CutMax argmax, the feedback embed (the one-hot z through the
LM head's transpose plus the next position's wpe row, refreshed) and the next forward entered from it: the generated
token never leaves the ciphertext domain. The client decrypts the logits and z of every step only to check them.

    .venv/bin/python -m perseus.examples.gpt2.run_generate --prompt 3 --tokens 4

Gate: at every step CutMax's token equals the argmax of the decrypted logits, and the tokens equal greedy decoding
of the plaintext model (numpy GPT-2 on the same weights) from the same prompt.

GPT2_SAMPLER=<head plan> samples by TFHE instead (tfhe_head.Sampler; a plan whose block 13 is that head, e.g.
bootstrap_placements/gpt2_gen_topk with --stc-cache 64): the gate is valid one-hots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from perseus.impl.driver import PlannedRun, Recorder

from .run_decode import gpu_busy


class GenRecorder(Recorder):
    """`Recorder` that also records the feedback stage (stage n_layers+2) after each token's CutMax: the embed of
    that token's z at the next position. The twin's own next token stays teacher-forced: only the op streams and
    plaintext values are recorded, and those do not depend on the token."""

    def token(self, t, row, argmax=True):
        blocks = super().token(t, row, argmax=True)
        m, n = self.model, self.n_layers
        z = blocks[n + 1].out
        z = list(z) if isinstance(z, (list, tuple)) else [z]
        x = m.feedback(z, t + self.first + 1)
        r = self.blocks[n + 2]
        r.x_in, r.out, r.out_names = z, x, [x.name]
        return self.blocks


class GenerateRun(PlannedRun):
    """`PlannedRun` with the generation stages: CutMax (n_layers+1), the feedback embed (n_layers+2) and block 0
    compiled for a fed-back token (`block_0_feedback`, entered at the feedback's refresh landing). The plan dir is
    overlaid (symlinks) in the work dir, where a plan without a feedback-stage placement gets one (`prepare`)."""

    FB_VARIANT = "block_0_feedback"

    def __init__(self, recorder, plan_dir, config, keys, *, work_dir, **kw):
        n = recorder.n_layers
        labels = [f"block_{b}" for b in range(n + 3)]
        plan_dir = os.path.abspath(str(plan_dir))
        if not os.path.exists(os.path.join(plan_dir, "block_0_feedback_placement.json")):
            raise RuntimeError(f"{plan_dir}: generation needs block_0_feedback_placement.json "
                               "(make_plan.sh <graph> <plan> feedback)")
        overlay = os.path.join(work_dir, "plan")
        os.makedirs(overlay, exist_ok=True)
        for f in os.listdir(plan_dir):
            if not os.path.lexists(os.path.join(overlay, f)):
                os.symlink(os.path.join(plan_dir, f), os.path.join(overlay, f))
        self.source_plan = plan_dir
        super().__init__(recorder, overlay, config, labels, keys, work_dir=work_dir,
                         variants={self.FB_VARIANT: 0}, **kw)

    def prepare(self, rows, argmax=True):
        """`PlannedRun.prepare`, after giving the overlay a feedback-stage placement when the plan has none: no
        planned refreshes, the code's hint fired (CutMax hands z over deep) and its closing refresh."""
        n = self.rec.n_layers
        fb = os.path.join(str(self.plan_dir), f"block_{n + 2}_placement.json")
        if not os.path.lexists(fb):
            self.rec.values = self.debug
            for t in range(min(self.probe, len(rows))):
                self._record(t, rows, argmax)
            with open(os.path.join(self.source_plan, f"block_{n + 1}_placement.json")) as fh:
                cm = json.load(fh)
            hints = [nd["output"] for nd in self.records[self.ref_token][n + 2].nodes if nd["op_type"] == "hint"]
            # the hint's input (CutMax's exit) sits past the StC-first entry: the shipped order refreshes it
            stub = {"version": cm["version"], "rules": cm["rules"], "placements": [], "hint_fire": hints,
                    "hint_decisions_bound": True, "shipped_order": hints, "rescale_after": [], "sparse_slots": {},
                    "correction_factor": {}, "real_route": [], "weight_levels": {}, "mask_levels": {}, "summary": {}}
            if "config" in cm:
                stub["config"] = cm["config"]
            with open(fb, "w") as fh:
                json.dump(stub, fh)
        return super().prepare(rows, argmax)

    tfhe_chunk = 0                         # > 0: the session serves the TFHE sampler (impl/tfhe.py)

    def open_session(self, kw):
        return super().open_session({**kw, "tfhe": self.tfhe_chunk} if self.tfhe_chunk else kw)

    def setup(self, rows, argmax=False):
        super().setup(rows, argmax)
        if self.tfhe_chunk:
            from perseus.impl import tfhe
            tfhe.setup_session(self.sess)
        return self

    def feedback(self, t):
        """Issue the feedback stage on token t's CutMax output; returns the register of position t+1's input."""
        n = self.rec.n_layers
        self._run_block(n + 2, t, self._zero, self.reg[n + 1]["out"][0][1])
        return self.reg[n + 2]["out"][0][1]


def greedy_reference(store, ids, n_tokens):
    """Greedy decoding of the plaintext GPT-2 (float64 numpy, exact GELU / softmax / LayerNorm) from `ids`."""
    from .weights import LNF, WPE, WTE, block_names
    t = store.tensor
    wte, wpe = t(WTE), t(WPE)
    n_layers = sum(1 for k in store.names() if k.endswith(".ln_1.weight"))
    H = 12

    def ln(x, w, b):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5) * w + b

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    ids, out = list(ids), []
    for _ in range(n_tokens):
        T = len(ids)
        x = wte[ids] + wpe[:T]
        for b in range(n_layers):
            nm = block_names(b)
            h = ln(x, t(nm["ln1_w"]), t(nm["ln1_b"]))
            q, k, v = np.split(h @ t(nm["qkv_w"]).T + t(nm["qkv_b"]), 3, axis=-1)
            q, k, v = (a.reshape(T, H, -1).transpose(1, 0, 2) for a in (q, k, v))
            s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1]) + np.triu(np.full((T, T), -np.inf), 1)
            p = np.exp(s - s.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            y = (p @ v).transpose(1, 0, 2).reshape(T, -1)
            x = x + y @ t(nm["out_w"]).T + t(nm["out_b"])
            h = ln(x, t(nm["ln2_w"]), t(nm["ln2_b"]))
            x = x + gelu(h @ t(nm["up_w"]).T + t(nm["up_b"])) @ t(nm["down_w"]).T + t(nm["down_b"])
        logits = ln(x[-1], t(LNF + ".weight"), t(LNF + ".bias")) @ wte.T
        out.append(int(np.argmax(logits)))
        ids.append(out[-1])
    return out


def _sampled_ref(smp, lg, pos):
    """The plaintext sample: the Gumbel-max (the sampler's draw at `pos`) over the bucket maxima in the exact set."""
    from types import SimpleNamespace
    from .tfhe_head import VOCAB, _exact_set
    a = smp.args
    lg = lg[:VOCAB]
    exact = _exact_set(SimpleNamespace(mode=smp.mode, k=a.k, minp=a.minp, topp=a.topp), lg)
    bt = smp.head.buckets(lg)
    g = smp.gumbel(pos)
    best = [(lg[t] + g[j], int(t)) for j, t in enumerate(bt) if int(t) in exact]
    return max(best)[1] if best else -1


def prompt_ids(store, rows):
    """The token ids of embedding rows wte[id] + wpe[pos] (the oracle stores rows, not ids)."""
    from .weights import WPE, WTE
    wte, wpe = store.tensor(WTE), store.tensor(WPE)
    d = wte.shape[1]
    return [int(np.argmin(((wte - (np.asarray(r[:d]) - wpe[p])) ** 2).sum(1))) for p, r in enumerate(rows)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=3, help="prompt length (the oracle's first rows)")
    ap.add_argument("--tokens", type=int, default=4, help="tokens to generate")
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index (default: the first of CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--plan", default=None, help="decode plan dir with block_0_feedback (default: the decode default)")
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS_PATH"))
    ap.add_argument("--configs", default=os.environ.get("CONFIGS_PATH"))
    ap.add_argument("--io-dir", default=os.environ.get("ALL_BLOCKS_IO_DIR"))
    ap.add_argument("--packing", choices=("cachemir", "cachemir_complex"), default="cachemir_complex")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one of the port's session env values (env.ENV), repeatable")
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--pt-rows", type=int, default=28, help="rows per plaintext slot (0 = every prime)")
    ap.add_argument("--stc-cache", type=int, default=8, help="StC-first last-stage cache entries")
    ap.add_argument("--no-graphs", action="store_true", help="replay every block eagerly (exec.use_graphs=0)")
    ap.add_argument("--work-dir", default=None, help="where the reference token's stream and the plan overlay go")
    ap.add_argument("--compile-only", action="store_true", help="record and compile on the CPU, no device")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tfhe-chunk", type=int, default=4096, help="GPT2_SAMPLER: values per TFHE switch pass")
    a = ap.parse_args(argv)
    from perseus.examples.defaults import default_plan
    a.plan = a.plan or default_plan("gpt2_decode")
    if not a.compile_only:
        busy = gpu_busy(a.device)
        if busy and not a.force:
            print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
            return 2

    from perseus import _pc, config as pcfg
    from perseus._paths import resolve
    from perseus.impl import config, fake, fakecap
    from perseus.impl.layout import Dims

    from . import env, oracle_io, weights
    from .model import Gpt2Model
    chain = "n32"
    over = {**env.plan_runtime(a.plan, chain), "GPT2_PACKING": a.packing, **dict(kv.split("=", 1) for kv in a.set)}
    env.export_env(a.device, chain=chain, CKKS_COMPLEX="1", **over)
    if not a.configs:
        a.configs = env.plan_configs(a.plan, chain, "configs/model/approximation/gpt2_base_n32/configs.json")
    # the CPU twin that records the op streams (as run_decode's planned path)
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over}, chain, True)
    from .tfhe_head import sampler_env
    smp = sampler_env()                 # GPT2_SAMPLER: the selection stage samples by TFHE
    d = Dims.gpt2()
    band, _ = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real)
    _, inf = fakecap.make_session(d, set(band.loaded_rot_steps) | fakecap.GPT2_BAND | set(smp.rotations() if smp else ()))
    store = weights.RawStore(a.weights)
    cfgs = config.load_configs(a.configs)
    rmodel = Gpt2Model(inf, store, cfgs, core=fakecap.core, packing=a.packing, kv_offload=False)
    rmodel.sampler = smp
    n = rmodel.n_layers

    # the twin records every position the run reaches with the oracle's rows; only the prompt's are run inputs
    os.environ["ALL_BLOCKS_IO_DIR"] = a.io_dir or ""
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(a.prompt + a.tokens)
    rows = oracle_io.read_teacher_forced_inputs(oracle_io.RunConfig.from_env())[:a.prompt + a.tokens]
    ids = prompt_ids(store, rows[:a.prompt])
    ref = greedy_reference(store, ids, a.tokens) if smp is None else [-1] * a.tokens
    print(f"[generate] prompt ids {ids}; plaintext greedy {ref}", flush=True)

    cfg = pcfg.plan_runtime(a.plan, chain, base=pcfg.preset("n32_gpt2_decode"))
    if a.no_graphs:
        cfg["exec.use_graphs"] = False
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"gen_ref_{os.getpid()}")
    keys = None if a.compile_only else _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    run = GenerateRun(GenRecorder(rmodel, n), resolve(a.plan), cfg, keys, work_dir=work, pt_rows=a.pt_rows,
                      stc_cache=a.stc_cache)
    run.tfhe_chunk = a.tfhe_chunk if smp is not None else 0
    t0 = time.time()
    run.prepare(rows, argmax=True)
    print(f"[generate] compiled {len(run.progs)} programs in {time.time() - t0:.1f}s; refreshes: feedback stage "
          f"{run.progs[run.labels[n + 2]].n_bootstraps}, block_0_feedback {run.progs[run.FB_VARIANT].n_bootstraps} "
          f"(block_0 {run.progs[run.labels[0]].n_bootstraps})", flush=True)
    if a.compile_only:
        return 0
    run.setup(rows, argmax=True)
    dmodel = Gpt2Model(run.fhe, store, cfgs, core=_pc, n_layers=n, packing=a.packing, kv_offload=False)

    t0 = time.time()
    for p in range(a.prompt):
        reg, _ = run.token(p, rows, argmax=True)
    res = []
    for j in range(a.tokens):
        pos = a.prompt - 1 + j
        ts = time.time()
        lg = dmodel.logits([run.ct(reg)])             # decrypting waits for the forward
        th = time.time()
        z = run.argmax(pos)
        zdec = dmodel.decode_z([run.ct(z)])
        head_s = time.time() - th                      # the selection stage (+ z's decryption)
        cm, am = int(np.argmax(zdec)), int(np.argmax(lg))
        r = {"pos": pos, "cutmax": cm, "fhe_argmax": am, "ref": ref[j], "z_mass": float(zdec[cm]),
             "z_off_max": float(np.abs(np.delete(zdec, cm)).max())}
        if smp is not None:             # the sample vs the same Gumbel draw's Gumbel-max over the exact set's buckets
            r["fhe_argmax"] = r["ref"] = _sampled_ref(smp, np.asarray(lg, dtype=np.float64), pos)
            r["ok_onehot"] = r["z_off_max"] < 0.1 and abs(r["z_mass"] - 1) < 0.1
        if j + 1 < a.tokens:
            x = run.feedback(pos)
            reg, _ = run.token(pos + 1, rows, argmax=True, entry=x, variant=run.FB_VARIANT)
        r["step_s"], r["head_s"] = time.time() - ts, head_s
        res.append(r)
        ok = r["cutmax"] == r["fhe_argmax"] == r["ref"]
        print(f"[generate] pos={pos} cutmax={cm} fhe_argmax={am} ref={r['ref']} {'OK' if ok else 'MISS'} "
              f"z_mass={r['z_mass']:.4f} z_off_max={r['z_off_max']:.3e} step={r['step_s']:.2f}s head={head_s:.2f}s", flush=True)
    ok = all(r["cutmax"] == r["fhe_argmax"] == r["ref"] for r in res)
    if smp is not None:                 # sampling: valid one-hots; agreement with the plaintext draw is reported
        ok = all(r["ok_onehot"] for r in res)
        print(f"[generate] {smp.mode} samples == plaintext Gumbel-max on the exact set {sum(r['cutmax'] == r['ref'] for r in res)}"
              f"/{len(res)}; steps median {np.median([r['step_s'] for r in res]):.2f}s"
              f", heads median {np.median([r['head_s'] for r in res]):.2f}s", flush=True)
    print(f"[generate] tokens={[r['cutmax'] for r in res]} ref={ref} wall={time.time() - t0:.0f}s "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    run.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
