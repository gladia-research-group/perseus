"""BERT-base SST-2 encrypted on perseus.impl: client embeddings -> 12 encrypted post-LN blocks (bidirectional
attention, blocked layout, PCMM linears) -> client pooler + classifier, gated against the float64 model.

    .venv/bin/python -m examples.bert_from_primitives.run_bert [--text 128 | --sst2 0,1,2] [--fake] [--layers 12]
        [--capture graphs/X --structure-only] [--plan bootstrap_placements/X] [--check]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

CONFIGS = "configs/model/approximation/bert_base_n32_t128/configs.json"
# the arm's switches (a plan's runtime line carries them; the shell wins)
ARM = {"PF_LAYOUT": "blocked", "PF_LINEAR": "pcmm", "LN_CHEB": "1", "SM_CHEB_GS": "1", "GELU_FOLD": "1",
       "IMPL_SCOPED_NAMES": "1"}


def inputs(a, T):
    """[(name, ids, n real tokens, label)]."""
    from . import data as D
    out = []
    if a.text == "128":
        ids, n = D.text128()
        out.append(("text128", ids, n, 1))
    if a.sst2:
        idx = (list(range(*[int(v) for v in a.sst2.split(":")])) if ":" in a.sst2
               else [int(i) for i in a.sst2.split(",")])
        sents = D.sentences(max(idx) + 1)
        for i in idx:
            s, lab = sents[int(i)]
            ids, n = D.encode(s, T)
            out.append((f"sst2[{i}]", ids, n, lab))
    return out


def main(argv=None):
    from examples.gpt2_from_primitives.run_decode import gpu_busy
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128, help="T: rows padded to it")
    ap.add_argument("--text", default="128", help="'128': the 128-token text; '' for none")
    ap.add_argument("--sst2", default="", help="SST-2 validation indices: comma-separated, or a range a:b")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", action="store_true", help="per-block maxerr against the float64 model")
    ap.add_argument("--fake", action="store_true", help="the numpy fake (perseus.impl.fakecap): no GPU")
    a = ap.parse_args(argv)
    os.environ.setdefault("KV_ARENA_GB", "0")
    busy = None if a.fake else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from examples.gpt2_from_primitives import env
    over = {"FIDESLIB_ENCODE_NOSYNC": "0", **ARM, **env.plan_runtime(a.plan, "n32"), "GPT2_PACKING": "cachemir",
            **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    if a.plan and not os.environ.get("PF_PCMM_CALIB_IN"):
        cal = os.path.join(a.plan, "pcmm_calib.json")
        if os.path.exists(cal):
            os.environ["PF_PCMM_CALIB_IN"] = cal
    over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
            "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    env.export_env(a.device, chain="n32", CKKS_COMPLEX="1", **over)
    if a.fake:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not a.configs:
        a.configs = CONFIGS
        cmd = os.path.join(a.plan, "PLAN_CMD.txt") if a.plan else ""
        if cmd and os.path.exists(cmd):
            for line in open(cmd).read().splitlines():
                if line.startswith("configs="):
                    a.configs = line[len("configs="):].split()[0]
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
    from perseus import _core
    from perseus.impl import config
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.layout import Dims, ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.prefill import diag_rot_steps, prefill_rot_steps
    from . import data as D
    from .model import BertPrefill
    d = Dims.gpt2()
    extra = prefill_rot_steps(d) | diag_rot_steps(d) | pcmm_rot_steps() | \
        blocked_rot_steps(Blocks(d, int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits())
    if os.environ.get("PF_ATTN", "") == "ccmm":
        from perseus.impl.sinc_mha import rot_steps as mha_rot_steps
        extra |= mha_rot_steps() | {16, d.N - 16}
    pruned = None
    if a.fake:
        from perseus.impl import fakecap
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
        rot = extra | fakecap.GPT2_BAND | (ladder_radix_steps(d, ladder_radix_bits()) if ladder_radix_bits() >= 2
                                           else set())
        _, inf = fakecap.make_session(d, rot)
        sess = fakecap.FakeSession(inf)
        core = fakecap.core
    else:
        from perseus.impl.magtape import plan_rot_steps
        pruned = plan_rot_steps(a.plan) if a.plan and os.environ.get("PF_ROT_PRUNE", "1") != "0" else None
        if pruned:
            extra = pruned
        sess = env.open_session(a.device, complex_payload=True, chain="n32", extra_rot_steps=extra, base=not pruned,
                                **over)
        inf = sess.inf
        env.bootstrap_setup(inf.fhe, over)
        core = _core
    st, cl = D.store(), D.client()
    cfgs = config.load_configs(a.configs)
    model = BertPrefill(inf, st, cfgs, core=core, n_layers=a.layers, profile=a.profile)
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    if a.plan:
        model.load_plans(a.plan, range(model.n_layers))
    t_pre = time.time()
    model.preload()
    from perseus.impl.pcmm import engine
    engine(model.rt).prepare(d.dim)
    if os.environ.get("PF_ATTN", "") == "ccmm":     # keys + calibration probes before the captured stages
        from perseus.impl.ccmm import engine as ccmm_engine
        ce = ccmm_engine(model.rt)
        ce.prepare(d.dim)
        print(f"[bert] ccmm c_unit={ce.c_unit} land={ce.land}", flush=True)
    print(f"[bert] configs={a.configs} weights+keys {time.time() - t_pre:.1f}s", flush=True)
    rows_out = []
    for name, ids, n, label in inputs(a, a.tokens):
        rows = D.embed(ids, cl)
        ref = D.exact_blocks(st, rows, n, model.n_layers) if (a.check or model.n_layers == 12) else None
        b0 = [int(inf.fhe.total_bootstraps)]

        def on_block(b, Xs, s):
            tb = int(inf.fhe.total_bootstraps)
            line = f"[bert] {name} block {b} {s:.2f}s bts={tb - b0[0]}"
            b0[0] = tb
            if a.check and ref is not None:
                h = model.hidden(Xs, ch_box[0])
                err = np.abs(h[:n] - ref[b][:n]).max(1)
                line += f" maxerr={err.max():.3g} (pos {int(err.argmax())}) |x|={np.abs(ref[b][:n]).max():.3g}"
            print(line, flush=True)
        ch_box = [None]
        _, ch_box[0] = model.encode(rows, n)
        bts0 = int(inf.fhe.total_bootstraps)
        t0 = time.time()
        Xs, ch, rec = model.forward(rows, n, on_block=on_block)
        wall = time.time() - t0
        line = f"[bert] SUMMARY {name} T={a.tokens} n={n} wall={wall:.1f}s per_token={wall / a.tokens:.3f}s " \
               f"bts={int(inf.fhe.total_bootstraps) - bts0}"
        if model.n_layers == 12:
            h = model.hidden(Xs, ch)
            got = D.head(h[0], cl)
            want = D.head(ref[-1][0], cl)
            kl = D.kl2(got, want)
            agree = int(np.argmax(got) == np.argmax(want))
            line += f" logits={np.round(got, 4).tolist()} ref={np.round(want, 4).tolist()} agree={agree} " \
                    f"label={label} KL={kl:.3g} cls_maxerr={np.abs(h[0] - ref[-1][0]).max():.3g}"
            rows_out.append((agree, kl))
        print(line, flush=True)
    pc = getattr(model.rt, "pcmm", None)
    if pc is not None:
        print(f"[pcmm] calls={pc.calls} cap={pc.cap} land={pc.land} | stc_in={pc.t_in:.2f}s pcmm={pc.t_mm:.2f}s "
              f"finish={pc.t_out:.2f}s", flush=True)
    if rows_out:
        print(f"[bert] AGREE {sum(r[0] for r in rows_out)}/{len(rows_out)} medianKL "
              f"{float(np.median([r[1] for r in rows_out])):.3g} maxKL {max(r[1] for r in rows_out):.3g}", flush=True)
    if a.profile and hasattr(model.rt.ops, "report"):
        print("[bert] per step:\n" + model.rt.ops.report_steps(1))
    model.close()
    sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
