"""BERT-base SST-2 encrypted on perseus.impl: client embeddings -> 12 encrypted post-LN blocks (bidirectional
attention, blocked layout, PCMM linears) -> client pooler + classifier, gated against the float64 model.

    .venv/bin/python -m perseus.examples.bert.run_bert --fake [--text 128 | --sst2 0,1,2] [--layers 12]
        [--capture graphs/X --structure-only] [--plan bootstrap_placements/X] [--check]
    tools/gpu.sh <log> .venv/bin/python -m perseus.examples.bert.run_bert --text 128 \
        [--plan bootstrap_placements/bert_sst2_128] [--check]   # default plan: examples.defaults

The device path runs planned only (examples.bert.planned.BertPlannedPrefill): the forward's op stream recorded once
on the CPU twin, every block a compiled Program run in Rust, the PCMM linears and the CCMM attention `Finish` ops
served by the session's matmul services.
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
# the fastest measured config is the default (sync-free encodes / PCMM / CCMM + StC-first power-of-two cache keys: BERT
# A/B 2026-10-18, -2.1 s, same accuracy); the shell or --set overrides each
FAST = {"FIDESLIB_ENCODE_NOSYNC": "1", "FIDESLIB_CCMM_ASYNC": "1", "FIDESLIB_PCMM_NOSYNC": "1",
        "FIDESLIB_ENCODE_PINNED_RING": "256", "FIDESLIB_STC_POW2_KEY": "1"}


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



def _planned(a, model, groups, cl, st, results=None):
    """PLANNED on the new runtime, one recorded stream per real-token count n: record + compile + session, an untimed
    warm-up forward of another input of the same length (BERT_WARMUP, default 1: the first input with the tokens
    between [CLS] and [SEP] reversed), then `--repeat` timed forwards per input (each: block 0's inputs encrypted,
    every block issued, the device drained)."""
    from perseus import _pc
    from perseus._paths import resolve
    from . import data as D
    from .planned import BertPlannedPrefill, plan_config
    rows_out = []
    for n, items in groups.items():
        model.rt.forget_masks()         # the previous token count's masks
        cfg = plan_config(a.plan, a.graphs)
        keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
        work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"m10_bert_{os.getpid()}_{n}")
        run = BertPlannedPrefill(model, resolve(a.plan), cfg, keys, n_keys=n, work_dir=work, pt_rows=a.pt_rows,
                                 stc_cache=a.stc_cache, check_meta=a.check_meta, log=lambda m: print(m, flush=True),
                                 ccmm_kappa=[float(x) for x in a.ccmm_kappa.split(",")],
                                 ccmm_sweep=[float(x) for x in a.ccmm_sweep.split(",") if x])
        name0, ids0, _n, _lab = items[0]
        rows0 = D.embed(ids0, cl)
        t_set = time.time()
        run.prepare(rows0)
        progs = [run.progs[lab] for lab in run.labels]
        n_bts = sum(p.n_bootstraps for p in progs)
        n_fin = [sum(p.n_finishes[i] for p in progs) for i in (0, 1)]
        print(f"[bert] compiled {len(progs)} blocks (n={n}): record {run.record_s:.1f}s compile {run.compile_s:.1f}s; "
              f"{n_bts} refresh sites + {n_fin[1]} PCMM/CCMM outputs ({n_fin[0]} finishes) = {n_bts + n_fin[1]} bts, "
              f"registers {run.n_regs}", flush=True)
        if a.compile_only:
            continue
        run.setup(rows0)
        free0 = run.fhe.mem_info()[0] / 2 ** 30
        print(f"[bert] planned setup {time.time() - t_set:.1f}s; device free {free0:.2f} GB", flush=True)
        if os.environ.get("BERT_WARMUP", "1") not in ("", "0"):
            wids = np.array(ids0)
            wids[1:n - 1] = wids[1:n - 1][::-1].copy()
            t_w = time.time()
            run.run(vals=run.input_values(D.embed(wids, cl)))
            run.fhe.device_sync()
            print(f"[bert] warm-up forward (other input) {time.time() - t_w:.2f}s", flush=True)
        for name, ids, _n, label in items:
            rows = D.embed(ids, cl)
            vals = run.input_values(rows)
            ref = D.exact_blocks(st, rows, n, model.n_layers)
            walls, frees = [], []
            for k in range(max(1, a.repeat)):
                info = []
                t0 = time.time()
                run.run(on_block=lambda b, i: info.append(i), vals=vals)
                run.fhe.device_sync()
                walls.append(time.time() - t0)
                frees.append(round(run.fhe.mem_info()[0] / 2 ** 30, 2))
                host = sum(i["host_us"] for i in info) / 1e6
                mm = sum(i["meta_mismatches"] for i in info)
                kl_k = ""
                if model.n_layers == 12:        # each forward's own encryption / refresh noise
                    kl_k = f", KL {D.kl2(D.head(run.hidden()[0], cl), D.head(ref[-1][0], cl)):.3g}"
                print(f"[bert] {name} forward {k}: wall {walls[-1]:.2f}s host issue {host:.2f}s"
                      + (f" metadata mismatches {mm}" if a.check_meta else "") + f", free {frees[-1]} GB{kl_k}",
                      flush=True)
            wall = sorted(walls)[len(walls) // 2]
            if a.check:
                for b in range(model.n_layers):
                    h = run.hidden(b)
                    err = np.abs(h[:n] - ref[b][:n]).max(1)
                    print(f"[bert] {name} block {b} maxerr={err.max():.3g} (pos {int(err.argmax())}) "
                          f"|x|={np.abs(ref[b][:n]).max():.3g}", flush=True)
            line = f"[bert] SUMMARY {name} T={a.tokens} n={n} wall={wall:.2f}s per_token={wall / a.tokens:.3f}s " \
                   f"bts={n_bts + n_fin[1]} walls={[round(w, 2) for w in walls]} free={frees}GB (setup {free0:.2f})"
            if model.n_layers == 12:
                h = run.hidden()
                got, want = D.head(h[0], cl), D.head(ref[-1][0], cl)
                kl = D.kl2(got, want)
                agree = int(np.argmax(got) == np.argmax(want))
                line += f" logits={np.round(got, 4).tolist()} ref={np.round(want, 4).tolist()} agree={agree} " \
                        f"label={label} KL={kl:.3g} cls_maxerr={np.abs(h[0] - ref[-1][0]).max():.3g}"
                rows_out.append((agree, kl))
                if results is not None:
                    results({"name": name, "agree": agree, "kl": kl, "wall": wall, "walls": walls, "free_gb": frees,
                             "bts": n_bts + n_fin[1]})
            print(line, flush=True)
        stp, stc = run.sess.pcmm_stats(), (run.sess.ccmm_stats() if run.ccmm else {})
        print(f"[pcmm] calls={stp.get('calls')} linears={stp.get('linears')} missing={stp.get('missing')}; "
              f"[ccmm] calls={stc.get('calls')}; planned {run.sess.planned_stats()}", flush=True)
        run.close()
    return rows_out


def main(argv=None, results=None):
    from perseus.examples.gpt2.run_decode import gpu_busy
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128, help="T: rows padded to it")
    ap.add_argument("--text", default="128", help="'128': the 128-token text; '' for none")
    ap.add_argument("--sst2", default="", help="SST-2 validation indices: comma-separated, or a range a:b")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index (default: the first of CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", action="store_true", help="per-block maxerr against the float64 model")
    ap.add_argument("--fake", action="store_true", help="the numpy fake (perseus.impl.fakecap): no GPU")
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed",
                    help="planned: rotation keys resident (dense) or 28-bit b + seed regenerated per use (packed)")
    ap.add_argument("--pt-rows", type=int, default=28, help="planned: rows per plaintext slot (0 = every prime)")
    ap.add_argument("--stc-cache", type=int, default=16, help="planned: StC-first last-stage cache entries")
    ap.add_argument("--check-meta", action="store_true", help="planned: compare every step's metadata with the compile")
    ap.add_argument("--repeat", type=int, default=1, help="planned: timed forwards per input (wall = median)")
    ap.add_argument("--graphs", action="store_true", help="planned: replay each block as CUDA graphs")
    ap.add_argument("--work-dir", default=None, help="planned: where the recorded stream is written")
    ap.add_argument("--compile-only", action="store_true", help="planned: record + compile on the CPU, then stop")
    ap.add_argument("--ccmm-kappa", default="256,16", help="planned: CCMM operating points QK,PV (SinCBoot input x kappa)")
    ap.add_argument("--ccmm-sweep", default="", help="planned: comma list of kappas: QK / PV probe precision per kappa")
    a = ap.parse_args(argv)
    from perseus.examples.defaults import default_plan
    if a.plan is None and not (a.fake or a.capture):
        a.plan = default_plan("bert", a.tokens)
    planned = not a.fake
    if planned and (not a.plan or a.capture):
        # the new runtime has no eager device path: a forward is a planned run (its stream recorded on the CPU twin)
        print("the device path runs planned (--plan); capture with --fake", file=sys.stderr)
        return 2
    os.environ.setdefault("KV_ARENA_GB", "0")
    busy = None if (a.fake or a.compile_only) else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from perseus._paths import resolve
    from perseus.examples.gpt2 import env
    if a.plan:
        a.plan = resolve(a.plan)
    over = {**{k: os.environ.get(k, v) for k, v in FAST.items()}, **ARM, **env.plan_runtime(a.plan, "n32"),
            "GPT2_PACKING": "cachemir", **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    if a.plan and not os.environ.get("PF_PCMM_CALIB_IN"):
        cal = os.path.join(a.plan, "pcmm_calib.json")
        if os.path.exists(cal):
            os.environ["PF_PCMM_CALIB_IN"] = cal
    over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
            "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    env.export_env(a.device, chain="n32", CKKS_COMPLEX="1", **over)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.fake else cvd
    if not a.configs:
        a.configs = CONFIGS
        cmd = os.path.join(a.plan, "PLAN_CMD.txt") if a.plan else ""
        if cmd and os.path.exists(cmd):
            for line in open(cmd).read().splitlines():
                if line.startswith("configs="):
                    a.configs = line[len("configs="):].split()[0]
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
    from perseus.impl import config, fakecap
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
    # the fake session: the capture (--fake) or the planned run's CPU twin
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    rot = extra | fakecap.GPT2_BAND | (ladder_radix_steps(d, ladder_radix_bits()) if ladder_radix_bits() >= 2
                                       else set())
    _, inf = fakecap.make_session(d, rot)
    sess = fakecap.FakeSession(inf)
    st, cl = D.store(), D.client()
    cfgs = config.load_configs(a.configs)
    model = BertPrefill(inf, st, cfgs, core=fakecap.core, n_layers=a.layers, profile=a.profile)
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    if a.plan and a.fake:
        model.load_plans(a.plan, range(model.n_layers))
    t_pre = time.time()
    model.preload()
    from perseus.impl.pcmm import engine
    engine(model.rt).prepare(d.dim)
    if os.environ.get("PF_ATTN", "") == "ccmm":
        from perseus.impl.ccmm import engine as ccmm_engine
        ccmm_engine(model.rt).prepare(d.dim)
    print(f"[bert] configs={a.configs} weights {time.time() - t_pre:.1f}s plan={a.plan}", flush=True)
    ins = inputs(a, a.tokens)
    if planned:
        groups = {}
        for item in ins:
            groups.setdefault(int(item[2]), []).append(item)
        rows_out = _planned(a, model, groups, cl, st, results)
    else:
        rows_out = _fake(a, model, inf, ins, cl, st)
    if rows_out:
        print(f"[bert] AGREE {sum(r[0] for r in rows_out)}/{len(rows_out)} medianKL "
              f"{float(np.median([r[1] for r in rows_out])):.3g} maxKL {max(r[1] for r in rows_out):.3g}", flush=True)
    model.close()
    sess.close()
    return 0


def _fake(a, model, inf, ins, cl, st):
    """The forward on the numpy fake (captures, CPU smokes)."""
    from . import data as D
    rows_out = []
    for name, ids, n, label in ins:
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
        print(f"[pcmm] calls={pc.calls} cap={pc.cap} land={pc.land}", flush=True)
    if a.profile and hasattr(model.rt.ops, "report"):
        print("[bert] per step:\n" + model.rt.ops.report_steps(1))
    return rows_out


if __name__ == "__main__":
    sys.exit(main())
