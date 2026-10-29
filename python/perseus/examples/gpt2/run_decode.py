"""Teacher-forced GPT-2 decode from primitives, gated like scripts/modes_baseline.py.

    source scripts/local_env.sh
    .venv/bin/python -m perseus.examples.gpt2.run_decode --tokens 16 [--argmax]
        [--capture graphs/gpt2_decode] [--plan bootstrap_placements/gpt2_decode]
        [--layers 12] [--device N]

The GPU must be free of other processes (the run refuses otherwise; --force)."""
from __future__ import annotations

import argparse
import os
import sys
import time


def gpu_busy(device=None):
    import subprocess
    from .env import physical_device
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader", "-i", physical_device(device)],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None
    # this process's own context (an earlier run in the same process, e.g. a test session) is not another process
    return [l for l in out.splitlines() if l.strip() and l.split(",")[0].strip() != str(os.getpid())]


class _Closer:
    def __init__(self, run):
        self.run = run

    def close(self):
        self.run.close()


def _planned(a, rmodel, cfgs, store, inputs, gt_logits, on_token):
    """PLANNED on the new runtime: compile + bind every block (impl.driver.PlannedRun), then the decode loop.
    Returns (results, the run, the device-side model that decodes the logits)."""
    from perseus import _pc, config as pcfg
    from perseus._paths import resolve
    from perseus.impl.driver import PlannedRun, Recorder
    from .model import Gpt2Model
    n = rmodel.n_layers
    cfg = pcfg.plan_runtime(a.plan, a.chain, base=pcfg.preset("n32_gpt2_decode"))
    if a.no_graphs:
        cfg["exec.use_graphs"] = False
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"m7c_ref_{os.getpid()}")
    labels = [f"block_{b}" for b in range(n + 2)]
    run = PlannedRun(Recorder(rmodel, n), resolve(a.plan), cfg, labels, keys, work_dir=work, pt_rows=a.pt_rows,
                     stc_cache=a.stc_cache, schedule=a.schedule)
    run.debug = a.debug_blocks
    run.setup(inputs, argmax=a.argmax)
    dmodel = Gpt2Model(run.fhe, store, cfgs, core=_pc, n_layers=n, packing=a.packing, kv_offload=False)
    t0 = time.time()
    res = dmodel.run_decode_planned(run, inputs, gt_logits, on_token=on_token, argmax=a.argmax)
    st = run.sess.planned_stats()
    print(f"[planned] decode {time.time() - t0:.1f} s; {st}", flush=True)
    return res, run, dmodel


def _planned_classed(a, rmodel, cfgs, store, inputs, gt_logits, on_token):
    """PLANNED for a plan that binds at every cache length (SM_PERIODIC=2, the 512-context plan): the stream changes
    structure as the cache grows (the score period Gp, a new K bucket), so every token is recorded at setup and
    grouped into program classes (impl.driver.StreamedPlannedRun without streamed weights: the weights are resident
    plaintexts, the K/V states live in pinned host units, each class holds the sparse routes its period needs)."""
    from perseus import _pc, config as pcfg
    from perseus._paths import resolve
    from perseus.impl.driver import Recorder, StreamedPlannedRun
    from .model import Gpt2Model
    n = rmodel.n_layers
    cfg = pcfg.plan_runtime(a.plan, a.chain, base=pcfg.preset("n32_gpt2_decode"))
    # eager replay: a token's plaintexts are retargeted to the slots holding their values, which a captured graph
    # cannot follow
    cfg["exec.use_graphs"] = False
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"ctx_ref_{os.getpid()}")
    labels = [f"block_{b}" for b in range(n + 2)]
    base = tuple(int(x) for x in (cfg["bts.sparse_bts_slots"] or ["512", "1"]))

    def routes_for(pos):
        r = rmodel.routes_for(pos)
        return set() if r is None else set(r[1])
    first, prefix = 0, None
    if a.plain_prefix:
        # the plain prefix: positions 0 .. N-1 run on the numpy twin of the algorithm (float64, CPU), their K / V
        # pushes replayed into the recording twin's caches from payloads encrypted as public values; the device's
        # caches start from those states (each encrypted at the twin's entry metadata); only positions >= N are
        # recorded and decoded encrypted
        import numpy as np
        first = a.plain_prefix
        from .capture_mags import fake_model
        t0 = time.time()
        import hashlib
        key = hashlib.sha256(repr((a.configs, open(resolve(a.configs), "rb").read(), open(os.path.join(resolve(a.plan), "PLAN_CMD.txt")).read(), a.packing, first,
                                   np.asarray(a.all_inputs[:first]).tobytes())).encode()).hexdigest()[:16]
        cpath = os.path.join(os.environ.get("PC_PREFIX_CACHE", os.path.expanduser("~/.cache/perseus_plain_prefix")),
                             f"gpt2_{key}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            logs = [[z[f"b{b}_{p}"] for p in range(first)] for b in range(n)]
            src = f"cached {cpath}"
        else:
            fmodel, _ffhe, _finf = fake_model(a.weights, a.configs, a.packing)
            for kv in fmodel.kv:
                kv.payload_log = []
            fmodel.run_decode(a.all_inputs[:first])
            logs = [kv.payload_log for kv in fmodel.kv]
            del fmodel
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            np.savez(cpath + ".tmp.npz", **{f"b{b}_{p}": logs[b][p] for b in range(n) for p in range(first)})
            os.replace(cpath + ".tmp.npz", cpath)
            src = f"computed, cached at {cpath}"
        if os.environ.get("PC_PREFIX_ONLY", "0") not in ("", "0"):
            print(f"[impl] plain prefix {src}; PC_PREFIX_ONLY: exit", flush=True)
            sys.exit(0)
        rmodel.start()
        rmodel.replay_kv_pushes(logs, a.prefix_level)
        snap = []
        for b in range(n):
            cts, ks = rmodel.kv[b].cts("")
            snap.append({k: np.array(getattr(c, "vec", c)) for k, c in zip(ks, cts)})
        prefix = lambda b: snap[b]
        print(f"[impl] plain prefix: positions 0..{first - 1} computed in plaintext on the CPU (numpy twin, float64; "
              f"{src}), K / V pushes encrypted as public values at level {a.prefix_level}; encrypted decode from "
              f"position {first} ({time.time() - t0:.0f}s)", flush=True)
    run = StreamedPlannedRun(Recorder(rmodel, n), resolve(a.plan), cfg, labels, keys, work_dir=work, streams={},
                             stream_rows=None, routes_for=routes_for, base_routes=base, stc_cache=a.stc_cache,
                             argmax=a.argmax, slim=True, log=lambda m: print(m, flush=True), schedule=a.schedule,
                             prefix=prefix, plan_windows=a.windows)
    run.debug = a.debug_blocks
    if first:
        run.rec.first, run.rec.restart = first, False
    run.prepare(inputs)
    # a token evicts the values it no longer reads before staging its new ones: one token's values + a margin
    run.pt_slots = run.n_pts + 64
    run.setup(inputs)
    dmodel = Gpt2Model(run.fhe, store, cfgs, core=_pc, n_layers=n, packing=a.packing, kv_offload=False)
    t0 = time.time()
    res = dmodel.run_decode_planned(run, inputs, gt_logits, on_token=on_token, argmax=a.argmax, first=first)
    print(f"[planned] decode {time.time() - t0:.1f} s; {run.sess.planned_stats()}", flush=True)
    run.progs = {lab: run.classes[run.class_of[len(inputs) - 1]]["progs"][b] for b, lab in enumerate(labels)
                 if b in run.classes[run.class_of[len(inputs) - 1]]["progs"]}
    return res, run, dmodel


def steady_report(res):
    """The 2026-10-19 table metric (old repo scripts/repro/decode_precise.py): per token `[precise] tok<p> decode=
    argmax= e2e=` (e2e = decode_s + argmax_s), then the steady tokens' (pos >= 1; token 0 fills the caches) e2e median /
    mean / range. Planned: decode_s = the token's issue + logits decrypt; `token` = its whole loop iteration."""
    import statistics
    for r in res:
        d, g = r.get("decode_s", 0.0), r.get("argmax_s", 0.0)
        print(f"[precise] tok{r.get('pos')} decode={d:.3f} argmax={g:.3f} e2e={d + g:.3f}", flush=True)
    st = [r for r in res if r.get("pos") is not None and r["pos"] >= 1]
    if not st:
        return None
    e = [r.get("decode_s", 0.0) + r.get("argmax_s", 0.0) for r in st]
    tok = [r["token_s"] for r in st if "token_s" in r]
    print(f"[precise] steady n={len(st)} e2e median={statistics.median(e):.3f} mean={statistics.mean(e):.3f} "
          f"min={min(e):.3f} max={max(e):.3f} | decode median={statistics.median([r.get('decode_s', 0.0) for r in st]):.3f} "
          f"argmax median={statistics.median([r.get('argmax_s', 0.0) for r in st]):.3f}"
          + (f" | token median={statistics.median(tok):.3f}" if tok else ""), flush=True)
    return statistics.median(e)


def main(argv=None, results=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=int(os.environ.get("MULTI_T", "16")))
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index (default: the first of CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--argmax", action="store_true", help="also run the encrypted CutMax per token")
    ap.add_argument("--capture", default=None, help="graph dir: capture block graphs at token 0")
    ap.add_argument("--fake-prefix", type=int, default=0, metavar="N",
                    help="run positions 0..N-1 on the numpy fake (CPU) and start the device at N from their caches "
                         "(their K/V pushes replayed from payloads encrypted as public values at --prefix-level): a capture of position N without "
                         "decoding the positions before it")
    ap.add_argument("--prefix-level", type=int, default=36, help="the level of the replayed push payloads")
    ap.add_argument("--plain-prefix", type=int, default=0, metavar="N",
                    help="planned: positions 0..N-1 computed in plaintext on the CPU (the numpy twin), their K / V "
                         "pushes replayed into the caches as public values; the encrypted decode runs N..--tokens-1")
    ap.add_argument("--structure-only", action="store_true",
                    help="capture without the per-node magnitude decrypts (fast); fill them with capture_mags.py")
    ap.add_argument("--plan", default=None, help="plan dir: run under block_<b>_placement.json")
    ap.add_argument("--plan-windows", default=None, metavar="B1:DIR1,B2:DIR2,...",
                    help="planned, SM_PERIODIC=2 plans: a plan per position window (DIRi serves the positions below "
                         "Bi), the plans sharing one runtime (only their magnitudes differ); --plan defaults to the "
                         "first window's")
    ap.add_argument("--no-plan-argmax", action="store_true", help="run CutMax / feedback eager even if planned")
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS_PATH"))
    ap.add_argument("--configs", default=os.environ.get("CONFIGS_PATH"))
    ap.add_argument("--io-dir", default=os.environ.get("ALL_BLOCKS_IO_DIR"))
    ap.add_argument("--packing", choices=("cachemir", "cachemir_complex"), default="cachemir_complex",
                    help="cachemir_complex (default): fused K+iV linear, output-packed up/down "
                         "projections, complex K/V buckets and paired LM-head tile (needs the "
                         "complex payload); cachemir: real linears, the complex payload only in the "
                         "K/V push, CutMax and the feedback (the C++ decode's Mode-A)")
    ap.add_argument("--payload", choices=("complex", "real"), default="complex",
                    help="complex: CKKS_COMPLEX=1, the C++ decode configuration (K/V pair bootstrap, "
                         "packed CutMax); real: real slots only (no shipped plan)")
    ap.add_argument("--cpp-bootstrap", action="store_true",
                    help="run FIDESlib's C++ bootstrap instead of the default Python stage orchestration "
                         "(perseus/impl/bootstrap.py); not with a SPRU plan")
    ap.add_argument("--py-bootstrap", action="store_true", help=argparse.SUPPRESS)  # the default now
    ap.add_argument("--chain", choices=("n32", "n64"),
                    default=os.environ.get("CHAIN") or "n32",
                    help="CKKS chain; source scripts/local_env.sh for the SAME chain")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one of the port's session env values (env.ENV), repeatable")
    ap.add_argument("--force", action="store_true", help="run even if GPU is busy")
    ap.add_argument("--fake", action="store_true", help="capture on the numpy fake (perseus.impl.fakecap): no GPU")
    ap.add_argument("--profile", action="store_true", help="per-primitive wall-time split")
    ap.add_argument("--check", action="store_true", help="exit 1 unless the dist gate passes (the default now)")
    ap.add_argument("--no-graphs", action="store_true", help="planned: replay every block eagerly (exec.use_graphs=0)")
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed",
                    help="planned: rotation keys resident (dense) or 28-bit b + seed regenerated per use (packed)")
    ap.add_argument("--pt-rows", type=int, default=28, help="planned: rows per plaintext slot (0 = every prime)")
    ap.add_argument("--stc-cache", type=int, default=8, help="planned: StC-first last-stage cache entries")
    ap.add_argument("--schedule", choices=("program", "siblings", "lockstep"), default="program",
                    help="planned: op order before K-batching (siblings: refresh pairs (PC_BTS_PAIR))")
    ap.add_argument("--work-dir", default=None, help="planned: where the reference token's stream is written")
    ap.add_argument("--debug-blocks", action="store_true", help="planned: decrypt every block output / K/V state vs the CPU twin (synchronous)")
    a = ap.parse_args(argv)
    from perseus._paths import resolve
    from perseus.examples.defaults import default_plan, default_windows
    a.windows = None
    if a.plan_windows:
        from perseus.impl.driver import parse_plan_windows
        if a.fake or a.capture:
            print("--plan-windows: planned device runs only", file=sys.stderr)
            return 2
        a.windows = [(b, resolve(d)) for b, d in parse_plan_windows(a.plan_windows)]
        if a.plan is None:
            a.plan = a.windows[0][1]
    if a.plan is None and not (a.fake or a.capture) and a.chain == "n32":
        # up to 32 cached keys the main plan; past them a plan per position window up to 512, then the 1,024-context
        # plan
        if a.tokens <= 32:
            a.plan = default_plan("gpt2_decode", a.tokens)
        elif default_windows("gpt2_decode", a.tokens):
            a.windows = [(b, resolve(d)) for b, d in default_windows("gpt2_decode", a.tokens)]
            a.plan = a.windows[0][1]
        else:
            a.plan = default_plan("gpt2_decode_ctx1024")

    busy = None if a.fake else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2

    from . import env
    # GPT2_PACKING is only read back by the plan contract: a plan cut under one packing
    # must not load under the other
    # the plan's landing-bound bootstrap levers (post-raise scaling, SPRU); --set wins
    over = {**env.plan_runtime(a.plan, a.chain), "GPT2_PACKING": a.packing,
            **dict(kv.split("=", 1) for kv in a.set)}
    env.export_env(a.device, chain=a.chain,
                   CKKS_COMPLEX="1" if a.payload == "complex" else "0", **over)
    if not a.configs:
        _cfg = "gpt2_base_n32" if a.chain == "n32" else "gpt2_base"
        a.configs = env.plan_configs(a.plan, a.chain, f"configs/model/approximation/{_cfg}/configs.json")
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
    from perseus.impl import config
    from . import oracle_io
    from . import weights
    from .model import Gpt2Model, dist_gate

    # `over` has to reach open_session as well: it calls export_env a second time, and a
    # second call re-applies env.ENV over anything set here.
    if a.fake:
        from perseus.impl import fake, fakecap
        from perseus.impl.layout import Dims
        env.export_env(a.device, chain=a.chain, CKKS_COMPLEX="1" if a.payload == "complex" else "0", **over)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1" if a.payload == "complex" else "0", **over}, a.chain,
                            a.payload == "complex")
        d = Dims.gpt2()
        band, _ = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real)
        _, inf = fakecap.make_session(d, set(band.loaded_rot_steps) | fakecap.GPT2_BAND)
        sess, core = fakecap.FakeSession(inf), fakecap.core
    else:
        # the new runtime runs planned only: every block is a compiled Program (impl.driver.PlannedRun), its
        # per-position stream recorded on the CPU twin below
        if not a.plan or a.capture:
            print("the device path runs planned (--plan); capture with --fake", file=sys.stderr)
            return 2
        from perseus.impl import fake, fakecap
        from perseus.impl.layout import Dims
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1" if a.payload == "complex" else "0", **over}, a.chain,
                            a.payload == "complex")
        d = Dims.gpt2()
        band, _ = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real)
        _, inf = fakecap.make_session(d, set(band.loaded_rot_steps) | fakecap.GPT2_BAND)
        sess, core = fakecap.FakeSession(inf), fakecap.core
    store = weights.RawStore(a.weights)
    cfgs = config.load_configs(a.configs)
    model = Gpt2Model(inf, store, cfgs, core=core, n_layers=a.layers, profile=a.profile, packing=a.packing,
                      kv_offload=False)
    missing = env.rotation_audit(inf.fhe, model.rt.dims)
    if missing:
        print(f"rotation keys missing from the band: {missing}", file=sys.stderr)
        return 3
    if a.capture:
        model.set_capture(a.capture)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    if a.plan and a.fake:
        model.load_plans(a.plan, argmax_blocks=not a.no_plan_argmax)

    os.environ["ALL_BLOCKS_IO_DIR"] = a.io_dir or ""   # --io-dir wins over an exported default
    os.environ.setdefault("STEPS_T", "128")   # the shipped per-step logits oracle horizon
    os.environ["MULTI_T"] = str(a.tokens)
    cfg = oracle_io.RunConfig.from_env()
    inputs = oracle_io.read_teacher_forced_inputs(cfg)[:a.tokens]
    a.all_inputs = inputs
    if a.plain_prefix:
        if a.fake or not a.plan:
            print("--plain-prefix: planned device runs only", file=sys.stderr)
            return 2
        inputs = inputs[a.plain_prefix:]
    gt = oracle_io.read_lm_head_steps(cfg)
    gt_logits = gt.logits if gt.T else None
    print(f"[impl] decode {len(inputs)} tokens, layers={model.n_layers}, "
          f"capture={a.capture} plan={a.plan}", flush=True)

    def on_token(r):
        if a.profile and hasattr(model.rt.ops, "report_steps") and r["pos"] > 0:
            top = model.rt.ops.report_steps(1).split("\n")[:7]
            print(f"[impl] tok{r['pos']} steps: " + " | ".join(t.split("%")[0].strip() for t in top), flush=True)
            for step in ("cutmax", "softmax"):
                print(f"[impl] tok{r['pos']} {step}: " + model.rt.ops.report_step_split(step), flush=True)
        if a.profile and hasattr(model.rt.ops, "reset"):
            model.rt.ops.reset()          # per-token tables; token 0 pays the encodes
        line = f"[impl] tok{r['pos']} top1={r['top1']} decode={r['decode_s']:.1f}s"
        if "kl" in r:
            line += f" ref={r['ref']} ref_rank={r['ref_rank']} KL={r['kl']:.4g} {'OK' if r['hit'] else 'MISS'}"
        if "free_gb" in r:
            line += f" free={r['free_gb']}GB"
        if "pool_gb" in r:
            line += " pool={}/{}GB(trim {}, recycler {})".format(*r["pool_gb"])
        if "rss_gb" in r:
            line += f" rss={r['rss_gb']}GB"
        if "cpu_s" in r:
            line += (f" cpu={r['cpu_s']}s thread={r['thread_s']}s ctx={r['ctx_vol']}/{r['ctx_invol']}"
                     f" flt={r['minflt']}/{r['majflt']}")
        if r.get("masks"):
            m = r["masks"]
            line += f" masks={m.get('adopted', 0)}/{m.get('evicted', 0)}/{m.get('misses', 0)} wait={r.get('mask_wait_s', 0):.2f}s"
        if "enc_cache" in r:
            line += f" enc={r['enc_cache']['size']} miss={r['enc_cache']['misses']} hit={r['enc_cache']['hits']}"
        if "ring_gap_s" in r:
            line += f" ring_gap={r['ring_gap_s']}s gaps={r['ring_gaps']}"
        if "cutmax" in r:
            line += f" cutmax={r['cutmax']} {'OK' if r['cutmax'] == r['top1'] else 'MISS'} z_mass={r['z_mass']:.4f} argmax={r['argmax_s']:.1f}s bts={r.get('argmax_bts', '?')}"
        if "arena_gb" in r:
            line += " arena={}/{}GB".format(*r["arena_gb"])
        if "served" in r:
            line += (f" token={r['token_s']:.2f}s host(record {r['record_s']}s patch {r['patch_s']}s issue "
                     f"{r['issue_s']}s staged {r['staged']}) served={r['served']}")
        print(line, flush=True)

    first = 0
    if a.fake_prefix:
        # the fake's run needs the session's switches, already exported above
        from .capture_mags import fake_model
        t0 = time.time()
        fmodel, _ffhe, _finf = fake_model(a.weights, a.configs, a.packing)
        for kv in fmodel.kv:
            kv.payload_log = []
        fmodel.run_decode(inputs[:a.fake_prefix])
        model.start()
        first = model.replay_kv_pushes([kv.payload_log for kv in fmodel.kv], a.prefix_level)
        print(f"[impl] positions 0..{first - 1} on the fake, their pushes replayed from level {a.prefix_level} "
              f"({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    if not a.fake:
        from perseus import config as pcfg
        classed = pcfg.plan_runtime(a.plan, a.chain, base=pcfg.preset("n32_gpt2_decode"))["model.sm_periodic"] == 2
        if a.windows and not classed:
            print("--plan-windows: the classed (SM_PERIODIC=2) planned run only", file=sys.stderr)
            return 2
        res, inf, model = (_planned_classed if classed else _planned)(a, model, cfgs, store, inputs, gt_logits,
                                                                      on_token)
        sess = _Closer(inf)
    else:
        res = model.run_decode(inputs, gt_logits, on_token=on_token, argmax=a.argmax, first=first)
    ok, bad = dist_gate(res)
    hits = sum(1 for r in res if r.get("hit"))
    kls = sorted(r["kl"] for r in res if "kl" in r)
    med = kls[len(kls) // 2] if kls else float("nan")
    total_bts = getattr(inf.fhe, "total_bootstraps", None)
    if not a.fake:
        n = model.n_layers
        total_bts = sum(inf.progs[f"block_{b}"].n_bootstraps for b in range(n + 1))
        argmax_bts = inf.progs[f"block_{n + 1}"].n_bootstraps if a.argmax else 0
        med_of = lambda k: sorted(r[k] for r in res)[len(res) // 2] if res else float("nan")
        steady = res[2:] if len(res) > 3 else res
        cm = sum(1 for r in res if r.get("cutmax") == r["top1"])
        print(f"[impl] PLANNED bts/token={total_bts} (+{argmax_bts} CutMax) median decode={med_of('decode_s'):.3f}s "
              f"median token={med_of('token_s'):.3f}s steady mean token={sum(r['token_s'] for r in steady) / max(1, len(steady)):.3f}s "
              f"cutmax={cm}/{len(res) if a.argmax else 0} free_gb={[r.get('free_gb') for r in res]}")
    steady_report(res)
    print(f"[impl] SUMMARY completed={len(res)}/{len(inputs)} top1={hits}/{len(res)} "
          f"medianKL={med:.4g} deliberate_bts={model.rt.ops.n_bootstraps} "
          f"total_bts={total_bts} relevels={getattr(inf.fhe, 'weight_relevel_count', None)} "
          f"wall={time.time() - t0:.0f}s")
    if a.profile:
        print("[impl] time split over the run:\n" + model.rt.ops.report())
        print("[impl] per step (steady tokens, token 0 excluded):\n" + model.rt.ops.report_steps(max(1, len(res) - 1)))
    print("[decode] PASS" if ok else f"[decode] FAIL {[(r['pos'], r['kl'], r['ref_rank']) for r in bad]}")
    if results is not None:
        results(res)
    model.close()
    sess.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
