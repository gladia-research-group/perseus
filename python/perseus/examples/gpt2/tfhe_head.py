"""The GPT-2 decode head's selection by TFHE signs (impl/tfhe_select.py) on logit tiles, gated against the plaintext
selection on the same logits: greedy argmax and top-k sampling; `--mode cutmax` runs the shipped CKKS CutMax on the
same inputs (the baseline).

    python -m perseus.examples.gpt2.tfhe_head --mirror --logits L.npy --mode argmax --n 256     # numpy mirror
    PF_TFHE_NOISE=7.9e-3 python -m perseus.examples.gpt2.tfhe_head --mirror ...               # with the switch's noise
    python -m perseus.examples.gpt2.tfhe_head --fake --logits L.npy --mode topk --capture graphs/<name>
    tools/gpu.sh <log> python -m perseus.examples.gpt2.tfhe_head --plan bootstrap_placements/gpt2_tfhe_topk_ship_hyb \
        --mode topk --logits L.npy --n 8

`--logits`: next-token logits (positions, 50257) of real text. The tiles are the LM head's layout (two tiles of N
slots, column cutmax_tile_col_of_slot(m) of tile k at slot m). A plan's PLAN_CMD.txt carries what the run repeats
(`runtime=`, `twin=`, `sign=`: PF_SELECT_SIGN). The plans: gpt2_tfhe_{argmax,topk}_ship (TFHE signs),
gpt2_tfhe_{argmax,topk}_ship_hyb (the big early rounds as CKKS signs, the rest TFHE), gpt2_tfhe_cutmax_ship (CutMax),
all on the gpt2_decode recipe and runtime line.

Measured (GPU 3, 8 positions, fresh inputs): argmax CutMax 0.45 s, TFHE 2.59 s, hybrid 1.29 s, the hybrid's all-CKKS
control 1.42 s; top-40 sampling TFHE 4.39 s, hybrid 3.10 s, all-CKKS control 3.88 s, all-CKKS fine 3.94 s. CutMax
wins greedy decoding; the tournament's edge is robustness (no calibration range: on 1792 positions CutMax's one-hot
is smeared at 8% of them and wrong by 22-39 logits at 0.5%, the TFHE argmax misses only near-ties < 0.08 logits).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

VOCAB = 50257
N = 1 << 15             # slots
HID = 1024          # Dims.gpt2().hid (the padded width the head interleaves by)


def tile_cols(k):
    """The vocabulary column of each slot of tile k (-1: padding)."""
    from perseus.impl.layout import cutmax_tile_col_of_slot
    col = k * N + cutmax_tile_col_of_slot(np.arange(N), HID, N)
    return np.where(col < VOCAB, col, -1)


class Head:
    """The selection stage on K = 2 logit tiles; `mode`: argmax | topk."""

    def __init__(self, mode="argmax", k=40, L=2048, W=32.0, probes=8, R=128.0, fine=2048, temp=1.0):
        self.mode, self.k, self.L, self.W, self.probes, self.R, self.fine, self.temp = mode, k, L, W, probes, R, fine, temp
        self.cols = [tile_cols(0), tile_cols(1)]
        self.valid = [c >= 0 for c in self.cols]

    def encode(self, logits, gumbel=None):
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape[0] > VOCAB:                      # logits ++ gumbel: the runners' one input vector
            logits, gumbel = logits[:VOCAB], logits[VOCAB:]
        out = [np.where(c >= 0, logits[np.maximum(c, 0)] / self.temp, 0.0) for c in self.cols]
        if self.mode == "topk" and gumbel is not None:
            self.g = np.zeros(N)              # a plaintext: one draw per run (re-encoded per token in a deployment)
            self.g[::N // self.L] = gumbel
        return out

    def stage(self, ops, xs):
        from perseus.impl import tfhe_select as sel
        if self.mode == "cutmax":           # the CKKS baseline: the shipped decode's packed CutMax, StC-first refreshes
            from perseus.examples.gpt2 import env
            from perseus.examples.gpt2.head import cutmax_argmax_packed, pack_tiles
            from perseus.impl import bootstrap as _pyb
            from perseus.impl.config import load_configs
            from perseus._paths import resolve
            cfg = load_configs(resolve(env.CONFIGS_N32)).cutmax
            with _pyb.stc_first(True):
                return [cutmax_argmax_packed(self.rt, pack_tiles(self.rt, xs[:2]), VOCAB, cfg, None)]
        if self.mode == "argmax":
            return sel.argmax(ops, xs[:2], self.valid, R=self.R, fine=self.fine)
        oh, mask = sel.topk_sample(ops, xs[:2], self.valid, self.g, k=self.k, L=self.L, W=self.W, probes=self.probes,
                                   R=self.R, fine=self.fine)
        return oh + [mask]

    def stages(self):
        return [("select", self.stage)]

    def rotations(self):
        """The rotation steps the stage issues (one mirror pass with a recording NumpyOps; CutMax: the powers of 2)."""
        if self.mode == "cutmax":
            return sorted({(1 << j) % N for j in range(15)} | {(-(1 << j)) % N for j in range(15)})
        from perseus.impl.ops import NumpyOps
        steps = set()

        class Rec(NumpyOps):
            def rotate(self, a, k):
                steps.add(int(k) % N)
                return super().rotate(a, k)

            def rotate_many(self, a, ks):
                return [self.rotate(a, k) for k in ks]

            def rotate_and_sum(self, a, start, stop):
                g = abs(int(start))
                while g < stop:
                    steps.add((g if start > 0 else -g) % N)
                    g *= 2
                return super().rotate_and_sum(a, start, stop)
        self.stage(Rec(), self.encode(np.zeros(VOCAB), np.zeros(self.L)))
        return sorted(steps - {0})

    def decode(self, outs):
        """(token one-hot over the vocabulary, bucket mask or None)."""
        v = np.zeros(VOCAB)
        if self.mode == "cutmax":           # the packed z: tile 0 on the real axis, tile 1 on the imaginary one
            z = np.asarray(outs[0])
            outs = [z.real, z.imag]
        for c, o in zip(self.cols, outs[:2]):
            ok = c >= 0
            v[c[ok]] = np.real(np.asarray(o))[ok]
        return v, (np.real(np.asarray(outs[2]))[::N // self.L] if self.mode == "topk" else None)

    def buckets(self, logits):
        """The token of each bucket's maximum (slot i of tile 0 folded with tile 1, then blocks of N / L slots)."""
        x = np.full((2, N), -np.inf)
        for k, c in enumerate(self.cols):
            x[k, c >= 0] = logits[c[c >= 0]]
        tok = np.where(x[1] > x[0], np.maximum(self.cols[1], 0), self.cols[0])
        v = x.max(0).reshape(self.L, -1)                    # row j: slots [j B, (j + 1) B)
        i = v.argmax(1)
        return tok.reshape(self.L, -1)[np.arange(self.L), i]


def _tv(lg, a, b):
    """The total variation between softmax sampling restricted to the token sets a and b."""
    p = np.zeros(len(lg))
    q = np.zeros(len(lg))
    for out, S in ((p, a), (q, b)):
        i = np.array(sorted(S))
        if len(i):
            e = np.exp(lg[i] - lg[i].max())
            out[i] = e / e.sum()
    return float(0.5 * np.abs(p - q).sum())


def _check_onehot(v):
    r = np.rint(v)
    return bool((np.abs(v - r).max() < 0.1) and r.sum() == 1 and r.min() >= 0)


def mirror(a, L):
    from perseus.impl.ops import NumpyOps
    ops = NumpyOps()
    h = Head(a.mode, k=a.k, L=a.L, W=a.W, probes=a.probes, fine=a.fine)
    rng = np.random.default_rng(1)
    res = []
    t0 = time.time()
    for p in range(min(a.n, len(L))):
        lg = L[p].astype(np.float64)
        top = np.sort(lg)[::-1]
        if a.mode == "argmax":
            v, _ = h.decode(h.stage(ops, h.encode(lg)))
            ok = _check_onehot(v)
            got = int(np.argmax(v))
            res.append((ok, got == int(np.argmax(lg)), float(top[0] - lg[got]), float(top[0] - top[1])))
        else:
            g = rng.gumbel(size=a.L)
            v, mask = h.decode(h.stage(ops, h.encode(lg, g)))
            bt = h.buckets(lg)
            members = set(bt[np.rint(mask) > 0.5].tolist())
            exact = set(np.argsort(-lg)[:a.k].tolist())
            got = int(np.argmax(v))
            # the sample the same Gumbel draw gives on the member set in plaintext
            mem = np.array(sorted(members))
            gb = np.zeros(VOCAB)
            gb[bt] = g
            ref = int(mem[np.argmax(lg[mem] + gb[mem])]) if len(mem) else -1
            res.append((_check_onehot(v), got == ref, len(members), len(members & exact) / a.k, got in exact,
                        _tv(lg, exact, members)))
    dt = time.time() - t0
    if a.mode == "argmax":
        valid = sum(x[0] for x in res)
        agree = sum(x[0] and x[1] for x in res)
        miss = [x[2] for x in res if not x[1]]
        print(f"[tfhe_head] mirror argmax n={len(res)}: valid one-hots {valid}, == argmax {agree}; misses' logit "
              f"deficit max {max(miss) if miss else 0:.3g} (top-2 gap of misses: "
              f"{[round(x[3], 3) for x in res if not x[1]][:10]}); {dt / len(res):.2f} s/pos", flush=True)
    else:
        print(f"[tfhe_head] mirror top-{a.k} (L={a.L}) n={len(res)}: valid one-hots {sum(x[0] for x in res)}, sample == "
              f"plaintext Gumbel-max on the members {sum(x[1] for x in res)}; |members| mean "
              f"{np.mean([x[2] for x in res]):.2f} min {min(x[2] for x in res)} max {max(x[2] for x in res)}; recall "
              f"mean {np.mean([x[3] for x in res]):.4f} min {min(x[3] for x in res):.3f}; sample in exact top-k "
              f"{sum(x[4] for x in res)}; TV(member sampling, exact top-k sampling) mean {np.mean([x[5] for x in res]):.4f} "
              f"p99 {np.percentile([x[5] for x in res], 99):.4f}; {dt / len(res):.2f} s/pos", flush=True)
    return res


def _inputs(a, L):
    """Logits ++ the Gumbel draw (one per run: the planned stage holds it as a plaintext)."""
    g = np.random.default_rng(1).gumbel(size=a.L)
    return [np.concatenate([L[p].astype(np.float64), g]) for p in range(min(a.n, len(L)))]


def _report(a, h, L, outs, walls):
    """The run's one-hots vs the plaintext selection on its input logits."""
    res = []
    for p, (o, x) in enumerate(zip(outs, _inputs(a, L))):
        lg, g = x[:VOCAB], x[VOCAB:]
        v, mask = h.decode(o)
        got, ok = int(np.argmax(v)), _check_onehot(v)
        if a.mode != "topk":
            res.append((ok, got == int(np.argmax(lg)), float(lg.max() - lg[got])))
            print(f"[tfhe_head] pos {p}: argmax {got} ref {int(np.argmax(lg))} valid {ok} deficit {res[-1][2]:.3g} "
                  f"max|v - onehot| {np.abs(v - np.rint(v)).max():.3g}", flush=True)
        else:
            bt = h.buckets(lg)
            members = set(bt[np.rint(mask) > 0.5].tolist())
            exact = set(np.argsort(-lg)[:a.k].tolist())
            mem = np.array(sorted(members))
            gb = np.zeros(VOCAB)
            gb[bt] = g
            ref = int(mem[np.argmax(lg[mem] + gb[mem])]) if len(mem) else -1
            res.append((ok, got == ref, len(members), len(members & exact) / a.k, _tv(lg, exact, members)))
            print(f"[tfhe_head] pos {p}: sample {got} (plaintext Gumbel-max on the members {ref}) valid {ok} |members| "
                  f"{len(members)} recall {res[-1][3]:.3f} TV {res[-1][4]:.4f} max|mask - 0/1| "
                  f"{np.abs(mask - np.rint(mask)).max():.3g}", flush=True)
    print(f"[tfhe_head] SUMMARY {a.mode} n={len(res)}: valid {sum(r[0] for r in res)}, agree {sum(r[1] for r in res)}"
          + (f", recall mean {np.mean([r[3] for r in res]):.4f}, TV mean {np.mean([r[4] for r in res]):.4f}"
             if a.mode == "topk" else f", max deficit {max(r[2] for r in res):.3g}")
          + (f"; s/token median {np.median(walls):.3f} (walls {' '.join(f'{w:.3f}' for w in walls)})" if walls else ""),
          flush=True)


def fake(a, L):
    from perseus.examples.resnet.run import _twin
    h = Head(a.mode, k=a.k, L=a.L, W=a.W, probes=a.probes, fine=a.fine)
    model = _twin(a, h)
    h.rt = model.rt
    outs = []
    for x in _inputs(a, L)[:1 if a.capture else None]:
        t0 = time.time()
        xs, _, _ = model.prefill(x)
        outs.append([np.asarray(model.rt.ops.decrypt_slots_complex(c)) for c in xs])
        print(f"[tfhe_head] twin {time.time() - t0:.1f}s", flush=True)
    if a.capture:
        print(f"[tfhe_head] captured -> {a.capture}", flush=True)
    _report(a, h, L, outs, [])


def _config(plan, preset):
    """bert.planned.plan_config on `preset` (CutMax: the decode preset, whose sparse 1-slot refreshes it plans)."""
    import json
    from pathlib import Path
    from perseus import config as pcfg
    cfg = pcfg.preset(preset)
    first = sorted(Path(plan).glob("block_*_placement.json"))[:1]
    if first:
        env = (json.loads(first[0].read_text()).get("capture_env") or {}).get("env") or {}
        cfg = cfg.with_env({k: str(v) for k, v in env.items() if v is not None})
    cfg = pcfg.plan_runtime(plan, "n32", base=cfg)
    cfg["exec.use_graphs"] = False
    return cfg


def planned(a, L):
    from perseus import _pc
    from perseus.examples.bert.planned import plan_config
    from perseus.examples.resnet.impl import ResnetPlanned
    from perseus.examples.resnet.run import _twin
    h = Head(a.mode, k=a.k, L=a.L, W=a.W, probes=a.probes, fine=a.fine)
    model = _twin(a, h)
    h.rt = model.rt
    keys = _pc.KeySource.generate([], conj=True, layout="packed")
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"tfhe_head_{os.getpid()}")
    cfg = _config(a.plan, "n32_gpt2_decode") if a.mode == "cutmax" else plan_config(a.plan)
    run = ResnetPlanned(model, a.plan, cfg, keys, work_dir=work, pt_rows=28, pool=a.pool,
                        log=lambda m: print(m, flush=True))
    from perseus.impl.tfhe_select import ckks_res
    run.tfhe_chunk = 0 if (a.mode == "cutmax" or ckks_res() is not None) else a.tfhe_chunk
    xs = _inputs(a, L)
    t0 = time.time()
    run.prepare(xs[0])
    p = run.progs[run.labels[0]]
    print(f"[tfhe_head] compiled: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; {p.n_bootstraps} "
          f"refreshes", flush=True)
    if a.compile_only:
        return
    run.setup(xs[0])
    print(f"[tfhe_head] planned setup {time.time() - t0:.1f}s", flush=True)
    outs, walls = [], []
    for x in xs:
        vals = run.input_values(x)
        t0 = time.time()
        regs = run.run(vals=vals)
        run.fhe.device_sync()
        walls.append(time.time() - t0)
        outs.append([np.asarray(v) for v in run.decrypt(regs)])
        st = run.sess.tfhe_stats() if run.tfhe_chunk else {}
        print(f"[tfhe_head] wall {walls[-1]:.3f}s; switch calls {st.get('calls')} values {st.get('values')} (cumulative)",
              flush=True)
    _report(a, h, L, outs, walls)
    run.close()


def _plan_settings(a):
    """The plan's PLAN_CMD.txt settings the run must repeat: `runtime=` (the twin's --set), `twin=` (the twin's
    reactive ceiling) and `sign=` (PF_SELECT_SIGN)."""
    path = os.path.join(a.plan, "PLAN_CMD.txt")
    lines = dict(ln.split("=", 1) for ln in open(path).read().splitlines() if "=" in ln) if os.path.exists(path) else {}
    a.set = lines.get("runtime", "").split() + a.set
    if "twin" in lines:
        a.twin_limit = int(lines["twin"].split()[-1])
    if "sign" in lines:
        os.environ.setdefault("PF_SELECT_SIGN", lines["sign"].strip())


def main(argv=None):
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mirror", action="store_true", help="plaintext mirror (NumpyOps; PF_TFHE_NOISE)")
    mode.add_argument("--fake", action="store_true", help="the CPU twin (perseus.impl.fakecap)")
    mode.add_argument("--plan", default=None, help="GPU, planned: the bootstrap_placements dir")
    ap.add_argument("--capture", default=None, help="--fake: write the stage graph here")
    ap.add_argument("--twin-limit", type=int, default=50)
    ap.add_argument("--twin-bts-out", type=int, default=32)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--tfhe-chunk", type=int, default=4096)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--mode", choices=("argmax", "topk", "cutmax"), default="argmax")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--W", type=float, default=32.0)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--fine", type=int, default=2048)
    a = ap.parse_args(argv)
    L = np.load(a.logits)
    if a.mirror:
        mirror(a, L)
    elif a.fake:
        fake(a, L)
    else:
        from perseus._paths import resolve
        a.plan = resolve(a.plan)
        _plan_settings(a)
        planned(a, L)
    return 0


if __name__ == "__main__":
    sys.exit(main())
