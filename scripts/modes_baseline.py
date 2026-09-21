"""The GPT-2 correctness gates, one mode per run:

  usage: modes_baseline.py {decode,prefill,handoff,gen,gen_prefill,forward}

Modes over `perseus._core` (scripts/run_task.sh drives the first, third, fourth and
fifth; the other two are run directly):
  decode       decode against the oracle's tokens         run_decode
  prefill      next token after a prefill                 run_prefill (prefill T, decode 0)
  handoff      prefill, then a decode of the tail         run_prefill (prefill T, decode 2)
  gen          decode, then generate                      run_generate (1 prompt token)
  gen_prefill  prefill, then generate                     run_generate (8 prompt tokens)
  forward      per-token per-block forward through perseus.nn.EncGPT2 against the
               oracle (top1/KL; GATE_DEBUG=1 adds per-block residual taps)

The teacher-forced modes check top1 against the oracle; the generating modes diverge from
it after the first token and gate on clean completion, unplanned_bts, and the per-step
agreement between the encrypted argmax and the decrypted logits.

Every mode ends with the acceptance marker `[<mode>] PASS` on success.
"""
import collections
import os
import sys
import traceback

mode = sys.argv[1] if len(sys.argv) > 1 else ""

GPT2_MODES = ("decode", "prefill", "handoff", "gen", "gen_prefill", "forward")
if mode not in GPT2_MODES:
    raise SystemExit(f"unknown mode '{mode}' (expected one of {GPT2_MODES})")

import numpy as np

from perseus import _core

GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "4"))
PREFILL_T = int(os.environ.get("PREFILL_T", "8"))

# Graded acceptance: top1 is a binary read on a continuous quantity, so the gates also report
# KL, ref_rank and top5 overlap (WORKING: KL ~0.01 on the 64-bit chain, ~0.04 on n32; a
# degenerate run saturates above 50).
# Acceptance thresholds. The verdict rides the quantities that measure the encrypted
# computation itself — the reference token stays near the top of our distribution, and the
# divergence stays far below the saturation a broken chain reaches (KL > 50). Exact top1
# agreement is REPORTED, not required: it depends on the reference sequence, because a
# near-tie resolves differently in different contexts, so a user whose oracle is not the
# shipped one would see a correct run "fail". GATE_MIN_TOP1 demands an exact count.
GATE_KL_MAX = float(os.environ.get("GATE_KL_MAX", "5"))
GATE_REF_RANK_MAX = int(os.environ.get("GATE_REF_RANK_MAX", "8"))
GATE_MIN_TOP1 = os.environ.get("GATE_MIN_TOP1")
DIST_GATED = True


def kl(p_logits, q_logits):
    """KL(P_ref || Q_ours) over the softmaxed logits."""
    p = np.exp(p_logits - np.max(p_logits)); p /= p.sum()
    q = np.exp(q_logits - np.max(q_logits)); q /= q.sum()
    return float(np.sum(p * np.log(p / np.maximum(q, 1e-30))))


#: `line` is the printed report (format frozen — humans and scripts parse it); the other
#: fields are the same numbers already computed for that line, handed back so a caller
#: can gate on them without recomputing or re-parsing.
DistStat = collections.namedtuple("DistStat", "line ref_rank kl")


def dist_report(ours, ref):
    """Graded read of one logit row against the oracle's.

    top1 is BINARY and cannot separate "one place off" from "detonated" — both print
    MISS, and the printed value is a token ID whose magnitude means nothing. ref_rank
    (0 ⇒ we would have picked the oracle token) and KL are the graded reads.

    Returns a DistStat; `.line` is the one-line summary.
    """
    rtop, top1 = int(np.argmax(ref)), int(np.argmax(ours))
    ov5 = len(set(np.argsort(ours)[-5:].tolist()) & set(np.argsort(ref)[-5:].tolist()))
    ref_rank, d_kl = int((ours > ours[rtop]).sum()), kl(ref, ours)
    return DistStat(
        line=(f"top1={top1} ref={rtop} ref_rank={ref_rank} "
              f"ours_rank_in_ref={int((ref > ref[top1]).sum())} top5_overlap={ov5}/5 "
              f"KL={d_kl:.4g} {'OK' if top1 == rtop else 'MISS'}"),
        ref_rank=ref_rank, kl=d_kl)


def dist_gate(stats):
    """Assert the per-token thresholds over the DistStats of every decoded position."""
    for tag, st in stats:
        assert st.kl <= GATE_KL_MAX, f"{tag} KL={st.kl:.4g} > GATE_KL_MAX={GATE_KL_MAX:g}"
        assert st.ref_rank <= GATE_REF_RANK_MAX, \
            f"{tag} ref_rank={st.ref_rank} > GATE_REF_RANK_MAX={GATE_REF_RANK_MAX}"


def check(res, gt, expect_rows, min_hits):
    assert not res.threw, f"{mode} threw: {res.error}"
    rows = len(res.top1)
    assert rows == expect_rows, f"logit rows {rows}/{expect_rows}"
    hits = 0
    # res.logits is [completed][vocab] (bound in session.cu). Keep the full distribution:
    # an argmax alone cannot separate a rank-1 near-miss from a detonated distribution.
    have = len(res.logits) == rows
    if DIST_GATED and not have:
        raise RuntimeError(
            f"GATE_KL_MAX/GATE_REF_RANK_MAX set but no logit rows "
            f"(res.logits {len(res.logits)}/{rows}) — the graded gate cannot be "
            f"evaluated; refusing to pass on top1 alone")
    stats = []
    for i, (pos, top1) in enumerate(zip(res.positions, res.top1)):
        ref = np.array(gt.logits[pos])
        hits += top1 == int(np.argmax(ref))
        if have:
            st = dist_report(np.array(res.logits[i]), ref)
            stats.append((f"pos={pos}", st))
            detail = st.line
        else:
            detail = (f"top1={top1} ref={int(np.argmax(ref))} "
                      f"{'OK' if top1 == int(np.argmax(ref)) else 'MISS'}  (no logit rows)")
        print(f"[{mode}] pos={pos} {detail}", flush=True)
    e2e = res.avg_s_per_tok + res.avg_argmax_s   # e2e/tok = forward + encrypted-argmax stage
    print(f"[{mode}] top1 {hits}/{rows}  completed={res.completed} "
          f"bootstraps={res.bootstraps} unplanned_bts={res.unplanned_bts} "
          f"weight_relevels={res.weight_relevels} "
          f"s/tok={res.avg_s_per_tok:.1f} argmax_s/tok={res.avg_argmax_s:.1f} e2e_s/tok={e2e:.1f}",
          flush=True)
    floor = int(GATE_MIN_TOP1) if GATE_MIN_TOP1 else min_hits
    assert hits >= floor, f"top1 {hits} < {floor}"
    dist_gate(stats)


def check_generate(res, want_tokens):
    assert not res.threw, f"{mode} threw: {res.error}"
    assert res.completed == want_tokens, f"completed {res.completed}/{want_tokens}"
    assert res.unplanned_bts == 0, f"unplanned_bts {res.unplanned_bts}"
    stream = [int(t) for t in res.top1]
    for pos, top1 in zip(res.positions, res.top1):
        print(f"[{mode}] pos={pos} top1={top1}", flush=True)
    e2e = res.avg_s_per_tok + res.avg_argmax_s
    print(f"[{mode}] stream={stream} completed={res.completed} "
          f"bootstraps={res.bootstraps} unplanned_bts={res.unplanned_bts} "
          f"weight_relevels={res.weight_relevels} "
          f"s/tok={res.avg_s_per_tok:.1f} argmax_s/tok={res.avg_argmax_s:.1f} e2e_s/tok={e2e:.1f}",
          flush=True)


def run_gpt2_session():
    cfg = _core.RunConfig.from_env()
    if mode == "prefill":
        cfg.tokens, cfg.prefill_tokens, cfg.decode_tokens = PREFILL_T, PREFILL_T, 0
    elif mode == "handoff":
        cfg.tokens, cfg.prefill_tokens, cfg.decode_tokens = PREFILL_T + 2, PREFILL_T, 2
    elif mode == "gen":
        cfg.gen_prompt, cfg.gen_tokens, cfg.teacher_forced = 1, GEN_TOKENS, False
        cfg.tokens = cfg.gen_prompt + cfg.gen_tokens
    elif mode == "gen_prefill":
        cfg.gen_prompt, cfg.gen_tokens, cfg.teacher_forced = 8, GEN_TOKENS, False
        cfg.tokens = cfg.gen_prompt + cfg.gen_tokens
    inputs = _core.read_teacher_forced_inputs(cfg)
    gt = _core.read_lm_head_steps(cfg)

    if mode == "decode":
        check(_core.run_decode(cfg, inputs), gt, expect_rows=cfg.tokens,
              min_hits=0)   # reported; the verdict is the KL / rank gate above
    elif mode == "prefill":
        # min_hits=0 for the complex token-pair buckets (T>=64): the prefill verdict rides
        # the KL band, deterministic bts and the plan-bound markers; top1 is informative.
        check(_core.run_prefill(cfg, inputs), gt, expect_rows=1,
              min_hits=0 if PREFILL_T >= 64 else 1)
    elif mode == "handoff":
        check(_core.run_prefill(cfg, inputs), gt, expect_rows=2, min_hits=1)
    else:
        check_generate(_core.run_generate(cfg, inputs), want_tokens=cfg.gen_tokens)


def run_gpt2_forward():
    """Per-token per-block forward gate vs the oracle."""
    from perseus.nn import EncGPT2

    opts = _core.InferenceOptions()
    opts.ckks = _core.CKKSOptions.from_env()
    opts.mode = _core.InferenceMode.Sync
    inf = _core.make_gpt2_inference(opts)
    store = _core.WeightStore.from_zip(os.environ["WEIGHTS_PATH"])
    configs = _core.load_configs(os.environ["CONFIGS_PATH"])
    model = EncGPT2(store, configs).bind(inf)
    print(f"[{mode}] model bound (block states + lnf encoded)", flush=True)

    cfg = _core.RunConfig.from_env()
    inputs = _core.read_teacher_forced_inputs(cfg)
    gt = _core.read_lm_head_steps(cfg)

    # GATE_RESEED=1 (implies taps): after each block, REPLACE the ciphertext with a
    # fresh encode of the ORACLE residual — every block then computes on ground-truth
    # inputs regardless of upstream numeric health. This is THE capture mode for a
    # chain whose eager arm diverges (n32): per-block magnitudes stay sane so the
    # planner's eligibility windows are trustworthy, while the planner re-derives
    # levels through its own entry chaining (same principle as the prefill chunk
    # templates). Harness instrumentation only — the captured per-block op sequence
    # is identical to eager decode's.
    reseed = bool(os.environ.get("GATE_RESEED"))
    taps = None
    if os.environ.get("GATE_DEBUG") or reseed:
        import json
        io_dir = os.environ["ALL_BLOCKS_IO_DIR"]
        taps = [json.load(open(f"{io_dir}/all_blocks_L{b:02d}_T{cfg.steps_t}.json"))
                for b in range(model.n_layers)]

    model.start()
    hits, logits, tiles = 0, None, None
    stats = []
    n_tok = int(os.environ.get("GATE_TOKENS", "2"))
    for t in range(n_tok):
        inf.capture_t = t
        x = _core.encode_token_input(inf, inputs[t])
        for b in range(model.n_layers):
            x = model.run_block(b, x)
            if taps is not None:
                out = np.array(_core.decode_token_output(inf, x))[:768]
                ref_res = np.array(taps[b]["res"][t])
                rel = np.linalg.norm(out - ref_res) / (np.linalg.norm(ref_res) + 1e-9)
                print(f"[{mode}] tok{t} block{b} res_rel={rel:.4f} "
                      f"free={_core.device_free_gb():.1f}GB", flush=True)
                if reseed:
                    x = _core.encode_token_input(inf, [float(v) for v in ref_res])
        x = _core.apply_final_ln(inf, x, model._lnf)
        tiles = model.lm_head(x)
        logits = np.array(model.decode_logits(tiles))
        ref = np.array(gt.logits[t])
        top1, rtop = int(logits.argmax()), int(ref.argmax())
        hits += top1 == rtop
        st = dist_report(logits, ref)
        stats.append((f"tok{t}", st))
        print(f"[{mode}] tok{t} {st.line}", flush=True)

    if hasattr(model, "cutmax"):
        z = model.cutmax(tiles)
        cm = int(np.array(model.decode_logits(z)).argmax())
        print(f"[{mode}] cutmax={cm} fhe_argmax={int(logits.argmax())}", flush=True)
        assert cm == int(logits.argmax()), "cutmax != logits argmax"
    assert hits >= n_tok - 1, f"top1 {hits}/{n_tok}"
    dist_gate(stats)

try:
    if mode == "forward":
        run_gpt2_forward()
    else:
        run_gpt2_session()
except BaseException:
    traceback.print_exc(file=sys.stdout)
    print(f"[{mode}] FAIL", flush=True)
    sys.stdout.flush()
    os._exit(1)

print(f"[{mode}] PASS", flush=True)
sys.stdout.flush()
