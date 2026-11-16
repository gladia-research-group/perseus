"""The runtime and planner levers of the shipped prefill plans, on the text-to-image pipeline.

RUNTIME is ViT's runtime line (bootstrap_placements/vit_b16_224: the dense image encoder, StC-first on every refresh)
with Llama prefill's runtime levers (bootstrap_placements/llama31_8b_prefill128: sync-free / cache, refresh pairing,
the key bundle's S2D merge / StC P' / level keys, SparseB K 16 + its w14 EvalMod table, the compiler's mult-sum and
sparse level budget, the warm-up), restricted to the levers whose consumer is on this pipeline's path: the bootstrap
and key-switch runtime, the program compiler, impl.poly, impl.pcmm, impl.blocked. The GPT-2 / Llama model switches
(SM_*, PF_SM_*, PF_ATTN*, PF_GLU_*, PF_SILU_SINGLE, ...) have no consumer here; their ideas are ported into the SANA /
Gemma kernels themselves. RECIPE is ViT's planner recipe (Llama prefill's whole recipe on SANA: +30% refreshes per
block and a destructive refresh in block 24's cross-attention, 11-10). A plan carries both (PLAN_CMD.txt `runtime=`
/ `recipe=`, written by make_plan.sh), and a run under `--plan` takes the plan's runtime.
"""
from __future__ import annotations

import os

RUNTIME = {
    # chain + bootstrap
    "CHAIN_PRESET": "n32a4", "BTS_MAX_INPUT_LEVEL": "50", "CHEB_BSGS": "1",
    "FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_CF": "13", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
    "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": "0", "FIDESLIB_BTS_REAL": "1", "FIDESLIB_BTS_SHIFT": "1",
    "FIDESLIB_BTS_RAISE_DROPS": "1,2,3,4,5,6,7,8", "FIDESLIB_BTS_RAISE_DROP": "512:1",
    "FIDESLIB_BTS_RAISE_NOSTC": "2,6", "FIDESLIB_BTS_RAISE_STC_ONUSE": "1",
    "FIDESLIB_BTS_SPARSE_B": "1", "FIDESLIB_BTS_SPARSE_B_K": "16", "FIDESLIB_BTS_SPARSE_B_MAX": "1024",
    "PC_BTS_SB_EM_TABLE": "w14", "SPARSE_BTS_SLOTS": "512,1", "SPARSE_LEVEL_BUDGET": "3,3",
    "FIDESLIB_CTS_ALT_BUDGET": "5", "FIDESLIB_ROT_KEY_BAND": "32", "FIDESLIB_SPRU": "64",
    "PC_BTS_PAIR": "1", "PC_BTS_PAIR_CF_MAX": "15", "PC_BTS_PAIR_LOOSE": "1", "PC_BTS_PAIR_WINDOW": "6144",
    "PC_BTS_MERGE_S2D": "1", "PC_BTS_STC_PS": "6", "PC_EM_LEVEL_KEYS": "1", "PC_EM_FUSE_RESCALE": "1",
    "PC_MULT_SUM": "1", "PF_WARMUP": "1",
    # sync-free / cache (Llama prefill)
    "FIDESLIB_STC_POW2_KEY": "1", "FIDESLIB_STC_CACHE_RELEASE": "1", "FIDESLIB_STC_LAST_CACHE_FREE_GB": "4",
    "FIDESLIB_GPU_ENCODE": "1", "FIDESLIB_ENCODE_NOSYNC": "1", "FIDESLIB_ENCODE_PINNED_RING": "256",
    "FIDESLIB_PCMM_ASYNC": "1", "FIDESLIB_PCMM_NOSYNC": "1", "FIDESLIB_PCMM_GEMM_MB": "256",
    "FIDESLIB_PCMM_OUT_CHUNK": "8", "FIDESLIB_DUMP_FULL": "1",
    # impl
    "IMPL_DEAD_OPS": "1", "IMPL_WSUM": "1", "PF_CHEB_LEAF_NATIVE": "1", "LADDER_RADIX_BITS": "3", "LN_CHEB": "1",
    "PF_LAYOUT": "blocked", "PF_LINEAR": "pcmm", "PF_PCMM_STAGED_ONLY": "1",
    # 24-bit weights (3 int8 planes). Llama prefill's 16 (2 planes) measured 11-11: -8% SANA bootstraps but the twin's
    # latent 3.9% -> 8.6% (the quantized weights; T rows 1.1e-5 -> 2.6e-3): kept at 24 for the image quality
    "PF_PCMM_WBITS": "24",
    # the output projections' finishes one raise drop down (ViT's out:2,down:2 on SANA's names); qx:2 measured -1%
    # on GPU 4 (block 0, 10-31: qkv:2 -1% at +18% error, ff_in:2 and all-of-them +1%)
    "PF_PCMM_RAISE": "out1:2,out2:2,ff_out:2,qx:6,ff_in:1,b.tv:5,g.s_pub:4,g.pv_pub:5",   # ff_in:1: act1 + dw still end at 46 (planner audit)
    # Llama prefill's raise drops on Gemma and SANA (11-11): each the most the bound plan's deliberate-drop probe
    # (PLAN_RAISE_DELIB=1, Gemma layers 1-24, SANA blocks of both steps: the same per layer) let a finish take on top
    # of the above, model-scoped (g. Gemma's g<k>.*, b. SANA's b<k>.* and s2.b<k>.*; impl/pcmm._raise_of)
    # the session's StC-first last-stage cache sized to the most keys one stage asks for, not every stage's (the text
    # and the image stages' keys differ: ~25 GB resident for the full pipeline; pc-py session.rs, eager only)
    "PC_STC_CACHE_PER_PROGRAM": "1",
    # the PCMM weights pooled per stage: the session holds the most one stage finishes (a Gemma layer's ~0.5 GB, not
    # the 16.7 GB of every stage's; pc-py pcmm.rs), uploaded before each stage (planned.SanaPlannedPrefill)
    "PC_PCMM_WEIGHTS_PER_PROGRAM": "1",
    "PF_PCMM_RAISE_HI": "g.gate_up:5",
    "PF_PCMM_RAISE_OUT": "g.qkv:0:1,g.qkv:1:1,g.qkv:4:5,b.qkv:0:1,b.qkv:1:1,b.qkv:2:1,b.qkv:6:2,b.qkv:7:2,b.qkv:8:3",
    # Llama prefill's model levers ported to the Gemma / SANA kernels (11-11; A/B in the session notes): the gate's
    # GELU domain folded into its weights, the softcap / exp domain maps folded into the score producers and the
    # softcap's coefficients (exact), the joint refreshes deliberate, the SiLU series by Llama's absolute error rule
    # PF_JOINT_FORCE off: on SANA it forced +1,304 deliberate joint refreshes per step against -350 hints (run E's plan,
    # 11-11: 12,818 vs 12,386 per step); the planner's hints skip most of them
    "PF_GELU_FOLD": "1", "PF_SM_FOLD": "1", "PF_JOINT_FORCE": "0", "PF_SILU_SINGLE_ERR": "2e-2",
    # a PCMM input whose stamp exceeds the 2^20 the StC-first factors are set for takes band shift 2 (pre-divided):
    # at shift 1 it skipped the pre-division and came out 2.7% wrong (ff_in at 24-bit weights; sana/impl.Pcmms.get)
    "PF_PREDIV_SHIFT_FIX": "1",
    # the cross-attention keys' 1 / (sqrt(DX) M) on the scores' window mask instead of the key rows:
    # the keys were refreshed at max |x| ~0.002 (rel. err ~2.5%); -16 refreshes per SANA block, exact
    "PF_XA_KEY_FOLD": "1",
}

# runtime keys the drivers' process environment carries (impl.pcmm / impl.poly's, the session's env knobs); the
# others reach the session as Config fields (bert.planned.plan_config)
ENV_KEYS = ("PF_", "PC_STC_CACHE_", "PC_PCMM_WEIGHTS_")

RECIPE = {
    "PLAN_DELIB_DEG": "captured", "BTS_LEVEL": "28", "PLAN_SPARSE_BTS_OUT": "1:16,512:26",
    "PLAN_RAISE_SEARCH": "1", "PLAN_RAISE_DROP_MAX": "13", "PLAN_RAISE_LANDING_MAX": "42",
    "PLAN_RAISE_ROUTES": "0,512,1", "PLAN_RAISE_DROP_SET": "1,2,3,4,5,6,7,8", "PLAN_RAISE_DROP_BASE": "512:1",
    "PLAN_SLIM_CUBIC": "1", "PLAN_STC_FIRST_ENTRY": "46", "PLAN_STC_FIRST_FALLBACK": "34:1",
    "PLAN_STC_FIRST_REROUTE": "1", "PLAN_REFRESH_ENV_CAP_ROUTES": "0:46", "PLAN_REFRESH_ENV_CAP": "48",
    "PLAN_ENVELOPE_STRICT": "1", "PLAN_ENV_EFFECTIVE": "1", "PLAN_PCMM_PREDIV_CAP": "42", "PLAN_DRIFT_RECUT": "1",
}


def save_capture_inputs(capture, configs, pcmm_calib_in):
    """A capture's model inputs (<capture>/sana_capture.json): its approximation configs (as hf.data reads them) and
    PCMM calibration, which make_plan.sh's plan carries (PLAN_CMD.txt configs=, pcmm_calib.json)."""
    import json
    from . import hf
    with open(os.path.join(capture, "sana_capture.json"), "w") as f:
        json.dump({"configs": configs, "pcmm_calib": hf.data(pcmm_calib_in) if pcmm_calib_in else None}, f, indent=1)


def plan_inputs(a):
    """`a.configs` and `a.pcmm_calib_in` unless given: a --plan run's from its plan (the Llama / ViT / BERT runners'
    convention: PLAN_CMD.txt configs=, the plan's pcmm_calib.json as PF_PCMM_CALIB_IN), else the shipped ones
    (hf.CONFIGS, hf.PCMM_CALIB)."""
    from . import hf
    from perseus import config
    from perseus._paths import resolve
    if a.plan:
        d = resolve(a.plan)
        if a.configs is None:
            a.configs = config.plan_configs(d)
        cal = os.path.abspath(os.path.join(d, "pcmm_calib.json"))
        if not a.pcmm_calib_in and not os.environ.get("PF_PCMM_CALIB_IN") and os.path.exists(cal):
            os.environ["PF_PCMM_CALIB_IN"] = cal
    if a.configs is None:
        a.configs = hf.CONFIGS
    if not a.pcmm_calib_in and not os.environ.get("PF_PCMM_CALIB_IN"):
        a.pcmm_calib_in = hf.PCMM_CALIB


def runtime(plan=None, sets=()):
    """The run's switches: the plan's runtime= line when it has one, else RUNTIME; `sets` (KEY=VALUE) win."""
    from perseus import config
    rt = dict(RUNTIME)
    if plan and config._plan_cmd(plan).get("runtime") is not None:
        rt = config._runtime_line(config._plan_cmd(plan)["runtime"])
    rt.update(kv.split("=", 1) for kv in sets)
    for k, v in rt.items():            # impl.pcmm / impl.poly / kernels read theirs at import
        if k.startswith(ENV_KEYS):
            os.environ[k] = str(v)
    return rt
