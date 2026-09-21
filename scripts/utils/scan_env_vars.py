#!/usr/bin/env python3
"""Regenerate ENV_VARS.md from the environment variables the tree actually reads.

Scans `getenv("X")` in src/ and include/, the FIDESlib API layer, and `os.environ` in
perseus/, and prints one table row per variable with the description below. A variable
found in the code with no description here exits non-zero, so the document cannot drift.

  python scripts/utils/scan_env_vars.py            # write ENV_VARS.md
  python scripts/utils/scan_env_vars.py --check    # only verify every knob is described
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

# name -> (group, description). Groups order the document.
DOC = {
    # ── what to run ───────────────────────────────────────────────────────────────────
    "CHAIN": ("Session", "`n32` (the paper's 32-bit composite chain, default) or `n64` (the 64-bit reference); selects deps_<chain>/, build_py_<chain>/ and the extension stash."),
    "CONFIGS_PATH": ("Session", "Approximation config the model runs with (configs/model/approximation/<name>/configs.json). Required."),
    "WEIGHTS_PATH": ("Session", "Exported weights archive (weights.bin.zip). Required."),
    "ALL_BLOCKS_IO_DIR": ("Session", "Teacher-forced decode oracle the gates compare against. Every run of the driver requires it, even one that feeds its own rows."),
    "PERSEUS_DATA": ("Session", "Root the run scripts resolve the weights and oracle under (default `.cache/`)."),
    "CUDA_VISIBLE_DEVICES": ("Session", "GPU this process uses; the runtime has no per-context device selection, so it must be set before the first session."),
    "GPT2_INFERENCE_MODE": ("Session", "`threaded` (default), `prefetch` or `sync`; sync is required for a capture."),
    "GPT2_PACKING": ("Session", "Slot packing of the GPT-2 driver (`cachemir`)."),
    "GPT2_CACHE": ("Session", "Keep the encoded block weights across tokens (default 1)."),
    "MULTI_T": ("Session", "Tokens a decode session runs (16 in the paper's row)."),
    "STEPS_T": ("Session", "Context length the KV cache is sized for (128)."),
    "PREFILL_TOKENS": ("Session", "Prompt length for the prefill modes."),
    "DECODE_TOKENS": ("Session", "Tokens decoded after a prefill hand-off."),
    "GEN_PROMPT": ("Session", "Prompt length for the generation modes."),
    "GEN_TOKENS": ("Session", "Tokens generated after the prompt."),
    "GEN_FEEDBACK": ("Session", "How a generation loop feeds the next token back: `encrypted` (the CutMax argmax, default) or `client` (decrypt, pick, re-encode). Read by notebooks/gpt2_perseus_nn.ipynb."),
    "GPT2_MODEL": ("Session", "HuggingFace checkpoint the notebooks load the tokenizer and embeddings from (openai-community/gpt2)."),
    "PERSEUS_PROMPT": ("Session", "Prompt text for the notebook generation demos (GEN_PROMPT is a token count, not text)."),
    "PIN_PROFILE": ("Session", "Pass SessionProfile.gpt2_decode_n32() explicitly instead of taking the CKKS parameters from the environment. Read by notebooks/gpt2_perseus_nn.ipynb."),
    "TEACHER_FORCED": ("Session", "Feed the oracle's tokens instead of the model's own argmax."),
    "PERSEUS_FATAL_EXIT": ("Session", "Install the terminate handler that exits the process on a fatal CUDA error instead of unwinding."),
    "PERSEUS_CLIENT_EXTENSION": ("Session", "Force the client role onto `core` or `client` instead of whichever extension loads."),
    # ── CKKS parameters ───────────────────────────────────────────────────────────────
    "LOGN": ("CKKS", "Ring dimension exponent (16)."),
    "CKKS_DEPTH": ("CKKS", "Usable multiplicative levels before the bootstrap overhead."),
    "SCALE_BITS": ("CKKS", "Scaling factor bits per level (54 = 2 x 27-bit primes on the composite chain)."),
    "BTP_SCALE_BITS": ("CKKS", "Scaling factor bits of the bootstrap's own levels."),
    "FIRST_MOD_BITS": ("CKKS", "Bits of q0."),
    "BTP_DEPTH_OVERHEAD": ("CKKS", "Levels the bootstrap consumes."),
    "NUM_LARGE_DIGITS": ("CKKS", "Hybrid key-switching digits (dnum)."),
    "COMPOSITE_DEGREE": ("CKKS", "Primes per level (2 on the 32-bit chain, 1 on the 64-bit one)."),
    "LEVEL_BUDGET": ("CKKS", "CoeffsToSlots:SlotsToCoeffs level budget of the bootstrap (`4:3`)."),
    "SPARSE_LEVEL_BUDGET": ("CKKS", "Level budget of the sparse bootstraps (default: the dense one)."),
    "CORRECTION_FACTOR": ("CKKS", "Default correction factor of a bootstrap when a plan does not type the site."),
    "AUTO_BTS_LEVEL": ("CKKS", "Depth at which an unplanned ciphertext is refreshed reactively."),
    "BTS_ITERATIONS": ("CKKS", "Meta-bootstrap iterations (1)."),
    "BTS_PRECISION": ("CKKS", "Target precision bits of the bootstrap."),
    "CKKS_COMPLEX": ("CKKS", "Carry a second real payload in the imaginary lane."),
    "H_WEIGHT": ("CKKS", "Hamming weight of the sparse secret (0 = uniform ternary)."),
    # ── plans, capture, sparse routing ────────────────────────────────────────────────
    "FHE_BOOTSTRAP_PLACEMENTS_DIR": ("Plan", "Plan directory a run binds to; unset = eager."),
    "FHE_DECODE_PLACEMENTS_DIR": ("Plan", "Plan used for the decode phase of a generation run."),
    "FHE_PREFILL_PLACEMENTS_DIR": ("Plan", "Plan used for the prefill phase of a generation run."),
    "FHE_GRAPH_DIR": ("Plan", "Directory a capture writes the op-graph into (STAGE=capture)."),
    "PLAN_HARD_ENV_CAP": ("Plan", "Planner: make the refresh-input depth envelope absolute (1) or payable (0, the shipped recipes)."),
    "FHE_ASYNC_MAG": ("Plan", "Measure capture magnitudes on worker threads instead of inline (default 1)."),
    "FHE_MAG_WORKERS": ("Plan", "Worker threads for that measurement."),
    "OPENFHE_DECODE_NO_THROW": ("Plan", "Let a capture decode a low-precision plaintext instead of throwing (capture only)."),
    "FHE_PREFILL_CAPTURE_RANGES": ("Plan", "Capture the per-chunk token ranges of a prefill."),
    "SPARSE_BTS_SLOTS": ("Plan", "Sparse bootstrap precomputations to build, e.g. `512,1`; 0 = none."),
    "SPARSE_AUTO": ("Plan", "Route a periodic payload to a sparse bootstrap automatically (2 = on)."),
    "SPARSE_LN_BTS": ("Plan", "Route the LayerNorm refreshes sparsely."),
    "SPARSE_SM_BTS": ("Plan", "Route the softmax refreshes sparsely."),
    "FIDESLIB_SPARSE_ARCSINE": ("Plan", "Arcsine-corrected sparse bootstrap (needed by the encrypted argmax)."),
    # ── model-level approximations ────────────────────────────────────────────────────
    "FUSED_SM_DEN": ("Model", "Refresh the softmax denominator inside the fold (default 1)."),
    "FUSED_LN_VAR": ("Model", "Fold the LayerNorm variance refresh (default 0)."),
    "GPT2_FOLD_LN1": ("Model", "Fold the first LayerNorm's affine part into the following weights."),
    "GPT2_FOLD_LN2": ("Model", "Same for the second LayerNorm."),
    "GPT2_FOLD_LNF": ("Model", "Same for the final LayerNorm (default 0)."),
    "GPT2_FOLD_LN_AFFINE": ("Model", "Fold the affine scale/bias generally."),
    "CUTMAX_PRECISE_SCOPED": ("Model", "Use the precise CutMax schedule inside the argmax scope."),
    "CUTMAX_VEC_BTS_ITERS": ("Model", "Bootstrap iterations of the vectorized CutMax step."),
    "CUTMAX_BTS_ITERS": ("Model", "Bootstrap iterations of the CutMax refreshes."),
    "CUTMAX_BTS_PRECISION": ("Model", "Precision bits of those refreshes."),
    "CUTMAX_SPARSE_BTS": ("Model", "Route the CutMax refreshes sparsely."),
    "CUTMAX_ARCSINE": ("Model", "Arcsine correction inside CutMax."),
    "CUTMAX_SPARSE_NO_ARCSINE": ("Model", "Sparse CutMax refreshes without the arcsine correction."),
    "FHE_DELTA_BLOCK": ("Model", "Delta-block attention for the filling (prefill) packing."),
    "FHE_LMHEAD_CAP": ("Model", "Level the LM-head weights are encoded at."),
    "CACHE_READ_LEVEL_K": ("Model", "Level the K cache is read at."),
    "CACHE_READ_LEVEL_V": ("Model", "Level the V cache is read at."),
    "GPT2_LMHEAD_GRANULARITY": ("Model", "`plaintext` or `linear` weight granularity for the LM head."),
    "GPT2_PREFILL_GRANULARITY": ("Model", "Same for the prefill linears."),
    # ── memory and staging ────────────────────────────────────────────────────────────
    "KV_ARENA_GB": ("Memory", "Pinned host arena for the KV cache."),
    "FHE_STAGE_ARENA_GB": ("Memory", "Pinned host arena each staging half gets."),
    "FHE_PT_STAGE_BLOCK": ("Memory", "Plaintext limbs staged per block (0 = no staging)."),
    "FHE_STAGE_RELEASE_CPU": ("Memory", "Release the host copy of a staged plaintext after upload."),
    "FHE_PT_COEFF_ENCODE": ("Memory", "Stage weights as raw coefficients and expand them on the GPU."),
    "FHE_PREFILL_ENCODE_THREADS": ("Memory", "Threads encoding prefill weights."),
    "FHE_MAG_RING_GB": ("Memory", "Pinned ring the asynchronous magnitude capture stores ciphertexts in."),
    "FIDESLIB_ROT_KEY_BAND": ("Memory", "Keep only rotation keys whose step is within +-2^band (-1 = all); the 32-bit decode row uses 22."),
    "MALLOC_ARENA_MAX": ("Memory", "glibc allocator arenas; 2 keeps host fragmentation down on the decode row."),
    "OMP_NUM_THREADS": ("Memory", "OpenMP threads the host-side encode and staging use."),
    # ── key-switching keys (paper Table 2) ────────────────────────────────────────────
    "FIDESLIB_KSK_REGEN": ("Keys", "Regenerate the `a` half of every key-switching key in-kernel from its seed (2 = both readers, the default; 0 = stored keys)."),
    "FIDESLIB_KSK_PACK": ("Keys", "Store the `b` half as a dense 28-bit bit-stream (default 1)."),
    "FIDESLIB_KS_DIGIT_INTT": ("Keys", "1 restores upstream's redundant per-digit INTT in the key switch."),
    # ── profiling ─────────────────────────────────────────────────────────────────────
    "FHE_PROFILE": ("Profiling", "Per-operation timing table on stderr (default off)."),
    "FHE_PROFILE_TOKEN": ("Profiling", "Restrict that profile to one token index."),
    "FHE_GRAPH_CAPTURE_TOKEN": ("Profiling", "Capture the op-graph of one token only."),
}
GROUPS = ["Session", "CKKS", "Plan", "Model", "Memory", "Keys", "Profiling"]

SCAN = [
    ("src", ("*.cu", "*.cpp")), ("include", ("*.h", "*.cuh")),
    ("perseus", ("*.py",)),
    ("third_party/FIDESlib/api", ("*.cpp", "*.hpp")),
    ("third_party/FIDESlib/src", ("*.cu", "*.cpp", "*.cuh")),
]
PAT = re.compile(r'getenv\("([A-Z_0-9]+)"|environ\.get\("([A-Z_0-9]+)"|environ\["([A-Z_0-9]+)"')


def found() -> set[str]:
    names = set()
    for sub, globs in SCAN:
        root = REPO / sub
        if not root.is_dir():
            continue
        for g in globs:
            for f in root.rglob(g):
                for m in PAT.finditer(f.read_text(errors="ignore")):
                    names.add(next(x for x in m.groups() if x))
    return names


def main() -> int:
    names = found()
    undocumented = sorted(n for n in names if n not in DOC)
    if undocumented:
        print("environment variables read by the code but not described in this script:", file=sys.stderr)
        for n in undocumented:
            print(f"  {n}", file=sys.stderr)
        return 1
    if "--check" in sys.argv:
        print(f"env vars: {len(names)} read, all described")
        return 0
    out = ["# Environment variables", "",
           "Every knob the published tree reads, grouped by what it configures. Values are",
           "defaults unless a script sets them; `scripts/local_env.sh` and `scripts/run_task.sh`",
           "carry the configuration the paper measured, and an exported variable always wins.",
           "Regenerate this file with `python scripts/utils/scan_env_vars.py`.", ""]
    for group in GROUPS:
        rows = sorted((n, d) for n, (g, d) in DOC.items() if g == group)
        if not rows:
            continue
        out += [f"## {group}", "", "| variable | meaning |", "|---|---|"]
        out += [f"| `{n}` | {d} |" for n, d in rows]
        out.append("")
    (REPO / "ENV_VARS.md").write_text("\n".join(out))
    print(f"ENV_VARS.md: {sum(1 for _ in DOC)} variables ({len(names)} read by the code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
