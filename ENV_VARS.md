# Environment variables

Every knob the published tree reads, grouped by what it configures. Values are
defaults unless a script sets them; `scripts/local_env.sh` and `scripts/run_task.sh`
carry the configuration the paper measured, and an exported variable always wins.
Regenerate this file with `python scripts/utils/scan_env_vars.py`.

## Session

| variable | meaning |
|---|---|
| `ALL_BLOCKS_IO_DIR` | Teacher-forced decode oracle the gates compare against. Every run of the driver requires it, even one that feeds its own rows. |
| `CHAIN` | `n32` (the paper's 32-bit composite chain, default) or `n64` (the 64-bit reference); selects deps_<chain>/, build_py_<chain>/ and the extension stash. |
| `CONFIGS_PATH` | Approximation config the model runs with (configs/model/approximation/<name>/configs.json). Required. |
| `CUDA_VISIBLE_DEVICES` | GPU this process uses; the runtime has no per-context device selection, so it must be set before the first session. |
| `DECODE_TOKENS` | Tokens decoded after a prefill hand-off. |
| `GEN_FEEDBACK` | How a generation loop feeds the next token back: `encrypted` (the CutMax argmax, default) or `client` (decrypt, pick, re-encode). Read by notebooks/gpt2_perseus_nn.ipynb. |
| `GEN_PROMPT` | Prompt length for the generation modes. |
| `GEN_TOKENS` | Tokens generated after the prompt. |
| `GPT2_CACHE` | Keep the encoded block weights across tokens (default 1). |
| `GPT2_INFERENCE_MODE` | `threaded` (default), `prefetch` or `sync`; sync is required for a capture. |
| `GPT2_MODEL` | HuggingFace checkpoint the notebooks load the tokenizer and embeddings from (openai-community/gpt2). |
| `GPT2_PACKING` | Slot packing of the GPT-2 driver (`cachemir`). |
| `MULTI_T` | Tokens a decode session runs (16 in the paper's row). |
| `PERSEUS_CLIENT_EXTENSION` | Force the client role onto `core` or `client` instead of whichever extension loads. |
| `PERSEUS_DATA` | Root the run scripts resolve the weights and oracle under (default `.cache/`). |
| `PERSEUS_FATAL_EXIT` | Install the terminate handler that exits the process on a fatal CUDA error instead of unwinding. |
| `PERSEUS_PROMPT` | Prompt text for the notebook generation demos (GEN_PROMPT is a token count, not text). |
| `PIN_PROFILE` | Pass SessionProfile.gpt2_decode_n32() explicitly instead of taking the CKKS parameters from the environment. Read by notebooks/gpt2_perseus_nn.ipynb. |
| `PREFILL_TOKENS` | Prompt length for the prefill modes. |
| `STEPS_T` | Context length the KV cache is sized for (128). |
| `TEACHER_FORCED` | Feed the oracle's tokens instead of the model's own argmax. |
| `WEIGHTS_PATH` | Exported weights archive (weights.bin.zip). Required. |

## CKKS

| variable | meaning |
|---|---|
| `AUTO_BTS_LEVEL` | Depth at which an unplanned ciphertext is refreshed reactively. |
| `BTP_DEPTH_OVERHEAD` | Levels the bootstrap consumes. |
| `BTP_SCALE_BITS` | Scaling factor bits of the bootstrap's own levels. |
| `BTS_ITERATIONS` | Meta-bootstrap iterations (1). |
| `BTS_PRECISION` | Target precision bits of the bootstrap. |
| `CKKS_COMPLEX` | Carry a second real payload in the imaginary lane. |
| `CKKS_DEPTH` | Usable multiplicative levels before the bootstrap overhead. |
| `COMPOSITE_DEGREE` | Primes per level (2 on the 32-bit chain, 1 on the 64-bit one). |
| `CORRECTION_FACTOR` | Default correction factor of a bootstrap when a plan does not type the site. |
| `FIRST_MOD_BITS` | Bits of q0. |
| `H_WEIGHT` | Hamming weight of the sparse secret (0 = uniform ternary). |
| `LEVEL_BUDGET` | CoeffsToSlots:SlotsToCoeffs level budget of the bootstrap (`4:3`). |
| `LOGN` | Ring dimension exponent (16). |
| `NUM_LARGE_DIGITS` | Hybrid key-switching digits (dnum). |
| `SCALE_BITS` | Scaling factor bits per level (54 = 2 x 27-bit primes on the composite chain). |
| `SPARSE_LEVEL_BUDGET` | Level budget of the sparse bootstraps (default: the dense one). |

## Plan

| variable | meaning |
|---|---|
| `FHE_ASYNC_MAG` | Measure capture magnitudes on worker threads instead of inline (default 1). |
| `FHE_BOOTSTRAP_PLACEMENTS_DIR` | Plan directory a run binds to; unset = eager. |
| `FHE_DECODE_PLACEMENTS_DIR` | Plan used for the decode phase of a generation run. |
| `FHE_GRAPH_DIR` | Directory a capture writes the op-graph into (STAGE=capture). |
| `FHE_MAG_WORKERS` | Worker threads for that measurement. |
| `FHE_PREFILL_CAPTURE_RANGES` | Capture the per-chunk token ranges of a prefill. |
| `FHE_PREFILL_PLACEMENTS_DIR` | Plan used for the prefill phase of a generation run. |
| `FIDESLIB_SPARSE_ARCSINE` | Arcsine-corrected sparse bootstrap (needed by the encrypted argmax). |
| `OPENFHE_DECODE_NO_THROW` | Let a capture decode a low-precision plaintext instead of throwing (capture only). |
| `PLAN_HARD_ENV_CAP` | Planner: make the refresh-input depth envelope absolute (1) or payable (0, the shipped recipes). |
| `SPARSE_AUTO` | Route a periodic payload to a sparse bootstrap automatically (2 = on). |
| `SPARSE_BTS_SLOTS` | Sparse bootstrap precomputations to build, e.g. `512,1`; 0 = none. |
| `SPARSE_LN_BTS` | Route the LayerNorm refreshes sparsely. |
| `SPARSE_SM_BTS` | Route the softmax refreshes sparsely. |

## Model

| variable | meaning |
|---|---|
| `CACHE_READ_LEVEL_K` | Level the K cache is read at. |
| `CACHE_READ_LEVEL_V` | Level the V cache is read at. |
| `CUTMAX_ARCSINE` | Arcsine correction inside CutMax. |
| `CUTMAX_BTS_ITERS` | Bootstrap iterations of the CutMax refreshes. |
| `CUTMAX_BTS_PRECISION` | Precision bits of those refreshes. |
| `CUTMAX_PRECISE_SCOPED` | Use the precise CutMax schedule inside the argmax scope. |
| `CUTMAX_SPARSE_BTS` | Route the CutMax refreshes sparsely. |
| `CUTMAX_SPARSE_NO_ARCSINE` | Sparse CutMax refreshes without the arcsine correction. |
| `CUTMAX_VEC_BTS_ITERS` | Bootstrap iterations of the vectorized CutMax step. |
| `FHE_DELTA_BLOCK` | Delta-block attention for the filling (prefill) packing. |
| `FHE_LMHEAD_CAP` | Level the LM-head weights are encoded at. |
| `FUSED_LN_VAR` | Fold the LayerNorm variance refresh (default 0). |
| `FUSED_SM_DEN` | Refresh the softmax denominator inside the fold (C++ decode default 1; the Python implementation's `env.py` sets 0). |
| `GPT2_FOLD_LN1` | Fold the first LayerNorm's affine part into the following weights. |
| `GPT2_FOLD_LN2` | Same for the second LayerNorm. |
| `GPT2_FOLD_LNF` | Same for the final LayerNorm (default 0). |
| `GPT2_FOLD_LN_AFFINE` | Fold the affine scale/bias generally. |
| `GPT2_LMHEAD_GRANULARITY` | `plaintext` or `linear` weight granularity for the LM head. |
| `GPT2_PREFILL_GRANULARITY` | Same for the prefill linears. |

## Memory

| variable | meaning |
|---|---|
| `FHE_MAG_RING_GB` | Pinned ring the asynchronous magnitude capture stores ciphertexts in. |
| `FHE_PREFILL_ENCODE_THREADS` | Threads encoding prefill weights. |
| `FHE_PT_COEFF_ENCODE` | Stage weights as raw coefficients and expand them on the GPU. |
| `FHE_PT_STAGE_BLOCK` | Plaintext limbs staged per block (0 = no staging). |
| `FHE_STAGE_ARENA_GB` | Pinned host arena each staging half gets. |
| `FHE_STAGE_RELEASE_CPU` | Release the host copy of a staged plaintext after upload. |
| `FIDESLIB_GPU_ENCODE` | Encode full-slot plaintexts on the GPU; an evicted one reloads from a pinned dump of its limbs. |
| `FIDESLIB_ROT_KEY_BAND` | Keep only rotation keys whose step is within +-2^band (-1 = all); the 32-bit decode row uses 22. |
| `KV_ARENA_GB` | Pinned host arena for the KV cache. |
| `PERSEUS_SLOTVEC_CACHE` | Disk cache of the GPT-2 port's prepared slot vectors (default on; 0 = off). |
| `MALLOC_ARENA_MAX` | glibc allocator arenas; 2 keeps host fragmentation down on the decode row. |
| `OMP_NUM_THREADS` | OpenMP threads the host-side encode and staging use. |

## Keys

| variable | meaning |
|---|---|
| `FIDESLIB_KSK_PACK` | Store the `b` half as a dense 28-bit bit-stream (default 1). |
| `FIDESLIB_KSK_REGEN` | Regenerate the `a` half of every key-switching key in-kernel from its seed (2 = both readers, the default; 0 = stored keys). |
| `FIDESLIB_KS_DIGIT_INTT` | 1 restores upstream's redundant per-digit INTT in the key switch. |

## Profiling

| variable | meaning |
|---|---|
| `FHE_GRAPH_CAPTURE_TOKEN` | Capture the op-graph of one token only. |
| `FHE_PROFILE` | Per-operation timing table on stderr (default off). |
| `FHE_PROFILE_TOKEN` | Restrict that profile to one token index. |
