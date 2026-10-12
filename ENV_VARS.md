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
| `FIDESLIB_BTS_SPARSE_B` | Depth-1 CtS/StC on the 512-slot bootstrap route (eprint 2026/1023): it lands 10 primes higher and takes no raise variants, so the model rotation keys need `FIDESLIB_ROT_KEY_BAND=32` (on in the GPT-2 port's n32 runtime; off for plans that do not declare it). |
| `FIDESLIB_BTS_SPARSE_B_MAX` | SparseB only on routes of at most this many slots; larger routes stay multi-stage (default: all). |
| `FIDESLIB_BTS_ROUTE_K24` | 0 keeps the shipped K = 16 EvalMod on multi-stage sparse routes (default: SparseB's K = 24 series). |
| `FIDESLIB_SPARSE_ARCSINE` | Arcsine-corrected sparse bootstrap (needed by the encrypted argmax). |
| `OPENFHE_DECODE_NO_THROW` | Let a capture decode a low-precision plaintext instead of throwing (capture only). |
| `PLAN_HARD_ENV_CAP` | Planner: make the refresh-input depth envelope absolute (1) or payable (0, the shipped recipes). |
| `PLAN_REFRESH_ENV_CAP_STEPS` | Planner: per-step refresh-input envelope, `name:cap,...` for nodes whose step contains `name` (e.g. `cutmax:48,tail:48` keeps the measured-safe 48 there while PLAN_REFRESH_ENV_CAP=50 serves the decode blocks). |
| `PLAN_DRIFT_RECUT` | Planner: 1 = un-place a cut-chosen refresh whose input drifted past the envelope after a later cut disarmed a hint upstream, and let the cut re-cover it (kept as forced if nothing envelope-respecting exists). Off by default: the shipped plans carry such sites and the runtime guard reads the nominal level. P3c reports them either way. |
| `PLAN_SHARED_REALIZE` | Planner: a deg-2 value read only by products or key switches (two or more) gets its pending rescale realized once, as a realize anchor, instead of one rescale per consumer (default 1 in make_plan.sh; empty or 0 = off). |
| `PLAN_KS_REALIZE` | Planner: realize anchors before rotations / conjugations that read a deg-2 value (off by default: neutral alone). |
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
| `IMPL_SCOPED_NAMES` | Step-scoped ciphertext names (`<step path>_<n>` on a per-step counter, the q.K^T scope keyed by the cache's group count) so one plan stays bound at every cache length; use with `SM_PERIODIC=0` (Python implementation; default 0, off for plans that do not declare it). |
| `IMPL_PT_SUM_FUSED` | A streamed weight's inner sums encoded on the device from its raw slot rows (`Inference.mult_pt_sum_raw`) instead of host plaintexts; superseded by the coefficient rows (Python implementation; default 0). |
| `LLAMA_MODEL` | Llama checkpoint the Llama port reads (HF repo id or directory; default `meta-llama/Llama-3.1-8B`). |
| `LLAMA_PACKING` | Slot packing of the Llama port (`cachemir_complex`, default, or `cachemir`). |
| `LLAMA_LINEAR_BOTTOM` | Llama port: these projections (qkv, out_proj, gate_up, down; `1` = all) run at the bottom of the chain (default 0). |
| `LLAMA_PLAN_SPARSE_SLOTS` | Llama `make_plan.sh`: the sparse routes the plan may use (default `256,1`, from Llama's 256-slot attention block). |
| `KV_LANES` | The K + iV linear output refreshed as one real payload, K on token lane 0 and V on lane 1, by the real-payload route instead of a complex refresh (Python implementation; default 1, off for plans that do not declare it). |
| `QKV_LANES` | The Q linear output rides the K/V push refresh on token lane 2 (one real refresh for Q, K and V), and the qkv / up input hints fire only past the session limit, so both linears run on the LayerNorm output as is and their OUTPUT is refreshed at the top of the envelope (Python implementation; default 1, off for plans that do not declare it; the plan's runtime line carries `BTS_MAX_INPUT_LEVEL=50` for the up output at nominal 50). |
| `CHEB_BSGS` | Every Chebyshev series (the GELU's P1/P2, the LayerNorm seed, the softmax exp) evaluated by baby-step giant-step: a degree-31 series costs 11 ciphertext products instead of 30 at the same depth (Python implementation; default 1, off for plans that do not declare it). |
| `LIN_PT_DOT` | A resident-weight linear's baby-step inner sums (sum of the rotated inputs times their weight plaintexts) in one fused kernel each instead of a plaintext product and an add per term; bit-exact, same recorded op stream (Python implementation; default 1, 0 = the loop). |
| `SM_HRS_PACK` | The softmax head sum and its seed-scaled copy (SM_FOLD) ride one complex mask `(0.5 + 0.5i seed) pos0` and one replicate ladder, split by one conjugation: four key switches fewer per reciprocal round (Python implementation; default 1, off for plans that do not declare it). |
| `CUTMAX_PAIR` | Two passes of the CutMax inverse-sigma cascade share one 1-slot refresh as `u_a + i u_b` (the refresh of u_j is off the path to u_{j+1}); 11 -> 7 cascade refreshes per token (Python implementation; default 1, off for plans that do not declare it). |
| `IMPL_DEAD_OPS` | Graph ops nothing reads are not emitted: the odd GELU series' T_6, the LayerNorm Newton constant when the seed takes no Newton iteration (Python implementation; default 1, off for plans that do not declare it). |
| `LADDER_RADIX_BITS` | Rotate-and-sum ladders as hoisted radix-2^b groups: each group of b rungs is one ModUp, 2^b - 1 key dots summed in the extended basis and one ModDown (`Ciphertext::rotateSumHoisted`); default 3, 0/1 = the plain ladder; adds the groups' rotation keys (off for plans that do not declare it). |
| `CHEB_ALIGN` | Chebyshev leaves read their babies pre-aligned to the deepest baby each leaf uses (an add of an exact zero there), instead of the runtime re-aligning the running sum at every depth step (Python implementation; default 0; a plan captured with it declares it). |
| `IMPL_REALIZE_SHARED` | Realize the pending rescale once on the BSGS Chebyshev bases and the LayerNorm's centered input in the runtime (default 0: the planner's PLAN_SHARED_REALIZE anchors cover these and more, and compose with CHEB_ALIGN). |
| `SM_FOLD` | The softmax's constant products folded into masks: the exp's input map into the q.K^T group and score masks, each reciprocal round's seed slope into a second head-sum mask (Python implementation; default 1, off for plans that do not declare it). |
| `SM_PERIODIC` | Attention scores kept 512-periodic while the cache fits one 32-token group: the softmax refreshes route sparse and P.V is one product (Python implementation; default 1, off for plans that do not declare it). `2` carries it past one group: with G groups, period Gp x 512 (Gp the power of two >= G; group g in every block b with b % Gp == g), the head sums ladder over one period and P.V takes one product per class of lanes; the Gp-dependent ops run in sub-steps keyed by Gp, so with `IMPL_SCOPED_NAMES=1` one token-0 plan (planned with `FORBID_STEPS=...,.gp`) serves every length, and the session builds the Gp x 512-slot route as the cache grows (`SM_ROUTE_AHEAD`). |
| `SM_ROUTE_AHEAD` | SM_PERIODIC=2: build the next length class's sparse route this many tokens early (default 0). |

## Memory

| variable | meaning |
|---|---|
| `FHE_MAG_RING_GB` | Pinned ring the asynchronous magnitude capture stores ciphertexts in. |
| `FHE_PREFILL_ENCODE_THREADS` | Threads encoding prefill weights. |
| `FHE_PT_COEFF_ENCODE` | Stage weights as raw coefficients and expand them on the GPU. |
| `FHE_PT_STAGE_BLOCK` | Plaintext limbs staged per block (0 = no staging). |
| `FHE_STAGE_ARENA_GB` | Pinned host arena each staging half gets. |
| `FHE_STAGE_RELEASE_CPU` | Release the host copy of a staged plaintext after upload. |
| `FIDESLIB_GPU_ENCODE` | Encode full-slot plaintexts on the GPU; an evicted one reloads from a pinned dump of its first composite group, lifted back on the GPU (on in the GPT-2 port's environment). |
| `FIDESLIB_GPU_ENCODE_CHECK` | Check every such reload against a fresh device encode, limb for limb. |
| `PERSEUS_LLAMA_CACHE` | Disk cache of the Llama slot vectors and their coefficient rows (default `$HF_HOME/perseus/llama_slots`; ~75 GB). |
| `FIDESLIB_COEFF_CHUNK` | Plaintexts per batch of a coefficient-row weight sum (`EvalMultCoeffRowsSum`; default 32 fused, 16 otherwise; at most 32). |
| `FIDESLIB_COEFF_FUSED` | Coefficient-row sums as one fused kernel (default 1). |
| `FIDESLIB_COEFF_WAVES` | Row groups of the fused sums, in waves of two blocks per SM (default 2). |
| `FIDESLIB_ROT_KEY_BAND` | Keep only rotation keys whose step is within +-2^band (-1 = all); the 32-bit decode row uses 22. |
| `KV_ARENA_GB` | Pinned host arena for the KV cache. |
| `IMPL_KV_OFFLOAD` | 1 parks each block's K/V cache ciphertexts in the pinned KV arena between uses (Python implementation); off by default: the caches stay on the device (2.2 GB on the n32 decode) and the per-stage fences go away. |
| `IMPL_NODUMP_ERASE` | A device-encoded plaintext erased from the encode cache (a per-step mask) is freed without its host dump (default 1; 0 restores the 2 x D2H + syncs per mask). |
| `IMPL_KEEP_MASKS` | The encode cache's device copies survive the end of the token instead of being evicted (default 1). |
| `IMPL_RESIDENT_BLOCKS` | Block stages whose weight plaintexts stay on the device across tokens (default all: no per-token reload, lift or per-block device drain; the GPT-2 n32 decode keeps ~29 GB free); 0 evicts every block after its stage. |
| `PERSEUS_SLOTVEC_CACHE` | Disk cache of the GPT-2 port's prepared slot vectors (default on; 0 = off). |
| `PERSEUS_SLOTVEC_CACHE_GB` | Size cap of that cache; least recently used model variants past it are dropped (default 32). |
| `PERSEUS_SLOTVEC_CACHE_VARIANTS` | Model variants (weights, configs, builder code) kept per model family in that cache (default 2). |
| `MALLOC_ARENA_MAX` | glibc allocator arenas; 2 keeps host fragmentation down on the decode row. |
| `OMP_NUM_THREADS` | OpenMP threads the host-side encode and staging use. |
| `FIDESLIB_PT_SCRATCH` | 0 keeps the NTT scratch of finished plaintexts (device-encoded weights and masks, bootstrap tables, SparseB copies); by default it is released, halving their footprint (a plaintext is only read once built). |
| `FIDESLIB_PT_COMPACT` | 0 stores the bootstrap stage tables in full (default: one value per NTT block). |
| `FIDESLIB_AUX_POLY_MAX` | Cap on FIDESlib's polynomial recycler (destroyed ciphertexts' polynomials kept with their limbs for reuse; default unbounded). Without recycling a token is ~4x slower; a full device frees the recycler before failing either way. |
| `FIDESLIB_BTS_RAISE_EAGER` | 1 builds every raise variant (`FIDESLIB_BTS_RAISE_DROPS`) at context load; by default each is built when a planted bootstrap first asks for it (each is a full copy of a route's tables). |
| `FIDESLIB_POOL_TRACE` | Record a backtrace per live pooled block: `_core.gpu_pool_trace()` lists the live bytes by allocation site (diagnostic, slow). |

## Keys

| variable | meaning |
|---|---|
| `FIDESLIB_KSK_PACK` | Store the `b` half as a dense 28-bit bit-stream (default 1). |
| `FIDESLIB_KSK_REGEN` | Regenerate the `a` half of every key-switching key in-kernel from its seed (2 = both readers, the default; 0 = stored keys). |
| `FIDESLIB_KS_DIGIT_INTT` | 1 restores upstream's redundant per-digit INTT in the key switch. |
| `FIDESLIB_SMALLINT_NOSYNC` | 0 restores the exact post-raise scaling's host readback of the limb pointer tables (a stream sync and two blocking device-to-host copies per call, two device drains per bootstrap). |
| `FIDESLIB_AUX_NOSYNC` | 0 restores the aux-buffer release's read-back of the limb pointer tables behind a stream sync (by default the live buffers are normalized on the device). |
| `FIDESLIB_SCRATCH_RETIRE` | 0 restores exact-size ciphertext scratch slots that grow behind a stream sync and a device-wide cudaFree inside bootstraps (by default a slot starts at a floor, grows geometrically and keeps outgrown buffers until the ciphertext dies). |
| `FIDESLIB_SPRU_NOSYNC` | 0 restores the SPRU bootstrap's three per-call cudaDeviceSynchronize fences (156 device drains per decode token). |

## Profiling

| variable | meaning |
|---|---|
| `FHE_GRAPH_CAPTURE_TOKEN` | Capture the op-graph of one token only. |
| `FHE_PROFILE` | Per-operation timing table on stderr (default off). |
| `FHE_PROFILE_TOKEN` | Restrict that profile to one token index. |
| `LLAMA_PROFILE_STEPS` | Llama `run_decode --profile`: also each step's primitive split. |
| `IMPL_PROFILE_SYNC` | `run_decode --profile` (Python implementation): a device fence after every timed primitive, so each is charged its own GPU time. |
| `PT_SUM_TIMING` | Host-side timing of the streamed-weight inner sums on stderr. |
| `FIDESLIB_PTDOT_VERIFY` | Checking only: every fused inner sum (`LIN_PT_DOT`) is recomputed as the product-and-add loop and compared residue for residue on stderr. |
| `PLAN_TRACE_STEP` | Every op recorded in that step path (e.g. `blk0.softmax`), as executed, on stderr (plan/run divergence hunts). |
| `PLAN_TRACE_DECRYPT` | With `PLAN_TRACE_STEP`: also each traced op's decrypted max \|z\| and max \|Im\| (where a planned run's values go wrong). |
| `FIDESLIB_ERRCHECK` | 1 polls cudaGetLastError after every FIDESlib launch (off by default: ~1.8M calls per decode token). |
| `FIDESLIB_OMP_REGIONS` | 1 enters FIDESlib's OpenMP regions on a single GPU (off by default: they run inline). |
| `FIDESLIB_AKS` | DEPRECATED: aggregated key switching for the first CoeffsToSlots stage (ePrint 2025/1403). Measured slower on the n32 decode (+0.03..+0.06 s/token) and its key generation uses a predictable error term, so it is refused unless `FIDESLIB_AKS_ALLOW_INSECURE=1`. |
| `FIDESLIB_AKS_ALLOW_INSECURE` | Lets the deprecated `FIDESLIB_AKS` run anyway, for timing experiments only (its keys are not secure). |
| `FIDESLIB_BCONV2` | Key-switch base conversions: 2 (default) = the shared-memory ModUp and ModDown kernels, 1 = ModUp only, 0 = the original kernels (same residues either way). |
| `FIDESLIB_LAZY_PS` | EvalMod's Chebyshev series sums its top-level products under one relinearization (lazy-BSGS Paterson-Stockmeyer, default 1; 0 = one relinearization per product). |
| `FIDESLIB_MODRAISE_COEFF` | ModRaise keeps the raised polynomials in coefficient form, because the exact post-raise scaling reads only the first limbs back in coefficient form and rewrites every limb before its own NTT (bit-exact; default 1, 0 = the NTT of all raised limbs). |
| `FIDESLIB_BTS_SPARSE_B_BSTEP` | Baby step of the SparseB CtS / StC linear transforms (default min(s, 16): on the 512-slot route 16 hoisted rotations and one giant step; s = every diagonal hoisted, the former default). |
| `FIDESLIB_GPU_ENCRYPT` | Public-key encryption on the GPU (default 1; 0 = OpenFHE host encryption): same algorithm and distributions as OpenFHE's, randomness from ChaCha12 keyed per context from the OS entropy pool. |
| `FIDESLIB_GPU_ENCRYPT_DEBUG` | Per-call stage timing of the device encryption on stderr. |
| `FIDESLIB_NOCLONE` | API wrappers that write their result once instead of cloning an operand first: 1 = rotations / conjugation / batch products, 2 (default) = also add / mult / mult-pt / square, 3 = also sub and scalar ops and fast rotations, 0 = always clone. |
| `FIDESLIB_CLONE_CENSUS` | Count every device clone per call stack, printed at exit. |
| `FIDESLIB_FAST_DECRYPT` | Decrypt a device ciphertext in an empty host container instead of one built by a dummy encryption (default 1; identical result, ~30 ms less per decrypt). |
| `FIDESLIB_DECRYPT_DEBUG` | Per-call stage timing of the host decrypt on stderr. |
