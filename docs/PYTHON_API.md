# Python API — quick reference

The public surface, one screen per concern. Everything here is importable from a
checkout with the extension built (README → Install); what works *without* the
extension is marked ⚙︎-free.

## Sessions

```python
from perseus import session, Session          # ⚙︎-free import
from perseus.profile import SessionProfile     # ⚙︎-free

with session(profile=SessionProfile.custom_n32(), family="gpt2") as s:
    s.inf            # the _core.Inference every Enc module binds to
    s.encrypt(v)     # 1-D vector (<= s.inf.size.dim real features) -> ciphertext
    s.decrypt(ct, d=768)
s.closed             # True; close() is idempotent
```

- `profile=` a `SessionProfile` (its env view is applied as *defaults*; exported env wins),
  `options=` an explicit `_core.InferenceOptions`, neither → the environment.
- `family` selects the rotation-key band: `"gpt2"` or `"generic"`.
- `device=` picks the GPU index (sets `CUDA_VISIBLE_DEVICES` before the runtime's first CUDA
  call; the runtime has no per-context device selection, so decide it for the first session).
- `mode=` overrides the residency scheduling (`"sync" | "prefetch" | "threaded"`); None
  follows the profile / options.
- Reproducibility: the computation is deterministic for fixed keys and inputs, but key
  generation and encryption draw from OpenFHE's RNG, which the binding cannot seed; a
  regression check compares against a plaintext reference (or a saved bundle), not against a
  previous run's ciphertexts.
- One live session per process (the runtime keeps one context): close it — or leave the
  `with` block — before opening the next; a second concurrent `session()` raises.
- `close()` releases what the session holds (weights, KV/mask/encode caches, the loaded
  rotation keys) to the runtime's device pool. The pool
  hands the memory to the next session in this process and returns it to the device at
  exit, so a driver-level free-memory counter (`nvidia-smi`, `cudaMemGetInfo`) is flat
  across a close; `Context.loaded_rot_steps` and `Inference.installed_weights` show what
  is held. The CKKS context (keys, bootstrap precomputation — including the rotation
  keys the bootstrap shares with the model band, 46 of 133 on the GPT-2 n32 band) lives
  until exit.

`SessionProfile` presets: `gpt2_decode_n32()`, `gpt2_decode_n64()`, `custom_n32()`,
`custom_n64()`. Fields are the semantics-bearing knobs (chain, levels, LN folds, cache read
levels); `mode` picks the residency scheduling and `extra` is a free-form env mapping the
presets also use for planner units and the rotation-key band. `profile.env()` shows the
exact exports.

## Modules (`perseus.nn`)

`EncModule` mirrors `torch.nn.Module` where it matters:

| | |
|---|---|
| `m(x, *a, **kw)` | forwards to `m.forward` |
| `m.bind(session_or_inf)` | attach the session, install owned weights (recursive); returns `m` |
| `m.named_children()`, `m.named_modules()`, `m.modules()`, `m.apply(fn)` | tree traversal |
| `repr(m)` | the tree, torch style |
| `m.residency()` | what the residency pipeline may stream for this module |
| `m.torch_mirror()` | a torch module with the same plaintext function (for calibration) |
| `m.apply_calibration(section, cfg, probe=None)` | install one fitted section under `m.cfg_name` |

Leaves and containers:

```python
EncLinear(name, d_in, d_out, weight=None, bias=None, hint=True)   # y = x @ W, W is (d_in, d_out)
EncLinear.from_torch(name, linear, d_in=None, d_out=None, hint=True)      # transposes + zero-pads
EncLayerNorm(cfg_name, d=None, weight=None, bias=None, scale=None)
EncGELU(cfg_name="mlp.act")
EncSequential(*modules, overlap=None)     # overlap: "sync" | "prefetch" | "threaded" | None
EncGPT2(store, configs, n_layers=None)    # .bind(inf, plan_dir=..., artifact_dir=..., coeff_encode=...)
EncGPT2.from_pretrained("gpt2")           # artifacts from the perseus cache, or weights= / configs=
```

Shapes are validated at construction: a torch-layout `(out, in)` weight, a wrong bias
length, NaNs, or a LayerNorm affine of the wrong width raise `ValueError` naming the
expected shape. `d_in`/`d_out` are the **packed** widths (powers of two); pad real
features with zeros.

Save / load a custom model (structure as JSON, parameters as .npz; calibrated sections
stay in `configs.json`):

```python
from perseus.nn import save, load
save(model, "my_mlp/"); model = load("my_mlp/").bind(session)
sd = model.state_dict(); model.load_state_dict(sd)      # torch-style keys ("1.weight")
```

Calibration on your own data:

```python
from perseus.nn import calibrate_sequential
parsed = calibrate_sequential(model, samples, approximation="gpt2", apply=True)
```

## Generation (`EncGPT2`)

```python
ids = model.generate(prompt_embeddings, max_new_tokens=8, eos_token_id=50256,
                     on_token=lambda t, j: print(t), feedback="encrypted")
for t in model.stream(prompt_embeddings, 8, feedback="client", do_sample=True,
                      temperature=0.8, top_k=40, seed=0):
    ...
```

`feedback="encrypted"` keeps the server's token feedback inside the ciphertext (CutMax
argmax + codebook re-embedding); `"client"` decrypts logits, picks the next token
(argmax or sampling), re-encodes `wte+wpe`. `prompt_mode="prefill"` runs the chunked
filling prefill for the prompt. `entry_bootstrap=` / `realize_entry=` choose how the first
refresh of a decode step lands; left at None they take the decode path's own defaults, and
the prefill path ignores them. Override `encode_input(values)` / `embed_token(token_id, position)` to feed embeddings from
the client side.

## Writing your own op (leaf primitives on the session)

```python
fhe, inf = s.inf.fhe, s.inf
y = fhe.rotate(ct, 1)                      # cyclic slot rotation (needs the rotation key)
y = fhe.conjugate(ct); y = fhe.negate(ct)
y = inf.mult_pt(ct, w)                     # slot-wise * plaintext vector (encoded at ct's level)
y = inf.add_pt(ct, b)
y = inf.sum_slots(ct, 4)                   # rotate-and-add over aligned groups of 4
y = inf.eval_chebyshev(ct, coeffs, a, b)   # sum_k coeffs[k] T_k(x), x in [a, b]
```

These are the pieces the runtime's own composites are made of (the GELU/softmax
polynomials go through the same Chebyshev evaluator). Every call consumes levels; plan
a bootstrap (`fhe.bootstrap_hint`) before a deep chain. `tests/gpu/test_primitives.py`
pins each one against numpy. A worked RMSNorm built from them is below
(`examples/rmsnorm_from_primitives.py`).

### Worked example: RMSNorm from primitives

`x * rsqrt(mean(x^2) + eps) * gamma` on a token ciphertext, in five calls:

```python
sq = fhe.square(ct)                                           # x^2 slot-wise        1 level
S = inf.sum_slots(sq, inf.slots)                              # all-slot sum, broadcast  0
r = inf.eval_chebyshev(S, coeffs, (a - eps) * d, (b - eps) * d)   # rsqrt   2 + ceil(log2 degree)
xg = inf.mult_pt(ct, gamma_slots)                             # x * gamma (parallel)  1
out = fhe.mult(xg, r)                                         # ciphertext product    1
```

* **Packing.** A token ciphertext keeps real lane `i` at slot `i * t` with
  `t = inf.slots // inf.size.hidDim` and every other slot zero
  (`src/packing/cachemir/cachemir_linear_utils.cu`, `encode_linear_input` /
  `decode_tokens`; `pack_tokens` in `src/model/gpt2/gpt2_io.cu` zero-pads to `hidDim`
  first). Summing every slot with
  `sum_slots(ct, inf.slots)` is therefore the sum over the `d` real lanes broadcast to all
  slots — exactly the ladder the runtime's LayerNorm variance uses
  (`cachemir_norm_utils.cu`). `gamma` is a slot vector with `gamma[i]` at `i * t`
  (`gamma_to_slots`); the steps `1, 2, 4, ..., slots/2` are in the gpt2 band
  (`Context.loaded_rot_steps`). The input must be a fresh token-basis ciphertext
  (`Session.encrypt`): a ciphertext that went through a linear carries partial sums in
  the off-lane slots and the ladder would sum them too.
* **Chebyshev.** The coefficients come from `numpy.polynomial.chebyshev.chebinterpolate`
  of `u ** -0.5` on `[a, b]` for `u = mean(x^2) + eps` (`c0` unhalved: numpy's and the
  runtime's convention). The evaluator maps its argument affinely onto `[-1, 1]`, so
  feeding it the sum of squares with the interval scaled to `[(a - eps) d, (b - eps) d]`
  is the same series with the `1/d` and `+eps` absorbed — one level saved
  (`fhe_interval`). The interval is the calibration window: the mirror `rmsnorm_ref`
  raises outside it, the FHE side cannot.
* **Levels.** `4 + ceil(log2 degree)` CKKS levels (square 1, affine map 1, `T_degree`
  tree, weighted sum 1, final product 1; the gamma product runs in parallel). The runtime
  bootstraps reactively once `ct.level >= fhe.level_limit()` (prime-granular on n32, one
  level = `composite_degree` primes), so with `custom_n32`'s landing at 34 and ceiling at
  49 a fresh encrypt has 7 levels: `degree <= 7` (`level_budget(sess, ct)` computes it;
  with `AUTO_BTS_LEVEL=46` exported it is 5, `degree <= 2`). The returned `RMSNormResult`
  reports `level_before` / `level_after`.

`rmsnorm_ref` applies the same series, so the approximation error (8e-5 relative at
degree 7 on `[0.04, 0.16]`) is separable from FHE noise; `rmsnorm_exact` is the closed
form. Tests: `tests/test_rmsnorm_example.py` (CPU: fit, mirror vs exact, guard, interval
fold, slot layout, the call sequence on a numpy fake) and `test_rmsnorm_example` in
`tests/gpu/test_primitives.py` (GPU: FHE vs mirror at `atol=1e-2`, level report).

Introspection: `WeightStore.names() / shape(name) / tensor(name)`, `CutMaxConfig.iters`,
`BootstrapPlan.placements / expected_levels`, `repr()` on most option structs
(`InferenceOptions`, `ModelSize`, `RunConfig`, `GeLUConfig`, `NormConfig`, `SoftmaxConfig`,
`CutMaxConfig`, `BootstrapPlan`, `WeightStore`; `CKKSOptions`, `ModelConfig` and
`ParsedConfigs` have none);
`run_decode / run_prefill / run_generate / DecodeSession.decode` raise on a token error
(`raise_on_error=False` returns the partial `RunResult`).

## Client / server (`perseus.nn.remote`)

```python
client = EncClient(profile=SessionProfile.custom_n32(), family="gpt2")   # CPU keygen
client.save_bundle("bundle/")            # context + public + eval keys + bundle.json

server = EncServer("bundle/")            # parameters FROM the manifest; cannot decrypt
client.accept(server.session_manifest())   # adopt the server's probed fresh-encode level
model = EncSequential(...).bind(server.inf)
y = client.decrypt(server.run(model, client.encrypt(x)), d=768)
```

`EncServer(bundle, options=|profile=)` cross-checks the caller's parameters against the
manifest and raises `BundleError` with a field diff (or warns with `strict=False`); a
bundle keyed for another family is refused before any session is built. See
`docs/SECURITY_MODEL.md` for what each party holds.

`EncGenerationServer` / `EncGenerationClient` (`perseus.nn.serve`) run the generation
loop over the same byte protocol (`prompt` / `step_encrypted` / `step_client`).

A GPU-less client derives its fresh-encode level from the parameter formula;
`EncServer.session_manifest()` carries the level the server actually probes
(`bootstrap_output_level`, 34 vs 32 on n32) and `EncClient.accept()` adopts it — required
for a strict plan (`Inference::weights_at` raises `[plan_weight_error]` for a client
ciphertext at the formula's level), harmless for eager runs; a manifest without the field
keeps the formula and logs once at INFO. `EncGenerationClient.generate` does this handshake
itself.

### GPU-less client (`perseus._client`)

A client machine without a CUDA driver cannot import `perseus._core` (the extension needs
libcuda/libcudart/libnccl whatever the code does). `perseus._client` is the client role on
OpenFHE alone — the same patched OpenFHE of the deps tree, no CUDA toolchain, no FIDESlib:

```bash
CHAIN=n32 bash scripts/local_build_client.sh      # -> perseus/_client.n32.so (+ import symlink)
python -c "from perseus.nn import EncClient, EncGenerationClient"
```

`EncClient(..., backend=None | "core" | "client")` picks the extension the role runs on
(None: `perseus._core` when it imports, else `perseus._client`; the `PERSEUS_CLIENT_EXTENSION`
environment variable overrides); `client.backend` says which one. Everything a `_client`
client produces interchanges with a `_core` server: the bundle layout (`context.bin` and its
`context.bin.dev` sidecar, `public.key`, `multkeys.bin`, `rotkeys.bin`) with byte-identical
context parameters and the same key set, the token packing, and the ciphertext bytes of
`serialize_ct` / `deserialize_ct` — `bundle.json` is unchanged. Its `bootstrap_output_level`
is the parameter formula (what a GPU-less `_core` session reports too), so the `accept()`
handshake above is the same. `EncServer`, `EncGenerationServer`, the Enc modules and
`perseus.session()` still need `perseus._core` (there is no `session(device="cpu")`: the CPU
client role is `EncClient`); on a `_client`-only machine those names raise `ImportError` with
the build recipe on first access, and `perseus.errors` exports `_client`'s error classes.

## Errors (`perseus.errors`, ⚙︎-free)

```
FHEError                 any runtime error from the FHE core
  PlanError              strict planned-mode violation (level / weight / bootstrap)
  MaskError              strict decode-mask miss
  LayoutError            slot-layout basis mismatch (_core.set_strict_layout(True))
PlanContractError        plan loaded under a different env than it was captured in
BundleError              key bundle disagrees with the requested session
SecurityError            a trust-boundary invariant would be violated
```

## Logging

Library modules log under the `perseus` logger and never print. Warnings surface on
stderr by default; to see diagnostics:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
```

The console scripts (`perseus-export`, `perseus-calibrate`, `perseus-plan`) configure
this themselves. The C++ runtime still prints its parameter table and per-token
diagnostics to stderr; there is no switch for that.

## Console scripts

| | |
|---|---|
| `perseus-export --model M --out DIR` | HF checkpoint → `weights.bin.zip` (server) + `client.npz` (client) |
| `perseus-calibrate model=gpt2 dataset=openwebtext ...` | Hydra CLI; fits `configs.json` |
| `perseus-plan --graph-dir G --out-dir P ...` | the min-cut bootstrap planner (`python -m perseus.plan` is the same entry point) |

## Environment knobs that still matter from Python

`SessionProfile` / `CKKSOptions` cover the options a Python caller normally sets. The C++
core reads about forty more names directly from the environment, so anything not exposed as
a field is passed through `SessionProfile.extra`, which the session applies as defaults
before the first CUDA call. The two most often reached for from Python are `FHE_GRAPH_DIR`
(capture protocol: write per-block graphs) and `FHE_BOOTSTRAP_PLACEMENTS_DIR` (run a
different plan). Strict slot-layout checking is not an environment knob: call
`_core.set_strict_layout(True)`. `CHAIN` selects the build
(`scripts/local_build_core.sh`) and is only consulted by `perseus._env.current_chain()`
when the loaded extension carries no `_core.chain` stamp.
`ENV_VARS.md`, at the repository root, documents the full set.
