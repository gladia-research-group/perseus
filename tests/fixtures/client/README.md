# perseus._client identity fixtures

Captured from a real `perseus._core` bundle so the CUDA-free client extension
(`perseus._client`) is pinned to what the client role produces through `_core` — without a
GPU and without a 30 GB keygen in the CPU tier (`tests/test_client_ext.py`).

Source bundle: written by
`scripts/utils/probe_client_server.py`'s client half (`_core.make_gpt2_inference` with
`skip_gpu_load=True`, default `InferenceOptions` = GPT-2 small sizes 768/3072/1024/4096,
16 padded / 12 real heads, Cachemir packing, no aux packings) under the n32 block of
`scripts/local_env.sh`; the FIDESlib submodule was at commit
773761d9ad985f168c9f29f0165d63b46833360b, the deps tree `deps_n32` (NATIVEINT=32).

The exact context-shaping environment (`_N32_ENV` in `tests/test_client_ext.py` and
`scripts/utils/capture_client_fixtures.py`):

```
LOGN=16 CKKS_DEPTH=10 BTP_DEPTH_OVERHEAD=16 SCALE_BITS=54 BTP_SCALE_BITS=54
FIRST_MOD_BITS=56 NUM_LARGE_DIGITS=6 LEVEL_BUDGET=4:3 CORRECTION_FACTOR=6
COMPOSITE_DEGREE=2 SPARSE_BTS_SLOTS=512,1 H_WEIGHT=192 BTS_ITERATIONS=1 BTS_PRECISION=12
CKKS_COMPLEX=1            # exported in the probe's shell: context.bin reproduces only with it
SPARSE_LEVEL_BUDGET / BTS_DIM1 unset
```

| file | what | pinned by |
|---|---|---|
| `options.json` | the bundle's option sections (a `bundle.json` without the stamps) | `test_from_env_matches_fixture` |
| `context.bin` | the serialized OpenFHE context (2489 B) | `test_context_bytes_match_core_fixture` (byte equality with `_client._debug.bundle_meta`) |
| `context.bin.dev` | FIDESlib's device sidecar: 133 rotation steps, `KeyDist: 3` | same test (text equality) |
| `rotkey_indexes.json` | the 165 automorphism-map keys of the bundle's 30 GiB `rotkeys.bin` (one key tag): the model band, the dense + 512 + 1 bootstrap precomps' rotations, M-1 (conjugation), M-2 / M-4 (ENCAPS switching pair) | `test_automorphism_index_set_matches_core` (set equality with `_client._debug.expected_automorphism_indexes`) |

Regenerate (verifies first, writes only on a match; `--indexes` reads the whole rotkeys.bin):

```
PERSEUS_CORE_BUNDLE=<bundle dir> numactl --cpunodebind=0,1,2 \
    .venv/bin/python scripts/utils/capture_client_fixtures.py --write --indexes
```

The fixtures go stale only if the wrapper's parameter derivation, the GPT-2 cachemir band
or the n32 preset changes — which then fails these tests loudly.
