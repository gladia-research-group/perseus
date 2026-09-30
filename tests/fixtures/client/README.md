# Client test fixtures

`perseus._client` is the client side of Perseus, built without CUDA: it generates keys,
encrypts and decrypts on a machine with no GPU. These files were taken from the GPU extension
(`perseus._core`) doing the same client work, and `tests/test_client_ext.py` checks that the
client reproduces them exactly. That way the CPU test suite covers the client without a GPU and
without generating the full 30 GB of keys.

Where they come from: the client half of `scripts/utils/probe_client_server.py`
(`_core.make_gpt2_inference` with `skip_gpu_load=True` and the default `InferenceOptions`:
GPT-2 small sizes 768/3072/1024/4096, 16 padded / 12 real attention heads, the default data
layout), run with the 32-bit settings of `scripts/local_env.sh`, with the FIDESlib submodule at
commit 773761d9ad985f168c9f29f0165d63b46833360b and the 32-bit dependency tree `deps_n32`.

The environment that shapes the encryption parameters (`_N32_ENV` in `tests/test_client_ext.py`
and `scripts/utils/capture_client_fixtures.py`):

```
LOGN=16 CKKS_DEPTH=10 BTP_DEPTH_OVERHEAD=16 SCALE_BITS=54 BTP_SCALE_BITS=54
FIRST_MOD_BITS=56 NUM_LARGE_DIGITS=6 LEVEL_BUDGET=4:3 CORRECTION_FACTOR=6
COMPOSITE_DEGREE=2 SPARSE_BTS_SLOTS=512,1 H_WEIGHT=192 BTS_ITERATIONS=1 BTS_PRECISION=12
CKKS_COMPLEX=1            # set in the probe's shell: context.bin reproduces only with it
SPARSE_LEVEL_BUDGET / BTS_DIM1 unset
```

| file | contents | checked by |
|---|---|---|
| `options.json` | the option sections of the key bundle (a `bundle.json` without its provenance stamps) | `test_from_env_matches_fixture` |
| `context.bin` | the serialized OpenFHE context (2489 B) | `test_context_bytes_match_core_fixture` (byte equality with `_client._debug.bundle_meta`) |
| `context.bin.dev` | FIDESlib's GPU-side description of it: 133 rotation steps, `KeyDist: 3` | same test (text equality) |
| `rotkey_indexes.json` | the 165 rotation keys in the bundle's 30 GiB `rotkeys.bin`: the rotations the model uses, those of the full-size and the two cheaper bootstraps (512 and 1 slots), conjugation (M-1), and the key-switching pair for the bootstrap's sparse secret (M-2 / M-4) | `test_automorphism_index_set_matches_core` (set equality with `_client._debug.expected_automorphism_indexes`) |

To regenerate them (the script first checks the current files, and writes only on a match;
`--indexes` reads the whole `rotkeys.bin`):

```
PERSEUS_CORE_BUNDLE=<bundle dir> numactl --cpunodebind=0,1,2 \
    .venv/bin/python scripts/utils/capture_client_fixtures.py --write --indexes
```

The fixtures only go out of date if the way the parameters are derived, the set of rotations
GPT-2 needs, or the 32-bit settings change, and then these tests fail loudly.
