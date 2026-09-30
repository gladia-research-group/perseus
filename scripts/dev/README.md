# scripts/dev: NumPy prototypes of the data layouts

Standalone scripts, run with `python scripts/dev/<name>.py` and kept for reference; pytest does
not collect them. They simulate in plain NumPy how data is laid out across ciphertext slots. The
self-checking matrix-vector model in `linear.py` also lives on as
`tests/test_cachemir_packing_model.py`.

| file | what it prototypes |
|---|---|
| `linear.py` | the encrypted vector-matrix product: lay out x and W, rotate and accumulate, read back, compared with `x @ W` |
| `python_kv_cache.py`, `gka_kv_cache.py` | layouts of the attention key/value cache (they import `linear.py`) |
| `polyeval.py` | polynomial evaluation |
