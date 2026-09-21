"""Two-process client/server demo: the server never holds the secret key.

client (this process, no GPU context: skip_gpu_load): keygen, write the server bundle
+ its own secret key, encrypt an input, hand the ciphertext over -> waits ->
decrypts the server's output and checks it against the plaintext linear.

server (subprocess, --server DIR): builds its GPU session FROM the bundle
(keys_dir, no KeyGen), installs its plaintext weights, runs the linear on the
received ciphertext, returns the result ciphertext. Asserts it holds no secret.
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from perseus import _core

d_pad, d_real, d_exp, e_real = 1024, 768, 4096, 3072
rng = np.random.default_rng(7)
W = rng.standard_normal((d_pad, d_exp)) * 0.5 / np.sqrt(d_real)
W[d_real:, :] = 0.0
W[:, e_real:] = 0.0
x_real = rng.standard_normal(d_real) * 0.3


def server(bundle):
    opts = _core.InferenceOptions()
    opts.ckks = _core.CKKSOptions.from_env()
    opts.ckks.keys_dir = bundle
    opts.mode = _core.InferenceMode.Sync
    t0 = time.perf_counter()
    inf = _core.make_gpt2_inference(opts)
    print(f"[server] session from bundle in {time.perf_counter() - t0:.0f}s "
          f"(level_limit={inf.fhe.level_limit()})", flush=True)
    assert not inf.fhe.has_secret_key, "server holds a secret key"
    print("[server] no secret key in this process", flush=True)
    inf.set_weight("fc", W.tolist(), d_pad, d_exp)
    x = _core.deserialize_ct(inf, open(f"{bundle}/input.ct", "rb").read())
    y = _core.linear(inf, x, "fc", d_pad, d_exp)
    open(f"{bundle}/output.ct", "wb").write(_core.serialize_ct(inf, y))
    print("[server] done", flush=True)


def client():
    opts = _core.InferenceOptions()
    opts.ckks = _core.CKKSOptions.from_env()
    opts.ckks.skip_gpu_load = True
    opts.mode = _core.InferenceMode.Sync
    t0 = time.perf_counter()
    inf = _core.make_gpt2_inference(opts)
    print(f"[client] CPU keygen in {time.perf_counter() - t0:.0f}s", flush=True)
    bundle = tempfile.mkdtemp(prefix="fhe_bundle_")
    t0 = time.perf_counter()
    _core.save_keys(inf, bundle)
    _core.save_secret_key(inf, f"{bundle}/secret.key")   # stays client-side (never read by server)
    sizes = {f: os.path.getsize(f"{bundle}/{f}") >> 20
             for f in ("context.bin", "public.key", "multkeys.bin", "rotkeys.bin", "secret.key")}
    print(f"[client] bundle written in {time.perf_counter() - t0:.0f}s: {sizes} MB", flush=True)
    x = _core.encode_token_input(inf, x_real.tolist())
    open(f"{bundle}/input.ct", "wb").write(_core.serialize_ct(inf, x))

    r = subprocess.run([sys.executable, __file__, "--server", bundle], text=True)
    # exit codes are not evidence here (the known teardown heap corruption lands after
    # the work is done); the artifact is: the server wrote its result ciphertext.
    assert os.path.exists(f"{bundle}/output.ct"), f"server produced no output (rc={r.returncode})"

    y = _core.deserialize_ct(inf, open(f"{bundle}/output.ct", "rb").read())
    out = np.array(_core.decode_linear_output(inf, y, d_pad, d_exp))
    x_pad = np.zeros(d_pad)
    x_pad[:d_real] = x_real
    ref = x_pad @ W
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    print(f"[client] decrypted server output vs plaintext: rel={rel:.3e}", flush=True)
    assert rel < 0.05
    print("[client] PASS", flush=True)
    # Success exit code made meaningful (teardown corruption lands after the work).
    # NOTE: this call MUST live inside the function — at module level it fires at
    # IMPORT and _Exit(0)s before main ever runs (the three 0-byte rc=0 runs).
    getattr(_core, "hard_exit", lambda c=0: None)(0)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--server":
        server(sys.argv[2])
        getattr(_core, "hard_exit", lambda c=0: None)(0)
    else:
        client()
