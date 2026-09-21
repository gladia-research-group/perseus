"""Keys and ciphertexts from perseus._client consumed by a perseus._core server (GPU).

The acceptance of the CUDA-free client extension: a bundle written by `_client` loads
through `_core`'s real keys_dir server path (context.bin + sidecar, multkeys.bin, the
multi-GB rotkeys.bin, the bootstrap setups), `_core` ciphertexts decrypt on `_client` with
the packing `_core.pack_tokens` uses, `_client` ciphertexts compute and bootstrap on the
server, and the results decrypt on the client. Slow: keygen + a ~32 GB bundle.

    cd <checkout> && CHAIN=n32 source scripts/local_env.sh
    CUDA_VISIBLE_DEVICES=<gpu> PERSEUS_SLOW=1 TMPDIR=<big-tmp> \\
        .venv/bin/python -m pytest -q -p no:cacheprovider -m gpu tests/gpu/test_client_roles.py
"""
import os
import shutil
import tempfile

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

D_PAD, D_REAL = 1024, 768


def test_client_extension_keys_and_ciphertexts_serve_a_core_session():
    _client = pytest.importorskip("perseus._client")
    from perseus import _core
    from perseus.nn import EncClient, EncLinear, EncSequential, EncServer
    from perseus.nn.remote import options_from_manifest
    from perseus.profile import SessionProfile

    prof = SessionProfile.custom_n32()
    client = EncClient(profile=prof, backend="client")
    assert client.backend is _client and isinstance(client.inf, _client.Inference)
    assert client.encode_level == 32                       # the parameter formula
    bundle = tempfile.mkdtemp(prefix="perseus_client_bundle_", dir=os.environ.get("TMPDIR"))
    try:
        client.save_bundle(bundle)
        assert sorted(os.listdir(bundle)) == [
            "bundle.json", "context.bin", "context.bin.dev", "multkeys.bin", "public.key", "rotkeys.bin"]
        meta = _client._debug.bundle_meta(options_from_manifest(client.manifest(), ext=_client), "gpt2")
        with open(os.path.join(bundle, "context.bin"), "rb") as f:
            assert meta[0] == f.read()

        # _core's server loader: DeserializeFromFile(context.bin) + the sidecar, multkeys.bin,
        # rotkeys.bin (band + bootstrap keys), the bootstrap setups, LoadContext
        server = EncServer(bundle, profile=prof)
        assert not server.inf.fhe.has_secret_key
        assert sorted(server.inf.fhe.loaded_rot_steps) == sorted(client.inf.fhe.loaded_rot_steps)
        have = int(server.inf.fhe.bootstrap_output_level())
        assert have == server.fresh_encode_level == 34
        assert client.accept(server.session_manifest()) == have

        rng = np.random.default_rng(0)
        x = rng.standard_normal(D_REAL) * 0.3

        # packing equality against _core.pack_tokens: the server encrypts with the client's
        # public key through _core's packer; the client's decoder recovers x, and the raw
        # slots sit exactly where _client's cachemir layout puts them (slot i*t)
        pc = _core.pack_tokens(server.inf, x[None, :], have)
        blob = _core.serialize_ct(server.inf, pc)
        np.testing.assert_allclose(client.decrypt(blob, d=D_REAL), x, atol=1e-5)
        cct = _client.deserialize_ct(client.inf, blob)
        assert cct.level == have
        slots = np.asarray(_client.decrypt_slots(client.inf, cct))
        t = client.inf.slots // D_PAD
        np.testing.assert_allclose(slots[::t][:D_REAL], x, atol=1e-5)
        own = np.asarray(_client.decrypt_slots(client.inf, _client.pack_tokens(client.inf, x[None, :], have)))
        np.testing.assert_allclose(own, slots, atol=1e-5)

        # a _client ciphertext through a _core linear
        W = rng.standard_normal((D_PAD, D_PAD)) * 0.5 / np.sqrt(D_REAL)
        W[D_REAL:, :] = 0; W[:, D_REAL:] = 0
        model = EncSequential(EncLinear("t_lin", D_PAD, D_PAD, weight=W)).bind(server.inf)
        cblob = client.encrypt(x)
        assert _core.deserialize_ct(server.inf, cblob).level == have
        y = client.decrypt(server.run(model, cblob), d=D_REAL)
        xp = np.zeros(D_PAD); xp[:D_REAL] = x
        ref = (xp @ W)[:D_REAL]
        assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-3

        # a _client ciphertext through a _core bootstrap: the client-made bootstrap rotation
        # keys, the conjugation key (M-1) and the ENCAPS switching pair (M-2 / M-4)
        pb = _core.deserialize_ct(server.inf, client.encrypt(x))
        server.inf.fhe.bootstrap(pb)
        assert pb.level == have
        yb = client.decrypt(_core.serialize_ct(server.inf, pb), d=D_REAL)
        assert np.linalg.norm(yb - x) / np.linalg.norm(x) < 1e-2
    finally:
        shutil.rmtree(bundle, ignore_errors=True)   # tens of GB
