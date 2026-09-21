"""Client/server roles on a real context (slow: keygen + a multi-GB bundle).

Parametrized over the extension backing the client: perseus._core (today's contract) and
perseus._client (the CUDA-free client extension) — the same server, the same bundle
layout, the same bytes. The bootstrap leg is the only exercise of the client-made
bootstrap rotation keys, conjugation key and ENCAPS switching pair VALUES.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.parametrize("backend", ["core", "client"])
def test_client_server_round_trip_with_manifest(backend):
    from perseus.nn import EncClient
    from perseus.profile import SessionProfile

    if backend == "client":
        pytest.importorskip("perseus._client")
    prof = SessionProfile.custom_n32()
    client = EncClient(profile=prof, backend=backend)
    assert client.backend.__name__ == f"perseus._{backend}"
    bundle = tempfile.mkdtemp(prefix="perseus_bundle_", dir=os.environ.get("TMPDIR"))
    try:
        _roles_round_trip(client, bundle, prof)
    finally:
        shutil.rmtree(bundle, ignore_errors=True)   # tens of GB


def _roles_round_trip(client, bundle, prof):
    from perseus import _core
    from perseus._backend import load
    from perseus.errors import BundleError
    from perseus.nn import EncLinear, EncSequential, EncServer
    from perseus.nn.remote import MANIFEST_NAME, options_from_manifest
    client.save_bundle(bundle)
    assert os.path.exists(os.path.join(bundle, MANIFEST_NAME))
    _client = load("_client")
    if _client is not None:
        # cross-extension parameter identity on the acceptance options: the context and
        # sidecar the bundle carries are what perseus._client derives for its manifest
        meta = _client._debug.bundle_meta(options_from_manifest(client.manifest(), ext=_client),
                                          client.family)
        with open(os.path.join(bundle, "context.bin"), "rb") as f:
            assert meta[0] == f.read()
        with open(os.path.join(bundle, "context.bin.dev")) as f:
            assert meta[1] == f.read()
    with pytest.raises(BundleError, match="keyed for family"):
        EncServer(bundle, family="generic")

    server = EncServer(bundle, profile=prof)        # cross-check passes
    assert not server.inf.fhe.has_secret_key
    d_pad, d_real = 1024, 768
    rng = np.random.default_rng(0)
    x = rng.standard_normal(d_real) * 0.3

    # The fresh-encode level handshake: the GPU-less client encodes at its parameter
    # formula (32 on n32) until it adopts the level the server probed (34 on n32) —
    # what Inference::weights_at requires of a client ciphertext under a strict plan.
    have = int(server.inf.fhe.bootstrap_output_level())
    assert server.fresh_encode_level == have
    ext = client.backend
    before = ext.deserialize_ct(client.inf, client.encrypt(x)).level
    assert before == client.encode_level, f"formula={before} client.encode_level={client.encode_level}"
    sm = json.loads(json.dumps(server.session_manifest()))     # JSON across the wire
    assert sm["bootstrap_output_level"] == have and sm["ckks"] == server.manifest["ckks"]
    assert client.accept(sm) == have, f"formula={before} probed={have}"
    assert client.encode_level == have, f"formula={before} probed={have}"
    after = ext.deserialize_ct(client.inf, client.encrypt(x)).level
    assert after == have, f"formula={before} probed={have} after_accept={after}"
    assert client.manifest()["bootstrap_output_level"] == have

    W = rng.standard_normal((d_pad, d_pad)) * 0.5 / np.sqrt(d_real)
    W[d_real:, :] = 0; W[:, d_real:] = 0
    model = EncSequential(EncLinear("t_lin", d_pad, d_pad, weight=W)).bind(server.inf)
    y = client.decrypt(server.run(model, client.encrypt(x)), d=d_real)   # at the adopted level
    xp = np.zeros(d_pad); xp[:d_real] = x
    ref = (xp @ W)[:d_real]
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-3

    # The bootstrap leg: a server-side bootstrap of a client ciphertext uses the client's
    # bootstrap rotation keys, conjugation key and ENCAPS switching pair (M-2 / M-4).
    pc = _core.deserialize_ct(server.inf, client.encrypt(x))
    server.inf.fhe.bootstrap(pc)
    assert pc.level == have
    yb = client.decrypt(_core.serialize_ct(server.inf, pc), d=d_real)
    assert np.linalg.norm(yb - x) / np.linalg.norm(x) < 1e-2
