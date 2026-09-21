"""perseus._client (the CUDA-free client extension) against what perseus._core produces.

Identity is pinned three ways: fixtures captured from a real _core bundle
(tests/fixtures/client: context bytes, sidecar text, the automorphism-map index set of its
rotkeys.bin), option/env parity against _core when both extensions import, and a small
CPU context (logN 14, no bootstrap) for the packing geometry, the bundle layout and the
ciphertext bytes. The slow tests (a real n32 keygen, the _core interchange) need
PERSEUS_SLOW=1; the GPU role test (tests/gpu/test_roles.py) proves the key VALUES.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_client = pytest.importorskip("perseus._client")
from perseus._backend import load
from perseus._env import scoped_env
from perseus.nn.remote import _CKKS_FIELDS, _INF_FIELDS, options_from_manifest, options_to_manifest

_core = load("_core")
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "client"

# The context-shaping environment the fixture bundle was written under
# (tests/fixtures/client/README.md); None unsets.
_N32_ENV = {
    "LOGN": "16", "CKKS_DEPTH": "10", "BTP_DEPTH_OVERHEAD": "16", "SCALE_BITS": "54",
    "BTP_SCALE_BITS": "54", "FIRST_MOD_BITS": "56", "NUM_LARGE_DIGITS": "6",
    "LEVEL_BUDGET": "4:3", "CORRECTION_FACTOR": "6", "COMPOSITE_DEGREE": "2",
    "SPARSE_BTS_SLOTS": "512,1", "H_WEIGHT": "192", "BTS_ITERATIONS": "1",
    "BTS_PRECISION": "12", "AUTO_BTS_LEVEL": "46", "CKKS_COMPLEX": "1",
    "SPARSE_LEVEL_BUDGET": None, "BTS_DIM1": None,
}
_SLOW = pytest.mark.skipif(os.environ.get("PERSEUS_SLOW") != "1", reason="slow: set PERSEUS_SLOW=1")
REPO = Path(__file__).resolve().parents[1]


def _fixture_options(ext=_client):
    fx = json.loads((FIXTURES / "options.json").read_text())
    return options_from_manifest({"format": 1, **fx}, ext=ext)


def _from_env(ext, env):
    with scoped_env(**env):
        o = ext.InferenceOptions()
        o.ckks = ext.CKKSOptions.from_env()
    return o


# ---- fixtures captured from the real _core bundle ----

def test_context_bytes_match_core_fixture():
    opts = _fixture_options()
    with scoped_env(**_N32_ENV):
        ctx_bytes, dev_text = _client._debug.bundle_meta(opts, "gpt2")
    assert ctx_bytes == (FIXTURES / "context.bin").read_bytes()
    assert dev_text == (FIXTURES / "context.bin.dev").read_text()


def test_automorphism_index_set_matches_core():
    opts = _fixture_options()
    with scoped_env(**_N32_ENV):
        got = list(_client._debug.expected_automorphism_indexes(opts, "gpt2"))
    want = json.loads((FIXTURES / "rotkey_indexes.json").read_text())
    assert got == want
    M = 2 << opts.ckks.logN
    assert {M - 1, M - 2, M - 4} <= set(got)          # conj key + the ENCAPS switching pair
    assert len(got) == 165


def test_bootstrap_output_level_formula():
    opts = _fixture_options()
    assert _client._debug.formula_level(opts.ckks) == 32
    off = _client.CKKSOptions(opts.ckks)
    off.enable_bootstrap = False
    assert _client._debug.formula_level(off) == 0
    two = _client.CKKSOptions(opts.ckks)
    two.bts_iterations = 2
    assert _client._debug.formula_level(two) == 34


def test_from_env_matches_fixture():
    fx = json.loads((FIXTURES / "options.json").read_text())
    assert options_to_manifest(_from_env(_client, _N32_ENV), "gpt2")["ckks"] == fx["ckks"]


# ---- parity with _core (skipped where it is not built, or built for the other chain) ----

def _need_same_chain():
    """Both extensions must be the same chain's build: they are separate stashes behind one
    import name, so a machine with n32 and n64 built can have them disagree, and their
    parameter defaults differ by chain."""
    if _core is None:
        pytest.skip("perseus._core not built")
    if getattr(_core, "chain", None) != getattr(_client, "chain", None):
        pytest.skip(f"_core is the {_core.chain} build, _client the {_client.chain} build")


def test_options_parity():
    _need_same_chain()
    a, b = _core.CKKSOptions(), _client.CKKSOptions()
    for k in (*_CKKS_FIELDS, "keys_dir", "skip_gpu_load", "sparse_bts_slots"):
        assert getattr(a, k) == getattr(b, k), k
    ia, ib = _core.InferenceOptions(), _client.InferenceOptions()
    for k in (*_INF_FIELDS, "parallel", "bench_mode"):
        assert getattr(ia, k) == getattr(ib, k), k
    assert ia.packing_kind.name == ib.packing_kind.name and ia.mode.name == ib.mode.name
    assert list(ia.aux_packing_kinds) == list(ib.aux_packing_kinds) == []
    for enum in ("PackingKind", "InferenceMode"):
        ea, eb = getattr(_core, enum).__members__, getattr(_client, enum).__members__
        assert list(ea) == list(eb), enum
        assert [int(v) for v in ea.values()] == [int(v) for v in eb.values()], enum
    # the sparse_bts_slots view and the copy constructors behave the same
    for ext in (_core, _client):
        o = ext.CKKSOptions()
        o.sparse_bts_slots = 512
        assert o.sparse_bts_slots_list == [512]
        o.sparse_bts_slots_list = [512, 1]
        assert o.sparse_bts_slots == 512
        c = ext.CKKSOptions(o)
        c.sparse_bts_slots_list = []
        assert o.sparse_bts_slots_list == [512, 1]
        io = ext.InferenceOptions()
        io.ckks = o
        ic = ext.InferenceOptions(io)
        ic.ckks.logN = 12
        assert io.ckks.logN == o.logN


@pytest.mark.parametrize("delta", [
    {},
    {"LEVEL_BUDGET": "4"},                 # malformed: ignored by both
    {"SPARSE_BTS_SLOTS": "512,0,32"},      # zeros dropped
    {"CKKS_COMPLEX": "0", "SPARSE_LEVEL_BUDGET": "3:3", "BTS_ITERATIONS": "2"},
])
def test_from_env_parity(delta):
    if _core is None:
        pytest.skip("perseus._core not built")
    env = {**_N32_ENV, **delta}
    want = options_to_manifest(_from_env(_core, env), "gpt2")
    got = options_to_manifest(_from_env(_client, env), "gpt2")
    assert got == want


# ---- rotation bands: a numpy transcription of the three rot-index TUs ----

def _cm(N, d_in, d_out):
    is_up = d_in <= d_out
    d = d_in if is_up else d_out
    alpha = max(d_in, d_out) // d
    t, tp = N // d, N // (alpha * d)
    tp_in, tp_out = (t, tp) if is_up else (tp, t)
    n_pt = (d if is_up else alpha * d) // tp_out
    r_i = min(max(1, d * d // N), n_pt)
    return dict(t=t, tp=tp, tp_in=tp_in, tp_out=tp_out, r_i=r_i)


def _pow2(lo, hi):
    s = lo
    while s < hi:
        yield s
        s *= 2


def _cachemir_band(N, hid, ff, heads):
    rots = set()
    for d_in, d_out in ((hid, hid), (hid, ff), (ff, hid), (hid, N)):
        p = _cm(N, d_in, d_out)
        rots |= {s * (p["t"] - 1) for s in _pow2(1, p["tp_in"])}
        rots |= {j * p["t"] * p["t"] for j in range(1, p["r_i"])}
        rots.add(p["t"] * p["tp"])
        rots |= set(_pow2(1, p["tp_out"]))
    t = N // hid
    rots |= set(_pow2(t, N)) | set(_pow2(1, N))
    tH, d_head = t * heads, hid // heads
    rots |= {-i for i in range(1, t)} | set(range(1, t)) | {-s for s in _pow2(1, t)}
    rots |= set(_pow2(tH, N)) | set(_pow2(1, t)) | {s - t for s in _pow2(1, t)}
    rots |= {i * tH for i in range(1, d_head)}
    return sorted(rots)


def _dg(N, d_in, d_out):
    s = int(math.sqrt(d_in))
    while s > 1 and d_in % s:
        s -= 1
    return dict(t_in=N // d_in, t_out=N // d_out, alpha=max(d_in, d_out) // min(d_in, d_out),
                s=s, G=d_in // s)


def _diagonal_band(N, hid, ff, heads):
    rots = set()

    def linear(d_in, d_out):
        p = _dg(N, d_in, d_out)
        rots.update(b * p["t_in"] for b in range(1, p["s"]))
        rots.update(g * p["s"] * p["t_in"] for g in range(1, p["G"]))

    linear(hid, hid)
    linear(hid, ff)
    pu = _dg(N, hid, ff)
    rots.update(-g * pu["t_out"] for g in range(1, pu["alpha"]))
    linear(ff, hid)
    linear(hid, N)
    rots |= set(_pow2(N // hid, N))
    return sorted(rots)


def _filling_band(N, hid, ff, heads):
    rots = set(_diagonal_band(N, hid, ff, heads))
    t = N // hid
    rots |= set(range(1, t)) | {-i for i in range(1, t)}
    rots |= set(_pow2(t * heads, N)) | {-s for s in _pow2(t * heads, N)}
    return sorted(rots)


_BANDS = {"cachemir": _cachemir_band, "diagonal": _diagonal_band, "cachemir_filling": _filling_band}


@pytest.mark.parametrize("shape", [(32768, 1024, 4096, 16), (2048, 64, 256, 4)])
@pytest.mark.parametrize("kind", sorted(_BANDS))
def test_bands_match_transcription(kind, shape):
    assert list(_client._debug.rot_band(kind, *shape)) == _BANDS[kind](*shape)


def test_fixture_sidecar_is_the_cachemir_band():
    dev = (FIXTURES / "context.bin.dev").read_text()
    steps = [int(x) for x in dev.split("RotationIndexes: {")[1].split("}")[0].split()]
    assert steps == _cachemir_band(32768, 1024, 4096, 16) and len(steps) == 133
    assert list(_client._debug.family_rot_band(_fixture_options(), "gpt2")) == steps
    o = _fixture_options()
    o.aux_packing_kinds = [_client.PackingKind.CachemirFilling]
    assert list(_client._debug.family_rot_band(o, "gpt2")) == sorted(
        set(steps) | set(_filling_band(32768, 1024, 4096, 16)))


# ---- a small CPU context: packing geometry, bundle layout, ciphertext bytes ----
# logN 14 is the smallest ring the HE-standard check accepts for a 56 + 2 x 54-bit chain
# with hybrid key switching; keygen takes seconds.

_N = 8192   # slots at logN 14
_D = 128    # hidDim: the cachemir band needs d*d >= N (n_pt = d / tp_out > 0)

def _small_options(ext=_client, packing="Cachemir", aux=()):
    o = ext.InferenceOptions()
    ck = ext.CKKSOptions()
    ck.logN, ck.enable_bootstrap, ck.depth, ck.composite_degree = 14, False, 2, 2
    ck.scale_bits, ck.btp_scale_bits, ck.first_mod_bits, ck.num_large_digits = 54, 54, 56, 6
    ck.h_weight = 192
    o.ckks = ck
    o.dim = o.hidDim = _D
    o.expanded = o.expDim = 4 * _D
    o.numHeads = o.numHeadsReal = 4
    o.seqLen = 16
    o.packing_kind = getattr(ext.PackingKind, packing)
    o.aux_packing_kinds = [getattr(ext.PackingKind, a) for a in aux]
    o.mode = ext.InferenceMode.Sync
    return o


@pytest.fixture(scope="module")
def small():
    opts = _small_options(aux=("CachemirFilling",))
    return opts, _client.make_gpt2_inference(opts)


def test_small_context_encode_geometry(small):
    opts, inf = small
    assert inf.slots == _N and inf.logN == 14 and inf.size.dim == _D and inf.packing == "cachemir"
    assert inf.fhe.has_secret_key and not inf.fhe.from_keys and inf.fhe.key_dist == 3
    assert inf.fhe.bootstrap_output_level() == 0
    band = list(_client._debug.family_rot_band(opts, "gpt2"))
    assert list(inf.fhe.loaded_rot_steps) == band == sorted(
        set(_cachemir_band(_N, _D, 4 * _D, 4)) | set(_filling_band(_N, _D, 4 * _D, 4)))
    rng = np.random.default_rng(3)
    x = rng.standard_normal(_D) * 0.3
    L = 2
    ct = _client.pack_tokens(inf, x[None, :], L)
    assert ct.level == L and ct.packing == "cachemir" and "cachemir" in repr(ct)
    slots = np.asarray(_client.decrypt_slots(inf, ct))
    t = _N // _D
    np.testing.assert_allclose(slots[::t], x, atol=1e-6)
    mask = np.ones(_N, bool)
    mask[::t] = False
    assert np.abs(slots[mask]).max() < 1e-6
    np.testing.assert_allclose(_client.decode_token_output(inf, ct), x, atol=1e-6)
    np.testing.assert_allclose(_client.decode_tokens_output(inf, ct, 1)[0], x, atol=1e-6)
    np.testing.assert_allclose(_client.decode_linear_output(inf, ct, _D, _D), x, atol=1e-6)
    short = _client.decode_token_output(inf, _client.pack_tokens(inf, [x[:10].tolist()], L))
    np.testing.assert_allclose(short[:10], x[:10], atol=1e-6)
    assert np.abs(short[10:]).max() < 1e-6
    fresh = _client.encode_token_input(inf, x)
    assert fresh.level == inf.fhe.bootstrap_output_level()
    with pytest.raises(ValueError, match="do not fit"):
        _client.encode_token_input(inf, np.zeros(_D + 1))


def test_small_context_ciphertext_bytes(small):
    _opts, inf = small
    x = np.linspace(-1, 1, _D)
    ct = _client.pack_tokens(inf, x[None, :], 2)
    blob = _client.serialize_ct(inf, ct)
    assert isinstance(blob, bytes) and len(blob) > 1000
    back = _client.deserialize_ct(inf, blob)
    assert back.level == 2 and back.packing == "cachemir"
    assert _client.serialize_ct(inf, back) == blob
    np.testing.assert_allclose(_client.decode_token_output(inf, back), x, atol=1e-6)
    with pytest.raises(RuntimeError):
        _client.deserialize_ct(inf, b"")


def test_small_context_bundle_layout_and_reload(small, tmp_path):
    opts, inf = small
    bundle = tmp_path / "bundle"
    _client.save_keys(inf, str(bundle))
    assert sorted(p.name for p in bundle.iterdir()) == [
        "context.bin", "context.bin.dev", "multkeys.bin", "public.key", "rotkeys.bin"]
    assert (bundle / "context.bin.dev").read_text() == _client._debug.dev_sidecar(inf)
    assert (bundle / "context.bin").read_bytes() == _client._debug.bundle_meta(opts, "gpt2")[0]
    secret = tmp_path / "secret.key"
    _client.save_secret_key(inf, str(secret))

    x = np.random.default_rng(5).standard_normal(_D)
    blob = _client.serialize_ct(inf, _client.pack_tokens(inf, x[None, :], 2))

    o2 = _client.InferenceOptions(opts)
    o2.ckks.keys_dir = str(bundle)
    inf2 = _client.make_gpt2_inference(o2)
    assert inf2.fhe.from_keys and not inf2.fhe.has_secret_key
    assert list(inf2.fhe.loaded_rot_steps) == list(inf.fhe.loaded_rot_steps)
    assert inf2.fhe.key_tag == inf.fhe.key_tag and inf2.fhe.key_dist == 3
    with pytest.raises(RuntimeError, match="no secret key"):
        _client.decode_token_output(inf2, _client.deserialize_ct(inf2, blob))
    # the bundle's public key encrypts; the first session decrypts (same key pair)
    y = np.random.default_rng(6).standard_normal(_D)
    blob2 = _client.serialize_ct(inf2, _client.pack_tokens(inf2, y[None, :], 2))
    np.testing.assert_allclose(_client.decode_token_output(inf, _client.deserialize_ct(inf, blob2)),
                               y, atol=1e-6)
    _client.load_secret_key(inf2, str(secret))
    assert inf2.fhe.has_secret_key
    np.testing.assert_allclose(_client.decode_token_output(inf2, _client.deserialize_ct(inf2, blob)),
                               x, atol=1e-6)

    got = _client._debug.automorphism_indexes_in_file(str(bundle / "rotkeys.bin"))
    assert list(got[inf.fhe.key_tag]) == list(inf.fhe.automorphism_key_indexes)
    assert list(got[inf.fhe.key_tag]) == list(_client._debug.expected_automorphism_indexes(opts, "gpt2"))
    # no bootstrap keys here: one automorphism key per DISTINCT automorphism index of the band
    # (steps s and s - N/2 share one at full slots), nothing else
    assert len(got[inf.fhe.key_tag]) <= len(inf.fhe.loaded_rot_steps)


def test_small_diagonal_context_packs_tokens_in_lanes():
    opts = _small_options(packing="Diagonal")
    inf = _client.make_gpt2_inference(opts)
    assert inf.packing == "diagonal"
    assert list(inf.fhe.loaded_rot_steps) == _diagonal_band(_N, _D, 4 * _D, 4)
    x = np.random.default_rng(8).standard_normal((2, _D))
    ct = _client.pack_tokens(inf, x, 2)
    assert inf.n_tok == 2 and ct.packing == "diagonal"
    slots = np.asarray(_client.decrypt_slots(inf, ct))
    t = _N // _D
    for tok in range(2):
        np.testing.assert_allclose(slots[tok::t][:_D], x[tok], atol=1e-6)
    np.testing.assert_allclose(_client.decode_tokens_output(inf, ct, 2), x, atol=1e-6)
    np.testing.assert_allclose(_client.decode_linear_output(inf, ct, _D, _D), x[0], atol=1e-6)


def _err(name):
    """_client registers MODULE-LOCAL pybind11 translators, so its C++ exceptions map to its
    own classes even with _core loaded in the same interpreter (and _core's stay _core's)."""
    return getattr(_client, name)


def test_stamp_and_errors():
    info = _client.build_info()
    assert info["chain"] == _client.chain in ("n32", "n64")
    assert info["native_int_bits"] == _client.native_int_bits in (32, 64)
    assert info["backend"] == "client" and info["cuda_runtime"] is None
    assert _client.__version__ == info["version"]
    # the two extensions are separate stashes behind one import name each, so on a machine
    # with both chains built they can be different builds; compare only when they agree
    if _core is not None and _core.chain == _client.chain:
        assert _core.native_int_bits == _client.native_int_bits
    assert issubclass(_client.FHEError, RuntimeError)
    assert issubclass(_client.PlanError, _client.FHEError)
    for kind, name in (("plan", "PlanError"), ("mask", "MaskError"),
                       ("layout", "LayoutError"), ("other", "FHEError")):
        with pytest.raises(_err(name)):
            _client._debug.throw_typed(kind, "no marker")
        with pytest.raises(RuntimeError):
            _client._debug.throw_typed(kind, "still a RuntimeError")
    with pytest.raises(RuntimeError, match=r"\[openfhe\]"):
        _client._debug.throw_typed("openfhe", "from OpenFHE")
    for marker, name in (("[plan_level_error] x", "PlanError"), ("[mask_gen] x", "MaskError"),
                         ("[layout_error] x", "LayoutError"), ("", "FHEError")):
        with pytest.raises(_err(name)):
            _client.throw_test(marker)


def test_client_errors_surface_as_perseus_errors(monkeypatch):
    """EncClient re-raises the extension's errors as perseus.errors' classes."""
    from perseus import errors
    from perseus.nn import remote
    with remote._translate(_client):
        pass
    for kind, cls in (("plan", errors.PlanError), ("mask", errors.MaskError),
                      ("layout", errors.LayoutError), ("other", errors.FHEError)):
        with pytest.raises(cls):
            with remote._translate(_client):
                _client._debug.throw_typed(kind, "x")
    with pytest.raises(ValueError):                 # foreign errors pass through untouched
        with remote._translate(_client):
            raise ValueError("not ours")


# ---- interchange with _core (opt-in) ----

def test_core_bundle_decrypts_on_client():
    """A bundle + secret key + input.ct written by _core's client (probe_client_server.py)
    open on _client: same files, same ciphertext bytes, same cachemir slot layout."""
    bundle = os.environ.get("PERSEUS_CORE_BUNDLE")
    need = ("context.bin", "context.bin.dev", "public.key", "secret.key", "input.ct")
    if not bundle or not all(os.path.exists(os.path.join(bundle, f)) for f in need):
        pytest.skip("set PERSEUS_CORE_BUNDLE to a probe_client_server.py bundle dir")
    opts = _from_env(_client, _N32_ENV)
    opts.ckks.keys_dir = bundle
    inf = _client.make_gpt2_inference(opts)
    assert not inf.fhe.has_secret_key and inf.fhe.from_keys
    assert len(inf.fhe.loaded_rot_steps) == 133
    _client.load_secret_key(inf, os.path.join(bundle, "secret.key"))
    ct = _client.deserialize_ct(inf, open(os.path.join(bundle, "input.ct"), "rb").read())
    assert ct.level == 32
    rng = np.random.default_rng(7)                   # probe_client_server.py's draw order
    rng.standard_normal((1024, 4096))
    x_real = rng.standard_normal(768) * 0.3
    np.testing.assert_allclose(_client.decode_token_output(inf, ct), x_real, atol=1e-6)


@_SLOW
def test_client_full_n32_keygen_matches_fixture_index_set():
    """The real n32 client keygen on _client alone (minutes, ~35 GB under TMPDIR): the
    context it writes is the fixture's byte-for-byte, its rotkeys.bin holds exactly the
    fixture's automorphism index set, and a level-32 token round-trips."""
    opts = _fixture_options()
    with scoped_env(**_N32_ENV):
        inf = _client.make_gpt2_inference(opts)
    assert inf.fhe.bootstrap_output_level() == 32
    assert list(inf.fhe.automorphism_key_indexes) == json.loads((FIXTURES / "rotkey_indexes.json").read_text())
    x = np.random.default_rng(1).standard_normal(768) * 0.3
    ct = _client.encode_token_input(inf, x)
    assert ct.level == 32
    blob = _client.serialize_ct(inf, ct)
    np.testing.assert_allclose(_client.decode_token_output(inf, _client.deserialize_ct(inf, blob)), x, atol=1e-5)
    bundle = tempfile.mkdtemp(prefix="perseus_client_bundle_", dir=os.environ.get("TMPDIR"))
    try:
        _client.save_keys(inf, bundle)
        assert open(os.path.join(bundle, "context.bin"), "rb").read() == (FIXTURES / "context.bin").read_bytes()
        assert open(os.path.join(bundle, "context.bin.dev")).read() == (FIXTURES / "context.bin.dev").read_text()
        got = _client._debug.automorphism_indexes_in_file(os.path.join(bundle, "rotkeys.bin"))
        assert list(got[inf.fhe.key_tag]) == json.loads((FIXTURES / "rotkey_indexes.json").read_text())
    finally:
        shutil.rmtree(bundle, ignore_errors=True)



def _isolated(request) -> bool:
    """Run the calling test in a fresh interpreter and report True to the caller (which
    then returns). perseus._core keeps process-global runtime state: a second _core
    session created after one that initialised the GPU re-serialises a host-resident
    ciphertext with different bytes (byte-identity holds in a fresh process, not after
    another _core session in the same one), so
    every slow test that builds a _core session gets its own process."""
    if os.environ.get("PERSEUS_ISOLATED_TEST"):
        return False
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        request.node.nodeid],
                       cwd=str(REPO), capture_output=True, text=True,
                       env={**os.environ, "PERSEUS_ISOLATED_TEST": "1"})
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-2000:]
    return True

@_SLOW
def test_core_client_bundle_interchange_cpu(request):
    """Both directions through the real loaders (orchestrator: needs _core, a GPU box for
    _core's cudaMemGetInfo, ~2 x 35 GB under TMPDIR): (a) a _core bundle + secret key opens
    on _client and decrypts a _core ciphertext; (b) a _client bundle loads through _core's
    keys_dir server path (context, sidecar, multkeys.bin, rotkeys.bin) and _core's bytes
    survive a _client round trip."""
    if _isolated(request):
        return
    if _core is None:
        pytest.skip("perseus._core not built")
    from perseus.nn import EncClient
    rng = np.random.default_rng(0)
    x = rng.standard_normal(768) * 0.3
    tmp = tempfile.mkdtemp(prefix="perseus_interchange_", dir=os.environ.get("TMPDIR"))
    try:
        with scoped_env(**_N32_ENV):
            # (a) _core client -> _client
            core_client = EncClient(backend="core")
            core_bundle = core_client.save_bundle(os.path.join(tmp, "core"))
            core_client.save_secret_key(os.path.join(tmp, "core.secret"))
            blob = core_client.encrypt(x)
            o = _from_env(_client, _N32_ENV)
            o.ckks.keys_dir = core_bundle
            cinf = _client.make_gpt2_inference(o)
            _client.load_secret_key(cinf, os.path.join(tmp, "core.secret"))
            ct = _client.deserialize_ct(cinf, blob)
            assert ct.level == 32
            np.testing.assert_allclose(_client.decode_token_output(cinf, ct), x, atol=1e-5)
            # (b) _client client -> _core server loader
            client = EncClient(backend="client")
            bundle = client.save_bundle(os.path.join(tmp, "client"))
            so = _from_env(_core, _N32_ENV)
            so.ckks.keys_dir = bundle
            so.ckks.skip_gpu_load = True
            sinf = _core.make_gpt2_inference(so)
            assert not sinf.fhe.has_secret_key
            cblob = client.encrypt(x)
            pc = _core.deserialize_ct(sinf, cblob)
            assert pc.level == 32
            assert _core.serialize_ct(sinf, pc) == cblob
            np.testing.assert_allclose(client.decrypt(_core.serialize_ct(sinf, pc), d=768), x, atol=1e-5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@_SLOW
def test_client_bundle_loads_through_core_server_loader_cpu(request):
    """A _client bundle through _core's keys_dir server loader on the CPU (no device: the
    dense-only bootstrap setup keeps _core off cudaMemGetInfo): context.bin + sidecar,
    public.key, multkeys.bin, the ~30 GB rotkeys.bin and the bootstrap setup all load; a
    _client ciphertext deserializes on _core byte-for-byte at level 32; _core's own
    pack_tokens on the loaded public key decrypts on _client with the same slot layout.
    Minutes, ~70 GB RAM; the sparse-slot variant and the bootstrap itself are the GPU
    tests (tests/gpu/test_client_roles.py)."""
    if _isolated(request):
        return
    if _core is None:
        pytest.skip("perseus._core not built")
    env = {**_N32_ENV, "SPARSE_BTS_SLOTS": None}
    opts = _from_env(_client, env)
    with scoped_env(**env):
        inf = _client.make_gpt2_inference(opts)
    tmp = tempfile.mkdtemp(prefix="perseus_client_bundle_", dir=os.environ.get("TMPDIR"))
    try:
        _client.save_keys(inf, tmp)
        x = np.random.default_rng(2).standard_normal(768) * 0.3
        blob = _client.serialize_ct(inf, _client.pack_tokens(inf, x[None, :], 32))
        so = _from_env(_core, env)
        so.ckks.keys_dir = tmp
        so.ckks.skip_gpu_load = True
        so.mode = _core.InferenceMode.Sync
        with scoped_env(**env):
            sinf = _core.make_gpt2_inference(so)
        assert not sinf.fhe.has_secret_key
        assert sorted(sinf.fhe.loaded_rot_steps) == sorted(inf.fhe.loaded_rot_steps)
        pc = _core.deserialize_ct(sinf, blob)
        assert pc.level == 32 and _core.serialize_ct(sinf, pc) == blob
        np.testing.assert_allclose(_client.decode_token_output(inf, _client.deserialize_ct(inf, blob)),
                                   x, atol=1e-5)
        y = np.random.default_rng(3).standard_normal(768) * 0.3
        cblob = _core.serialize_ct(sinf, _core.pack_tokens(sinf, y[None, :], 32))
        cct = _client.deserialize_ct(inf, cblob)
        assert cct.level == 32
        np.testing.assert_allclose(_client.decode_token_output(inf, cct), y, atol=1e-5)
        t = inf.slots // 1024
        np.testing.assert_allclose(np.asarray(_client.decrypt_slots(inf, cct))[::t][:768], y, atol=1e-5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
