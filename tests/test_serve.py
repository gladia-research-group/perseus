"""Generation across the trust boundary, driven with fakes (CPU): the wire framing, the
client's embedding/argmax roles, the protocol loop, and from_pretrained's resolution."""
import json
import zipfile

import numpy as np
import pytest

pytest.importorskip("perseus._core")
from perseus.nn import serve
from perseus.nn.pretrained import resolve_artifacts


class _FakeCore:
    """serialize_ct/deserialize_ct on tagged fakes, decode_lm_head_logits from a script."""

    def __init__(self):
        self.script = []

    def serialize_ct(self, inf, ct):
        return f"ct:{ct}".encode()

    def deserialize_ct(self, inf, blob):
        return blob.decode()[3:]

    def decode_lm_head_logits(self, inf, tiles, vocab):
        z = np.zeros(vocab)
        z[self.script.pop(0)] = 9.0
        return z.tolist()


@pytest.fixture
def fake_core(monkeypatch):
    fc = _FakeCore()
    monkeypatch.setattr(serve, "_ext", fc)      # the module default the client half runs on
    return fc


def test_tile_framing_round_trip(fake_core):
    blob = serve.serialize_tiles(None, ["t0", "t1", "tile-two"])
    assert blob.startswith(serve._MAGIC)
    assert serve.deserialize_tiles(None, blob) == ["t0", "t1", "tile-two"]
    with pytest.raises(ValueError, match="not a perseus tile payload"):
        serve.deserialize_tiles(None, b"garbage")


class _FakeClient:
    inf = object()

    def __init__(self):
        self.encrypted = []
        self.accepted = None
        self.accepted_before = None     # how many encrypts preceded accept()

    def encrypt(self, v):
        self.encrypted.append(np.asarray(v))
        return f"ct:x{len(self.encrypted)}".encode()

    def accept(self, m):
        self.accepted = m
        self.accepted_before = len(self.encrypted)
        return m.get("bootstrap_output_level")


class _FakeServer:
    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append("start")

    def prompt(self, blobs):
        self.calls.append(("prompt", len(blobs)))
        return serve.serialize_tiles(None, ["p"])

    def step_encrypted(self):
        self.calls.append("enc")
        return serve.serialize_tiles(None, ["e"])

    def step_client(self, blob):
        self.calls.append(("cli", blob))
        return serve.serialize_tiles(None, ["c"])


class _FakeServerWithManifest(_FakeServer):
    """A server that, like EncGenerationServer, hands back its probed encode level."""

    def session_manifest(self):
        self.calls.append("manifest")
        return {"bootstrap_output_level": 34}


def test_client_embeds_and_drives_the_protocol(fake_core):
    wte = np.arange(20, dtype=float).reshape(5, 4)      # vocab 5, d 4
    wpe = 100 * np.arange(12, dtype=float).reshape(3, 4)  # 3 positions
    fc = _FakeClient()
    c = serve.EncGenerationClient(fc, wte, wpe)
    np.testing.assert_array_equal(c.embed(2, 1), wte[2] + wpe[1])
    with pytest.raises(ValueError, match="outside the vocabulary"):
        c.embed(7, 0)
    with pytest.raises(ValueError, match="beyond the 3 positions"):
        c.embed(1, 3)

    s = _FakeServer()
    fake_core.script = [4, 1, 3]
    ids = c.generate(s, [0, 1], max_new_tokens=3, feedback="encrypted")
    assert ids == [4, 1, 3]
    assert s.calls == ["start", ("prompt", 2), "enc", "enc"]     # no token crosses to the server
    assert len(fc.encrypted) == 2                                  # only the prompt was encrypted
    assert fc.accepted is None                                     # a server without the handshake

    s = _FakeServer(); fc = _FakeClient(); c = serve.EncGenerationClient(fc, wte, wpe)
    fake_core.script = [4, 1, 3]
    ids = c.generate(s, [0], max_new_tokens=3, feedback="client", eos_token_id=1)
    assert ids == [4, 1]                                           # eos stops
    assert s.calls == ["start", ("prompt", 1), ("cli", b"ct:x2")]
    np.testing.assert_array_equal(fc.encrypted[1], wte[4] + wpe[1])  # re-embedded at pos 1


def test_generate_adopts_the_servers_level_before_encrypting_the_prompt(fake_core):
    wte = np.arange(20, dtype=float).reshape(5, 4)
    wpe = 100 * np.arange(12, dtype=float).reshape(3, 4)
    fc = _FakeClient(); c = serve.EncGenerationClient(fc, wte, wpe)
    s = _FakeServerWithManifest()
    fake_core.script = [2]
    ids = c.generate(s, [0, 1], max_new_tokens=1, feedback="client")
    assert ids == [2]
    assert s.calls[:2] == ["start", "manifest"]                    # handshake right after start
    assert fc.accepted == {"bootstrap_output_level": 34}
    assert fc.accepted_before == 0                                 # ...and before any encrypt
    assert len(fc.encrypted) == 2                                  # the prompt


def test_generation_server_forwards_the_session_manifest():
    class Inf:
        pass

    inf = Inf()

    class Srv:
        def session_manifest(self):
            return {"bootstrap_output_level": 34, "family": "gpt2"}

    class M:
        pass

    srv, m = Srv(), M()
    srv.inf = m.inf = inf
    gs = serve.EncGenerationServer(srv, m)
    assert gs.session_manifest() == {"bootstrap_output_level": 34, "family": "gpt2"}


def test_generate_argument_validation(fake_core):
    c = serve.EncGenerationClient(_FakeClient(), np.zeros((3, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="feedback must be"):
        c.generate(_FakeServer(), [0], 1, feedback="x")
    with pytest.raises(ValueError, match="sampling needs"):
        c.generate(_FakeServer(), [0], 1, feedback="encrypted", do_sample=True)
    with pytest.raises(ValueError, match="at least one prompt"):
        c.generate(_FakeServer(), [], 1)


def test_server_requires_a_model_bound_to_its_session():
    class M:
        inf = object()

    class S:
        inf = object()

    with pytest.raises(ValueError, match="bound to the server's session"):
        serve.EncGenerationServer(S(), M())


def test_resolve_artifacts_names_the_producing_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="perseus-export --model openai-community/gpt2"):
        resolve_artifacts("openai-community/gpt2", configs=tmp_path)
    w = tmp_path / "w"; w.mkdir()
    with zipfile.ZipFile(w / "weights.bin.zip", "w") as zf:
        zf.writestr("manifest.json", json.dumps({"tensors": []}))
    with pytest.raises(FileNotFoundError, match="configs= is required"):
        resolve_artifacts("x", weights=str(w / "weights.bin.zip"))
    (tmp_path / "configs.json").write_text("{}")
    got = resolve_artifacts("x", weights=str(w / "weights.bin.zip"), configs=str(tmp_path))
    assert got == (str(w / "weights.bin.zip"), str(tmp_path / "configs.json"))
