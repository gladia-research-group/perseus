import json
import struct

import numpy as np

from .. import _backend
from ._sampling import _make_sampler, _stop_ids

_ext = None   # None = resolve on first use (see remote._default_ext)


def _default_ext():
    return _ext if _ext is not None else _backend.default()

_MAGIC = b"PRSTILE1"


def serialize_tiles(inf, tiles, ext=None):
    """[PackedCtx] -> bytes (magic, u32 header length, JSON header, blobs); `ext` is the
    extension `inf` belongs to (default: the loaded extension)."""
    ext = ext or _default_ext()
    blobs = [ext.serialize_ct(inf, t) for t in tiles]
    hdr = json.dumps({"n": len(blobs), "lengths": [len(b) for b in blobs]}).encode()
    return _MAGIC + struct.pack("<I", len(hdr)) + hdr + b"".join(blobs)


def deserialize_tiles(inf, blob, ext=None):
    """bytes -> [PackedCtx] (inverse of serialize_tiles)."""
    ext = ext or _default_ext()
    if blob[:8] != _MAGIC:
        raise ValueError("not a perseus tile payload")
    hlen = struct.unpack("<I", blob[8:12])[0]
    hdr = json.loads(blob[12:12 + hlen])
    out, off = [], 12 + hlen
    for n in hdr["lengths"]:
        out.append(ext.deserialize_ct(inf, blob[off:off + n]))
        off += n
    return out


class EncGenerationServer:
    """Blind generation: a bound EncGPT2 on an EncServer session, bytes in / bytes out."""

    def __init__(self, server, model):
        if model.inf is not server.inf:
            raise ValueError("the model must be bound to the server's session "
                             "(model.bind(server.inf))")
        self.server = server
        self.model = model
        self._tiles = None
        self._position = 0

    @property
    def _core(self):
        from .. import _core  # the server role is _core-only (ImportError with the recipe)
        return _core

    def start(self):
        self.model.start()
        self._tiles, self._position = None, 0
        return self

    def session_manifest(self) -> dict:
        """The EncServer handshake document (the bundle manifest with the probed
        `bootstrap_output_level`); the client adopts it with `EncClient.accept()`."""
        return self.server.session_manifest()

    def prompt(self, x_blobs):
        """Consume the prompt (one encrypted embedding per position); returns the
        encrypted logit tiles of the last position."""
        inf = self.model.inf
        P = len(x_blobs)
        core = self._core
        for p, blob in enumerate(x_blobs):
            inf.capture_t = p
            self._tiles = self.model.forward(core.deserialize_ct(inf, blob), head=(p == P - 1))
        self._position = P - 1
        return serialize_tiles(inf, self._tiles, core)

    def step_encrypted(self):
        """Encrypted feedback: CutMax argmax + codebook re-embedding on the server (no
        token id is ever visible here); returns the next position's logit tiles."""
        m = self.model
        inf = m.inf
        self._position += 1
        inf.capture_t = self._position
        m.prepare_feedback()
        core = self._core
        x, _z, _am_s = core.cutmax_feedback(inf, self._tiles, m.store, m.lm_head.vocab,
                                            m.cutmax.config, m._fb, self._position,
                                            m._plan13, m._plan14)
        self._tiles = m.forward(x)
        return serialize_tiles(inf, self._tiles, core)

    def step_client(self, x_blob):
        """Client feedback: the client re-embedded its chosen token; run one position."""
        inf = self.model.inf
        self._position += 1
        inf.capture_t = self._position
        core = self._core
        self._tiles = self.model.forward(core.deserialize_ct(inf, x_blob))
        return serialize_tiles(inf, self._tiles, core)


class EncGenerationClient:
    """Key owner for generation: embeds token ids, decrypts logit tiles to the next id."""

    def __init__(self, client, wte, wpe, vocab=None):
        self.client = client
        self.wte = np.asarray(wte, dtype=np.float64)
        self.wpe = np.asarray(wpe, dtype=np.float64)
        if self.wte.ndim != 2 or self.wpe.ndim != 2 or self.wte.shape[1] != self.wpe.shape[1]:
            raise ValueError(f"wte {self.wte.shape} and wpe {self.wpe.shape} must be "
                             f"(vocab, d) and (positions, d)")
        self.vocab = int(vocab if vocab is not None else self.wte.shape[0])

    def embed(self, token_id, position):
        """wte[id] + wpe[position] (the GPT-2 input embedding), as a plaintext row."""
        if not 0 <= int(token_id) < self.wte.shape[0]:
            raise ValueError(f"token id {token_id} outside the vocabulary of {self.wte.shape[0]}")
        if not 0 <= int(position) < self.wpe.shape[0]:
            raise ValueError(f"position {position} beyond the {self.wpe.shape[0]} positions")
        return self.wte[int(token_id)] + self.wpe[int(position)]

    def encrypt_tokens(self, ids, start_position=0):
        """[ids] -> [ciphertext bytes], one per position."""
        return [self.client.encrypt(self.embed(t, start_position + i)) for i, t in enumerate(ids)]

    def logits(self, tiles_blob):
        """Encrypted logit tiles (bytes) -> the [vocab] logits, on the client's extension."""
        inf = self.client.inf
        ext = getattr(self.client, "backend", None) or _default_ext()
        tiles = deserialize_tiles(inf, tiles_blob, ext)
        return np.asarray(ext.decode_lm_head_logits(inf, tiles, self.vocab), dtype=np.float64)

    def next_token(self, tiles_blob, sampler=None):
        z = self.logits(tiles_blob)
        return int(sampler(z)) if sampler is not None else int(np.argmax(z))

    def generate(self, server, prompt_ids, max_new_tokens, feedback="encrypted", *,
                 eos_token_id=None, on_token=None, do_sample=False, temperature=1.0,
                 top_k=None, seed=None):
        """Drive the protocol against `server` (an EncGenerationServer or anything with
        start/prompt/step_encrypted/step_client); returns the emitted token ids. A server
        that also exposes session_manifest() (EncGenerationServer does) has its probed
        fresh-encode level adopted before the prompt is encrypted, so client encodes
        match a strict plan's block-0 entry."""
        if feedback not in ("encrypted", "client"):
            raise ValueError(f"feedback must be 'encrypted' or 'client', got {feedback!r}")
        if do_sample and feedback != "client":
            raise ValueError("sampling needs feedback='client' (the server's CutMax is an argmax)")
        if not prompt_ids:
            raise ValueError("generate needs at least one prompt token")
        stops = _stop_ids(eos_token_id)
        sampler = _make_sampler(do_sample, temperature, top_k, seed)
        server.start()
        session_manifest = getattr(server, "session_manifest", None)
        if callable(session_manifest):
            self.client.accept(session_manifest())
        tiles = server.prompt(self.encrypt_tokens(prompt_ids))
        ids = []
        for j in range(int(max_new_tokens)):
            tok = self.next_token(tiles, sampler)
            ids.append(tok)
            if on_token is not None:
                on_token(tok, j)
            if j + 1 == int(max_new_tokens) or tok in stops:
                break
            if feedback == "encrypted":
                tiles = server.step_encrypted()
            else:
                pos = len(prompt_ids) + j
                tiles = server.step_client(self.client.encrypt(self.embed(tok, pos)))
        return ids
