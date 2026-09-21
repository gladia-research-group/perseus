"""CPU tests for EncGPT2's generation loop, driven through a fake session.

The loop's contract (bounded by max_new_tokens, stopped by eos, callbacks in order,
client re-embedding, deterministic sampling) is exercised without a WeightStore, a
context or a GPU: forward/decode_logits/embed_token are stubbed on a subclass.
"""
import numpy as np
import pytest

pytest.importorskip("perseus._core")
from perseus.nn import EncGPT2, EncModule
from perseus.nn.gpt2 import _make_sampler, _stop_ids


class _FakeInf:
    capture_t = None


class _Fake(EncGPT2):
    """Scripted logits: step j emits `script[j]` as the argmax."""

    def __init__(self, script, vocab=16):
        EncModule.__init__(self)          # skip EncGPT2.__init__ (needs a store)
        self.inf = _FakeInf()
        self.script = list(script)
        self.vocab = vocab
        self.calls = []
        self._planned = False
        self._plan14 = None
        self._ran_decode = False
        self.n_layers = 1
        self._step = 0

    def start(self):
        self.calls.append("start")

    def forward(self, x, head=True):
        self.calls.append(("forward", x, head))
        return ("tiles", self._step)

    def decode_logits(self, tiles):
        j = self._step
        self._step += 1
        z = np.zeros(self.vocab)
        z[self.script[j]] = 10.0
        return z.tolist()

    def encode_input(self, values):
        return ("x", list(values))

    def embed_token(self, token_id, position):
        self.calls.append(("embed", token_id, position))
        return ("x", token_id)


def test_bounded_by_max_new_tokens_and_reembeds_on_the_client_path():
    m = _Fake([5, 7, 9, 11])
    ids = m.generate([[0.0]] * 2, max_new_tokens=3, feedback="client")
    assert ids == [5, 7, 9]
    # 2 prompt forwards (head only on the last) + 2 feedback forwards; re-embed at pos+1
    fwd = [c for c in m.calls if c[0] == "forward"]
    assert [c[2] for c in fwd] == [False, True, True, True]
    assert [c for c in m.calls if c[0] == "embed"] == [("embed", 5, 2), ("embed", 7, 3)]
    assert m.calls[0] == "start" and m._ran_decode


def test_positional_n_tokens_is_an_alias():
    assert _Fake([1, 2, 3]).generate([[0.0]], 2, feedback="client") == [1, 2]
    with pytest.raises(ValueError, match="not both"):
        _Fake([1]).generate([[0.0]], 1, max_new_tokens=1, feedback="client")
    with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
        _Fake([1]).generate([[0.0]], feedback="client")


def test_eos_stops_early_and_is_returned():
    m = _Fake([4, 2, 8, 8])
    assert m.generate([[0.0]], max_new_tokens=4, feedback="client", eos_token_id=2) == [4, 2]
    m = _Fake([4, 2, 8, 8])
    assert m.generate([[0.0]], max_new_tokens=4, feedback="client",
                      eos_token_id={8, 3}) == [4, 2, 8]


def test_stream_yields_incrementally_and_on_token_sees_each_step():
    m = _Fake([3, 1, 4])
    seen = []
    gen = m.stream([[0.0]], 3, feedback="client", on_token=lambda t, j: seen.append((t, j)))
    assert next(gen) == 3 and seen == [(3, 0)]
    assert list(gen) == [1, 4] and seen == [(3, 0), (1, 1), (4, 2)]


def test_argument_validation():
    m = _Fake([1])
    with pytest.raises(ValueError, match="feedback must be"):
        m.generate([[0.0]], 1, feedback="server")
    with pytest.raises(ValueError, match="prompt_mode must be"):
        m.generate([[0.0]], 1, prompt_mode="chunked", feedback="client")
    with pytest.raises(ValueError, match="at least one prompt"):
        m.generate([], 1, feedback="client")
    with pytest.raises(ValueError, match="sampling needs feedback='client'"):
        m.generate([[0.0]], 1, feedback="encrypted", do_sample=True)


def test_sampling_is_seeded_and_respects_top_k():
    logits = [0.0, 1.0, 50.0, 49.9]
    s1, s2 = _make_sampler(True, 1.0, None, 0), _make_sampler(True, 1.0, None, 0)
    a = [s1(logits) for _ in range(20)]
    assert a == [s2(logits) for _ in range(20)]      # same seed, same draws
    assert set(a) <= {2, 3}                            # the two dominant tokens
    top1 = _make_sampler(True, 1.0, 1, 0)
    assert all(top1(logits) == 2 for _ in range(10))  # top_k=1 is argmax
    assert _make_sampler(False, 1.0, None, None) is None
    with pytest.raises(ValueError, match="temperature"):
        _make_sampler(True, 0.0, None, None)
    m = _Fake([0, 0, 0])
    m.script = [0, 0, 0]
    ids = m.generate([[0.0]], 3, feedback="client", do_sample=True, temperature=1e-3,
                     seed=1)
    assert ids == [0, 0, 0]                           # cold sampling reduces to argmax


def test_stop_ids_accepts_int_or_collection():
    assert _stop_ids(None) == frozenset()
    assert _stop_ids(7) == frozenset({7})
    assert _stop_ids([1, 2]) == frozenset({1, 2})


def test_prefill_chunk_schedule():
    assert EncGPT2._prefill_chunks(5, 32, False) == [(5, False)]
    assert EncGPT2._prefill_chunks(70, 32, False) == [(32, False), (32, False), (6, False)]
    # complex payload: chunks carry up to 2t while more than t remain, then a real tail
    assert EncGPT2._prefill_chunks(70, 32, True) == [(64, True), (6, False)]
    assert EncGPT2._prefill_chunks(100, 32, True) == [(64, True), (36, True)]


def test_configure_feeds_a_plain_bind():
    m = _Fake([1])
    m.configure(overlap="sync", cache_states=False)
    assert m._bind_kwargs == {"overlap": "sync", "cache_states": False}
    with pytest.raises(TypeError, match="unknown setting"):
        m.configure(plan_dirr="x")
