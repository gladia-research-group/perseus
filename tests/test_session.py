"""perseus.Session lifecycle, without a real context."""
import types

import numpy as np
import pytest

pytest.importorskip("perseus._core")
from perseus import Session
from perseus.nn import EncGELU, EncModule


class _FakeInf:
    def __init__(self):
        self.cleared = 0
        self.size = types.SimpleNamespace(dim=8)

    def clear_enc_cache(self):
        self.cleared += 1
        return 0


def test_close_is_idempotent_and_clears_the_cache():
    inf = _FakeInf()
    s = Session(inf, family="generic")
    assert not s.closed and s.inf is inf and "open" in repr(s)
    s.close()
    s.close()
    assert s.closed and inf.cleared == 1
    with pytest.raises(RuntimeError, match="closed"):
        _ = s.inf


def test_context_manager_closes_and_bind_accepts_a_session():
    inf = _FakeInf()
    with Session(inf, family="generic") as s:
        m = EncGELU("act").bind(s)
        assert m.inf is inf and m.bound
        root = EncModule()
        root.act = EncGELU()
        root.bind(s)
        assert root.act.inf is inf
    assert s.closed and inf.cleared == 1


def test_close_survives_a_runtime_that_is_already_gone():
    class Dead:
        def clear_enc_cache(self):
            raise RuntimeError("context gone")

    s = Session(Dead())
    s.close()
    assert s.closed


def test_encrypt_validates_before_touching_the_runtime():
    s = Session(_FakeInf(), family="generic")
    with pytest.raises(ValueError, match="1-D"):
        s.encrypt(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="do not fit the session's 8-wide"):
        s.encrypt(np.zeros(9))
    with pytest.raises(ValueError, match="NaN"):
        s.encrypt([1.0, float("nan")])
