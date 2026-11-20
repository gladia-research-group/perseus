"""impl.math / impl.stats / impl.learn with functional bootstrapping (impl/fbs_fit.py): the FBS candidates and the
consumers that take them only when asked (today's series by default)."""
from __future__ import annotations

import numpy as np
import pytest

_pc = pytest.importorskip("perseus._pc")


def test_math_fbs_candidates():
    from perseus.impl import math as M
    cs = M.fbs_candidates("sigmoid", -8.0, 8.0, 1e-3)
    forms = {dict(a.params)["form"] for a in cs}
    assert forms == {"sine", "sine-paired", "trig", "trig-low"}
    for a in cs:
        assert a.method.startswith("fbs-") and a.err <= 1e-3
    # the plain choice is untouched (no FBS unless asked)
    assert M.approx("sigmoid", -8.0, 8.0, 1e-3).method == "cheb"
    a, uses = M.refreshed("sigmoid", -8.0, 8.0, 1e-3, fbs=False)
    assert not uses and a.method == "cheb"


def test_consumers_choose_fbs_only_when_asked():
    """impl.learn's link (an FBS by default for the logistic loss) and impl.stats' refreshed reciprocal (today's
    series by default): with `fbs` the functional bootstrap that lands no deeper than the series after a refresh
    (stats: also no slower, unless "force")."""
    from perseus.impl import learn as LN
    from perseus.impl import math as M
    from perseus.impl import stats as S
    from perseus.impl.ops import NumpyOps
    assert LN.Trainer(fbs=False).link().method == "cheb"
    assert LN.Trainer(loss="svm").link().method == "cheb"           # the hinge keeps the series by default
    assert LN.Trainer().link().method.startswith("fbs-")            # the logistic link is an FBS by default
    a = LN.Trainer(fbs=True).link()
    assert a.method.startswith("fbs-") and a.depth <= M.approx("sigmoid", -8.0, 8.0, 1e-3).depth
    rng = np.random.default_rng(0)
    lay = S.Batch(16, 4)
    X = rng.uniform(-4, 4, (4, 16))
    ref = np.exp(X - X.max(1, keepdims=True))
    ref /= ref.sum(1, keepdims=True)
    for fbs_ in (False, "force"):
        y = lay.decode(S.softmax(NumpyOps(), lay.encode(X), lay, -4.0, 4.0, 1e-3, shift="max", method="ckks",
                                 fbs=fbs_))
        assert np.abs(y - ref).max() < 1e-3
