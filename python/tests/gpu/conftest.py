"""Every GPU test imports perseus' Python modules afresh: many read their switches from the environment at import time
(impl/prefill.py, impl/pcmm.py, impl/activation.py, ...), so a model run after another in the same process would keep
the first one's. The extension `perseus._pc` stays: it cannot be initialised twice."""
import sys

import pytest


def _purge():
    for name in [m for m in sys.modules if m.startswith("perseus.") and not m.startswith("perseus._pc")]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _fresh_perseus_modules():
    _purge()
    yield
    _purge()
