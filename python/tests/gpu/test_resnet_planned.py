"""ResNet-20 / CIFAR-10 planned on the new runtime (`examples.resnet.impl.ResnetPlanned`): the convs as plaintext
diagonals (impl/conv.py), the composite-sign ReLU, every stage a compiled `Program`; 8 test images vs the float64
model with the same polynomial ReLU.

Gate (the full one is `resnet.run --plan bootstrap_placements/resnet20_cifar10 --n 64`): argmax agreement 8/8, logits
within 0.05. ~2 min with the setup.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

PLAN = "bootstrap_placements/resnet20_cifar10"


@pytest.mark.skipif(not os.environ.get("HF_HOME"), reason="HF_HOME (CIFAR-10, the checkpoint) missing")
def test_resnet_planned_cifar8(tmp_path):
    from perseus.examples.resnet import run
    got = {}
    rc = run.main(["--plan", PLAN, "--n", "8", "--work-dir", str(tmp_path / "w")], results=got.update)
    assert rc == 0
    assert got["agree"] == got["n"] == 8, got
    assert got["max_dlogit"] < 0.05, got


STCF = ["--one-stage", "--twin-limit", "400", "--set", "FIDESLIB_BTS_STC_FIRST=2", "--set", "FIDESLIB_BTS_STC_FIRST_ROUTES=dense",
        "--set", "FIDESLIB_BTS_STC_FIRST_CF=13", "--set", "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY=0"]


@pytest.mark.skipif(not os.environ.get("HF_HOME"), reason="HF_HOME (CIFAR-10, the checkpoint) missing")
def test_resnet_stc_first_planned_cifar8(tmp_path):
    """The one-stage StC-first plan (CFs from the StC-first accuracy curves): 8/8, logits within 0.01."""
    from perseus.examples.resnet import run
    got = {}
    rc = run.main(["--plan", "bootstrap_placements/resnet20_stcf2", *STCF, "--n", "8", "--work-dir", str(tmp_path / "w")],
                  results=got.update)
    assert rc == 0
    assert got["agree"] == got["n"] == 8, got
    assert got["max_dlogit"] < 0.01, got


@pytest.mark.skipif(not os.environ.get("HF_HOME"), reason="HF_HOME (CIFAR-10, the checkpoint) missing")
def test_resnet_tfhe_argmax_planned_cifar8(tmp_path):
    """The ReLU's sign and an argmax head by CKKS -> TFHE -> CKKS (impl/tfhe.py): 8/8 against the exact model, the
    encrypted one-hot right on all 8."""
    from perseus.examples.resnet import run
    got = {}
    rc = run.main(["--plan", "bootstrap_placements/resnet20_tfhe_argmax", "--tfhe", "--tfhe-argmax", "--n", "8",
                   "--work-dir", str(tmp_path / "w")], results=got.update)
    assert rc == 0
    assert got["agree"] == got["n"] == 8, got
    assert got["max_dlogit"] < 1.0, got
    assert got["onehot_agree"] == 8, got
