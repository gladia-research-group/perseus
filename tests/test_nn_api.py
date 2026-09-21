"""CPU tests for the torch-style module surface: no GPU, no context, milliseconds.

Covers EncModule's tree protocol (`__call__` forwarding, repr, named traversal,
child registration) and the construction-time validation that turns a wrong-shaped
weight into a ValueError instead of an out-of-bounds read in C++.
"""
import numpy as np
import pytest

pytest.importorskip("perseus._core")   # module classes import the extension
from perseus.nn import (
    EncGELU,
    EncLayerNorm,
    EncLinear,
    EncModule,
    EncSequential,
)


class _Toy(EncModule):
    def forward(self, a, b, scale=1):
        return (a + b) * scale


def test_call_forwards_positional_and_keyword_arguments():
    m = _Toy()
    assert m(1, 2) == 3 and m(1, 2, scale=10) == 30 and m(a=1, b=2) == 3


def test_base_forward_is_a_named_not_implemented():
    with pytest.raises(NotImplementedError, match="EncModule.forward"):
        EncModule()(1)


def test_repr_prints_the_tree():
    m = EncSequential(EncLinear("fc1", 1024, 4096), EncGELU("act"),
                      EncLinear("fc2", 4096, 1024, weight=np.zeros((4096, 1024))))
    r = repr(m)
    assert r.splitlines()[0] == "EncSequential("
    assert "(0): EncLinear('fc1', 1024->4096, shared)" in r
    assert "(1): EncGELU('act')" in r
    assert "(2): EncLinear('fc2', 4096->1024, bias=False)" in r
    assert repr(EncLayerNorm("ln", 768)) == "EncLayerNorm('ln', d=768)"


def test_named_modules_apply_and_children():
    m = EncSequential(EncLinear("fc1", 8, 8), EncGELU("act"))
    names = [n for n, _ in m.named_modules()]
    assert names == ["", "0", "1"]
    assert [n for n, _ in m.named_children()] == ["0", "1"]
    seen = []
    assert m.apply(lambda mod: seen.append(type(mod).__name__)) is m
    assert seen == ["EncSequential", "EncLinear", "EncGELU"]
    assert not m.bound and all(not c.bound for c in m.children())


def test_setattr_registers_and_unregisters_children():
    m = EncModule()
    m.child = EncGELU()
    assert "child" in dict(m.named_children())
    m.child = None                      # replaced by a non-module: leaves the tree
    assert "child" not in dict(m.named_children())
    m.other = EncGELU()
    del m.other
    assert list(m.children()) == []


def test_subclass_without_super_init_gets_a_clear_error():
    class Bad(EncModule):
        def __init__(self):
            self.x = 1

    with pytest.raises(AttributeError, match="super\\(\\).__init__"):
        Bad()


def test_linear_rejects_the_torch_layout_by_name():
    with pytest.raises(ValueError, match="torch's \\(out_features, in_features\\)"):
        EncLinear("fc", 1024, 4096, weight=np.zeros((4096, 1024)))


def test_linear_rejects_wrong_rank_shape_nan_and_bias_length():
    with pytest.raises(ValueError, match="2-D"):
        EncLinear("fc", 8, 8, weight=np.zeros(8))
    with pytest.raises(ValueError, match="!= \\(d_in, d_out\\)"):
        EncLinear("fc", 8, 8, weight=np.zeros((8, 4)))
    bad = np.zeros((8, 8)); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        EncLinear("fc", 8, 8, weight=bad)
    with pytest.raises(ValueError, match="bias must have shape"):
        EncLinear("fc", 8, 8, weight=np.zeros((8, 8)), bias=np.zeros(4))
    m = EncLinear("fc", 8, 8, weight=np.ones((8, 8)).tolist(), bias=range(8))
    assert m.weight.dtype == np.float64 and m.bias.shape == (8,)


def test_linear_from_torch_transposes_and_pads():
    torch = pytest.importorskip("torch")
    lin = torch.nn.Linear(6, 5)
    m = EncLinear.from_torch("fc", lin, d_in=8, d_out=8)
    assert m.weight.shape == (8, 8) and m.bias.shape == (8,)
    np.testing.assert_allclose(m.weight[:6, :5], lin.weight.detach().numpy().T, atol=1e-6)
    assert (m.weight[6:, :] == 0).all() and (m.weight[:, 5:] == 0).all()
    np.testing.assert_allclose(m.bias[:5], lin.bias.detach().numpy(), atol=1e-6)
    x = torch.randn(3, 8); x[:, 6:] = 0
    ref = torch.nn.functional.pad(lin(x[:, :6]), (0, 3))
    np.testing.assert_allclose(m.torch_mirror()(x).detach().numpy(), ref.detach().numpy(),
                               atol=1e-5)
    with pytest.raises(ValueError, match="larger than the packed"):
        EncLinear.from_torch("fc", lin, d_in=4, d_out=4)


def test_layernorm_validates_the_affine():
    with pytest.raises(ValueError, match="come together"):
        EncLayerNorm("ln", 4, weight=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="same length"):
        EncLayerNorm("ln", weight=[1, 1, 1, 1], bias=[0, 0])
    with pytest.raises(ValueError, match="entries but d=8"):
        EncLayerNorm("ln", 8, weight=[1, 1, 1, 1], bias=[0, 0, 0, 0])
    m = EncLayerNorm("ln", weight=[1, 1, 1, 1], bias=[0, 0, 0, 0])
    assert m.d == 4 and m.weight.dtype == np.float64


def test_unbind_detaches_the_tree():
    m = EncSequential(EncGELU("a"), EncGELU("b"))
    m.bind(object())
    assert m.bound and all(c.bound for c in m.children())
    assert m.unbind() is m
    assert not m.bound and not any(c.bound for c in m.children())
