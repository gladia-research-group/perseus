"""perseus/impl/magtape.py on the CPU: the coefficient statistic inverts the canonical embedding, the tape records the
fake's leaf ops with their step paths, and fill_graph merges a tape into a graph (refresh nodes inherit their
input's statistics) and refuses a tape that does not describe the graph."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.impl import fake, layout, magtape  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402


def test_coeffs_invert_the_embedding():
    n = 16
    z = np.random.default_rng(3).standard_normal(n) + 1j * np.random.default_rng(4).standard_normal(n)
    m, tw = magtape._coeff_perm(n)
    y = np.empty(n, complex); y[m] = z
    u = np.fft.fft(y) * tw / n
    coeffs = np.concatenate([u.real, u.imag])            # sum_k Re u_k X^k + Im u_k X^(k+n)
    zeta = np.exp(2j * np.pi / (4 * n))
    for j in range(n):
        assert abs(np.polyval(coeffs[::-1], zeta ** pow(5, j, 4 * n)) - z[j]) < 1e-12
    st = magtape.node_stats(z)
    assert np.isclose(st["output_max_coeff"], np.abs(coeffs).max())
    assert np.isclose(st["output_max_abs"], np.abs(z).max())


def _taped():
    fhe, inf = fake.make_fake(complex_payload=True)
    tape = magtape.attach(fhe, inf)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    return rt, tape


def test_tape_records_ops_and_steps():
    rt, tape = _taped()
    ops = rt.ops
    x = fake.FakeCt(np.arange(rt.dims.N, dtype=float), 34)
    tape.active = True
    with rt.step("blk0"):
        with rt.step("ln_1"):
            y = ops.mult(ops.add(x, 1.0), x)
            ops.inplace_add(y, ops.rotate(y, 1))
    assert [(op, st) for op, st, _ in tape.records] == [
        ("add", "blk0.ln_1"), ("mult", "blk0.ln_1"), ("rotate", "blk0.ln_1"), ("add_inplace", "blk0.ln_1")]
    assert tape.records[-1][2]["output_max_abs"] == pytest.approx(np.abs(y.vec).max())


def test_fill_graph(tmp_path):
    st = lambda v: magtape.node_stats(np.full(8, v))
    recs = [("add", "s", st(1.0)), ("mult", "s", st(2.0))]
    nodes = [{"op_type": "add", "inputs": ["a"], "output": "v_1", "step": "s"},
             {"op_type": "auto_bootstrap", "inputs": ["v_1"], "output": "v_1_auto_bootstrapped", "step": "s"},
             {"op_type": "mult", "inputs": ["v_1_auto_bootstrapped"], "output": "v_2", "step": "s"}]
    g = tmp_path / "graph.json"
    g.write_text(json.dumps({"version": 1, "nodes": nodes}))
    assert magtape.fill_graph(str(g), recs) == 2
    out = json.loads(g.read_text())["nodes"]
    assert out[1]["output_max_abs"] == pytest.approx(1.0)       # the refresh inherits its input's
    assert out[2]["output_max_abs"] == pytest.approx(2.0)
    g.write_text(json.dumps({"version": 1, "nodes": nodes}))
    with pytest.raises(ValueError):
        magtape.fill_graph(str(g), list(reversed(recs)))
