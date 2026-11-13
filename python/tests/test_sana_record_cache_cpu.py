"""The SANA planned prefill's record cache: save_record / load_record round-trip a record exactly; another key, or
another stage count, loads nothing."""
from types import SimpleNamespace

import numpy as np

from perseus.examples.sana.planned import SanaPlannedPrefill as P, record_key
from perseus.impl.driver import BlockRec
from perseus.impl.pcmm import PcmmLinear


def _run(rng, n_stages=3):
    """A planned prefill's record fields, made up."""
    pts = {f"v/{i}": (rng.standard_normal(64) + 1j * rng.standard_normal(64)) if i % 2 else rng.standard_normal(64)
           for i in range(5)}
    recs = {}
    for b in range(n_stages):
        r = BlockRec([{"op_type": "mult", "inputs": [f"x{b}", "pt_0"], "output": f"y{b}"}], {"pt_0": f"v/{b}"}, {},
                     {f"x{b}": (40, 1)})
        r.out_names = [f"y{b}"]
        recs[b] = r
    lin = PcmmLinear(1, 2, rng.integers(-128, 128, (2, 32, 16), dtype=np.int8), 2.0 ** 10, 1,
                     [rng.standard_normal(8), rng.standard_normal(8)], "b0.qkv", 4)
    lin.m_orig, lin.n_out_orig = [0, 2], 3
    return SimpleNamespace(n_stages=n_stages, pt_vals=pts, recs=recs, in_names={0: ["x0"]}, in_metas={0: [(40, 1)]},
                           in_vals=[rng.standard_normal(64)], record_s=1.5,
                           model=SimpleNamespace(lin=SimpleNamespace(_enc={"b0.qkv": lin})))


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    a = _run(rng)
    d = str(tmp_path / "rec")
    P.save_record(a, d, "k1")
    b = _run(np.random.default_rng(1))                  # other values everywhere: the load must replace them all
    assert P.load_record(b, d, "k1")
    assert set(b.pt_vals) == set(a.pt_vals)
    for k, v in a.pt_vals.items():
        assert b.pt_vals[k].dtype == v.dtype and np.array_equal(b.pt_vals[k], v)
    for i, r in a.recs.items():
        q = b.recs[i]
        assert (q.nodes, q.pts, q.metas, q.out_names) == (r.nodes, r.pts, r.metas, r.out_names)
    la, lb = a.model.lin._enc["b0.qkv"], b.model.lin._enc["b0.qkv"]
    assert np.array_equal(la.Wd, lb.Wd) and lb.Wd.dtype == np.int8 and lb.m_orig == [0, 2] and lb.w_scale == la.w_scale
    assert all(np.array_equal(x, y) for x, y in zip(la.bias, lb.bias))
    assert (b.in_names, b.in_metas) == (a.in_names, a.in_metas) and np.array_equal(b.in_vals[0], a.in_vals[0])
    assert b.recorded_s == 1.5
    P.save_record(a, d, "k2")                            # a second save replaces the first
    assert not P.load_record(_run(rng), d, "k1") and P.load_record(_run(rng), d, "k2")


def test_mismatch_loads_nothing(tmp_path):
    rng = np.random.default_rng(0)
    d = str(tmp_path / "rec")
    P.save_record(_run(rng), d, "k1")
    c = _run(rng, n_stages=4)
    keep = c.pt_vals
    assert not P.load_record(c, d, "k1") and c.pt_vals is keep
    assert not P.load_record(_run(rng), str(tmp_path / "none"), "k1")


def test_key(tmp_path, monkeypatch):
    f = tmp_path / "configs.json"
    f.write_text("{}")
    k = record_key(87, [str(f)])
    assert k == record_key(87, [str(f)]) and k != record_key(86, [str(f)])
    monkeypatch.setenv("PERSEUS_RECORD_WORKERS", "3")    # how a record runs, not what it records
    assert record_key(87, [str(f)]) == k
    monkeypatch.setenv("SANA_LA_QNORM", "key-test")
    assert record_key(87, [str(f)]) != k
    monkeypatch.delenv("SANA_LA_QNORM")             # (its value before the test comes back after it)
    f.write_text('{"x": 1}')
    assert record_key(87, [str(f)]) != k
