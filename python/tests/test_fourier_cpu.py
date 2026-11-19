"""impl/fourier.py on the plaintext mirror (NumpyOps) and on the CPU twin (the strict fake: the Shadow's levels, the
rotation-key band) against numpy / scipy in float64: DFT and inverse (natural and bit-reversed order, direct and
planned), the real-input FFTs, DCT-II / DCT-III (real and pair packings), 2-D, both layouts."""
import numpy as np
import pytest
import scipy.fft as sf

from perseus.impl import fourier as F
from perseus.impl.ops import FheOps, NumpyOps

S = 1 << 12
TOL = 1e-12


def _run(T, x, ops=None):
    return T.unpack(T.apply(ops or NumpyOps(), T.pack(x)))


def _close(got, want):
    np.testing.assert_allclose(got, want, atol=TOL * max(1.0, np.abs(want).max()))


@pytest.fixture
def rng():
    return np.random.default_rng(5)


def _cplx(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("n", [8, 64, 256])
@pytest.mark.parametrize("mode", ["block", "stride"])
@pytest.mark.parametrize("levels", [None, 1, 2, 3])
def test_dft_natural(rng, n, mode, levels):
    B = S // n if mode == "stride" else 3
    lay = F.Layout(n, S, B, mode)
    x = _cplx(rng, (B, n))
    T = F.dft(n, levels, layout=lay)
    assert levels is None or T.levels <= levels
    _close(_run(T, x), np.fft.fft(x))
    _close(_run(F.idft(n, levels, layout=lay), x), np.fft.ifft(x))


@pytest.mark.parametrize("n", [16, 128])
def test_dft_direct_and_exact_levels(rng, n):
    lay = F.Layout(n, S, S // n, "stride")
    x = _cplx(rng, (S // n, n))
    T = F.dft(n, layout=lay, direct=True, norm="ortho")
    assert T.levels == 1 and T.steps[0].n_pt == n          # cyclic layout: n diagonals
    _close(_run(T, x), np.fft.fft(x, norm="ortho"))
    T = F.dft(n, 4, layout=lay, exact=True)
    assert T.levels == 4
    _close(_run(T, x), np.fft.fft(x))


@pytest.mark.parametrize("n", [64, 512])
def test_dft_bitrev_pair(rng, n):
    """Forward with bit-reversed output, inverse with bit-reversed input: a convolution needs no permutation."""
    lay = F.Layout(n, S, 2)
    x, h = _cplx(rng, (2, n)), _cplx(rng, (2, n))
    Fw = F.dft(n, layout=lay, order="bitrev")
    X = _run(Fw, x)
    _close(X[:, np.argsort(F.bit_reverse(n))], np.fft.fft(x))
    Hb = np.fft.fft(h)[:, F.bit_reverse(n)]
    y = _run(F.idft(n, layout=lay, order="bitrev"), X * Hb)
    _close(y, np.fft.ifft(np.fft.fft(x) * np.fft.fft(h)))


def test_bit_swaps_are_cheap():
    n = 1024
    lay = F.Layout(n, S, 4)
    assert all(len(lay.offsets(f.A)) == 3 for f in F.bit_reverse_factors(n))
    P = F._compose(F.bit_reverse_factors(n)).A
    assert abs(P - F._perm(F.bit_reverse(n))).max() == 0
    assert len(lay.offsets(F._perm(F.bit_reverse(n)))) > 200     # the one-shot permutation


@pytest.mark.parametrize("n", [8, 64, 256])
@pytest.mark.parametrize("norm", ["backward", "ortho"])
def test_rfft(rng, n, norm):
    B = 4
    lay = F.Layout(n, S, B)
    x, y = rng.standard_normal((B, n)), rng.standard_normal((B, n))
    h = n // 2
    T = F.rfft_pair(n, layout=lay, norm=norm)
    assert T.needs_conj()
    Y = _run(T, F.pack_real_pair(x, y))
    _close(F.unpack_rfft(Y[:, :h], n), np.fft.rfft(x, norm=norm))
    _close(F.unpack_rfft(Y[:, h:], n), np.fft.rfft(y, norm=norm))
    _close(_run(F.rfft_pair(n, layout=lay, norm=norm, inverse=True), Y), F.pack_real_pair(x, y))
    lh = F.Layout(h, S, B)
    T = F.rfft_half(n, layout=lh, norm=norm)
    Z = _run(T, F.pack_real_half(x))
    _close(F.unpack_rfft(Z, n), np.fft.rfft(x, norm=norm))
    _close(F.pack_rfft(np.fft.rfft(x, norm=norm), n), Z)
    _close(_run(F.rfft_half(n, layout=lh, norm=norm, inverse=True), Z), F.pack_real_half(x))


@pytest.mark.parametrize("n", [8, 64, 256])
@pytest.mark.parametrize("norm", ["ortho", "backward"])
@pytest.mark.parametrize("packing", ["real", "pair"])
def test_dct(rng, n, norm, packing):
    B = S // n
    lay = F.Layout(n, S, B, "stride")
    x, y = rng.standard_normal((B, n)), rng.standard_normal((B, n))
    z = x + 0j if packing == "real" else F.pack_real_pair(x, y)
    for typ, mk in ((2, F.dct2), (3, F.dct3)):
        got = _run(mk(n, norm=norm, layout=lay, packing=packing), z)
        _close(got.real, sf.dct(x, typ, norm=norm))
        if packing == "pair":
            _close(got.imag, sf.dct(y, typ, norm=norm))
        else:
            _close(got.imag, np.zeros_like(x))
    T = F.dct2(n, layout=lay, direct=True, packing=packing)
    assert T.levels == 1
    _close(_run(T, z).real, sf.dct(x, 2, norm="ortho"))


def test_makhoul_factors_are_cheap():
    n = 512
    lay = F.Layout(n, S, 8)
    assert all(len(lay.offsets(f.A)) == 3 for f in F.makhoul_factors(n))


@pytest.mark.parametrize("r", [8, 16])
@pytest.mark.parametrize("kw", [dict(), dict(separable=True), dict(direct=True), dict(levels=3)])
def test_two_d(rng, r, kw):
    n = r * r
    B = S // n
    lay = F.Layout(n, S, B, "stride")
    a, b = rng.standard_normal((B, r, r)), rng.standard_normal((B, r, r))
    z = F.pack_real_pair(a.reshape(B, n), b.reshape(B, n))
    got = _run(F.dct2_2d(r, layout=lay, **kw), z).reshape(B, r, r)
    _close(got.real, sf.dctn(a, 2, axes=(1, 2), norm="ortho"))
    _close(got.imag, sf.dctn(b, 2, axes=(1, 2), norm="ortho"))
    back = _run(F.dct3_2d(r, layout=lay, **kw), got.reshape(B, n))
    _close(back, z)
    got = _run(F.dct2_2d(r, layout=lay, packing="real", **kw), a.reshape(B, n) + 0j).reshape(B, r, r)
    _close(got.real, sf.dctn(a, 2, axes=(1, 2), norm="ortho"))
    c = _cplx(rng, (B, r, r))
    got = _run(F.dft_2d(r, layout=lay, **kw), c.reshape(B, n)).reshape(B, r, r)
    _close(got, np.fft.fft2(c))


def test_separable_has_two_levels():
    T = F.dct2_2d(32, layout=F.Layout(1024, S, 4), separable=True)
    assert T.levels == 2


@pytest.mark.parametrize("make", [
    lambda lay: F.dft(64, 3, layout=lay, exact=True),
    lambda lay: F.rfft_pair(64, layout=lay),
    lambda lay: F.dct2(64, layout=lay),
    lambda lay: F.dct2(64, 2, layout=lay, packing="pair", exact=True),
])
def test_twin_levels_and_keys(rng, make):
    """On the strict fake: exactly one level per step, every rotation inside the transform's own key set (a missing
    key raises), the conjugation only where the transform says so, and the float64 values."""
    from perseus.impl.fake import FakeCt, core, make_fake
    lay = F.Layout(64, S, S // 64, "stride")
    T = make(lay)
    steps = T.rot_steps() | {s % S for s in T.rot_steps()}
    fhe, inf = make_fake(N=S, rot_steps=steps, complex_payload=True, limit=10 ** 6, strict=True)
    ops = FheOps(inf, core)
    x = _cplx(rng, (S // 64, 64))
    y = T.apply(ops, FakeCt(T.pack(x), 0))
    assert ops.lvl(y) == T.levels * ops.unit
    assert ("conjugate" in fhe.calls) == T.needs_conj()
    _close(T.unpack(y.vec), T.reference(x))
    bad = sorted(T.rot_steps())[0]
    fhe2, inf2 = make_fake(N=S, rot_steps=steps - {bad, bad % S}, complex_payload=True, limit=10 ** 6, strict=True)
    with pytest.raises(KeyError):
        T.apply(FheOps(inf2, core), FakeCt(T.pack(x), 0))


def test_rlinear_compose():
    """Factor.then on R-linear factors matches applying them one after the other."""
    rng = np.random.default_rng(0)
    n = 16
    def rnd():
        import scipy.sparse as sp
        return F.Factor(sp.csr_matrix(_cplx(rng, (n, n))), sp.csr_matrix(_cplx(rng, (n, n))))
    f, g = rnd(), rnd()
    z = _cplx(rng, (3, n))
    _close(f.then(g).apply(z), g.apply(f.apply(z)))
