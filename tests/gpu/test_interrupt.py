"""Interruptibility: a Python signal unwinds a running C++ forward."""
import signal
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

D_PAD, D_REAL, D_EXP, E_REAL = 1024, 768, 4096, 3072


def _toy(sess):
    from perseus.nn import EncGELU, EncLinear, EncSequential
    rng = np.random.default_rng(0)

    def w(di, do, ir, orr, s=0.5):
        m = rng.standard_normal((di, do)) * s / np.sqrt(ir)
        m[ir:, :] = 0.0
        m[:, orr:] = 0.0
        return m

    return EncSequential(EncLinear("z_fc1", D_PAD, D_EXP, weight=w(D_PAD, D_EXP, D_REAL, E_REAL)),
                         EncGELU("z_act"),
                         EncLinear("z_fc2", D_EXP, D_PAD, weight=w(D_EXP, D_PAD, E_REAL, D_REAL)),
                         overlap="sync").bind(sess)


def test_ctrl_c_unwinds_a_running_forward(sess):
    from perseus import _core
    from perseus.nn import calibrate_sequential
    if not hasattr(_core, "close_session"):
        pytest.skip("extension predates close_session")
    model = _toy(sess)
    rng = np.random.default_rng(1)
    calibrate_sequential(model, rng.standard_normal((16, D_REAL)) * 0.3)
    x = sess.encrypt(rng.standard_normal(D_REAL) * 0.3)

    def alarm(signum, frame):
        raise KeyboardInterrupt("simulated Ctrl-C")

    old = signal.signal(signal.SIGALRM, alarm)
    t0 = time.time()
    try:
        signal.setitimer(signal.ITIMER_REAL, 1.0)   # fires inside the pipelined forward
        with pytest.raises(KeyboardInterrupt):
            model(x)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    assert time.time() - t0 < 12, "the interrupt was delivered only after the call returned"
    # the session is still usable afterwards
    y = sess.decrypt(model(x), d=D_REAL)
    assert np.isfinite(y).all()
