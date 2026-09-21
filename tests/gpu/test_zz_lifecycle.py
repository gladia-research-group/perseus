"""Session.close() releases what the session holds. Sorted last: it ends
the module's session for good.

The rotation keys (~14 GB on the n32 GPT-2 band) and the weights go back to the
runtime's device pool, not to the driver: FIDESlib keeps freed limbs in per-size free
lists and pins the CUDA pool's release threshold, so `cudaMemGetInfo` is flat across a
close and the next session in this process reuses the memory. The test therefore checks
what close() guarantees — every key and weight released, nothing grows — not a driver
counter.
"""
import pytest

pytestmark = pytest.mark.gpu


def test_close_releases_keys_and_weights(sess):
    from perseus import _core
    if not hasattr(_core, "close_session"):
        pytest.skip("extension predates close_session")
    inf = sess.inf
    steps = list(inf.fhe.loaded_rot_steps)
    assert steps, "the GPT-2 session should hold rotation keys"
    before = _core.device_free_gb()
    freed = _core.close_session(inf)
    # the DFT automorphism keys shared with the bootstrap precomputation are protected
    # (they belong to the context), so freed < len(steps): 87 of 133 on the n32 band
    assert 0 < freed <= len(steps), (freed, len(steps))
    assert inf.fhe.loaded_rot_steps == []
    assert inf.installed_weights == []
    assert _core.close_session(inf) == 0            # nothing left to free
    sess.close()
    assert sess.closed
    assert _core.device_free_gb() >= before - 0.25, "close() must not grow device usage"
    sess.close()                                   # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        _ = sess.inf
    del inf
