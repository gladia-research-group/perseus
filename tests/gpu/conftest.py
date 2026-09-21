"""The GPU tier: one real session per test module, opt-in (`pytest -m gpu tests/gpu`).

Every test here is marked `gpu`; the default `addopts` deselects the marker so the CPU
tier never touches a device. Set CUDA_VISIBLE_DEVICES to the GPU you are allowed to use
and CHAIN to the extension build that is linked (`readlink perseus/_core.cpython-*.so`).
"""
import os
import sys

import pytest

pytestmark = pytest.mark.gpu


def _have_cuda_extension():
    try:
        import perseus._core  # noqa: F401
    except ImportError:
        return False
    return os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in str(item.fspath) and not item.get_closest_marker("gpu"):
            item.add_marker(pytest.mark.gpu)
        if item.get_closest_marker("slow") and os.environ.get("PERSEUS_SLOW") != "1":
            item.add_marker(pytest.mark.skip(reason="slow: set PERSEUS_SLOW=1"))
        if item.get_closest_marker("gpu") and not _have_cuda_extension():
            item.add_marker(pytest.mark.skip(
                reason="needs perseus._core and CUDA_VISIBLE_DEVICES set to a usable GPU"))


def pytest_runtest_call(item):
    if item.get_closest_marker("gpu"):
        item.session.config._perseus_gpu_ran = True


def pytest_sessionfinish(session, exitstatus):
    session.config._perseus_exitstatus = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """The C++ runtime's cross-library static destruction can crash AFTER pytest has
    written its report (the documented teardown corruption). Once a GPU test has run, leave through the runtime's hard exit so the
    process status reflects the tests, not the teardown. This runs after the terminal
    reporter's summary (sessionfinish is wrapped by it). PERSEUS_GPU_HARD_EXIT=0 opts out."""
    if not getattr(config, "_perseus_gpu_ran", False):
        return
    if os.environ.get("PERSEUS_GPU_HARD_EXIT", "1") == "0":
        return
    try:
        import perseus._core as core
    except ImportError:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    core.hard_exit(int(getattr(config, "_perseus_exitstatus", 0)))


@pytest.fixture(scope="module")
def sess():
    """One session per module (keygen ~1 min on the n32 chain)."""
    from perseus import session
    from perseus.profile import SessionProfile
    prof = (SessionProfile.custom_n32() if os.environ.get("CHAIN", "n32") == "n32"
            else SessionProfile.custom_n64())
    with session(profile=prof) as s:
        yield s
