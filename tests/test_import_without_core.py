"""The package must degrade gracefully when the CUDA extension is not built.

Runs a subprocess against a copy of ``perseus/`` with every ``_core*.so`` removed
(the editable-install finder is dropped so the real tree cannot leak in) and checks
that the pure-Python layers import, and that the extension-dependent surface fails
with the build recipe in the message rather than a bare ``cannot import name``.

Scenario 2 (skipped when perseus._client is not built): the same copy WITH the CUDA-free
client extension — a GPU-less client machine. The client role must import and run on
it, while the _core-only surface still fails with the recipe.
"""
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_PROBE = r"""
import importlib, sys
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__name__.lower()
                    and "editable" not in getattr(f, "__module__", "").lower()]
import perseus
assert perseus.__file__.startswith(sys.argv[1]), perseus.__file__
for m in ("perseus.errors", "perseus.profile", "perseus.plan", "perseus.plan.placer",
          "perseus.plan.contract", "perseus.calibrate"):
    importlib.import_module(m)
from perseus.errors import FHEError, PlanError, SecurityError, BundleError
assert issubclass(PlanError, FHEError)
from perseus.profile import SessionProfile
SessionProfile.gpt2_decode_n32().env()          # no extension needed for the env view
import perseus.nn                                # the module surface imports without any extension
assert "EncClient" in dir(perseus.nn)
try:
    perseus.nn.remote._default_ext()             # using it is what needs one, with the build recipe
except ImportError as e:
    assert "local_build_core.sh" in str(e), str(e)
else:
    raise SystemExit("an extension resolved without one being built")
try:
    SessionProfile().options()
except ImportError as e:
    assert "local_build_core.sh" in str(e), str(e)
else:
    raise SystemExit("options() worked without the extension")
print("OK")
"""

_PROBE_CLIENT_ONLY = r"""
import sys
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__name__.lower()
                    and "editable" not in getattr(f, "__module__", "").lower()]
import perseus
assert perseus.__file__.startswith(sys.argv[1]), perseus.__file__
try:
    from perseus import _core
except ImportError as e:
    assert "local_build_core.sh" in str(e), str(e)
else:
    raise SystemExit("perseus._core imported in the client-only copy")
import perseus._client as client
assert client.__file__.startswith(sys.argv[1]), client.__file__
import perseus.nn
from perseus.nn import EncClient, EncGenerationClient, remote, serve
assert remote._default_ext() is client and serve._default_ext() is client
import perseus.errors
assert perseus.errors.FHEError is client.FHEError
import perseus._env
assert perseus._env.current_chain() == client.chain, (perseus._env.current_chain(), client.chain)
from perseus.profile import SessionProfile
opts = SessionProfile().options()
assert isinstance(opts, client.InferenceOptions), type(opts)
assert opts.mode == client.InferenceMode.Threaded
try:
    getattr(perseus.nn, "EncLinear")
except ImportError as e:
    assert "local_build_core.sh" in str(e), str(e)
else:
    raise SystemExit("EncLinear resolved without perseus._core")
assert "EncLinear" in dir(perseus.nn) and "EncClient" in dir(perseus.nn)
print("OK")
"""


def _copy(tmp_path, ignore):
    shutil.copytree(REPO / "perseus", tmp_path / "perseus",
                    ignore=shutil.ignore_patterns(*ignore, "__pycache__"))


def _run(tmp_path, probe):
    return subprocess.run([sys.executable, "-c", probe, str(tmp_path)],
                          cwd=tmp_path, capture_output=True, text=True,
                          env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"})


def test_pure_python_layers_import_without_the_extension(tmp_path):
    _copy(tmp_path, ("*.so", "*.so.*"))
    r = _run(tmp_path, _PROBE)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stdout + r.stderr


def test_client_role_imports_with_only_the_client_extension(tmp_path):
    if not list((REPO / "perseus").glob("_client*.so")):
        pytest.skip("perseus._client is not built (scripts/local_build_client.sh)")
    _copy(tmp_path, ("_core*.so", "_core*.so.*"))
    r = _run(tmp_path, _PROBE_CLIENT_ONLY)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stdout + r.stderr


_PROBE_STALE_CORE = r"""
import os, sys
os.environ["CHAIN"] = "n99"
from perseus import _backend, _env
assert _backend.load("_core") is None, "a present-but-unloadable _core.so must read as absent"
import perseus.nn
assert "EncClient" in dir(perseus.nn)
if sys.argv[2] == "with-client":
    assert _env.current_chain() == _backend.load("_client").chain   # the stamp, not $CHAIN
    mod = _backend.default()
    assert mod.__name__ == "perseus._client", mod.__name__
    assert _backend.client_extension().__name__ == "perseus._client"
else:
    assert _env.current_chain() == "n99", _env.current_chain()      # env fallback
print("OK")
"""


def test_stale_core_extension_reads_as_absent(tmp_path):
    """A _core.so that cannot be loaded here (libcuda missing on a client box, or a stale
    build) must not break the package: CPython names that ImportError by the LEAF name."""
    have_client = bool(list((REPO / "perseus").glob("_client*.so")))
    _copy(tmp_path, ("_core*.so", "_core*.so.*") if have_client else ("*.so", "*.so.*"))
    suffix = sysconfig.get_config_var("EXT_SUFFIX")          # the name THIS interpreter would load
    (tmp_path / "perseus" / f"_core{suffix}").write_bytes(b"not an ELF\n")
    probe = _PROBE_STALE_CORE
    r = subprocess.run([sys.executable, "-c", probe, str(tmp_path),
                        "with-client" if have_client else "no-client"],
                       cwd=tmp_path, capture_output=True, text=True,
                       env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stdout + r.stderr
