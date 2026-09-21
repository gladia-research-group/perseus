"""The extension's own surface: version/chain stamp, typed-error routing, boundary checks.

Needs an extension that exposes `_core.build_info`; older builds skip. No context is created: everything here fails or answers before keygen.
"""
import pytest

_core = pytest.importorskip("perseus._core")
if not hasattr(_core, "build_info"):
    pytest.skip("perseus._core predates the version/chain stamp",
                allow_module_level=True)

from perseus._env import current_chain
from perseus.errors import FHEError, LayoutError, MaskError, PlanError


def test_stamp_is_consistent():
    info = _core.build_info()
    assert info["chain"] == _core.chain in ("n32", "n64")
    assert info["native_int_bits"] == _core.native_int_bits in (32, 64)
    assert _core.__version__ == info["version"]
    assert current_chain() == _core.chain          # the stamp wins over CHAIN in the env


@pytest.mark.parametrize("marker, exc", [
    ("[plan_level_error] v_1 expected 34 actual 36", PlanError),
    ("[plan] refused", PlanError),
    ("[mask_gen] no mask for step", MaskError),
    ("[mask_miss] strict", MaskError),
    ("[layout_error] basis mismatch", LayoutError),
    ("something else entirely", FHEError),
    ("", FHEError),
])
def test_marker_routing(marker, exc):
    with pytest.raises(exc):
        _core.throw_test(marker)
    if exc is not FHEError:
        with pytest.raises(FHEError):          # every typed error is still an FHEError
            _core.throw_test(marker)


@pytest.mark.parametrize("kind, exc", [
    ("plan", PlanError), ("mask", MaskError), ("layout", LayoutError), ("other", FHEError),
])
def test_typed_cpp_errors_map_by_type_not_by_message(kind, exc):
    dbg = getattr(_core, "_debug", None)
    if dbg is None or not hasattr(dbg, "throw_typed"):
        pytest.skip("extension predates the typed errors")
    with pytest.raises(exc):
        dbg.throw_typed(kind, "no marker in this message")
    with pytest.raises(RuntimeError):              # FHEError is a RuntimeError
        dbg.throw_typed(kind, "still a RuntimeError")


def test_openfhe_exceptions_are_fhe_errors():
    dbg = getattr(_core, "_debug", None)
    if dbg is None or not hasattr(dbg, "throw_typed"):
        pytest.skip("extension predates the typed errors")
    with pytest.raises(FHEError, match=r"\[openfhe\]"):
        dbg.throw_typed("openfhe", "from OpenFHE")


def test_the_version_is_one_number():
    """perseus.__version__ and the two C++ stamps must agree: they are separate literals."""
    import re
    from pathlib import Path

    import perseus
    repo = Path(__file__).resolve().parents[1]
    pat = re.compile(r'"(\d+\.\d+\.\d+)"')
    stamps = {}
    for rel in ("include/build_stamp.h", "include/client_context.h"):
        text = (repo / rel).read_text()
        m = re.search(r'VERSION\w*\s*(?:=|\[\]\s*=)?\s*' + pat.pattern, text)
        if m:
            stamps[rel] = m.group(1)
    assert stamps, "no version literal found in the C++ headers"
    for rel, v in stamps.items():
        assert v == perseus.__version__, f"{rel} says {v}, perseus.__version__ is {perseus.__version__}"
