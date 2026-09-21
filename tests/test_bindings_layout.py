"""Source-layout invariants of the pybind11 layer (src/bindings) and its runtime boundary.

Text-level checks only (the extension is not rebuilt here), so they run in the CPU tier
and catch what a bindings-file split leaves behind: a ``bind_*`` registered in module.cu
but defined nowhere (or twice), a translation unit missing from the ``_core`` source list
in CMakeLists.txt, runtime code (cachemir_lib) that includes pybind11, the documented
registration order in module.cu drifting from the calls it describes, or the block-artifact
format sliding back into the binding layer.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / "src" / "bindings"
CMAKE = (ROOT / "CMakeLists.txt").read_text()
MODULE = (BINDINGS / "module.cu").read_text()

_PYBIND_INCLUDE = re.compile(r'^\s*#include\s*[<"]pybind11/', re.M)
_BIND_DECL = re.compile(r"^void (bind_\w+)\(py::module_& m\);", re.M)
_BIND_DEF = re.compile(r"^void (bind_\w+)\(py::module_& m\) \{", re.M)
_BIND_CALL = re.compile(r"^\s+(bind_\w+)\(m\);", re.M)


def _cmake_sources(target_open: str) -> set[str]:
    """The source paths listed inside ``target_open( ... )`` (first matching block)."""
    start = CMAKE.index(target_open)
    body = CMAKE[start + len(target_open):CMAKE.index(")", start)]
    return {tok for tok in body.split() if tok.endswith((".cu", ".cpp", ".cc"))}


def _bindings_tus() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(BINDINGS.glob("*.cu"))}


def test_every_bindings_tu_is_a_core_source():
    listed = {Path(s).name for s in _cmake_sources("pybind11_add_module(_core")}
    on_disk = set(_bindings_tus())
    assert on_disk == listed, (
        f"src/bindings/*.cu and the _core source list in CMakeLists.txt disagree: "
        f"not listed {sorted(on_disk - listed)}, listed but missing {sorted(listed - on_disk)}")


def test_every_bind_function_is_declared_defined_and_called_exactly_once():
    declared = _BIND_DECL.findall(MODULE)
    called = _BIND_CALL.findall(MODULE)
    defined: dict[str, list[str]] = {}
    for name, text in _bindings_tus().items():
        for fn in _BIND_DEF.findall(text):
            defined.setdefault(fn, []).append(name)
    assert declared, "module.cu declares no bind_* functions"
    assert sorted(declared) == sorted(called), (declared, called)
    assert len(called) == len(set(called)), f"a bind_* is called twice: {called}"
    for fn in declared:
        assert defined.get(fn, []) and len(defined[fn]) == 1, (
            f"{fn} is declared in module.cu but defined in {defined.get(fn, [])}")
    orphans = set(defined) - set(declared)
    assert not orphans, f"bind_* defined but never registered by module.cu: {sorted(orphans)}"


def test_registration_list_in_module_cu_matches_the_call_order():
    # The forward declarations carry the documented step numbers (`// 4`); the calls in
    # PYBIND11_MODULE must run in exactly that order, else the comment lies.
    steps = re.findall(r"^void (bind_\w+)\(py::module_& m\);\s*// (\d+)", MODULE, re.M)
    assert steps, "module.cu's bind_* declarations no longer carry their step numbers"
    by_step = [fn for fn, _ in sorted(steps, key=lambda s: int(s[1]))]
    numbers = [int(n) for _, n in steps]
    assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers), numbers
    assert _BIND_CALL.findall(MODULE) == by_step
    for fn, n in steps:   # every step is spelled out in the list at the top of the file
        assert re.search(rf"^//\s+{n} {fn}\b", MODULE, re.M), f"step {n} ({fn}) missing from the list"


def test_runtime_library_never_includes_pybind11():
    offenders = []
    for path in list((ROOT / "include").rglob("*")) + list((ROOT / "src").rglob("*")):
        if path.suffix not in (".h", ".cuh", ".cu", ".cpp", ".cc") or BINDINGS in path.parents:
            continue
        if _PYBIND_INCLUDE.search(path.read_text(errors="replace")):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"cachemir_lib code includes pybind11: {offenders}"


def test_block_artifact_format_lives_in_the_runtime():
    header = (ROOT / "include" / "block_artifact.h").read_text()
    runtime = {"encode_block_state_coeff", "save_block_state", "read_block_artifact",
               "block_state_diff"}
    for fn in runtime:
        assert re.search(rf"\b{fn}\(", header), f"{fn} not declared in include/block_artifact.h"
    assert "src/block_artifact.cu" in _cmake_sources("add_library(cachemir_lib")
    for name, text in _bindings_tus().items():
        for fn in runtime:
            assert not re.search(rf"^\S.*\b{fn}\([^;]*\)\s*\{{", text, re.M), (
                f"{name} defines {fn}: the artifact format belongs to cachemir_lib")
    serial = (BINDINGS / "serial.cu").read_text()
    assert '#include "block_artifact.h"' in serial
    assert "block_artifact::read_block_artifact(" in serial   # the KeyError tail stays a wrapper
