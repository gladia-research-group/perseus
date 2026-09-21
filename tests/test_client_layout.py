"""Source-layout invariants of the CUDA-free client extension (perseus._client), text only.

perseus._client re-declares the option structs, enums and the env parser of the CUDA
runtime (the originals live in CUDA-bound headers); these checks catch the two copies
drifting apart, a CUDA/FIDESlib include creeping into the client TUs, pybind11 leaking
into src/client, the CMake source list disagreeing with the files on disk, and the build
script losing its readelf guard. No extension needed.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENT_H = (ROOT / "include" / "client_context.h").read_text()
CLIENT_TUS = {p.relative_to(ROOT).as_posix(): p.read_text()
              for p in sorted((ROOT / "src" / "client").glob("*.cpp"))}
BINDING = (ROOT / "src" / "bindings" / "client.cpp").read_text()
WRAPPER = (ROOT / "include" / "fideslib_wrapper.h").read_text()
INFERENCE_H = (ROOT / "include" / "inference.h").read_text()
PACKED_CTX_H = (ROOT / "include" / "packing" / "packed_ctx.h").read_text()
CMAKE = (ROOT / "CMakeLists.txt").read_text()
BUILD_SH = (ROOT / "scripts" / "local_build_client.sh").read_text()

_FIELD = re.compile(r"^\s*([A-Za-z_][\w:<>]*)\s+(\w+)\s*=\s*([^;]+);", re.M)
_INCLUDE = re.compile(r'^\s*#include\s*[<"]([^>"]+)[>"]', re.M)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _struct_body(text: str, name: str) -> str:
    m = re.search(rf"\bstruct {name}\s*\{{(.*?)\n[ \t]*\}};", _strip_comments(text), re.S)
    assert m, f"struct {name} not found"
    return m.group(1)


def _enum_body(text: str, name: str) -> list[str]:
    m = re.search(rf"\benum class {name}\s*\{{(.*?)\}};", _strip_comments(text), re.S)
    assert m, f"enum class {name} not found"
    return [tok.strip() for tok in m.group(1).split(",") if tok.strip()]


def _fields(body: str) -> list[tuple[str, str, str]]:
    return [(t, n, re.sub(r"\s+", "", v)) for t, n, v in _FIELD.findall(body)]


def test_option_structs_are_verbatim_copies():
    for name, origin in (("CKKSContextOptions", WRAPPER), ("ModelSize", INFERENCE_H),
                         ("InferenceOptions", INFERENCE_H)):
        theirs = _fields(_struct_body(origin, name))
        ours = _fields(_struct_body(CLIENT_H, name))
        assert theirs, f"no `type name = default;` fields parsed from the original {name}"
        assert ours == theirs, f"{name} drifted between include/client_context.h and its original"
    # the string field without an initializer
    assert "std::string keys_dir;" in _struct_body(CLIENT_H, "CKKSContextOptions")
    assert "std::vector<PackingKind> aux_packing_kinds;" in _struct_body(CLIENT_H, "InferenceOptions")


def test_enums_are_verbatim_copies():
    assert _enum_body(CLIENT_H, "PackingKind") == _enum_body(PACKED_CTX_H, "PackingKind")
    assert _enum_body(CLIENT_H, "InferenceMode") == _enum_body(INFERENCE_H, "InferenceMode")
    for kind in _enum_body(PACKED_CTX_H, "PackingKind"):
        assert f'.value("{kind}", PackingKind::{kind})' in BINDING, kind


def _env_names(text: str) -> set[str]:
    start = text.index("ckks_options_from_env() {")
    body = text[start:text.index("return o;", start)]
    return set(re.findall(r'getenv\("([A-Z_]+)"\)', body))


def test_env_parser_reads_the_same_knobs():
    theirs, ours = _env_names(WRAPPER), _env_names(CLIENT_TUS["src/client/client_context.cpp"])
    assert theirs and ours == theirs, (sorted(ours ^ theirs))
    assert "std::cerr" not in _strip_comments(CLIENT_TUS["src/client/client_context.cpp"])


def test_client_sources_are_cuda_free():
    banned = re.compile(r"cuda|fideslib|\.cuh$|fideslib_wrapper\.h|^inference\.h$|^model/|^packing/",
                        re.I)
    for name, text in {"include/client_context.h": CLIENT_H, "src/bindings/client.cpp": BINDING,
                       **CLIENT_TUS}.items():
        for inc in _INCLUDE.findall(text):
            assert not banned.search(inc), f"{name} includes {inc}"
    for name, text in CLIENT_TUS.items():
        assert not re.search(r"#include\s*[<\"]pybind11/", text), f"{name} includes pybind11"
    assert 'getenv("CHAIN")' not in CLIENT_H


def test_cmake_client_target_lists_the_client_tus():
    start = CMAKE.index("pybind11_add_module(_client")
    block = CMAKE[start:CMAKE.index(")", start)]
    listed = {tok for tok in block.split() if tok.endswith(".cpp")}
    on_disk = {"src/bindings/client.cpp", *CLIENT_TUS}
    assert listed == on_disk, (sorted(listed ^ on_disk))
    client_block = CMAKE[CMAKE.index("if(CACHEMIR_BUILD_CLIENT)"):]
    client_block = client_block[:client_block.index("endif()")]
    assert "-Wl,--exclude-libs,ALL" in client_block
    assert "OPENFHEpke_static" in client_block and "fideslib" not in client_block
    assert "project(cuda_cachemir LANGUAGES CXX)" in CMAKE
    assert re.search(r"if\(NOT CACHEMIR_CLIENT_ONLY\)\s*enable_language\(CUDA\)", CMAKE)


def test_build_script_guards_the_module():
    assert "--target _client" in BUILD_SH
    assert "readelf -d" in BUILD_SH and "libcuda|libcudart|libnccl" in BUILD_SH
    assert "_client.${CHAIN}.so" in BUILD_SH


def _chain_defaults(text):
    """{chain: {name: value}} from a `namespace chain_defaults { #if NATIVEINT == 32 … #else … }`."""
    i = text.index("namespace chain_defaults")
    body = text[i:text.index("}  // namespace chain_defaults", i)]
    n32, n64 = body.split("#else")
    out = {}
    for chain, part in (("n32", n32), ("n64", n64)):
        out[chain] = dict(re.findall(r"inline constexpr \w+\s+(\w+)\s*=\s*([^;]+);", part))
    return out


def test_chain_defaults_hold_the_same_numbers_on_both_sides():
    """The two chains' parameters are written twice; the client's copy decides what a
    GPU-less keygen produces, so a drift would hand the server keys for another chain."""
    theirs = _chain_defaults(WRAPPER)
    ours = _chain_defaults(CLIENT_H)
    assert theirs["n32"] and theirs["n64"], "no chain_defaults parsed from the original"
    for chain in ("n32", "n64"):
        assert ours[chain] == theirs[chain], (
            f"{chain} chain_defaults drifted: include/client_context.h has {ours[chain]}, "
            f"the original has {theirs[chain]}")
