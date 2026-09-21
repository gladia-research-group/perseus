#!/usr/bin/env python3
"""Regenerate the type stub of a compiled extension (perseus/_core.pyi by default).

    .venv/bin/python scripts/utils/gen_core_stub.py                   # -> perseus/_core.pyi
    .venv/bin/python scripts/utils/gen_core_stub.py perseus._client   # -> perseus/_client.pyi

Runs pybind11-stubgen (through `uvx`, with numpy available so NDArray return types
render) against the extension currently linked into perseus/, then folds the
`<module>._debug` submodule into the single stub file as a class of static methods. A stub
*package* (perseus/_core/__init__.pyi) is deliberately avoided: a directory of that name
next to the .so is importable as a namespace package when the extension is missing,
which would turn the "extension not built" ImportError into confusing AttributeErrors.
"""
import importlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(module: str = "perseus._core"):
    leaf = module.rsplit(".", 1)[1]
    out = Path(tempfile.mkdtemp(prefix="perseus_stubs_"))
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    subprocess.run(["uvx", "--quiet", "--with", "numpy", "--python", sys.executable,
                    "pybind11-stubgen", module, "-o", str(out), "--ignore-all-errors"],
                   check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pkg = out / "perseus" / leaf
    single = out / "perseus" / f"{leaf}.pyi"
    if single.exists():
        text = single.read_text(encoding="utf-8")
    else:
        init = (pkg / "__init__.pyi").read_text(encoding="utf-8")
        dbg = (pkg / "_debug.pyi").read_text(encoding="utf-8")
        # the submodule's defs, as a namespace class of static methods
        body = dbg[dbg.index("\ndef "):] if "\ndef " in dbg else ""
        blocks = re.split(r"\n(?=def )", body.strip("\n"))
        cls = ["class _debug:", '    """Harness and research taps: for probes, tests and diagnostics."""',
               "    decrypt_slots = decrypt_slots"]
        for b in blocks:
            if not b.startswith("def "):
                continue
            cls.append("    @staticmethod")
            cls += ["    " + ln if ln else ln for ln in b.splitlines()]
        lines = [ln for ln in init.splitlines()
                 if not ln.startswith(f"from {module}._debug import") and ln != "from . import _debug"]
        text = "\n".join(lines) + "\n" + "\n".join(cls) + "\n"
        # the top-level aliases the runtime keeps (only those this module actually exports)
        sys.path.insert(0, str(REPO))
        mod = importlib.import_module(module)
        for name in ("hard_exit", "install_fatal_exit_handler", "throw_test"):
            if hasattr(mod, name):
                text += f"{name} = _debug.{name}\n"
    dest = REPO / "perseus" / f"{leaf}.pyi"
    dest.write_text(text, encoding="utf-8")
    stale = REPO / "perseus" / leaf
    if stale.is_dir() and not (stale / "__init__.py").exists():
        for f in stale.glob("*.pyi"):
            f.unlink()
        stale.rmdir()
    compile(text, str(dest), "exec")   # the stub must parse as Python
    print(f"{dest}: {text.count('def ')} defs")


if __name__ == "__main__":
    main(*sys.argv[1:2])
