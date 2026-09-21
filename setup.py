"""Build hook: `pip install .` builds the CUDA extension (perseus._core) when it can.

The pure-Python package installs anywhere (setuptools, pyproject.toml). The extension is
a CMake target that needs the FIDESlib/OpenFHE deps tree and a CUDA toolchain; this hook
runs that CMake build during `pip install` / `pip wheel` when

    FIDESLIB_ROOT=/path/to/deps_n32   (the tree scripts/install_deps.sh produces)

is set, and skips it otherwise (or with PERSEUS_BUILD_EXT=0), so a client machine or CI
runner still gets the planner, calibration and export layers. Knobs, all optional:

    PERSEUS_CMAKE_BUILD_DIR   reuse a configured build dir (default build_py_<chain>)
    PERSEUS_CUDA_ARCH         CMAKE_CUDA_ARCHITECTURES (default: 120-real)
    PERSEUS_CMAKE_ARGS        extra -D flags, whitespace-separated
    CMAKE_BUILD_PARALLEL_LEVEL  jobs

The module lands in perseus/ next to the package (the same place scripts/local_build_core.sh
writes it), stamped with the chain of the deps tree it was linked against (_core.chain).
"""
import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        root = os.environ.get("FIDESLIB_ROOT")
        if os.environ.get("PERSEUS_BUILD_EXT", "1") == "0" or not root:
            self.announce("perseus._core not built: set FIDESLIB_ROOT to the deps tree "
                          "(scripts/install_deps.sh) to build the CUDA extension", level=3)
            return
        chain = "n32" if root.rstrip("/").endswith("n32") else ("n64" if root.rstrip("/").endswith("n64") else "n32")
        build_dir = Path(os.environ.get("PERSEUS_CMAKE_BUILD_DIR", HERE / f"build_py_{chain}"))
        build_dir.mkdir(parents=True, exist_ok=True)
        py = sys.executable
        args = [
            "cmake", "-S", str(HERE), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={os.environ.get('PERSEUS_CUDA_ARCH', '120-real')}",
            f"-DFIDESLIB_ROOT={root}",
            "-DCACHEMIR_BUILD_PYTHON=ON", "-DCACHEMIR_BUILD_TESTS=OFF",
            "-DPYBIND11_FINDPYTHON=NEW", f"-DPython_EXECUTABLE={py}",
            f"-DPython_INCLUDE_DIR={sysconfig.get_paths()['include']}",
        ]
        cuda = os.environ.get("CUDA_HOME")
        if cuda:
            args.append(f"-DCMAKE_CUDA_COMPILER={cuda}/bin/nvcc")
        try:
            import pybind11
            args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
        except ImportError:
            pass
        args += shlex.split(os.environ.get("PERSEUS_CMAKE_ARGS", ""))
        if not (build_dir / "CMakeCache.txt").exists():
            subprocess.check_call(args)
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "8")
        subprocess.check_call(["cmake", "--build", str(build_dir), "--parallel", jobs,
                               "--target", "_core"])
        # the target writes perseus/_core<EXT_SUFFIX> into the source tree; for a wheel
        # build copy it where setuptools expects the extension
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        built = HERE / "perseus" / f"_core{suffix}"
        if not built.exists():
            raise RuntimeError(f"CMake reported success but {built} is missing")
        dest = Path(self.get_ext_fullpath("perseus._core"))
        if dest.resolve() != built.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(built.read_bytes())


def _wants_extension():
    return bool(os.environ.get("FIDESLIB_ROOT")) and os.environ.get("PERSEUS_BUILD_EXT", "1") != "0"


# Declaring the extension only when it will be built keeps the pure-Python wheel
# py3-none-any; with FIDESLIB_ROOT set the wheel is platform-tagged and carries _core.
setup(ext_modules=[CMakeExtension("perseus._core")] if _wants_extension() else [],
      cmdclass={"build_ext": CMakeBuild})
