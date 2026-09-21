"""Headless notebook executor (kernel pinned to sys.executable): run $NB_PATH via nbclient, save *_executed.ipynb.

Prints [notebook] PASS on success; any cell error propagates as a failure.
"""
import json
import os
import sys
import tempfile

# Force the kernel to THIS interpreter: a user-level "python3" kernelspec (e.g.
# another project's venv) shadows the native resolution and imports a different
# perseus (or none). A throwaway spec dir prepended via JUPYTER_PATH wins.
_spec = tempfile.mkdtemp(prefix="nbkernel_")
os.makedirs(os.path.join(_spec, "kernels", "python3"), exist_ok=True)
with open(os.path.join(_spec, "kernels", "python3", "kernel.json"), "w") as f:
    json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher",
                        "-f", "{connection_file}"],
               "display_name": "python3 (pinned)", "language": "python"}, f)
os.environ["JUPYTER_PATH"] = _spec + os.pathsep + os.environ.get("JUPYTER_PATH", "")

import nbclient  # noqa: E402  (must follow the JUPYTER_PATH pin above)
import nbformat  # noqa: E402

path = os.environ["NB_PATH"]
nb = nbformat.read(path, as_version=4)
client = nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3")
print(f"[notebook] kernel pinned to {sys.executable}", flush=True)
try:
    client.execute()
except Exception as e:
    msg = str(e)
    # CellExecutionError embeds the cell SOURCE first and the traceback LAST —
    # print both ends so the actual exception is never truncated away.
    print(f"[notebook] FAIL {path}: {type(e).__name__}: {msg[:300]}", flush=True)
    err_path = path + ".error.txt"
    with open(err_path, "w") as f:
        f.write(msg)
    print(f"[notebook] full error -> {err_path}", flush=True)
    sys.exit(1)
out = path.replace(".ipynb", "_executed.ipynb")
nbformat.write(nb, out)
print(f"[notebook] executed -> {out}", flush=True)
print("[notebook] PASS", flush=True)
