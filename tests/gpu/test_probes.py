"""The PASS-gated GPU probes as tests: one subprocess per script, exit status + marker.

Each probe keeps its own contract (env, PASS marker, hard_exit) — running it in a
subprocess means the documented teardown crash cannot flip a green run, and the probes
stay usable as standalone scripts. A probe is skipped when the env it reads is not set
(they need the artifacts a box has: CONFIGS_PATH, WEIGHTS_PATH, PLAN_DIR, ...).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]
PROBES = sorted(p.name for p in (REPO / "scripts" / "utils").glob("probe_*.py"))


def _required_env(script):
    src = (REPO / "scripts" / "utils" / script).read_text(encoding="utf-8")
    return sorted(set(re.findall(r'os\.environ\["([A-Z0-9_]+)"\]', src)))


@pytest.mark.parametrize("script", PROBES)
def test_probe_passes(script):
    missing = [k for k in _required_env(script) if k not in os.environ]
    if missing:
        pytest.skip(f"{script} needs {', '.join(missing)} in the environment")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "utils" / script)],
                       cwd=REPO, capture_output=True, text=True, timeout=7200)
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-30:])
    assert r.returncode == 0, f"{script} exited {r.returncode}\n{tail}"
    assert "PASS" in r.stdout, f"{script} printed no PASS marker\n{tail}"
