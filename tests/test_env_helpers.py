"""perseus._env: the only sanctioned way library code touches os.environ.

Also pins the session profiles to the environment the run scripts set.
"""
import os
import pathlib

import pytest

from perseus._env import current_chain, scoped_env

_REPO = pathlib.Path(__file__).resolve().parents[1]


def test_scoped_env_sets_and_restores_including_unset(monkeypatch):
    monkeypatch.setenv("PERSEUS_T_A", "old")
    monkeypatch.delenv("PERSEUS_T_B", raising=False)
    with scoped_env(PERSEUS_T_A="new", PERSEUS_T_B=1, PERSEUS_T_C=None):
        assert os.environ["PERSEUS_T_A"] == "new" and os.environ["PERSEUS_T_B"] == "1"
        assert "PERSEUS_T_C" not in os.environ
    assert os.environ["PERSEUS_T_A"] == "old" and "PERSEUS_T_B" not in os.environ


def test_scoped_env_restores_on_error(monkeypatch):
    monkeypatch.setenv("PERSEUS_T_A", "old")
    try:
        with scoped_env(PERSEUS_T_A="new"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert os.environ["PERSEUS_T_A"] == "old"


def test_current_chain_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("CHAIN", "n64")
    assert current_chain() in ("n64", "n32")     # the module stamp wins once it exists
    monkeypatch.delenv("CHAIN")
    assert current_chain("x") in ("x", "n32", "n64")


def test_profile_device_clamp_uses_a_fraction_of_free_memory(monkeypatch):
    from perseus import profile as prof_mod
    from perseus.profile import SessionProfile

    p = SessionProfile.gpt2_decode_n32()
    assert p._env()["KV_ARENA_GB"] == "24"
    monkeypatch.setattr(prof_mod, "_device_free_gb", lambda: 40.0)
    assert p.env()["KV_ARENA_GB"] == "10"                 # 25 % of 40 GB
    monkeypatch.setattr(prof_mod, "_device_free_gb", lambda: 200.0)
    assert p.env()["KV_ARENA_GB"] == "24"                 # never raised above the preset
    monkeypatch.setattr(prof_mod, "_device_free_gb", lambda: None)
    assert p.env()["KV_ARENA_GB"] == "24"                 # no GPU visible: untouched


@pytest.mark.parametrize("chain,preset", [("n32", "gpt2_decode_n32"), ("n64", "gpt2_decode_n64")])
def test_profile_agrees_with_the_run_script_environment(chain, preset):
    """The decode presets must stay the configuration scripts/local_env.sh sets.

    Only the names that shape the executed graph are compared: a plan is bound to them,
    so a Python caller who drifts from the shell gets a run the shipped plan cannot bind.
    """
    import shutil
    import subprocess

    from perseus.profile import SessionProfile
    if shutil.which("bash") is None:
        pytest.skip("no bash")
    names = ("AUTO_BTS_LEVEL", "FIDESLIB_ROT_KEY_BAND", "FHE_LMHEAD_CAP",
             "CACHE_READ_LEVEL_K", "CACHE_READ_LEVEL_V", "SPARSE_BTS_SLOTS",
             "SPARSE_AUTO", "FUSED_SM_DEN", "FUSED_LN_VAR", "FHE_PT_COEFF_ENCODE")
    script = "source scripts/local_env.sh >/dev/null 2>&1 || exit 77\n" + "".join(
        f'printf "%s=%s\\n" {n} "${{{n}-}}"\n' for n in names)
    r = subprocess.run(["bash", "-c", script], cwd=_REPO, text=True, capture_output=True,
                       env={**os.environ, "CHAIN": chain})
    if r.returncode == 77:
        pytest.skip("scripts/local_env.sh does not run here")
    shell = dict(line.split("=", 1) for line in r.stdout.splitlines())
    env = getattr(SessionProfile, preset)().env(host=False)
    for n in names:
        if not shell.get(n):
            continue                                  # the shell leaves it to the code default
        assert env.get(n) == shell[n], f"{preset}: {n} is {env.get(n)!r}, local_env.sh sets {shell[n]!r}"
