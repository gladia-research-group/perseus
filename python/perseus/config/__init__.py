"""The typed run configuration (``_pc.Config``, crate pc-config) for the Python side: presets, the plan-bound
runtime of a plan dir, the capture stamp, and the bridge to the legacy env the rest of the port still reads.

    from perseus import config
    cfg = config.preset("n32_gpt2_decode", LN_CHEB="0")        # a preset + legacy-env overrides
    cfg = config.plan_runtime("bootstrap_placements/gpt2_decode", base=cfg)
    config.to_env(cfg)                                          # {"LN_CHEB": "0", ...} for os.environ

Presets (``config.PRESETS``) are the configurations the old run paths exported, derived from the old env dicts
(crates/pc-config/tools/derive_presets.py): ``n32_gpt2_decode``, ``n32_gpt2_prefill``, ``n32_llama_decode``,
``n32_llama_prefill``.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from perseus import _pc
from perseus._paths import resolve

Config = _pc.Config
PlanConfig = _pc.PlanConfig
ConfigError = _pc.ConfigError
PRESETS = tuple(_pc.Config.presets())
STAMP = "capture_config.json"


def preset(name: str, **env_overrides: str) -> _pc.Config:
    """A named preset with legacy-env overrides layered on (`LN_CHEB="0"`)."""
    cfg = _pc.Config.preset(name)
    return cfg.with_env({k: str(v) for k, v in env_overrides.items()}) if env_overrides else cfg


def plan_preset(name: str) -> _pc.PlanConfig:
    """The planner knobs the preset's run path exported (PLAN_* of the session profile)."""
    return _pc.PlanConfig.preset(name)


def _plan_cmd(plan) -> dict[str, str]:
    """The `key=value` lines of a plan dir's PLAN_CMD.txt (runtime=, recipe=, configs=, ...)."""
    cmd = Path(resolve(str(plan))) / "PLAN_CMD.txt"
    out = {}
    if cmd.exists():
        for line in cmd.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.isidentifier():
                out[k] = v
    return out


# runtime= switches of the C++ runtime that have no meaning here: block-weight residency (the arena is sized from the
# loaded programs, the weights are resident plaintexts)
RUNTIME_DROPPED = frozenset({"IMPL_RESIDENT_BLOCKS"})


def _runtime_line(line: str) -> dict[str, str]:
    return {k: v for k, v in (kv.split("=", 1) for kv in shlex.split(line)) if k not in RUNTIME_DROPPED}


def runtime_env(plan=None, chain: str = "n32") -> dict[str, str]:
    """The plan-bound switches of `plan` as legacy env (`examples.gpt2.env.plan_runtime`, which this replaces):
    a plan's `runtime=` line over the legacy values of the switches it does not declare; no plan on n32 = the
    eager defaults (RUNTIME_N32); n64 = legacy."""
    from perseus.examples.gpt2.env import RUNTIME_LEGACY, RUNTIME_N32
    if chain != "n32":
        return dict(RUNTIME_LEGACY)
    if not plan:
        return dict(RUNTIME_N32)
    line = _plan_cmd(plan).get("runtime")
    if line is None:
        return dict(RUNTIME_LEGACY)
    return {**RUNTIME_LEGACY, **_runtime_line(line)}


def plan_runtime(plan=None, chain: str = "n32", base: _pc.Config | None = None) -> _pc.Config:
    """`base` (default: empty) with the plan's runtime switches set. A plan whose placements carry the Config stamp
    (`placement.json["config"]`) is read from the stamp with PLAN_CMD.txt's `runtime=` line over it (the stamp
    holds the capture's load-bearing fields, not the runtime levers; pc-program `PlanDir::config` reads it the same
    way); older plans from the `runtime=` line alone."""
    cfg = base.copy() if base is not None else _pc.Config()
    stamp = _plan_stamp(plan) if plan else None
    if stamp is not None:
        cfg = cfg.overlay(stamp)
        line = _plan_cmd(plan).get("runtime")
        return cfg.overlay(_config_switches(_runtime_line(line))) if line else cfg
    return cfg.overlay(_config_switches(runtime_env(plan, chain)))


def _config_switches(env: dict[str, str]) -> _pc.Config:
    """The switches of a runtime line the Config has fields for; the others (model switches such as SM_HRS_CLEANSE,
    which the CPU twin reads) reach the run through the example drivers' process environment (examples.*.env)."""
    c, _pl, _rest = _pc.Config.split_legacy_env(env)
    return c


def _plan_stamp(plan):
    d = Path(resolve(str(plan)))
    for p in sorted(d.glob("block_*_placement.json"))[:1]:
        doc = json.loads(p.read_text())
        if isinstance(doc.get("config"), dict):
            return _pc.Config.from_json(json.dumps(doc["config"]))
    return None


def plan_configs(plan=None, default: str | None = None) -> str | None:
    """The approximation configs.json a plan declares (`configs=` line), else `default`."""
    line = _plan_cmd(plan).get("configs") if plan else None
    return line.split()[0] if line else default


def capture_config(graph_dir) -> _pc.Config:
    """The Config a graph was captured under: `capture_config.json`, else the legacy `capture_env.json` stamp."""
    d = Path(resolve(str(graph_dir)))
    if (d / STAMP).exists():
        return _pc.Config.from_json((d / STAMP).read_text())
    env = json.loads((d / "capture_env.json").read_text()).get("env") or {}
    return _pc.Config.from_legacy_env(env)


def write_stamp(graph_dir, cfg: _pc.Config) -> str:
    p = Path(graph_dir) / STAMP
    p.write_text(cfg.to_json())
    return str(p)


def to_env(cfg: _pc.Config) -> dict[str, str]:
    """The legacy env spelling of every set field (for code that still reads os.environ)."""
    return dict(cfg.to_legacy_env())


__all__ = ["Config", "PlanConfig", "ConfigError", "PRESETS", "preset", "plan_preset", "runtime_env",
           "plan_runtime", "plan_configs", "capture_config", "write_stamp", "to_env"]
