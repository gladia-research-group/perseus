"""Provenance for the on-disk artifacts and a validator for loading them together.

Three files turn a checkpoint into an encrypted model — ``weights.bin.zip`` (export),
``configs.json`` (calibration) and, optionally, ``block_<b>.enc`` weight artifacts — and
until now none of them said which perseus, model, dataset or chain produced it, so a
mismatch surfaced as a wrong number deep in a run. This module writes a small ``meta``
block into each and checks them against each other and the session before use.
"""
import datetime
import json
import os
import zipfile

from . import __version__
from ._env import current_chain

CONFIGS_FORMAT = 1
WEIGHTS_FORMAT = 1


def _now():
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def calibration_meta(model, dataset, approximation, n_batches, sections, inherited=()):
    """The ``meta`` section written by ``perseus-calibrate``."""
    return {
        "format": CONFIGS_FORMAT,
        "perseus_version": __version__,
        "model": model,
        "dataset": dataset,
        "approximation": approximation,
        "n_calib_batches": int(n_batches),
        "sections_fit": list(sections),
        "sections_inherited": list(inherited),
        "chain": current_chain(),
        "created": _now(),
    }


def weights_meta(model_id, model_type):
    """The extra keys ``perseus-export`` writes into ``manifest.json``."""
    return {
        "format_version": WEIGHTS_FORMAT,
        "source": {"perseus_version": __version__, "model_id": model_id,
                   "model_type": model_type, "created": _now()},
    }


def read_configs_meta(path):
    """The ``meta`` section of a configs.json (``{}`` for files written before it existed)."""
    p = path if os.path.isfile(path) else os.path.join(path, "configs.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("meta", {})


def read_weights_manifest(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read("manifest.json"))


def check(weights_zip=None, configs_path=None, chain=None, model_type=None):
    """Cross-check the artifacts a session is about to load. Returns a list of
    human-readable problems (empty = consistent); missing provenance is reported as a
    note, never as an error, so pre-provenance artifacts keep loading."""
    problems = []
    chain = chain or current_chain()
    cm = read_configs_meta(configs_path) if configs_path else {}
    wm = read_weights_manifest(weights_zip) if weights_zip else {}
    if configs_path and not cm:
        problems.append(f"note: {configs_path} carries no meta section (calibrated before "
                        f"perseus {__version__}); cannot verify its provenance")
    if weights_zip and "format_version" not in wm:
        problems.append(f"note: {weights_zip} carries no format_version (exported before "
                        f"perseus {__version__})")
    if cm.get("format", CONFIGS_FORMAT) > CONFIGS_FORMAT:
        problems.append(f"error: configs.json format {cm['format']} is newer than this "
                        f"perseus understands ({CONFIGS_FORMAT})")
    if wm.get("format_version", WEIGHTS_FORMAT) > WEIGHTS_FORMAT:
        problems.append(f"error: weights.bin.zip format {wm['format_version']} is newer than "
                        f"this perseus understands ({WEIGHTS_FORMAT})")
    if cm.get("chain") and chain and cm["chain"] != chain:
        problems.append(f"error: configs.json was calibrated for chain {cm['chain']!r} but the "
                        f"session is {chain!r} (Chebyshev basis / nr_iters differ per chain)")
    src = wm.get("source", {})
    if model_type and src.get("model_type") and src["model_type"] != model_type:
        problems.append(f"error: weights.bin.zip holds a {src['model_type']!r} model, "
                        f"expected {model_type!r}")
    if cm.get("model") and src.get("model_id") and cm["model"] != src["model_id"]:
        problems.append(f"warning: configs.json was calibrated on {cm['model']!r} but the "
                        f"weights come from {src['model_id']!r}")
    return problems


def require(**kw):
    """check(...) that raises on any 'error:' line."""
    problems = check(**kw)
    errors = [p for p in problems if p.startswith("error:")]
    if errors:
        raise ValueError("artifact mismatch:\n  " + "\n  ".join(errors))
    return problems
