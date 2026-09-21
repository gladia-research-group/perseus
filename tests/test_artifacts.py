"""Artifact provenance: written by export/calibrate, checked before a session loads them."""
import json
import zipfile

import pytest

from perseus import artifacts


def _zip(tmp_path, manifest):
    p = tmp_path / "weights.bin.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    return p


def _configs(tmp_path, meta=None):
    p = tmp_path / "configs.json"
    doc = {"model": {"n_layers": 1}, "softgelu": {}}
    if meta is not None:
        doc["meta"] = meta
    p.write_text(json.dumps(doc))
    return p


def test_meta_writers_carry_version_and_provenance():
    m = artifacts.calibration_meta("gpt2", "openwebtext", "gpt2", 8, ["softgelu", "norm"], ["cutmax"])
    assert m["format"] == 1 and m["perseus_version"] and m["sections_inherited"] == ["cutmax"]
    w = artifacts.weights_meta("openai-community/gpt2", "gpt2")
    assert w["format_version"] == 1 and w["source"]["model_type"] == "gpt2"


def test_pre_provenance_artifacts_load_with_a_note(tmp_path):
    z = _zip(tmp_path, {"model": "classic", "tensors": []})
    c = _configs(tmp_path)
    notes = artifacts.check(weights_zip=z, configs_path=c)
    assert len(notes) == 2 and all(n.startswith("note:") for n in notes)
    artifacts.require(weights_zip=z, configs_path=c)      # notes never raise


def test_chain_and_model_type_mismatches_are_errors(tmp_path):
    z = _zip(tmp_path, {"model": "classic", "tensors": [],
                        **artifacts.weights_meta("bigscience/bloom-560m", "bloom")})
    c = _configs(tmp_path, {**artifacts.calibration_meta("gpt2", "owt", "gpt2", 8, []), "chain": "n64"})
    problems = artifacts.check(weights_zip=z, configs_path=c, chain="n32", model_type="gpt2")
    kinds = sorted(p.split(":")[0] for p in problems)
    assert kinds == ["error", "error", "warning"]
    with pytest.raises(ValueError, match="artifact mismatch"):
        artifacts.require(weights_zip=z, configs_path=c, chain="n32", model_type="gpt2")


def test_newer_formats_are_refused(tmp_path):
    z = _zip(tmp_path, {"tensors": [], "format_version": 99})
    assert any("newer" in p for p in artifacts.check(weights_zip=z))
