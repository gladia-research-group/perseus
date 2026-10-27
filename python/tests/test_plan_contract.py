"""The plan/config contract (perseus.plan.contract): the planner stamps `Config.load_bearing()` of the capture
into `placement.json["config"]`; plans stamped before the typed Config (`capture_env`) still validate, read through
`Config.from_legacy_env`."""
import json
import os
from pathlib import Path

import pytest

from perseus import _pc
from perseus._paths import ASSETS
from perseus.plan import contract
from perseus.plan.placer import emit

BP = ASSETS / "bootstrap_placements"
GRAPH = ASSETS / "graphs" / "gpt2_decode"


def _shipped_stamps():
    """(file, legacy stamp) of every shipped placement file that carries one."""
    out = []
    files = [Path(d) / f for d, _, fs in os.walk(BP, followlinks=True) for f in fs
             if f.startswith("block_") and f.endswith("_placement.json") and "/_regen_" not in d]
    for p in sorted(files):
        st = contract.read_stamp(p)
        if st is not None:
            out.append((p, st))
    return out


def test_every_shipped_plan_validates_through_the_config():
    stamps = _shipped_stamps()
    assert len(stamps) > 100
    for p, st in stamps:
        if "env" not in st:
            # a plan made on the new runtime carries the Config stamp itself: it states its own session
            from perseus import _pc
            import json
            assert contract.validate_contract(st, session=_pc.Config.from_json(json.dumps(st)), source=str(p)) == []
            continue
        session = contract.config_of(st)
        # the legacy stamp against the session it states, and the Config stamp derived from it against the same
        assert contract.validate_contract(st, session=session, source=str(p)) == []
        assert contract.validate_contract(contract.stamp_of(st), session=session, source=str(p)) == []


def test_mismatch_names_the_knob():
    st = json.loads((GRAPH / "capture_env.json").read_text())        # the legacy stamp (`capture_env`)
    session = contract.config_of(st).with_env({"LN_CHEB": "0"})
    for stamp in (st, contract.stamp_of(st)):
        bad = contract.validate_contract(stamp, session=session, strict=False)
        assert len(bad) == 1 and "model.ln_cheb (LN_CHEB)" in bad[0], bad
        with pytest.raises(contract.PlanContractError, match="LN_CHEB"):
            contract.validate_contract(stamp, session=session)
    # set <-> unset is a mismatch too (an unset fold flag IS a value)
    unset = contract.config_of({"env": {**st["env"], "CHEB_ALIGN": None}})
    have = contract.config_of({"env": {**st["env"], "CHEB_ALIGN": "1"}})
    assert contract.validate_contract(contract.stamp_of(unset), session=have, strict=False)


def test_capture_stamp_is_the_load_bearing_config():
    # the block's own capture_env.json first, then the graph dir's (the planner's order)
    st = contract.capture_stamp(GRAPH / "block_1")
    legacy = json.loads((GRAPH / "block_1" / "capture_env.json").read_text())
    assert st == json.loads(_pc.Config.from_legacy_env(legacy["env"]).load_bearing().to_json())
    assert contract.config_of(st) == contract.config_of(legacy).load_bearing()


def test_emit_and_stamp_file_write_the_config_key(tmp_path):
    doc = {"placements": [], "capture_env": {"env": {"LN_CHEB": "1"}}}
    st = contract.capture_stamp(GRAPH)          # the graph dir's capture_env.json (what make_plan.sh stamps)
    emit.stamp(doc, st)
    assert "capture_env" not in doc and doc["config"] == st
    p = tmp_path / "block_0_placement.json"
    p.write_text(json.dumps({"placements": [], "capture_env": json.loads((GRAPH / "capture_env.json").read_text())}))
    contract.stamp_file(p, json.loads((GRAPH / "capture_env.json").read_text()))
    d = json.loads(p.read_text())
    assert "capture_env" not in d and d["config"] == st
    assert contract.read_stamp(p) == st


def test_read_stamp_prefers_the_config(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"config": {"model": {"ln_cheb": True}}, "capture_env": {"env": {}}}))
    assert contract.read_stamp(p) == {"model": {"ln_cheb": True}}
    assert contract.read_stamp(tmp_path / "missing.json") is None
