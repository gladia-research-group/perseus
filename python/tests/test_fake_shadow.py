"""The Shadow-backed fakes: `_pc` bindings, the fake's level rules as named quirks, the fake capture's graph v2
and stamps, the typed Config. The GPT-2 capture gate (decode + prefill fake captures == the shipped graphs) runs with
PC_M4B_GATE=1 (it needs the old repo's GPT-2 export and oracle; ~2 min)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from perseus import _pc, config
from perseus.impl import fake, fakecap

T = _pc.tag
ROOT = Path(__file__).resolve().parents[2]
OLD = Path("/data02/users/mithrillo/projects/perseus")
D3 = Path("/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/d3")


# ── Config ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_config_fields_and_presets():
    # + bts.em_fuse_rescale, bts.lt_tails, bts.pair, bts.sparse_b_k, model.prefill.{last_query,qkv_pack,attn_bsgs,sm_hpack},
    # chain.preset;
    # plan.fallback_cf,
    # plan.argmax_mincut, plan.ilp_tiebreak, plan.raise_drop_base, plan.raise_drop_search, plan.pack_cf,
    # plan.keep_placements_exact, plan.mag_safety_steps, plan.resume; chain.preset
    assert len(_pc.Config.fields()) == 261 and len(_pc.PlanConfig.fields()) == 104
    assert all(f["env"] == "" or f"`{f['env']}`" in f["doc"] for f in _pc.Config.fields())
    assert set(config.PRESETS) == {"n32_gpt2_decode", "n32_gpt2_prefill", "n32_llama_decode", "n32_llama_prefill"}
    c = config.preset("n32_gpt2_decode")
    assert c["model.ln_cheb"] is True and c["LN_CHEB"] is True and c.env("LN_CHEB") == "1"
    assert c["bts.auto_bts_level"] == 44 and c["bts.sparse_bts_slots"] == ["512", "1"]
    assert c["quirks.fakecap_landings"] == "0:32,1:14,256:22,512:24"
    assert config.preset("n32_gpt2_prefill")["model.prefill.linear"] == "pcmm"
    assert config.preset("n32_llama_decode")["model.sm_periodic"] == 2
    assert _pc.Config.from_json(c.to_json()) == c
    assert config.preset("n32_gpt2_decode", LN_CHEB="0")["model.ln_cheb"] is False
    lb = c.load_bearing()
    assert lb["exec.gpt2_cache"] is True and lb["host.omp_num_threads"] is None
    assert config.plan_preset("n32_gpt2_decode").get("plan.max_level") == "46" or \
        config.plan_preset("n32_gpt2_decode").get("plan_max_level") == 46


def test_config_errors_name_the_knob():
    with pytest.raises(_pc.ConfigError, match="LN_CHEB"):
        _pc.Config.from_legacy_env({"LN_CHEB": "yes"})
    with pytest.raises(_pc.ConfigError):
        _pc.Config()["model.nope"]
    with pytest.raises(_pc.ConfigError):
        _pc.Config().with_env({"NOT_A_KNOB": "1"})
    c, p, rest = _pc.Config.split_legacy_env({"LN_CHEB": "1", "BTS_LEVEL": "28", "PATH": "/x", "CHEB_ALIGN": None})
    assert c["model.ln_cheb"] and p.get("bootstrap_level") == "28" and rest == ["PATH"]
    assert c["model.cheb_align"] is None


def test_plan_runtime_matches_the_env_dicts():
    from perseus.examples.gpt2 import env
    for plan in ("bootstrap_placements/python/dacapo", None):          # runtime= line only
        assert config.to_env(config.plan_runtime(plan)) == env.plan_runtime(plan, "n32")
    for plan in ("bootstrap_placements/gpt2_decode", "bootstrap_placements/gpt2_prefill128"):
        # Config-stamped plans: the stamp adds the capture's load-bearing fields, the runtime levers agree
        got, want = config.to_env(config.plan_runtime(plan)), env.plan_runtime(plan, "n32")
        for k, v in want.items():
            assert got.get(k) == v or (got.get(k) is None and v == env.RUNTIME_LEGACY.get(k)), (plan, k)
    assert config.plan_configs("bootstrap_placements/gpt2_decode") == \
        "configs/model/approximation/gpt2_base_n32_cm/configs.json"


def test_capture_env_stamps_bridge():
    for g in sorted((ROOT / "graphs").glob("*/capture_env.json")):
        env = json.loads(g.read_text())["env"]
        c = config.capture_config(g.parent)
        want = {k: v for k, v in env.items() if v is not None}
        got = c.to_legacy_env()
        if (g.parent / "capture_config.json").exists():
            # a v2 capture's capture_config.json is the whole session Config: it agrees on every stamped name
            got = {k: got.get(k) for k in want}
        assert got == want, g


# ── tags, metas, routes ─────────────────────────────────────────────────────────────────────────────────────────────

def test_tag_algebra_bindings():
    a = T.from_signature(1024, 32, 32, 32, 0, 3)          # 32 live slots, stride 32, from 3
    assert a.known() and a.support.kind == T.AP and a.support.offset == 3
    r = T.t_rotate(a, 5)
    assert r.support.offset == (3 - 5) % 1024
    assert T.t_add(a, _pc.Tag()).known() is False
    assert T.t_bootstrap(a, 512, 1024).period == 512
    c = T.t_conjugate(a)
    c.conj_of = 7
    assert T.conj_pair(T.t_add(a, c), 7, 9, a, c, False).field == T.F_REAL
    assert T.conj_pair(T.t_add(a, c), 8, 9, a, c, True).field == T.F_UNKNOWN
    t = fakecap.pt_tag(np.r_[np.zeros(3), np.ones(4), np.zeros(1017)])
    assert t.field == T.F_REAL and (t.support.kind, t.support.offset, t.support.count) == (T.AP, 3, 4)
    assert _pc.top_limb(0) == 53 and _pc.level_from_top_limb(53) == 0
    assert _pc.planner_consumed(30, 28) == 2 and _pc.level_from_planner(-2.0, 28) == 26
    with pytest.raises(_pc.LevelError):
        _pc.level_from_planner(0.5, 28)


def test_shadow_modes_and_landings():
    cfg = config.preset("n32_gpt2_decode")
    cap = _pc.Shadow(cfg, "fakecap")
    assert cap.quirks == ["AddPtRealizes", "CapStcEntryNominal", "CapMultPtMax"]
    m = _pc.CtMeta(44, 2, slots=32768)
    assert cap.landing(_pc.Route(stc_first=True), m) == (26, 2)            # StC-first: 32 - FAKECAP_STC_DROP
    assert cap.landing(_pc.Route(stc_first=True), m.replace(level=48)) == (32, 2)   # nominal input past the entry
    assert cap.landing(_pc.Route(), m) == (32, 2)                           # shipped order
    assert cap.landing(_pc.Route.sparse(512), m) == (24, 2)
    rt = _pc.Shadow(cfg, "runtime", plan=_pc.PlanConfig.from_legacy_env(
        {"BTS_LEVEL": "28", "PLAN_SPARSE_BTS_OUT": "1:16,512:26", "PLAN_STC_FIRST_ENTRY": "46",
         "PLAN_STC_FIRST_FALLBACK": "34:1"}))
    assert rt.quirks == [] and rt.bootstrap_level == 28
    assert rt.landing(_pc.Route(stc_first=True), _pc.CtMeta(44, 1)) == (28, 1)
    assert rt.landing(_pc.Route.sparse(1), _pc.CtMeta(44, 1)) == (16, 1)
    with pytest.raises(_pc.LevelError, match="max_input_level"):
        rt.landing(_pc.Route(), _pc.CtMeta(50, 1))
    # the C++ product: degree 2 at the deeper effective level; the fake's: rescaled one level deeper
    assert rt.mult(_pc.CtMeta(30, 2), _pc.CtMeta(32, 1)).out.level == 32
    f = _pc.Shadow(fake.fake_config(), "fake")
    r = f.mult(_pc.CtMeta(30, 1), _pc.CtMeta(32, 1))
    assert (r.out.level, r.out.deg) == (34, 1) and r.pre == []
    p = rt.add(_pc.CtMeta(30, 2), _pc.CtMeta(34, 1))            # FLEXIBLEAUTO adjust: preludes on a
    assert (p.out.level, p.out.deg) == (34, 1) and {x.kind for x in p.pre} <= {"rescale", "drop_to", "scalar"}
    h = rt.hint(_pc.CtMeta(40, 1), 36)                         # dense, shipped order
    assert h.fired and h.out.level == 34
    assert rt.hint(_pc.CtMeta(40, 1), 36, route=_pc.Route(stc_first=True)).out.level == 28
    assert not rt.hint(_pc.CtMeta(36, 2), 36).fired and rt.hint(_pc.CtMeta(36, 2), 36, True).fired
    with pytest.raises(_pc.ConfigError):
        _pc.Shadow(cfg, "nope")


def test_shadow_scale_tables_and_consts():
    s = _pc.Shadow(config.preset("n32_gpt2_decode"), "runtime", tables=True)
    a = _pc.CtMeta(10, 1, nf=2.0 ** 54)
    r = s.mult_scalar(a, 0.5)
    assert r.consts_kind == "mult" and len(r.consts) == 54 - 10 and r.out.deg == 2
    assert s.add_scalar(a, -1.0).consts_kind == "add"


@pytest.mark.skipif(not os.access(D3 / "params.json", os.R_OK), reason="d3 dump not readable")
def test_em_program_view():
    p = json.loads((D3 / "params.json").read_text())
    cts3 = json.loads((D3 / "it0.cts3.json").read_text())
    em = json.loads((D3 / "it0.em.json").read_text())
    prog = _pc.EmProgram.build(cts3["level"], cts3["noise_level"], cts3["noise_factor"], json.dumps(p["cheb_ps"]),
                               int(p.get("double_angle_its", p.get("evalmod_its", 3))) if "double_angle_its" in p
                               or "evalmod_its" in p else 3)
    assert prog.n_ops > 0 and prog.n_consts > 0 and len(prog.ops) == prog.n_ops
    assert len(prog.fused_ops) <= prog.n_ops
    top, nl, nf = prog.out
    assert (top, nl) == (em["level"], em["noise_level"])
    assert _pc.EmProgram.from_json(prog.to_json()).n_ops == prog.n_ops


# ── the fake ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_fake_rules_are_the_old_fake():
    fhe, inf = fake.make_fake()
    assert fhe.shadow.quirks == ["PtLevelIgnored", "FakeProductRescaled", "FakeAddMaxDeg", "FakeNegateFree",
                                 "FakeAddPtFree", "FakeBootstrapRealized"]
    x = fake.FakeCt(np.linspace(-1, 1, 1024), 34)
    y = fhe.mult(x, x)
    assert (y.level, y.noise_deg) == (36, 1) and np.allclose(y.vec, x.vec ** 2)
    assert fhe.add(x, y).level == 36 and fhe.negate(y).level == 36 and inf.add_pt(y, np.ones(4)).level == 36
    assert inf.mult_pt(y, np.ones(4)).level == 38 and fhe.mult(y, 2.0).level == 38
    z = fake.FakeCt(x.vec, 44)
    w = fhe.square(z)                                   # 46 = the limit: the reactive refresh to bts_out
    assert (w.level, fhe.n_reactive) == (34, 1)
    fhe.bootstrap_hint(z, 44)
    assert z.level == 44
    fhe.bootstrap_hint(z, 40)
    assert z.level == 34
    z.noise_deg = 2
    fake.core.realize_pending_rescale(inf, z)
    assert (z.level, z.noise_deg) == (36, 1)
    s, _ = fake.make_fake(strict=True)
    with pytest.raises(_pc.LevelError):
        s.square(fake.FakeCt(x.vec, 44))
    with pytest.raises(RuntimeError):               # LevelError is a RuntimeError (the old strict contract)
        s.square(fake.FakeCt(x.vec, 44))


# ── the fake capture ────────────────────────────────────────────────────────────────────────────────────────────────

def _small_capture(tmp_path, cfg):
    from perseus.impl.layout import Dims
    d = Dims(1024, 32, 24, 4, 3, 128, 96)
    os.environ["FHE_GRAPH_DIR"] = str(tmp_path)
    fhe, inf = fakecap.make_session(d, range(1, 1024), config=cfg)
    core = fakecap.core
    assert core.begin_subgraph_capture(inf, 0)
    x = core.encode_token_input(inf, np.linspace(-1, 1, 24))
    with inf.step("blk0.t"):
        y = fhe.mult(x, x)
        y = inf.mult_pt(y, np.full(1024, 0.1))
        r = fhe.rotate_sum(y, [1, 2, 3])
        c = fhe.conjugate(r)
        z = fhe.add(r, c)
        fhe.bootstrap_hint(z, 26)
        for _ in range(10):
            z = fhe.square(z)
        fhe.level_hint(z, 40)
    core.end_subgraph_capture(inf, 0)
    return fhe, json.loads((tmp_path / "block_0" / "graph.json").read_text())


def test_fakecap_graph_v2(tmp_path):
    cfg = config.preset("n32_gpt2_decode")
    fhe, g = _small_capture(tmp_path, cfg)
    assert g["version"] == 2
    ops = [n["op_type"] for n in g["nodes"]]
    assert ops[:5] == ["mult", "mult", "rotate_sum", "conjugate", "add"] and "auto_bootstrap" in ops
    by = {n["op_type"]: n for n in g["nodes"]}
    assert by["rotate_sum"]["args"] == {"steps": [1, 2, 3]}
    assert by["mult"]["shadow"]["deg"] == 2 and "tag" in by["mult"]
    assert g["nodes"][1]["args"]["pt"]["kind"] == "values"
    assert by["hint"]["args"]["thr"] == 26 and by["hint"]["args"]["fired"] is True
    assert (by["hint"]["output_level"], by["hint"]["output_noise_level"]) == (28, 1)   # StC-first 26, realized
    assert by["auto_bootstrap"]["args"]["route"]["slots"] in (0, 1, 512)
    assert by["level_hint"]["args"] == {"cap": 40}
    assert by["add"]["pack_field"] == T.F_REAL                 # x + conj(x)
    # v1 fields intact for the planner, both loaders take v2
    from perseus.plan.placer.ir import Graph
    assert len(Graph.load(tmp_path / "block_0" / "graph.json", level_unit=2).nodes) == len(g["nodes"])
    stamp = _pc.Config.from_json((tmp_path / "capture_config.json").read_text())
    assert stamp["bts.auto_bts_level"] == 44 and (tmp_path / "block_0" / "capture_config.json").exists()
    # compare_graphs: 0 against itself, counts a perturbation per field
    assert set(fakecap.compare_graphs(tmp_path, tmp_path, log=lambda *a: None).values()) == {0}
    other = tmp_path / "other"
    (other / "block_0").mkdir(parents=True)
    g["nodes"][3]["output_level"] += 2
    g["nodes"][4]["op_type"] = "sub"
    g["nodes"][4]["step"] = "x"
    (other / "block_0" / "graph.json").write_text(json.dumps(g))
    assert fakecap.compare_graphs(tmp_path, other, log=lambda *a: None) == {"0": 3}


def test_fakecap_landings_follow_the_config(tmp_path):
    cfg = config.preset("n32_gpt2_decode").with_env({"FAKECAP_LANDINGS": "0:30,1:14,256:22,512:22",
                                                      "SPARSE_AUTO": "0", "FIDESLIB_BTS_STC_FIRST": "0"})
    _, g = _small_capture(tmp_path, cfg)
    assert {n["output_level"] for n in g["nodes"] if n["op_type"] == "auto_bootstrap"} == {30}


def test_finish_landings_from_the_shadow(tmp_path, monkeypatch):
    """The PCMM / CCMM finishes land where the Shadow says (`finish_landing`: the StC-first route under the raise
    drop, + the CCMM's StC stage), in every mode: the fake capture records what the runtime pins."""
    cfg = config.preset("n32_llama_prefill")
    want = {("pcmm", d): (26 + 2 * d, 2) for d in range(7)} | {("ccmm", d): (30 + 2 * d, 1) for d in range(7)}
    for mode in ("runtime", "fakecap", "fake"):
        sh = _pc.Shadow(cfg, mode)
        assert {k: sh.finish_landing(*k) for k in want} == want, mode
    with pytest.raises(_pc.ConfigError, match="finish_landing"):
        _pc.Shadow(cfg, "runtime").finish_landing("gemm")
    from perseus.impl import ccmm
    monkeypatch.delenv("PF_CCMM_FAKE_LAND", raising=False)
    assert ccmm.fake_landing() == 30
    monkeypatch.setenv("PF_CCMM_FAKE_LAND", "28")
    assert ccmm.fake_landing() == 28
    monkeypatch.delenv("PF_CCMM_FAKE_LAND")
    fhe, _ = _small_capture(tmp_path, config.preset("n32_gpt2_decode"))
    fhe.nodes = []
    for op, d in (("pcmm", 0), ("pcmm", 5), ("ccmm", 4)):
        x = fakecap.CapCt(np.zeros(1024), 42, packing=None)
        fhe.record_external_bootstrap(x, ["a"], [42], desc={"op": op, "raise_drop": d})
        n = fhe.nodes[-1]
        assert (n["output_level"], n["output_noise_level"]) == want[(op, d)] == tuple(n["args"]["landing"])
    fhe.pcmm_landing = (24, 2)                                  # an explicit override keeps the drop
    fhe.record_external_bootstrap(fakecap.CapCt(np.zeros(1024), 42, packing=None), ["a"], [42],
                                  desc={"op": "pcmm", "raise_drop": 1})
    assert fhe.nodes[-1]["output_level"] == 26


@pytest.mark.skipif(not (ROOT / "graphs" / "llama31_8b_prefill128" / "block_0").is_dir(),
                    reason="the Llama prefill recording (release artifact) is not unpacked")
def test_llama_prefill_finishes_land_where_the_runtime_pins():
    """Every finish of the shipped Llama prefill recording (PCMM raise drops 0-6, CCMM drops 0 and 4) sits at the
    runtime Shadow's `finish_landing` for its descriptor."""
    rt = _pc.Shadow(config.preset("n32_llama_prefill"), "runtime")
    seen = set()
    for b in sorted((ROOT / "graphs" / "llama31_8b_prefill128").glob("block_*")):
        for n in json.loads((b / "graph.json").read_text())["nodes"]:
            e = (n.get("args") or {}).get("external")
            if isinstance(e, dict):
                d = int(e.get("raise_drop", 0))
                got = (n["output_level"], n["output_noise_level"])
                assert got == rt.finish_landing(e["op"], d), (b.name, n["output"])
                seen.add((e["op"], d))
    if not seen:
        pytest.skip("a v1 recording (no finish descriptors)")
    assert {("pcmm", d) for d in (0, 1, 2, 4, 5, 6)} | {("ccmm", 0), ("ccmm", 4)} <= seen


# ── the gate ────────────────────────────────────────────────────────────────────────────────────────────────────────

GATE = os.environ.get("PC_M4B_GATE") == "1"
CACHE = OLD / ".cache"


def _gate_capture(tmp_path, mode):
    env = {**{k: v for k, v in os.environ.items() if not k.startswith(("FAKECAP_", "PF_", "FHE_", "PLAN_"))},
           "CUDA_VISIBLE_DEVICES": "", "PERSEUS_SLOTVEC_CACHE": "0", "PYTHONPATH": str(ROOT / "python"),
           "WEIGHTS_PATH": str(CACHE / "models/openai-community/gpt2/classic/weights.bin.zip"),
           "ALL_BLOCKS_IO_DIR": str(CACHE / "oracle/gpt2/all_blocks_io")}
    preset = "n32_gpt2_decode" if mode == "decode" else "n32_gpt2_prefill"
    plan = "gpt2_decode" if mode == "decode" else "gpt2_prefill128"
    env["FAKECAP_LANDINGS"] = config.preset(preset)["quirks.fakecap_landings"]
    sets = [x for kv in config.runtime_env(f"bootstrap_placements/{plan}").items() for x in ("--set", "=".join(kv))]
    out = tmp_path / mode
    if mode == "decode":
        cmd = ["-m", "perseus.examples.gpt2.run_decode", "--fake", "--tokens", "1", "--argmax",
               "--structure-only", "--configs", "configs/model/approximation/gpt2_base_n32/configs.json"]
    else:
        env["PF_PCMM_CALIB_IN"] = str(ROOT / "bootstrap_placements" / plan / "pcmm_calib.json")
        cmd = ["-m", "perseus.examples.gpt2.run_prefill", "--fake"]
    subprocess.run([sys.executable, *cmd, "--capture", str(out), *sets], env=env, check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return out


@pytest.mark.skipif(not (GATE and CACHE.exists()), reason="PC_M4B_GATE=1 and the GPT-2 export needed")
@pytest.mark.parametrize("mode,ref", [("decode", "gpt2_decode"), ("prefill", "gpt2_prefill128")])
def test_gate_fake_capture_equals_shipped(tmp_path, mode, ref):
    out = _gate_capture(tmp_path, mode)
    diffs = fakecap.compare_graphs(ROOT / "graphs" / ref, out)
    print(f"[gate] {mode} vs {ref}: {diffs}")
    assert diffs and set(diffs.values()) == {0}
    if mode == "prefill":      # every PCMM finish names its linear and output (args.external)
        ext = [n["args"]["external"] for b in out.glob("block_*") for n in json.loads((b / "graph.json").read_text())
               ["nodes"] if "external" in n.get("args", {})]
        assert ext and all(e["op"] == "pcmm" and e["linear"] and 0 <= e["out"] < e["n_out"] for e in ext)
