"""Probe: EncSequential residency-pipeline parity (eager vs sync/prefetch/threaded).

Covers the three stage kinds (opaque, streamed inf.w keys, EncodedBlock state), a
mixed chain, repeat forwards (evict/reload), and eager-after-pipeline (lazy reload).
"""
import os
import time

import numpy as np

from perseus import _core
from perseus.nn import EncGELU, EncLinear, EncModule, EncSequential

opts = _core.InferenceOptions()
opts.ckks = _core.CKKSOptions.from_env()
opts.mode = _core.InferenceMode.Sync
inf = _core.make_gpt2_inference(opts)
print(f"[probe] slots={inf.slots} logN={inf.logN} level_limit={inf.fhe.level_limit()}",
      flush=True)

configs = _core.load_configs(os.environ["CONFIGS_PATH"])
inf.set_gelu_cfg("act", configs.softgelu["transformer.h.0.mlp.act"])

d_pad, d_real = 1024, 768
d_exp, e_real = 4096, 3072
rng = np.random.default_rng(0)


def make_weight(d_in, d_out, in_real, out_real, scale=0.5):
    w = rng.standard_normal((d_in, d_out)) * scale / np.sqrt(in_real)
    w[in_real:, :] = 0.0
    w[:, out_real:] = 0.0
    return w


def gelu_exact(v):
    return 0.5 * v * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (v + 0.044715 * v**3)))


def rel(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


x_real = rng.standard_normal(d_real) * 0.3
x_pad = np.zeros(d_pad)
x_pad[:d_real] = x_real


def run(model, tag):
    t0 = time.perf_counter()
    y = model(_core.encode_token_input(inf, x_real.tolist()))
    out = np.array(_core.decode_token_output(inf, y))[:d_real]
    print(f"[probe] {tag}: {time.perf_counter() - t0:.2f}s", flush=True)
    return out


# ── A: streamed-key stages (EncLinear residency, with bias) ──────────────────
W1 = make_weight(d_pad, d_exp, d_real, e_real)
W2 = make_weight(d_exp, d_pad, e_real, d_real)
b1 = np.zeros(d_exp); b1[:e_real] = rng.standard_normal(e_real) * 0.05
b2 = np.zeros(d_pad); b2[:d_real] = rng.standard_normal(d_real) * 0.05
mods = [EncLinear("fc1", d_pad, d_exp, weight=W1.tolist(), bias=b1.tolist()),
        EncGELU("act"),
        EncLinear("fc2", d_exp, d_pad, weight=W2.tolist(), bias=b2.tolist())]
ref_a = (gelu_exact(x_pad @ W1 + b1) @ W2 + b2)[:d_real]

eager = EncSequential(*mods).bind(inf)
out_eager = run(eager, "A eager")
print(f"[probe] A eager vs ref: rel={rel(out_eager, ref_a):.3e}", flush=True)
assert rel(out_eager, ref_a) < 0.05

for overlap in ("sync", "prefetch", "threaded"):
    piped = EncSequential(*mods, overlap=overlap)
    piped.inf = inf   # modules already bound; the container only drives the runner
    out = run(piped, f"A {overlap}")
    # fresh encryption noise per run: arms agree to the chain's noise floor, not bitwise
    r = rel(out, out_eager)
    print(f"[probe] A {overlap} vs eager: rel={r:.3e}  vs ref: rel={rel(out, ref_a):.3e}", flush=True)
    assert rel(out, ref_a) < 0.05, f"A {overlap} lost the plaintext ref: {rel(out, ref_a)}"
    assert r < 2e-2, f"A {overlap} diverged from eager beyond noise: {r}"

piped = EncSequential(*mods, overlap="threaded")
piped.inf = inf
out = run(piped, "A threaded (2nd pass)")     # evict -> reload path
assert rel(out, ref_a) < 0.05
out = run(eager, "A eager (after pipeline)")  # reload after key evict
r = rel(out, out_eager)
print(f"[probe] A eager-after-pipeline vs eager: rel={r:.3e}", flush=True)
assert rel(out, ref_a) < 0.05


# ── B: EncodedBlock state stages (custom module + builder) ───────────────────
class StageMLP(EncModule):
    """up -> gelu -> down with its own EncodedBlock stage state."""

    def __init__(self, prefix, Wu, Wd):
        super().__init__()
        self.prefix = prefix
        self.Wu, self.Wd = Wu, Wd
        self.state = None

    def bind(self, inf):
        super().bind(inf)
        st = _core.EncodedBlock()
        st.set_weight(inf, f"{self.prefix}.up", self.Wu.tolist(), d_pad, d_exp)
        st.set_weight(inf, f"{self.prefix}.down", self.Wd.tolist(), d_exp, d_pad)
        self.state = st
        return self

    def residency(self):
        return self.state

    def forward(self, x):
        fhe = self.inf.fhe
        fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
        fhe.level_hint(x, fhe.level_limit() - 1)
        x = _core.linear(self.inf, x, f"{self.prefix}.up", d_pad, d_exp)
        x = _core.gelu_approx(self.inf, x, "act")
        fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
        fhe.level_hint(x, fhe.level_limit() - 1)
        return _core.linear(self.inf, x, f"{self.prefix}.down", d_exp, d_pad)


# A deep chain walks the GELU past its fitted xmax (run_notebooks.sh custom NOTE), so
# the plaintext ref is reported for info only; the assert is pipelined-vs-eager PARITY,
# which is what validates the runner. scale=0.3 keeps magnitudes tame regardless.
W3 = make_weight(d_pad, d_exp, d_real, e_real, scale=0.3)
W4 = make_weight(d_exp, d_pad, e_real, d_real, scale=0.3)
h = gelu_exact(x_pad @ W1) @ W2
ref_b = (gelu_exact(h @ W3) @ W4)[:d_real]

stage1 = StageMLP("s1", W1, W2)
stage2 = StageMLP("s2", W3, W4)
model_b = EncSequential(stage1, stage2, overlap="threaded").bind(inf)
out_b = run(model_b, "B threaded (2 state stages)")
print(f"[probe] B threaded vs plain ref (info): rel={rel(out_b, ref_b):.3e}", flush=True)

model_b.overlap = None   # states installed by the threaded pass; eager reuses them
out_b_eager = run(model_b, "B eager")
# The out-of-range gelu amplifies encryption noise; measure the SAME-PATH floor
# (two eager runs, fresh encryptions) and gate arm parity against it.
floor_b = rel(run(model_b, "B eager (repeat)"), out_b_eager)
r = rel(out_b, out_b_eager)
print(f"[probe] B threaded vs eager: rel={r:.3e} (same-path floor {floor_b:.3e})", flush=True)
assert r < max(3 * floor_b, 2e-2), f"B threaded beyond noise: {r} vs floor {floor_b}"

# ── C: mixed chain (key stage + opaque stage + state stage) ──────────────────
model_c = EncSequential(
    EncLinear("fc1", d_pad, d_exp, weight=None),   # pre-installed name: opaque by design
    EncGELU("act"),
    EncLinear("fc2", d_exp, d_pad, weight=W2.tolist()),
    StageMLP("s3", W3, W4),
    overlap="threaded",
).bind(inf)
out_c = run(model_c, "C threaded (mixed)")
model_c.overlap = None
out_c_eager = run(model_c, "C eager")
floor_c = rel(run(model_c, "C eager (repeat)"), out_c_eager)
r = rel(out_c, out_c_eager)
print(f"[probe] C threaded vs eager: rel={r:.3e} (same-path floor {floor_c:.3e})", flush=True)
assert r < max(3 * floor_c, 2e-2), f"C threaded beyond noise: {r} vs floor {floor_c}"

# ── D: capture -> plan -> load_plans (the EncSequential planned-mode flow) ───
# Block-shaped stages (the plannable granularity — a bare linear/gelu graph has no
# feasible min-cut). No plaintext anchor for a chained toy (the missing head-style
# rearrangement makes the chain a different, deterministic function), so the gate
# is planned-threaded vs planned-eager parity against the measured same-path floor.
import dataclasses
import tempfile

from perseus.plan import PlanConfig, plan_graph_dir

# PlanConfig defaults are the n64 chain's units; on n32 the runner exports the
# prime-granular recipe (make_plans.sh n32 arm: BTS/SRC/CACHE=34, unit=2, MAX_LEVEL
# = runtime AUTO_BTS_LEVEL) or the flow network is infeasible by construction.
# PLAN_MAX_ABS is not read any more: v1's flat magnitude ceiling has no v2 counterpart
# (v2 prices each refresh with the CF/band accuracy model -- err_target, mag_safety).
popts = PlanConfig()
if os.environ.get("PLAN_MAX_LEVEL"):
    popts = dataclasses.replace(
        popts,
        max_level=int(os.environ["PLAN_MAX_LEVEL"]),
        bootstrap_level=int(os.environ.get("PLAN_BTS_LEVEL", popts.bootstrap_level)),
        source_level=int(os.environ.get("PLAN_SRC_LEVEL", popts.source_level)),
        cache_read_level=int(os.environ.get("PLAN_CACHE_READ_LEVEL", popts.cache_read_level)),
        level_unit=int(os.environ.get("PLAN_LEVEL_UNIT", 1)),
    )

root = tempfile.mkdtemp(prefix="probe_pipe_graphs_")
plans = tempfile.mkdtemp(prefix="probe_pipe_plans_")
model_d = EncSequential(StageMLP("d1", W3, W4), StageMLP("d2", W3, W4)).bind(inf)
model_d.export_graphs(_core.encode_token_input(inf, x_real.tolist()), root)
summaries = plan_graph_dir(root, plans, popts)
print(f"[probe] D planned {len(summaries)} stages, "
      f"placements={sum(s['num_placements'] for s in summaries)}", flush=True)
assert len(summaries) == 2

model_d.overlap = _core.InferenceMode.Threaded
model_d.load_plans(plans)
out_dp = run(model_d, "D threaded (planned)")
model_d.overlap = None
out_de = run(model_d, "D eager (planned)")
floor_d = rel(run(model_d, "D eager (planned, repeat)"), out_de)
r = rel(out_dp, out_de)
print(f"[probe] D planned threaded vs eager: rel={r:.3e} (same-path floor {floor_d:.3e})",
      flush=True)
assert r < max(3 * floor_d, 2e-2), f"D planned beyond noise: {r} vs floor {floor_d}"

# check the contract stamp round-trip on D's plans
import json as _json

from perseus.plan.contract import PlanContractError

stamp = _json.load(open(os.path.join(plans, "block_0_placement.json")))
assert "capture_env" in stamp, "planner did not stamp the capture contract"
_saved = os.environ.get("GPT2_FOLD_LN1")
os.environ["GPT2_FOLD_LN1"] = "0" if _saved != "0" else "1"
try:
    EncSequential(EncGELU("act")).load_plans(plans)
    raise AssertionError("contract mismatch not caught")
except PlanContractError as e:
    print(f"[probe] D contract mismatch caught: {str(e).splitlines()[-1].strip()}", flush=True)
finally:
    if _saved is None:
        os.environ.pop("GPT2_FOLD_LN1", None)
    else:
        os.environ["GPT2_FOLD_LN1"] = _saved

# ── E: slot-layout tracking + strict mode + error taxonomy ───────────────────
xe = _core.encode_token_input(inf, x_real.tolist())
print(f"[probe] E layout(encode)={_core.layout_of(xe)}", flush=True)
assert _core.layout_of(xe) == "token"
ye = _core.linear(inf, xe, "fc1", d_pad, d_exp)
assert _core.layout_of(ye) == "expanded", _core.layout_of(ye)
ge = _core.gelu_approx(inf, ye, "act")
assert _core.layout_of(ge) == "expanded", _core.layout_of(ge)
ze = _core.linear(inf, ge, "fc2", d_exp, d_pad)
assert _core.layout_of(ze) == "derived", _core.layout_of(ze)
_core.set_strict_layout(True)
try:
    _core.linear(inf, ze, "fc1", d_pad, d_exp)   # the chained-MLP trap
    raise AssertionError("strict layout did not catch the chained linear")
except _core.LayoutError:
    print("[probe] E chained linear caught as LayoutError (strict)", flush=True)
finally:
    _core.set_strict_layout(False)

# ── F: generic calibration on THIS model's distribution ──────────────────────
from perseus.nn import calibrate_sequential

cal_samples = rng.standard_normal((64, d_real)) * 0.3
parsed_cal = calibrate_sequential(eager, cal_samples)
out_cal = run(eager, "F eager (own-fit act)")
r = rel(out_cal, ref_a)
print(f"[probe] F own-calibrated vs ref: rel={r:.3e}", flush=True)
assert r < 0.05, f"own-fit calibration degraded the chain: {r}"

# ── G: ciphertext serialization round-trip ───────────────────────────────────
import tempfile as _tf

xg = _core.encode_token_input(inf, x_real.tolist())
blob = _core.serialize_ct(inf, xg)
print(f"[probe] G ct bytes={len(blob)}", flush=True)
xg2 = _core.deserialize_ct(inf, blob)
d0 = np.array(_core.decode_token_output(inf, xg))[:d_real]
d1 = np.array(_core.decode_token_output(inf, xg2))[:d_real]
assert rel(d1, d0) < 1e-9, "serialize/deserialize round-trip drifted"
yg = _core.linear(inf, xg2, "fc1", d_pad, d_exp)   # GPU load path from host-only ct
yd = np.array(_core.decode_linear_output(inf, yg, d_pad, d_exp))
print(f"[probe] G linear-on-deserialized vs ref: rel={rel(yd, x_pad @ W1 + b1):.3e}", flush=True)
assert rel(yd, x_pad @ W1 + b1) < 0.05
skp = os.path.join(_tf.mkdtemp(prefix="probe_keys_"), "secret.key")
_core.save_secret_key(inf, skp)
assert os.path.getsize(skp) > 0
print(f"[probe] G secret key serialized: {os.path.getsize(skp)} bytes", flush=True)

# ── H: LayerNorm (gamma/beta) + own-distribution norm calibration ────────────
# Runs on BOTH chains. The n32 failure history: the standalone affine used to run
# at the chain cliff (tiles encoded at level 0, mult at 40-42 of limit 46) where
# auto-bootstraps amplify encryption noise chaotically — fixed by refreshing the
# normed ct before ln_affine + encoding tiles at bts_out (probe_ln_toy2.py).
from perseus.nn import EncLayerNorm

gamma = beta = None
gamma = 1.0 + 0.1 * rng.standard_normal(d_real)
beta = 0.05 * rng.standard_normal(d_real)
model_h = EncSequential(
    EncLayerNorm("cln", d_real, weight=gamma.tolist(), bias=beta.tolist()),
    EncLinear("h1", d_pad, d_exp, weight=W1.tolist(), bias=b1.tolist()),
    EncGELU("hact"),
    EncLinear("h2", d_exp, d_pad, weight=W2.tolist(), bias=b2.tolist()),
).bind(inf)
xh = rng.standard_normal(d_real) * 0.3 + 0.2
mu, var = xh.mean(), xh.var()
ln = (xh - mu) / np.sqrt(var + 1e-5) * gamma + beta
ln_pad = np.zeros(d_pad); ln_pad[:d_real] = ln
ref_h = (gelu_exact(ln_pad @ W1 + b1) @ W2 + b2)[:d_real]
if os.environ.get("CHAIN", "") == "n32":
    # n32: the standalone norm chain is run-to-run chaotic (scalar inv-sqrt noise
    # amplification at 27-limb precision; probe_ln_toy4) — the SUPPORTED behavior
    # is a loud refusal from fit_scale's repeatability gate, not a wrong answer.
    try:
        calibrate_sequential(model_h, rng.standard_normal((64, d_real)) * 0.3 + 0.2)
    except RuntimeError as e:
        assert "repeatable" in str(e) or "chaotic" in str(e), f"wrong refusal: {e}"
        print("[probe] H n32: standalone LN correctly REFUSED (unstable chain): "
              f"{str(e)[:90]}", flush=True)
    else:
        raise AssertionError("n32 standalone LN fit unexpectedly passed the "
                             "repeatability gate — re-validate the numeric arm")
else:
    calibrate_sequential(model_h, rng.standard_normal((64, d_real)) * 0.3 + 0.2)   # fits cln (+ scale) + hact
    y = model_h(_core.encode_token_input(inf, xh.tolist()))
    out_h = np.array(_core.decode_token_output(inf, y))[:d_real]
    r = rel(out_h, ref_h)
    print(f"[probe] H LN(gamma,beta)+MLP own-calibrated vs ref: rel={r:.3e}", flush=True)
    assert r < 0.05, f"LN chain off: {r}"

# ── telemetry taps (print one line each; validity is "does not throw") ───────
_core.residency_perf_report(0)
_core.block_extract_perf_report(0)
_core.mask_perf_report(inf, 0)

print("[probe] PASS", flush=True)
# Success exit code made meaningful: cross-library static teardown (OpenFHE global key
# maps vs the FIDESlib CUDA context) corrupts AFTER the results are complete; skip it.
getattr(_core, "hard_exit", lambda c=0: None)(0)
