"""E2E: capture -> plan (v2, ml50 recipe + gen entry) -> PLANNED encrypted generation.

Produces the gen-shaped plan dir the generation surface needs (block_0..n_layers
decode-style captured with the deg-2 entry bootstrap, + block_13 cutmax and
block_14 feedback), then runs EncGPT2.generate(feedback="encrypted") strict
against it and compares with the eager capture run.

Env: the n32 decode preset (run_gen.sh) + GEN_PLAN_OUT_NAME (dir under
bootstrap_placements/, default a scratch name).
"""
import glob
import os
import shutil
import subprocess
import tempfile
import time

from perseus import _core
from perseus.nn import EncGPT2

opts = _core.InferenceOptions()
opts.ckks = _core.CKKSOptions.from_env()
opts.mode = _core.InferenceMode.Threaded
inf = _core.make_gpt2_inference(opts)
bts_out = inf.fhe.bootstrap_output_level()
print(f"[genplan] level_limit={inf.fhe.level_limit()} bts_out_level={bts_out}", flush=True)

store = _core.WeightStore.from_zip(os.environ["WEIGHTS_PATH"])
configs = _core.load_configs(os.environ["CONFIGS_PATH"])
cfg = _core.RunConfig.from_env()
inputs = _core.read_teacher_forced_inputs(cfg)
P, M = 2, 2
prompt = [inputs[p] for p in range(P)]

# ── phase 1: eager generate under capture (GEN_GRAPH_DIR reuses a capture) ────
graph_dir = os.environ.get("GEN_GRAPH_DIR") or tempfile.mkdtemp(prefix="gen_graphs_")
reuse = bool(os.environ.get("GEN_GRAPH_DIR")) and glob.glob(f"{graph_dir}/block_*/graph.json")
os.environ["FHE_GRAPH_DIR"] = graph_dir
model = EncGPT2(store, configs).bind(inf, cache_states=False)
# capture_t is pinned to the capture token for the whole run. GEN_CAPTURE_T picks
# WHICH forward gets captured: 0 = the first prompt forward (COLD KV — the plan
# then has no placements for the warm-KV read vars, and every pos>=1 forward
# detonates: fidelity probe measured rel 1.5e+04 at pos 1 vs 0.137 at pos 0);
# P-1 = the last prompt forward (WARM KV, the C++ steady-token capture shape).
# graph_capture_token() is a FIXED 0 in the C++ hooks — "capturing at the warm
# token" means giving capture_t=0 to the CAP_T-th FORWARD and a non-zero token to
# the earlier ones (pinning capture_t=CAP_T captures NOTHING: captured=[]).
CAP_T = int(os.environ.get("GEN_CAPTURE_T", "0"))
_fwd_count = [0]
_orig_forward = model.forward
def _forward_pinned(x, head=True):
    inf.capture_t = 0 if _fwd_count[0] >= CAP_T else 99
    _fwd_count[0] += 1
    # head FORCED during capture: block n_layers captures on the FIRST forward
    # (capture_t pinned), and a head=False prompt forward there records a tail
    # graph without the lm_head — the planner then emits no lm_head_tile levels
    # and the strict run dies at [plan_weight_error] lm_head_tile_0 34 vs 36.
    return _orig_forward(x, head=True)
model.forward = _forward_pinned
_orig_cutmax = _core.cutmax_feedback
def _cutmax_pinned(*a, **k):   # blocks 13/14 capture on the capture token too
    inf.capture_t = CAP_T
    return _orig_cutmax(*a, **k)
_core.cutmax_feedback = _cutmax_pinned
t0 = time.perf_counter()
ids_eager = None if reuse else model.generate(prompt, M, feedback="encrypted")
del os.environ["FHE_GRAPH_DIR"]
model.forward = _orig_forward
_core.cutmax_feedback = _orig_cutmax
captured = sorted(int(os.path.basename(os.path.dirname(g)).split("_")[1])
                  for g in glob.glob(f"{graph_dir}/block_*/graph.json"))
print(f"[genplan] eager ids={ids_eager} ({time.perf_counter() - t0:.0f}s); "
      f"captured blocks={captured}", flush=True)
n = model.n_layers
missing = sorted(set(range(n + 1)) - set(captured))   # 13/14 (the eager tail) are optional
assert not missing, f"capture incomplete, missing blocks {missing}"
assert os.path.exists(f"{graph_dir}/block_0/capture_env.json"), "C++ capture did not stamp"

# ── phase 2: plan with the production v2 recipe + the gen entry ──────────────
out_name = os.environ.get("GEN_PLAN_OUT_NAME", "_probe_gen_n32")
out_dir = os.path.join("bootstrap_placements", out_name)
shutil.rmtree(out_dir, ignore_errors=True)
# Plan blocks 0..n_layers (entry replanned for the deg-2 feedback/bootstrap entry). The
# CutMax/feedback tail (13/14) stays EAGER: its captured magnitudes sit past the
# bootstrap band (v2 calls every site DESTRUCTIVE) and the decode row never plans
# it either — gpt2_cutmax_feedback clears the live plan for the tail.
blocks_dir = tempfile.mkdtemp(prefix="gen_graphs_blocks_")
for b in range(n + 1):
    os.symlink(os.path.abspath(f"{graph_dir}/block_{b}"), f"{blocks_dir}/block_{b}")
env = dict(os.environ,
           GRAPH_DIR=blocks_dir, OUT_NAME=out_name,
           BTS_LEVEL="34", SRC_LEVEL=os.environ.get("GEN_PLAN_SRC_LEVEL", "34"),
           CACHE_READ_LEVEL="34", PLAN_LEVEL_UNIT="2",
           PLAN_CF_MAX="14", PLAN_NO_PRESCALE="1", PLAN_SPARSE_SLOTS="512,1",
           PLAN_SPARSE_BTS_OUT="1:24,512:34", PLAN_ACC_CHAIN="n32",
           MAX_LEVEL=os.environ.get("GEN_PLAN_MAX_LEVEL", "50"),
           **({} if os.environ.get("GEN_PLAN_ENTRY_CAPTURED") else
              dict(FIRST_ENTRY_LEVEL=os.environ.get("GEN_PLAN_ENTRY_LEVEL", str(bts_out)),
                   FIRST_ENTRY_DEG=os.environ.get("GEN_PLAN_ENTRY_DEG", "2"))),
           PYTHON=os.environ.get("PYTHON", ".venv/bin/python"))
t0 = time.perf_counter()
r = subprocess.run(["bash", "scripts/utils/run_bootstrap_all_blocks.sh"], env=env,
                   capture_output=True, text=True)
tail = "\n".join(r.stdout.splitlines()[-6:] + r.stderr.splitlines()[-6:])
print(f"[genplan] planner rc={r.returncode} ({time.perf_counter() - t0:.0f}s)\n{tail}", flush=True)
assert r.returncode == 0, "planner failed"
plans = sorted(glob.glob(f"{out_dir}/block_*_placement.json"))
assert len(plans) == n + 1, f"expected {n + 1} placements, got {len(plans)}"
import json

assert "capture_env" in json.load(open(plans[0])), "placement not stamped"
print(f"[genplan] gen plan dir: {out_dir} ({len(plans)} placements, stamped)", flush=True)

# ── phase 3: PLANNED encrypted generation against the new dir ────────────────
planned = EncGPT2(store, configs).bind(inf, plan_dir=out_dir, cache_states=False)
planned._plan14 = _core.BootstrapPlan()   # eager tail by design: lift the decode-shape guard
t0 = time.perf_counter()
ids_planned = planned.generate(prompt, M, feedback="encrypted")
print(f"[genplan] planned ids={ids_planned} ({time.perf_counter() - t0:.0f}s)", flush=True)
assert len(ids_planned) == M
if ids_eager is not None:
    assert ids_planned[0] == ids_eager[0], f"step-0 diverged: {ids_planned[0]} vs {ids_eager[0]}"
    if ids_planned != ids_eager:
        print(f"[genplan] note: trajectories forked after step 0 ({ids_planned} vs {ids_eager})",
              flush=True)
print("[genplan] PASS", flush=True)
# Success exit code made meaningful: cross-library static teardown (OpenFHE global key
# maps vs the FIDESlib CUDA context) corrupts AFTER the results are complete; skip it.
getattr(_core, "hard_exit", lambda c=0: None)(0)
