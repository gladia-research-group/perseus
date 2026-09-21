"""S4 gate: python EncBlock graph parity vs the C++ transformer_block (block 0)."""
import json
import os

from perseus import _core
from perseus.nn import EncBlock

out_dir = os.environ["PARITY_OUT"]

opts = _core.InferenceOptions()
opts.ckks = _core.CKKSOptions.from_env()
opts.mode = _core.InferenceMode.Sync
inf = _core.make_gpt2_inference(opts)

store = _core.WeightStore.from_zip(os.environ["WEIGHTS_PATH"])
configs = _core.load_configs(os.environ["CONFIGS_PATH"])
print("[parity] store+configs loaded", flush=True)
state0 = _core.load_block_state(inf, store, configs, _core.BootstrapPlan(), 0)
print("[parity] block-0 state encoded", flush=True)
_core.install_block_state(inf, state0)
inf.block_prefix = _core.gpt2_block_scope(0)
inf.capture_t = 0
inf.capture_b = 0

inputs = _core.read_teacher_forced_inputs(_core.RunConfig.from_env())
block = EncBlock().bind(inf)


x0 = _core.encode_token_input(inf, inputs[0])  # one ct; both arms consume the same input


def capture(fn, path):
    _core.reset_kv_cache(inf, 1)
    _core.reset_graph_runtime(inf)
    inf.enable_graph_capture()
    y = fn(x0)
    inf.export_graph_json(path)
    inf.disable_graph_capture()
    return y


_core.reset_kv_cache(inf, 1)
_core.transformer_block(inf, x0)  # warm-up: masks/weights cached for both arms
print("[parity] warm-up done", flush=True)
y_py = capture(block, f"{out_dir}/py_block0.json")
y_cc = capture(lambda t: _core.transformer_block(inf, t), f"{out_dir}/cc_block0.json")

a = _core.decode_token_output(inf, y_py)
b = _core.decode_token_output(inf, y_cc)
md = max(abs(u - v) for u, v in zip(a, b))
print(f"[parity] output max_diff={md:.3e}")

ga = json.load(open(f"{out_dir}/py_block0.json"))["nodes"]
gb = json.load(open(f"{out_dir}/cc_block0.json"))["nodes"]
print(f"[parity] nodes: py={len(ga)} cc={len(gb)}")

mismatches = 0
for i, (p, c) in enumerate(zip(ga, gb)):
    for k in ("op_type", "inputs", "output", "step", "input_levels", "output_level"):
        if p.get(k) != c.get(k):
            if mismatches < 12:
                print(f"[parity] MISMATCH node {i} {k}: py={p.get(k)} cc={c.get(k)}")
            mismatches += 1

assert len(ga) == len(gb), f"node count differs: {len(ga)} vs {len(gb)}"
assert mismatches == 0, f"{mismatches} node-field mismatches"
assert md < 1e-5, f"output diff too high for identical input ct: {md}"
print("[parity] PASS")
