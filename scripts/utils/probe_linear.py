"""Probe: cachemir linear weight layout + standalone GELU accuracy for custom models."""
import os

import numpy as np

from perseus import _core

opts = _core.InferenceOptions()
opts.ckks = _core.CKKSOptions.from_env()
opts.mode = _core.InferenceMode.Sync
inf = _core.make_gpt2_inference(opts)

d_pad, d_real = 1024, 768
rng = np.random.default_rng(0)
x_real = rng.standard_normal(d_real) * 0.3
x_pad = np.zeros(d_pad)
x_pad[:d_real] = x_real


def run_linear(name, W):
    inf.set_weight(name, W.tolist(), d_pad, d_pad)
    x = _core.encode_token_input(inf, x_real.tolist())
    y = _core.linear(inf, x, name, d_pad, d_pad)
    return np.array(_core.decode_token_output(inf, y))[:d_real]


def rel(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


I = np.eye(d_pad)
out = run_linear("w_eye", I)
print(f"[probe] identity: rel={rel(out, x_real):.3e}", flush=True)

W = rng.standard_normal((d_pad, d_pad)) * 0.5 / np.sqrt(d_real)
W[d_real:, :] = 0.0
W[:, d_real:] = 0.0
out = run_linear("w_rand", W)
print(f"[probe] random: vs x@W rel={rel(out, (x_pad @ W)[:d_real]):.3e}  "
      f"vs W@x rel={rel(out, (W @ x_pad)[:d_real]):.3e}", flush=True)

configs = _core.load_configs(os.environ["CONFIGS_PATH"])
cfg = configs.softgelu["transformer.h.0.mlp.act"]
inf.set_gelu_cfg("act", cfg)
print(f"[probe] gelu cfg: method={cfg.method} xmax={cfg.xmax}", flush=True)


def gelu_exact(v):
    return 0.5 * v * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (v + 0.044715 * v**3)))


for scale in (0.3, 1.0, 3.0):
    g_in = rng.standard_normal(d_real) * scale
    x = _core.encode_token_input(inf, g_in.tolist())
    y = _core.gelu_approx(inf, x, "act")
    out = np.array(_core.decode_token_output(inf, y))[:d_real]
    print(f"[probe] gelu scale={scale}: rel={rel(out, gelu_exact(g_in)):.3e}", flush=True)

x = _core.encode_token_input(inf, x_real.tolist())
y = _core.linear(inf, _core.linear(inf, x, "w_eye", d_pad, d_pad), "w_eye", d_pad, d_pad)
out = np.array(_core.decode_token_output(inf, y))[:d_real]
print(f"[probe] chained square identity: rel={rel(out, x_real):.3e}", flush=True)

d_exp, e_real = 4096, 3072
Wu = rng.standard_normal((d_pad, d_exp)) * 0.5 / np.sqrt(d_real)
Wu[d_real:, :] = 0.0
Wu[:, e_real:] = 0.0
Wd = rng.standard_normal((d_exp, d_pad)) * 0.5 / np.sqrt(e_real)
Wd[e_real:, :] = 0.0
Wd[:, d_real:] = 0.0
inf.set_weight("up_p", Wu.tolist(), d_pad, d_exp)
inf.set_weight("down_p", Wd.tolist(), d_exp, d_pad)

x = _core.encode_token_input(inf, x_real.tolist())
h = _core.linear(inf, x, "up_p", d_pad, d_exp)
up_dec = np.array(_core.decode_linear_output(inf, h, d_pad, d_exp))
up_ref = x_pad @ Wu
print(f"[probe] up output: rel={rel(up_dec, up_ref):.3e}", flush=True)

h = _core.gelu_approx(inf, h, "act")
g_dec = np.array(_core.decode_linear_output(inf, h, d_pad, d_exp))
g_ref = gelu_exact(up_ref)
print(f"[probe] gelu(up) output: rel={rel(g_dec, g_ref):.3e}", flush=True)

y = _core.linear(inf, h, "down_p", d_exp, d_pad)
out = np.array(_core.decode_token_output(inf, y))[:d_real]
ref = (g_ref @ Wd)[:d_real]
print(f"[probe] mlp chain 1024->4096->gelu->1024: rel={rel(out, ref):.3e}", flush=True)

fhe = inf.fhe
x = _core.encode_token_input(inf, x_real.tolist())
fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
fhe.level_hint(x, fhe.level_limit() - 1)
h = _core.linear(inf, x, "up_p", d_pad, d_exp)
h = _core.gelu_approx(inf, h, "act")
fhe.bootstrap_hint(h, fhe.level_limit() - 1, True)
fhe.level_hint(h, fhe.level_limit() - 1)
y = _core.linear(inf, h, "down_p", d_exp, d_pad)
out = np.array(_core.decode_token_output(inf, y))[:d_real]
print(f"[probe] mlp chain HINTED: rel={rel(out, ref):.3e}", flush=True)

print("[probe] DONE")
