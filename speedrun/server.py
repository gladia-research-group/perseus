"""Perseus on its Rust runtime (perseus-core) as an Encrypted GPT-2 Speedrun server: the harness protocol. Decode-8
drives perseus.serve.gpt2_decode's handler (the served planned decode) instead of its socket; Prefill-128 runs
PlannedPrefill with GPT-2's encrypted tail (14 stages, the shipped prefill plan).

    bash speedrun/serve.sh --track {prefill,decode} --inbox IN --outbox OUT [--device N]

The wire is the speedrun's (harness/wire.py, harness/protocol.py): every file is an 8-byte little-endian length and
the payload, written under a temporary name and renamed. Setup, untimed: compile the plan's programs from the CPU
twin, write the key request out/keyspec (pc_client::KeySpec JSON, docs/WIRE.md), wait for the client's bundle
(in/keys/keys.pckb, complete once in/keys.done exists), open the session on it (no secret), and write out/ready
with the level block 0 takes its input at. Decode-8: request k is the 16 positions in/k/0..15 from an empty cache,
each a ciphertext frame of the token row; position j answers out/k/j with [logits frame, CutMax z frame] in
speedrun/record1.py's list framing. Prefill-128: setup also warms up on a public prompt (seeded token ids), and
out/ready lists the (level, degree) of each blocked input; request k is in/k/0, the prompt's blocked ciphertext
frames, answered with the last position's [logits frame, CutMax z frame]. The session ends at in/quit.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import types

from speedrun.record1 import PREFILL_PLAN, PROMPT, pack, unpack

HEADER = struct.Struct("<Q")
STEPS = 16


def write(root, rel, payload):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.tmp")
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(len(payload)))
        f.write(payload)
    os.rename(tmp, path)


def read(root, rel):
    with open(os.path.join(root, rel), "rb") as f:
        data = f.read()
    return data[HEADER.size:HEADER.size + HEADER.unpack_from(data)[0]]


def wait(root, *rels):
    while True:
        for rel in rels:
            if os.path.exists(os.path.join(root, rel)):
                return rel
        time.sleep(0.0005)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=("prefill", "decode"), required=True)
    p.add_argument("--inbox", required=True)
    p.add_argument("--outbox", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--plan", help="Decode-8's plan (default: perseus.examples.defaults')")
    a = p.parse_args()
    return decode(a) if a.track == "decode" else prefill(a)


def decode(a):
    from perseus.examples.defaults import default_plan
    from perseus.client import contract_hash
    from perseus.serve import gpt2_decode as served
    args = types.SimpleNamespace(plan=a.plan or default_plan("gpt2_decode"), chain="n32", layers=None, packing="cachemir_complex",
                                 weights=os.environ["WEIGHTS_PATH"], key_layout="packed", pt_rows=28, stc_cache=8,
                                 work_dir=os.path.join(os.environ.get("HOME", "/tmp"), "speedrun"))
    h = served.Gpt2Handler(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.device)      # the handler exports its own device; the GPU is ours
    man = h.manifest({"tokens": STEPS, "argmax": True, "contract": contract_hash(served._config(args.plan, "n32"))})
    write(a.outbox, "keyspec", man["spec"].encode())
    wait(a.inbox, "keys.done")
    h.load(os.path.join(a.inbox, "keys", "keys.pckb"), None)
    write(a.outbox, "ready", json.dumps({"input": man["input"]}).encode())
    k = 0
    while wait(a.inbox, f"{k}/0", "quit") != "quit":
        for j in range(STEPS):
            wait(a.inbox, f"{k}/{j}")
            _h, outs = h.run({"t": j, "argmax": True}, [read(a.inbox, f"{k}/{j}")], None)
            write(a.outbox, f"{k}/{j}", pack(outs))
        k += 1
    h.run_.close()
    return 0


def prefill(a):
    import numpy as np
    from perseus.examples.gpt2 import env, weights
    plan = PREFILL_PLAN
    os.environ.setdefault("KV_ARENA_GB", "0")
    fast = {"FIDESLIB_ENCODE_NOSYNC": "1", "FIDESLIB_CCMM_ASYNC": "1", "FIDESLIB_PCMM_NOSYNC": "1",
            "FIDESLIB_ENCODE_PINNED_RING": "256", "FIDESLIB_STC_POW2_KEY": "1"}
    over = {**{k: os.environ.get(k, v) for k, v in fast.items()}, **env.plan_runtime(plan, "n32"),
            "GPT2_PACKING": "cachemir"}
    for k, v in over.items():                  # as run_prefill: the plan's PF_* switches
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    os.environ.setdefault("PF_PCMM_CALIB_IN", os.path.join(plan, "pcmm_calib.json"))
    over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
            "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    env.export_env(a.device, chain="n32", CKKS_COMPLEX="1", **over)
    from perseus import _pc, config as pcfg
    from perseus.impl import config, fakecap
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.layout import Dims, ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.prefill import PlannedPrefill, diag_rot_steps, prefill_rot_steps
    from perseus.examples.gpt2.prefill import Gpt2Prefill
    d = Dims.gpt2()
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    rot = prefill_rot_steps(d) | diag_rot_steps(d) | pcmm_rot_steps() | blocked_rot_steps(
        Blocks(d, int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits()) | fakecap.GPT2_BAND \
        | ladder_radix_steps(d, ladder_radix_bits())
    _fhe, inf = fakecap.make_session(d, rot)
    store = weights.RawStore(os.environ["WEIGHTS_PATH"])
    model = Gpt2Prefill(inf, store, config.load_configs(env.plan_configs(plan, "n32", env.CONFIGS_N32_LEGACY)),
                        core=fakecap.core)
    model.with_tail = True

    class Served(PlannedPrefill):
        bundle = None

        def open_session(self, kw):
            return _pc.Session.from_bundle(self.cfg, self.keys, self.bundle, **kw)

    cfg = pcfg.plan_runtime(plan, "n32", base=pcfg.preset("n32_gpt2_prefill"))
    cfg["exec.use_graphs"] = False             # as run_prefill: a one-shot forward only pays a graph capture
    run = Served(model, plan, cfg, _pc.KeySource.generate([], conj=True, layout="packed"), pt_rows=28, stc_cache=16,
                 work_dir=os.path.join(os.environ.get("HOME", "/tmp"), "speedrun"), log=lambda m: print(m, flush=True))
    ids = np.random.default_rng(0).integers(0, store.shape(weights.WTE)[0], PROMPT)   # public placeholder prompt
    rows = store.tensor(weights.WTE)[ids] + store.tensor(weights.WPE)[:PROMPT]
    run.prepare(rows)
    write(a.outbox, "keyspec", _pc.Session.key_manifest(run.cfg, run.keys, **run.session_kwargs()).encode())
    wait(a.inbox, "keys.done")
    run.bundle = os.path.join(a.inbox, "keys", "keys.pckb")
    run.setup(rows)
    assert not run.sess.has_secret and not run.fhe.has_secret_key
    run.run(vals=run.input_values(rows))       # warm-up on the public prompt
    run.fhe.device_sync()
    write(a.outbox, "ready", json.dumps({"inputs": [list(m) for m in run.in_metas[0]]}).encode())
    tail, cutmax = run.reg[model.n_layers][0][1], run.reg[model.n_layers + 1][0][1]
    k = 0
    while wait(a.inbox, f"{k}/0", "quit") != "quit":
        x0 = [_pc.import_ct(run.sess, f) for f in unpack(read(a.inbox, f"{k}/0"))]
        run.run(x0=x0)
        write(a.outbox, f"{k}/0", pack([_pc.export_ct(run.sess, run.ct(r)) for r in (tail, cutmax)]))
        del x0
        k += 1
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
