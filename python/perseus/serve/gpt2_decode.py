"""GPT-2 planned decode served to a separate client process: the server never holds the secret.

    # server (GPU, under tools/gpu.sh): the plan's programs compiled from a CPU twin fed with zero rows (the
    # per-position op stream does not depend on the token's values), the session on the client's key bundle
    python -m perseus.serve.gpt2_decode server --addr /tmp/x.sock [--plan bootstrap_placements/gpt2_decode]
    # client (no device): keygen of the server's request, per token encrypt -> run -> decrypt, gate vs the oracle
    python -m perseus.serve.gpt2_decode client --addr /tmp/x.sock --plan ... --tokens 16 --argmax --out res.json

Per token the client sends block 0's input (`pack_tokens` layout at the compiled input level) and gets back the
LM-head tile and, with --argmax, CutMax's z; it decodes logits / z itself (`examples.gpt2.head`).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

PRESET = "n32_gpt2_decode"


def _config(plan, chain):
    from perseus import config as pcfg
    from perseus._paths import resolve
    return pcfg.plan_runtime(resolve(plan), chain, base=pcfg.preset(PRESET))


# ── server ──────────────────────────────────────────────────────────────────────────────────────────────────────

class Gpt2Handler:
    def __init__(self, a):
        from perseus import _pc
        from perseus._paths import resolve
        from perseus.impl import config, fake, fakecap
        from perseus.impl.driver import PlannedRun, Recorder
        from perseus.impl.layout import Dims
        from perseus.examples.gpt2 import env, weights
        from perseus.examples.gpt2.model import Gpt2Model
        self.a, self._pc = a, _pc
        plan = resolve(a.plan)
        over = {**env.plan_runtime(plan, a.chain), "GPT2_PACKING": a.packing}
        env.export_env(3, chain=a.chain, CKKS_COMPLEX="1", **over)
        configs = env.plan_configs(plan, a.chain, "configs/model/approximation/gpt2_base_n32/configs.json")
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over}, a.chain, True)
        d = Dims.gpt2()
        band, _ = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real)
        _, inf = fakecap.make_session(d, set(band.loaded_rot_steps) | fakecap.GPT2_BAND)
        model = Gpt2Model(inf, weights.RawStore(a.weights), config.load_configs(configs), core=fakecap.core,
                          n_layers=a.layers, packing=a.packing, kv_offload=False)
        self.n = model.n_layers
        self.dim = d.dim
        cfg = _config(a.plan, a.chain)
        keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
        labels = [f"block_{b}" for b in range(self.n + 2)]

        class Served(PlannedRun):
            """PlannedRun on a client's bundle: no secret; block 0's input is the client's frame."""
            bundle, inputs = None, {}

            def manifest(self):
                return _pc.Session.key_manifest(self.cfg, self.keys, **self.session_kwargs())

            def open_session(self, kw):
                return _pc.Session.from_bundle(self.cfg, self.keys, self.bundle, **kw)

            def input_ct(self, t, rows, p):
                x = self.inputs.pop(t)
                if x.level != p.inputs[0][1]:
                    raise ValueError(f"token {t}: input at level {x.level}, block 0 takes {p.inputs[0][1]}")
                return x

        self.run_ = Served(Recorder(model, self.n), plan, cfg, labels, keys, work_dir=os.path.join(a.work_dir, "ref"),
                           pt_rows=a.pt_rows, stc_cache=a.stc_cache)
        self.rows = None
        self.argmax = False

    def manifest(self, hello):
        t0 = time.time()
        self.rows = [np.zeros(self.dim)] * int(hello["tokens"])      # public placeholders for the CPU twin
        self.argmax = bool(hello.get("argmax"))
        r = self.run_
        r.prepare(self.rows, argmax=self.argmax)
        spec = r.manifest()
        if json.loads(spec)["contract"] != hello["contract"]:
            raise ValueError(f"client plan contract {hello['contract']}, server {json.loads(spec)['contract']}")
        p0 = r.progs[r.labels[0]]
        print(f"[server] prepared (record + compile) {time.time() - t0:.1f} s; key request: "
              f"{len(json.loads(spec)['keys'])} switching keys", flush=True)
        return {"spec": spec, "input": {"level": int(p0.inputs[0][1]), "deg": 1}, "n_layers": self.n}

    def load(self, path, ask):
        r = self.run_
        r.bundle = path
        r.setup(self.rows, argmax=self.argmax)
        r.precompute(self.rows, argmax=self.argmax)
        assert not r.sess.has_secret and not r.fhe.has_secret_key
        s, k = r.sess.setup_seconds()
        return {"session_s": round(s, 2), "import_s": round(k, 2), "keys": r.sess.n_keys(), "has_secret": r.sess.has_secret}

    def run(self, h, frames, ask):
        _pc, r = self._pc, self.run_
        t = int(h["t"])
        t0 = time.time()
        r.inputs[t] = _pc.import_ct(r.sess, frames[0])
        t1 = time.time()
        reg, rec = r.token(t, self.rows, self.argmax)
        lg = _pc.export_ct(r.sess, r.ct(reg))
        t2 = time.time()
        out = [lg]
        if h.get("argmax"):
            out.append(_pc.export_ct(r.sess, r.ct(r.argmax(t))))
        t3 = time.time()
        return {"t": t, "import_s": t1 - t0, "fwd_s": t2 - t1, "argmax_s": t3 - t2, "server_s": t3 - t1, **rec}, out


def server_main(a):
    from .server import serve_one
    h = Gpt2Handler(a)
    st = serve_one(a.addr, h, os.path.join(a.work_dir, "server"))
    print(f"[server] done: {st}", flush=True)
    return 0


# ── client ──────────────────────────────────────────────────────────────────────────────────────────────────────

def client_main(a):
    import types
    from perseus.client import Client, ClientSecret, Remote, contract_hash, pack_token_rows
    from perseus.impl.layout import Dims
    from perseus.examples.gpt2 import oracle_io, weights
    from perseus.examples.gpt2.head import decode_logits, decode_z
    from perseus.examples.gpt2.model import dist_gate, dist_report
    os.environ["ALL_BLOCKS_IO_DIR"] = a.io_dir or ""
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(a.tokens)
    oc = oracle_io.RunConfig.from_env()
    rows = oracle_io.read_teacher_forced_inputs(oc)[:a.tokens]
    gt = oracle_io.read_lm_head_steps(oc)
    d = Dims.gpt2()
    vocab = weights.vocab_size(weights.RawStore(a.weights))
    contract = contract_hash(_config(a.plan, a.chain))
    t0 = time.time()
    cl = Client(ClientSecret.generate(), complex=True)
    remote = Remote(a.addr, cl, os.path.join(a.work_dir, "client"))
    man = remote.handshake(contract, tokens=len(rows), argmax=a.argmax)
    level = man["input"]["level"]
    kg = remote.provide_keys(threads=a.threads)
    shim = types.SimpleNamespace(ops=types.SimpleNamespace(decrypt_slots_complex=lambda f: cl.decrypt(f)[0]), dims=d)
    res = []
    for t, row in enumerate(rows):
        ts = time.time()
        x = cl.encrypt(pack_token_rows([row], d.N, d.hid, d.dim), level)
        te = time.time()
        h, outs = remote.run([x], t=t, argmax=a.argmax)
        tr = time.time()
        lg = decode_logits(shim, [outs[0]], vocab, d.N)
        r = {"pos": t, "top1": int(np.argmax(lg)), "encrypt_s": te - ts, "roundtrip_s": tr - te, **h}
        if gt.T and t < len(gt.logits) and len(gt.logits[t]):
            r.update(dist_report(lg, np.asarray(gt.logits[t])))
        if a.argmax:
            z = decode_z(shim, [outs[1]], vocab, d.N)
            r.update(cutmax=int(np.argmax(z)), z_mass=float(z.max()))
        r["decrypt_s"] = time.time() - tr
        res.append(r)
        print(f"[client] tok{t} top1={r['top1']} ref={r.get('ref')} KL={r.get('kl', float('nan')):.4g} "
              f"{'OK' if r.get('hit') else 'MISS'} cutmax={r.get('cutmax')} server fwd={h['fwd_s']:.3f} "
              f"argmax={h['argmax_s']:.3f} roundtrip={r['roundtrip_s']:.3f} encrypt={r['encrypt_s']:.3f} "
              f"decrypt={r['decrypt_s']:.3f}", flush=True)
    remote.close()
    ok, bad = dist_gate(res)
    kls = sorted(r["kl"] for r in res if "kl" in r)
    steady = [r for r in res if r["pos"] >= 1]
    med = lambda k: statistics.median(r[k] for r in steady) if steady else float("nan")
    summ = {"tokens": len(res), "top1": sum(1 for r in res if r.get("hit")), "median_kl": kls[len(kls) // 2] if kls else None,
            "cutmax": sum(1 for r in res if r.get("cutmax") == r["top1"]), "server_s_tok": med("server_s"),
            "server_fwd_s": med("fwd_s"), "server_argmax_s": med("argmax_s"), "roundtrip_s": med("roundtrip_s"),
            "encrypt_s": med("encrypt_s"), "decrypt_s": med("decrypt_s"), "keygen": kg, "wall_s": time.time() - t0,
            "gate": ok}
    print(f"[client] SUMMARY {json.dumps({k: v for k, v in summ.items() if k != 'keygen'})}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summ, "tokens": [{k: v for k, v in r.items() if k != "logits"} for r in res]}, f, indent=1, default=float)
    print("[client] PASS" if ok else f"[client] FAIL {[(r['pos'], r.get('kl')) for r in bad]}", flush=True)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=("server", "client"))
    ap.add_argument("--addr", required=True, help="Unix socket path or host:port")
    from perseus.examples.defaults import default_plan
    ap.add_argument("--plan", default=default_plan("gpt2_decode"))
    ap.add_argument("--chain", default="n32")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--argmax", action="store_true")
    ap.add_argument("--packing", default="cachemir_complex")
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS_PATH"))
    ap.add_argument("--io-dir", default=os.environ.get("ALL_BLOCKS_IO_DIR"))
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--pt-rows", type=int, default=28)
    ap.add_argument("--stc-cache", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8, help="client keygen threads")
    ap.add_argument("--work-dir", default=os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), "m12_served"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return server_main(a) if a.role == "server" else client_main(a)


if __name__ == "__main__":
    sys.exit(main())
