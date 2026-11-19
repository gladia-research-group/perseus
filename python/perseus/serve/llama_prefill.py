"""Llama planned prefill served to a separate client process: the server never holds the client's secret.

    python -m perseus.serve.llama_prefill server --addr SOCK -- --tokens 127 --plan bootstrap_placements/llama31_8b_prefill128 --force
    python -m perseus.serve.llama_prefill client --addr SOCK [--out res.json] -- <the same run_prefill arguments>

Both processes build `examples.llama.run_prefill`'s CPU twin from the same arguments. The server records and
compiles with placeholder tokens (BOS everywhere: the prefill's op stream depends on the prompt length only), asks
for its keys (PCMM + CCMM included), calibrates the CCMM on its own (`c_unit` under a throwaway secret of its own:
nothing the client decrypts ever goes back to the server) and runs forwards on the client's encrypted block-0
inputs, returning the last block's hidden ciphertexts. The client packs its prompt (`LlamaPlannedPrefill.input_values`'s
packing), decrypts the hidden rows and applies the final norm + LM head itself, gated like run_prefill.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

from . import private_dir


def _work_dir():
    return private_dir(os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), "m12_llama"))


def _pack(model, toks):
    """Block 0's input slot vectors of a prompt (BOS + tokens), packed as page 0 of `prefill_paged`, and the chunks."""
    from perseus.impl import blocked as BL
    from perseus.impl.ccmm_paged import page_tokens
    from perseus.impl.prefill import Chunks
    prompt = list(toks[1:])
    n = page_tokens(0)
    chunk = prompt + [prompt[-1]] * (n - len(prompt))
    rows = np.stack([model.w.embedding(t) for t in chunk])
    ch = Chunks(len(chunk), model.blocks.t, base=1)
    Xs = BL.encode_blocked(model.rt, rows, ch, model.blocks)
    return [np.array(c.vec) for blk in Xs for c in blk], ch


# ── server ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _server(addr, a, model, toks, gt, ref, results=None):
    from perseus import _pc
    from perseus._paths import resolve
    from perseus.examples.llama import run_prefill as RP
    from perseus.examples.llama.planned import LlamaPlannedPrefill
    from perseus.impl import ccmm as CC
    from .server import serve_one
    cfg = RP._plan_config(a.plan)
    cfg["exec.use_graphs"] = bool(a.graphs)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    ptoks = [int(toks[0])] * len(toks)
    work = _work_dir()

    class Served(LlamaPlannedPrefill):
        bundle, ccmm_ref = None, None

        def open_session(self, kw):
            return _pc.Session.from_bundle(self.cfg, self.keys, self.bundle, **kw)

        def setup(self, toks):
            if not hasattr(self, "progs"):
                self.prepare(toks)
            if self.ccmm:
                self.ccmm_ref = self.throwaway_fit()
            return super().setup(toks)

        def throwaway_fit(self):
            """`ccmm_fit` on a session of the same config and programs under a fresh secret of the server's own
            (no PCMM keys), closed before the client's session opens. c_unit does not depend on the key (see
            `ccmm_fit`), so the client is never asked to decrypt anything (a decryption oracle: the server
            picks the ciphertexts)."""
            t0 = time.time()
            own = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
            sess = _pc.Session(self.cfg, own, **dict(self.session_kwargs(), pcmm=None))
            sess.complete_setup()
            cal = self.ccmm_fit(sess)
            del sess, own
            gc.collect()
            self.log(f"[ccmm] throwaway-key fit: c_unit {cal['c_unit']!r} (probe rel. err {cal['probe_rel_err']:.2e}), "
                     f"landing {cal['land']}, {time.time() - t0:.1f} s")
            return cal

        def calibrate_ccmm(self):
            """`LlamaPlannedPrefill.calibrate_ccmm` without a secret: nf_ref / nf_split from a token on the client's
            session (scales only, no decryption), c_unit and the landing from `throwaway_fit`."""
            t0 = time.time()
            ref, sess = self.ccmm_ref, self.sess
            tok = self.fhe.encrypt(np.zeros(len(self.in_vals[0]), dtype=np.complex128), int(self.in_metas[0][0][0]), 1)
            nf_ref, nf_split = sess.ccmm_prepare(tok)
            del tok
            if not np.allclose([nf_ref, nf_split], [ref["nf_ref"], ref["nf_split"]], rtol=1e-12, atol=0):
                raise RuntimeError(f"CCMM scales nf_ref / nf_split {nf_ref} / {nf_split}, the throwaway session's "
                                   f"{ref['nf_ref']} / {ref['nf_split']}")
            sess.ccmm_set(ref["c_unit"], *self.ccmm_kappa)
            land = ref["land"]
            fake_land = CC.fake_landing(getattr(getattr(self.model, "inf", None), "fhe", None))
            self.ccmm_cal = dict(ref, s=time.time() - t0)
            self.log(f"[ccmm] calibrated on the server: c_unit {ref['c_unit']:.6g}, landing {land}")
            if land[0] != fake_land:
                raise RuntimeError(f"CCMM lands at level {land[0]} on the device, the CPU twin recorded {fake_land}")

        def forward(self, x0):
            """`PlannedPrefill.run` on block-0 inputs already encrypted (the client's)."""
            info = []
            for b in range(self.n_layers):
                p = self.progs[self.labels[b]]
                ins = [x0[j] for j in self.in_pos[0]] if b == 0 else [[reg for _j, reg in self.reg[b - 1]][j] for j in self.in_pos[b]]
                info.append(p.run(ins, list(self.reg[b])))
            return info

    run = Served(model, resolve(a.plan), cfg, keys, work_dir=work, pt_rows=a.pt_rows, stc_cache=a.stc_cache,
                 check_meta=a.check_meta, log=lambda m: print(m, flush=True),
                 ccmm_kappa=[float(x) for x in a.ccmm_kappa.split(",")])

    class Handler:
        def manifest(self, hello):
            if int(hello["tokens"]) != len(toks):
                raise ValueError(f"client prompt of {hello['tokens']} tokens, the server twin was built for {len(toks)}")
            run.prepare(ptoks)
            spec = _pc.Session.key_manifest(run.cfg, run.keys, **run.session_kwargs())
            if json.loads(spec)["contract"] != hello["contract"]:
                raise ValueError(f"client plan contract {hello['contract']}, server {json.loads(spec)['contract']}")
            return {"spec": spec, "inputs": [list(m) for m in run.in_metas[0]], "warmup": bool(cfg["model.prefill.warmup"])}

        def load(self, path):
            run.bundle = path
            run.setup(ptoks)
            assert not run.sess.has_secret and not run.fhe.has_secret_key
            s, k = run.sess.setup_seconds()
            return {"session_s": round(s, 2), "import_s": round(k, 2), "keys": run.sess.n_keys(), "has_secret": run.sess.has_secret}

        def run(self, h, frames):
            t0 = time.time()
            x0 = [_pc.import_ct(run.sess, f) for f in frames]
            t1 = time.time()
            info = run.forward(x0)
            run.fhe.device_sync()
            wall = time.time() - t1
            del x0
            out = [] if h.get("warmup") else [_pc.export_ct(run.sess, run.ct(r)) for r in run.hid_reg[run.n_layers - 1]]
            return {"import_s": t1 - t0, "wall": wall, "export_s": time.time() - t1 - wall,
                    "host_s": sum(i["host_us"] for i in info) / 1e6, "served": "".join(i["served"][0] for i in info)}, out

    st = serve_one(addr, Handler(), os.path.join(work, "server"))
    print(f"[server] done: {st}; pcmm {run.sess.pcmm_stats()}", flush=True)
    return 0


# ── client ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _client(addr, out_path, a, model, toks, gt, ref, results=None):
    import types
    from perseus.client import Client, ClientSecret, Remote, contract_hash
    from perseus.examples.llama import hf
    from perseus.examples.llama import run_prefill as RP
    from perseus.examples.llama.prefill import plain_logits, report
    from perseus.impl.blocked import decrypt_blocked
    from perseus.impl.prefill import Chunks
    cfg = RP._plan_config(a.plan)
    cfg["exec.use_graphs"] = bool(a.graphs)
    t0 = time.time()
    cl = Client(ClientSecret.generate(), chain=cfg["chain.preset"] or "n32", complex=True)
    remote = Remote(addr, cl, os.path.join(_work_dir(), "client"))

    man = remote.handshake(contract_hash(cfg), tokens=len(toks))
    kg = remote.provide_keys()
    metas = man["inputs"]

    def frames_of(tk):
        vals, ch = _pack(model, tk)
        if len(vals) != len(metas):
            raise RuntimeError(f"{len(vals)} packed inputs, the server's block 0 takes {len(metas)}")
        return [cl.encrypt(v, int(lv), int(dg)) for v, (lv, dg) in zip(vals, metas)], ch

    if man.get("warmup"):
        wtoks = [int(t) for t in hf.text_windows(2, 512)[1][:len(toks)]]
        hw, _ = remote.run(frames_of(wtoks)[0], warmup=True)
        print(f"[client] warm-up forward (other prompt) server wall {hw['wall']:.2f}s", flush=True)
    te = time.time()
    frames, ch = frames_of(toks)
    enc_s = time.time() - te
    tr = time.time()
    h, outs = remote.run(frames)
    rt_s = time.time() - tr
    td = time.time()
    vals = [cl.decrypt(x)[0] for x in outs]
    shim = types.SimpleNamespace(ops=types.SimpleNamespace(decrypt_slots_complex=lambda v: v), dims=model.rt.dims)
    Bk = model.blocks
    hid = decrypt_blocked(shim, [vals[g * Bk.K:(g + 1) * Bk.K] for g in range(len(vals) // Bk.K)], ch, Bk,
                          model.rt.dims.dim)[:a.tokens]
    dec_s = time.time() - td
    # PF_LAST_QUERY: the last position alone is computed past the last block's K / V (as run_prefill reports)
    lo = len(hid) - 1 if model.last_position(Chunks(len(hid), model.blocks.t)) is not None else 0
    rows_, s = report(plain_logits(model, hid[lo:]), gt[1 + lo:1 + len(hid)])
    remote.close()
    summ = {"T": a.tokens, "server_wall": h["wall"], "server_host_s": h["host_s"], "served": h["served"],
            "import_s": h["import_s"], "export_s": h["export_s"], "encrypt_s": enc_s, "roundtrip_s": rt_s,
            "decrypt_s": dec_s, **s, "keygen": kg, "client_wall_s": time.time() - t0}
    print("[client] per-position KL " + " ".join(f"{r['kl']:.3g}" for r in rows_), flush=True)
    print(f"[client] SUMMARY T={a.tokens} server wall={h['wall']:.2f}s per_token={h['wall'] / a.tokens:.3f}s "
          f"top1={s.get('top1')}/{s['n']} medianKL={s.get('medianKL', float('nan')):.4g} maxKL={s.get('maxKL', float('nan')):.4g} "
          f"encrypt {enc_s:.2f}s roundtrip {rt_s:.2f}s decrypt+head {dec_s:.2f}s", flush=True)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(summ, f, indent=1, default=float)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sep = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=("server", "client"))
    ap.add_argument("--addr", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv[:sep])
    from perseus.examples.llama import run_prefill as RP
    if a.role == "server":
        RP._planned = lambda *x, **k: _server(a.addr, *x, **k)
    else:
        RP._planned = lambda *x, **k: _client(a.addr, a.out, *x, **k)
    return RP.main(argv[sep + 1:])


if __name__ == "__main__":
    sys.exit(main())
