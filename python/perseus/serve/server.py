"""The server loop: one client, one secret-free session (docs/WIRE.md, session protocol).

    hello (client: plan contract, chain, run shape) -> manifest (server: `Session.key_manifest` + the input layout)
    bundle (client: the key bundle, once) -> ready (server: `Session.from_bundle`, programs bound)
    run (client: input frames) [-> ask / answer ...] -> result (server: output frames)   (repeated)
    bye

A `Handler` supplies the model: `manifest(hello) -> dict` (with "spec"), `load(bundle_path, ask) -> dict`,
`run(header, frames, ask) -> (dict, frames)`; `ask(what, frames, **header) -> (header, frames)` sends a request
back to the client (e.g. decrypting calibration probes) and waits for its answer.
"""
from __future__ import annotations

import os
import time
import traceback

from . import accept, listen, private_dir


def serve_one(addr: str, handler, work_dir: str, log=print) -> dict:
    """Serve one client session on `addr` to its bye; returns the transport counters."""
    private_dir(work_dir)
    srv = listen(addr)
    log(f"[serve] listening on {addr}")
    conn = accept(srv)
    try:
        _k, hello, _ = conn.recv("hello")
        log(f"[serve] hello {hello}")
        conn.send("manifest", **handler.manifest(hello))
        path = os.path.join(work_dir, "keys.pckb")
        t0 = time.time()
        conn.recv_file(path, "bundle")
        recv_s = time.time() - t0

        def ask(what, frames=(), **h):
            conn.send("ask", what=what, blobs=list(frames), **h)
            _k, ah, ab = conn.recv("answer")
            return ah, ab

        t1 = time.time()
        ready = handler.load(path, ask)
        os.remove(path)
        ready.update(bundle_recv_s=round(recv_s, 2), load_s=round(time.time() - t1, 2))
        log(f"[serve] ready {ready}")
        conn.send("ready", **ready)

        while True:
            kind, h, blobs = conn.recv()
            if kind == "bye":
                break
            if kind != "run":
                raise RuntimeError(f"unexpected '{kind}' message")
            rh, rb = handler.run(h, blobs, ask)
            conn.send("result", blobs=list(rb), **rh)
        return {"sent": conn.sent, "received": conn.received}
    except Exception as e:
        try:
            conn.send("error", message=f"{type(e).__name__}: {e}")
        except OSError:
            pass
        traceback.print_exc()
        raise
    finally:
        conn.close()
        srv.close()
