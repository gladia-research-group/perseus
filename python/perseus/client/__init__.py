"""The client of a perseus server: it holds the secret and does everything that needs it, without a device.

`ClientSecret` / `Client` (pc-client through `_pc`): keygen of what the server's plan asks for (the server's
`Session.key_manifest`) into a key bundle, encryption of encoded inputs, decryption of outputs. `Remote` drives the
session protocol with a server (`perseus.serve`; docs/WIRE.md): hello -> manifest -> bundle -> ready -> run*.
A client process never opens a `Session`.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from perseus._pc import Client, ClientSecret, bundle_info, contract_hash

__all__ = ["Client", "ClientSecret", "Remote", "bundle_info", "contract_hash", "pack_token_rows"]


def pack_token_rows(rows, slots: int, hid: int, dim: int) -> np.ndarray:
    """Tokens `rows[tok]` (<= dim features each) at slots i * t + tok, t = slots / hid: the layout of
    `Fhe.pack_tokens` (real parts)."""
    t = slots // hid
    if len(rows) > t:
        raise ValueError(f"{len(rows)} tokens > {t} token lanes")
    v = np.zeros(slots, dtype=np.complex128)
    for tok, r in enumerate(rows):
        r = np.real(np.asarray(r, dtype=np.complex128))[:dim]
        v[np.arange(len(r)) * t + tok] = r
    return v


class Remote:
    """The client's end of a served session. `ask`: handler of the server's requests during a run
    (`ask(header, frames) -> (header, frames)`), e.g. decrypting calibration probes."""

    def __init__(self, addr: str, client: Client, work_dir: str, log=print):
        from perseus import serve
        self.client, self.work_dir, self.log = client, work_dir, log
        serve.private_dir(work_dir)
        self.conn = serve.connect(addr)
        self.spec = None
        self.stats = {}

    def handshake(self, contract: str, **hello) -> dict:
        """Send hello; receive the server's key request; check it is for our plan contract and chain."""
        self.conn.send("hello", contract=contract, params_hash=self.client.params_hash, **hello)
        _k, h, _ = self.conn.recv("manifest")
        spec = json.loads(h["spec"])
        if spec["contract"] != contract:
            raise RuntimeError(f"server runs plan contract {spec['contract']}, this client {contract}")
        if spec["params_hash"] != self.client.params_hash:
            raise RuntimeError(f"server chain {spec['params_hash']}, client chain {self.client.params_hash}")
        self.spec, self.manifest = h["spec"], h
        return h

    def provide_keys(self, threads: int = 8, ask=None) -> dict:
        """Generate the requested keys into a bundle file, send it, wait for the server's ready (answering its
        setup-time asks, e.g. a calibration, with `ask`)."""
        path = os.path.join(self.work_dir, "keys.pckb")
        t0 = time.time()
        st = dict(self.client.keygen(self.spec, path, threads=threads))
        st["wall_s"] = time.time() - t0
        st["info"] = bundle_info(path)
        self.log(f"[client] keygen {st['wall_s']:.1f} s ({int(st['keys'])} switching keys), bundle "
                 f"{st['info']['bytes'] / 2**30:.2f} GiB {st['info']['bytes_by_kind']}")
        t1 = time.time()
        self.conn.send_file("bundle", path)
        st["send_s"] = time.time() - t1
        st["ready"] = self._wait("ready", ask)[0]
        self.stats["keygen"] = st
        os.remove(path)
        return st

    def run(self, frames=(), ask=None, **meta):
        """One request; returns (header, frames) of the result (answering the server's asks on the way)."""
        self.conn.send("run", blobs=list(frames), **meta)
        return self._wait("result", ask)

    def _wait(self, want, ask):
        while True:
            kind, h, blobs = self.conn.recv()
            if kind == want:
                return h, blobs
            if kind == "ask":
                if ask is None:
                    raise RuntimeError(f"server asked {h.get('what')!r} and this client has no handler")
                rh, rb = ask(h, blobs)
                self.conn.send("answer", blobs=list(rb), **rh)
                continue
            raise RuntimeError(f"unexpected '{kind}' message while waiting for '{want}'")

    def close(self):
        try:
            self.conn.send("bye")
        finally:
            self.conn.close()
