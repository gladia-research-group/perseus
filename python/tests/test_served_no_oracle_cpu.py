"""A served client answers no server request: a malicious server that sends ciphertexts of its choice for the client
to decrypt (or fit, e.g. the old CCMM calibration's `ccmm_calibrate`) gets nothing back, the client ends the
session. The served servers calibrate on keys of their own (`serve.llama_prefill`)."""
from __future__ import annotations

import inspect
import os
import socket
import threading

import pytest

from perseus import serve
from perseus.client import Remote
from perseus.serve import server


def test_server_loop_has_no_request_channel():
    assert 'conn.send("ask"' not in inspect.getsource(server)
    for f in (Remote.provide_keys, Remote.run):
        assert "ask" not in inspect.signature(f).parameters


@pytest.mark.parametrize("phase", ["ready", "result"])
def test_client_refuses_a_server_request(tmp_path, phase):
    addr = str(tmp_path / "s.sock")
    srv = serve.listen(addr)
    got = {}

    def malicious():
        conn = serve.accept(srv)
        conn.recv("hello" if phase == "ready" else "run")
        conn.send("ask", what="ccmm_calibrate", blobs=[b"\0" * 64], f=0.125)
        conn.sock.settimeout(10)
        data = b""
        try:
            while chunk := conn.sock.recv(1 << 16):
                data += chunk
        except (socket.timeout, OSError):
            pass
        got["bytes"] = len(data)
        conn.close()

    th = threading.Thread(target=malicious)
    th.start()
    rem = Remote(addr, None, str(tmp_path / "c"), log=lambda m: None)
    with pytest.raises(RuntimeError, match="'ask'.*refused"):
        if phase == "ready":
            rem.conn.send("hello", contract="x")
            rem._wait("ready")
        else:
            rem.run([b"frame"])
    rem.conn.close()
    th.join()
    srv.close()
    assert got["bytes"] == 0
