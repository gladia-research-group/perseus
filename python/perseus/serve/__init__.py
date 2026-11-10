"""The server shim's transport: length-prefixed messages over a stream socket (Unix path or host:port).

A message is `u64 length | u32 header length | header (JSON: {"kind", ...}) | blobs`, the header listing the blob
sizes (`"blobs": [n0, n1, ...]`); blobs are ciphertext wire frames (`_pc.export_ct` / `Client.encrypt`) or raw
bytes. A key bundle travels once as one message whose single blob is streamed from / to a file. The session
protocol (hello -> manifest -> bundle -> ready -> run* -> bye, and the server's `ask` requests back to the client)
is in docs/WIRE.md.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import struct

CHUNK = 1 << 24
# The largest message `recv` holds in memory (headers and ciphertext frames; a key bundle streams to a file).
MAX_MESSAGE = 1 << 34


class ProtocolError(RuntimeError):
    pass


def _addr(addr: str):
    if ":" in addr and not addr.startswith("/"):
        host, port = addr.rsplit(":", 1)
        return socket.AF_INET, (host, int(port))
    return socket.AF_UNIX, addr


class Conn:
    """One connected peer: `send(kind, blobs=(), **header)`, `recv() -> (kind, header, blobs)`."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.sent = self.received = 0

    def _read(self, n: int) -> bytes:
        if n > MAX_MESSAGE:
            raise ProtocolError(f"peer declares {n} bytes, above the {MAX_MESSAGE} of a message held in memory")
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            k = self.sock.recv_into(view[got:], min(n - got, CHUNK))
            if k == 0:
                raise ProtocolError(f"peer closed after {got} of {n} bytes")
            got += k
        self.received += n
        return bytes(buf)

    def send(self, kind: str, blobs=(), **header):
        header = {"kind": kind, **header, "blobs": [len(b) for b in blobs]}
        h = json.dumps(header).encode()
        total = 4 + len(h) + sum(len(b) for b in blobs)
        self.sock.sendall(struct.pack("<QI", total, len(h)) + h)
        for b in blobs:
            self.sock.sendall(b)
        self.sent += 12 + len(h) + sum(len(b) for b in blobs)

    def recv(self, expect: str | None = None):
        total, hl = struct.unpack("<QI", self._read(12))
        header = json.loads(self._read(hl))
        sizes = header.pop("blobs")
        if total != 4 + hl + sum(sizes) or total > MAX_MESSAGE:
            raise ProtocolError(f"message length {total} does not match its header, or is above {MAX_MESSAGE}")
        blobs = [self._read(n) for n in sizes]
        kind = header.pop("kind")
        if kind == "error":
            raise ProtocolError(f"peer error: {header.get('message')}")
        if expect is not None and kind != expect:
            raise ProtocolError(f"expected a '{expect}' message, got '{kind}'")
        return kind, header, blobs

    def send_file(self, kind: str, path: str, **header):
        """One message whose single blob is the file at `path`, streamed."""
        size = os.path.getsize(path)
        header = {"kind": kind, **header, "blobs": [size]}
        h = json.dumps(header).encode()
        self.sock.sendall(struct.pack("<QI", 4 + len(h) + size, len(h)) + h)
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK):
                self.sock.sendall(chunk)
        self.sent += 12 + len(h) + size

    def recv_file(self, path: str, expect: str):
        """Receive a one-blob message into the file at `path`; returns its header."""
        total, hl = struct.unpack("<QI", self._read(12))
        header = json.loads(self._read(hl))
        kind, sizes = header.pop("kind"), header.pop("blobs")
        if kind == "error":
            raise ProtocolError(f"peer error: {header.get('message')}")
        if kind != expect or len(sizes) != 1 or total != 4 + hl + sizes[0]:
            raise ProtocolError(f"expected a one-blob '{expect}' message, got '{kind}' {sizes}")
        left = sizes[0]
        if left > shutil.disk_usage(os.path.dirname(path)).free // 2:
            raise ProtocolError(f"'{kind}' of {left} bytes is more than half the free space under {path}")
        with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600), "wb") as f:
            while left:
                chunk = self.sock.recv(min(left, CHUNK))
                if not chunk:
                    raise ProtocolError(f"peer closed with {left} bundle bytes missing")
                f.write(chunk)
                left -= len(chunk)
        self.received += 12 + hl + sizes[0]
        return header

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def private_dir(path: str) -> str:
    """`path` as this user's own directory, mode 0700: a work directory under a shared /tmp must not be one
    another user made, or a link to somewhere else."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise PermissionError(f"work directory {path} is not this user's own directory")
    os.chmod(path, 0o700)
    return path


def listen(addr: str) -> socket.socket:
    fam, a = _addr(addr)
    if fam == socket.AF_UNIX and os.path.exists(a):
        os.unlink(a)
    s = socket.socket(fam, socket.SOCK_STREAM)
    if fam == socket.AF_INET:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(a)
    s.listen(1)
    return s


def accept(srv: socket.socket) -> Conn:
    c, _ = srv.accept()
    return Conn(c)


def connect(addr: str, timeout: float = 600.0) -> Conn:
    """Connect, retrying while the server is still starting (up to `timeout` s)."""
    import time
    fam, a = _addr(addr)
    t0 = time.time()
    while True:
        s = socket.socket(fam, socket.SOCK_STREAM)
        try:
            s.connect(a)
            return Conn(s)
        except (FileNotFoundError, ConnectionRefusedError):
            s.close()
            if time.time() - t0 > timeout:
                raise
            time.sleep(1.0)
