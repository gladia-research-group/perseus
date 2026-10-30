"""Record 1 of the Encrypted GPT-2 Speedrun (Perseus on perseus-core): what its server (speedrun/server.py) and the
harness client's crypto (harness/perseus_core_client.py, run on this checkout) share. Keys and ciphertexts are
perseus-core's own (docs/WIRE.md); a list of byte strings is a u32 count, the u64 lengths, then the strings
(little-endian), as record 0's."""
from __future__ import annotations

import struct

PREFILL_PLAN = "bootstrap_placements/gpt2_prefill128"        # the shipped prefill plan: stages 12-13 are the tail
PROMPT = 128
NCH = 512                                    # the prefill's channel blocks (PF_NCH)
_COUNT, _LEN = struct.Struct("<I"), struct.Struct("<Q")


def pack(blobs) -> bytes:
    blobs = [bytes(b) for b in blobs]
    return _COUNT.pack(len(blobs)) + b"".join(_LEN.pack(len(b)) for b in blobs) + b"".join(blobs)


def unpack(payload: bytes, most: int = 64) -> list[bytes]:
    """pack's inverse; refuses a malformed payload (the client reads it from the server)."""
    if len(payload) < _COUNT.size:
        raise ValueError("record1.unpack: no payload header")
    (n,) = _COUNT.unpack_from(payload)
    off = _COUNT.size + n * _LEN.size
    if n > most or off > len(payload):
        raise ValueError("record1.unpack: bad payload header")
    lens = [_LEN.unpack_from(payload, _COUNT.size + i * _LEN.size)[0] for i in range(n)]
    if off + sum(lens) != len(payload):
        raise ValueError("record1.unpack: lengths do not match the payload")
    out = []
    for m in lens:
        out.append(payload[off:off + m])
        off += m
    return out
