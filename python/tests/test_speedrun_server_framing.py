"""speedrun/server.py's in-place framing, without Perseus or a GPU: `write_packed` writes exactly the file `write` writes
for `record1.pack`'s payload, and `unpacked` returns `record1.unpack`'s blobs (as views) and refuses what it refuses.
"""
import importlib
import os
import sys
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "speedrun" / "server.py").is_file():
    pytest.skip("no speedrun/server.py in this checkout", allow_module_level=True)
sys.path.insert(0, str(ROOT))
server = importlib.import_module("speedrun.server")
record1 = importlib.import_module("speedrun.record1")

BLOBS = [b"", b"\x00" * 7, os.urandom(4096), bytes(range(256)) * 3]


class Framing(unittest.TestCase):
    def test_write_packed_is_write_of_pack(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for blobs in ([], BLOBS[:1], BLOBS):
                server.write(d, "a/joined", record1.pack(blobs))
                server.write_packed(d, "a/packed", blobs)
                self.assertEqual(Path(d, "a/joined").read_bytes(), Path(d, "a/packed").read_bytes())
                self.assertEqual(server.read(d, "a/packed"), record1.pack(blobs))
                self.assertEqual(sorted(os.listdir(Path(d, "a"))), ["joined", "packed"])   # no temporary left

    def test_unpacked_is_unpack(self):
        for blobs in ([], BLOBS[:1], BLOBS):
            payload = record1.pack(blobs)
            views = server.unpacked(payload)
            self.assertEqual([bytes(v) for v in views], record1.unpack(payload))
            self.assertTrue(all(isinstance(v, memoryview) for v in views))

    def test_unpacked_refuses_what_unpack_refuses(self):
        payload = record1.pack(BLOBS)
        for bad in (b"", payload[:3], payload[:-1], payload + b"\x00", (65).to_bytes(4, "little") + payload[4:]):
            with self.assertRaises(ValueError):
                record1.unpack(bad)
            with self.assertRaises(ValueError):
                server.unpacked(bad)


if __name__ == "__main__":
    unittest.main()
