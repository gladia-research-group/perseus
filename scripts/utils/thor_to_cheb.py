#!/usr/bin/env python
"""thor_to_cheb.py — CLI shim; the implementation lives in perseus.calibrate.thor_to_cheb.

Usage (unchanged):
  .venv/bin/python scripts/utils/thor_to_cheb.py SRC.json DST.json [--p1-domain M]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from perseus.calibrate.thor_to_cheb import main  # noqa: E402

if __name__ == "__main__":
    main()
