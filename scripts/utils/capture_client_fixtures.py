#!/usr/bin/env python3
"""Capture / verify the perseus._client identity fixtures from a real perseus._core bundle.

    PERSEUS_CORE_BUNDLE=<dir> .venv/bin/python scripts/utils/capture_client_fixtures.py [--write] [--indexes]

The bundle is one written by _core's client role (scripts/utils/probe_client_server.py or
EncClient.save_bundle) under the n32 preset of scripts/local_env.sh with the default GPT-2
InferenceOptions. The script rebuilds the context on perseus._client under the same
environment (tests/test_client_ext.py::_N32_ENV) and compares:

  * context.bin      byte-for-byte against `_client._debug.bundle_meta`
  * context.bin.dev  text against the same call
  * rotkeys.bin      (--indexes; reads the whole multi-GB file) its automorphism index set
                     against `_client._debug.expected_automorphism_indexes`

With --write the matching artifacts are copied into tests/fixtures/client/ (context.bin,
context.bin.dev, options.json, rotkey_indexes.json). Nothing is written on a mismatch.
"""
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perseus import _client  # noqa: E402
from perseus._env import scoped_env  # noqa: E402
from perseus.nn.remote import options_to_manifest  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "client"

# scripts/local_env.sh's n32 block (the context-shaping subset) + the wrapper defaults the
# probe left untouched; AUTO_BTS_LEVEL and the FIDESLIB_* knobs never reach the context.
N32_ENV = {
    "LOGN": "16", "CKKS_DEPTH": "10", "BTP_DEPTH_OVERHEAD": "16", "SCALE_BITS": "54",
    "BTP_SCALE_BITS": "54", "FIRST_MOD_BITS": "56", "NUM_LARGE_DIGITS": "6",
    "LEVEL_BUDGET": "4:3", "CORRECTION_FACTOR": "6", "COMPOSITE_DEGREE": "2",
    "SPARSE_BTS_SLOTS": "512,1", "H_WEIGHT": "192", "BTS_ITERATIONS": "1",
    "BTS_PRECISION": "12", "AUTO_BTS_LEVEL": "46",
    # the bundle's context.bin reproduces only with the COMPLEX data type
    "CKKS_COMPLEX": "1", "SPARSE_LEVEL_BUDGET": None, "BTS_DIM1": None,
}


def options(env):
    with scoped_env(**env):
        opts = _client.InferenceOptions()
        opts.ckks = _client.CKKSOptions.from_env()
    return opts


def compare(bundle, env, label):
    opts = options(env)
    with scoped_env(**env):        # BTS_DIM1 is read at setup time
        ctx_bytes, dev_text = _client._debug.bundle_meta(opts, "gpt2")
    want_ctx = (bundle / "context.bin").read_bytes()
    want_dev = (bundle / "context.bin.dev").read_text()
    ok_ctx, ok_dev = ctx_bytes == want_ctx, dev_text == want_dev
    print(f"[{label}] context.bin {'MATCH' if ok_ctx else 'DIFF'} ({len(ctx_bytes)} vs {len(want_ctx)} B); "
          f"context.bin.dev {'MATCH' if ok_dev else 'DIFF'}")
    if not ok_ctx:
        first = next((i for i, (a, b) in enumerate(zip(ctx_bytes, want_ctx)) if a != b), min(len(ctx_bytes), len(want_ctx)))
        print(f"    first differing byte at {first}")
    if not ok_dev:
        print("    ours:", dev_text[:200].replace("\n", "\\n"))
        print("    want:", want_dev[:200].replace("\n", "\\n"))
    return ok_ctx and ok_dev, opts, ctx_bytes, dev_text


def main():
    bundle = Path(os.environ.get("PERSEUS_CORE_BUNDLE", ""))
    if not bundle.is_dir():
        sys.exit("set PERSEUS_CORE_BUNDLE to a bundle directory (context.bin, context.bin.dev, rotkeys.bin)")
    write = "--write" in sys.argv
    indexes = "--indexes" in sys.argv

    ok, opts, ctx_bytes, dev_text = compare(bundle, N32_ENV, "n32 env")
    if not ok:
        for label, delta in (("CKKS_COMPLEX unset", {"CKKS_COMPLEX": None}),
                             ("CORRECTION_FACTOR unset", {"CORRECTION_FACTOR": None})):
            ok, opts, ctx_bytes, dev_text = compare(bundle, {**N32_ENV, **delta}, label)
            if ok:
                print(f"NOTE: the bundle matches the '{label}' variant, not the documented env")
                break
    if not ok:
        sys.exit("no environment variant reproduces the bundle's context.bin: see the design's fallback "
                 "(precompute=true in src/client/client_context.cpp) or regenerate the fixtures on _core")

    manifest = options_to_manifest(opts, "gpt2")
    fixture_opts = {"family": "gpt2", "ckks": manifest["ckks"], "inference": manifest["inference"]}

    if write:
        FIXTURES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / "context.bin", FIXTURES / "context.bin")
        shutil.copyfile(bundle / "context.bin.dev", FIXTURES / "context.bin.dev")
        (FIXTURES / "options.json").write_text(json.dumps(fixture_opts, indent=1) + "\n")
        print(f"wrote {FIXTURES}/context.bin, context.bin.dev, options.json")

    if indexes:
        print(f"reading {bundle / 'rotkeys.bin'} ({os.path.getsize(bundle / 'rotkeys.bin') >> 30} GiB) ...", flush=True)
        got = _client._debug.automorphism_indexes_in_file(str(bundle / "rotkeys.bin"))
        tags = sorted(got)
        print(f"key tags: {tags}; sizes: {[len(got[t]) for t in tags]}")
        with scoped_env(**N32_ENV):
            expected = _client._debug.expected_automorphism_indexes(opts, "gpt2")
        for t in tags:
            same = list(got[t]) == list(expected)
            print(f"[{t}] index set {'MATCH' if same else 'DIFF'}: file {len(got[t])}, expected {len(expected)}")
            if not same:
                a, b = set(got[t]), set(expected)
                print("    only in file:", sorted(a - b)[:40])
                print("    only expected:", sorted(b - a)[:40])
        if write and len(tags) == 1:
            (FIXTURES / "rotkey_indexes.json").write_text(json.dumps(list(got[tags[0]])) + "\n")
            print(f"wrote {FIXTURES}/rotkey_indexes.json")


if __name__ == "__main__":
    main()
