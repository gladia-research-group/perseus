"""Generate `crates/pc-config/src/fields.rs` from `docs/ENV_INVENTORY.md` (tables 1 + 2, 5a/5b, 6a/6b, 7a, 8a).

    python3 crates/pc-config/tools/gen_fields.py > crates/pc-config/src/fields.rs

One field per inventory row, named as the inventory proposes (`group.field`, `model.prefill.*`, `exec.debug.*`,
`plan.*` -> `PlanConfig`), typed from the inventory's type column (first reading wins; the rows the scan could not
type are typed below by hand), documented with the env name, where it is read, its default(s) and the shipped plans
that stamp it. Rows whose proposal is "(stays env ...)" / "(drop ...)" / "(old upstream ...)" get no field.
Fields with no env name are written by hand in `config.rs` (marked `env: ""`).
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# rows the AST scan could not type (table 2: set only, read by the C++), from the values the port sets / plans stamp
MANUAL_KIND = {
    "CUTMAX_SPARSE_BTS": "Flag", "FHE_STAGE_ARENA_GB": "Float", "FIDESLIB_BTS_RAISE_DROP": "Str",
    "FIDESLIB_BTS_RAISE_DROPS": "List", "FIDESLIB_BTS_SHIFT": "Flag", "FIDESLIB_BTS_SPARSE_B": "Flag",
    "FIDESLIB_BTS_SPARSE_B_MAX": "Int", "FIDESLIB_BTS_STC_FIRST_CF": "Int", "FIDESLIB_GPU_ENCODE": "Flag",
    "FIDESLIB_ROT_KEY_BAND": "Int", "FIDESLIB_SPRU": "Int", "GPT2_INFERENCE_MODE": "Str", "LLAMA_PACKING": "Str",
    "MALLOC_ARENA_MAX": "Int", "PLAN_BTS_LEVEL": "Int", "PLAN_CACHE_READ_LEVEL": "Int", "PLAN_MAX_LEVEL": "Int",
    "PLAN_SRC_LEVEL": "Int", "SPARSE_LEVEL_BUDGET": "List", "SPARSE_LN_BTS": "Flag", "SPARSE_SM_BTS": "Flag",
    # mixed readings resolved: an enum of ints, a path, an optional path, a flag compared as a string
    "SM_PERIODIC": "Int", "FHE_GRAPH_DIR": "Str", "PF_PCMM_CALIB": "Str", "PF_PCMM_CALIB_IN": "Str",
    "PF_PCMM_SHIFT": "Int", "PLAN_HINT_ENV_VETO": "Flag",
    # contract-only names (compared verbatim as strings), typed from every value the 39 shipped plans stamp
    "BTS_ITERATIONS": "Int", "CACHE_READ_LEVEL_K": "Int", "CACHE_READ_LEVEL_V": "Int", "CKKS_COMPLEX": "Flag",
    "CORRECTION_FACTOR": "Int", "FHE_LMHEAD_CAP": "Int", "FHE_PT_COEFF_ENCODE": "Flag", "GPT2_CACHE": "Flag",
    "GPT2_FOLD_LN1": "Flag", "GPT2_FOLD_LN2": "Flag", "GPT2_FOLD_LNF": "Flag", "LOGN": "Int",
    # 5b / 6b (set only, read by the perseus C++ / FIDESlib), from the values the plans / drivers set
    "FIDESLIB_CCMM_ASYNC": "Flag", "FIDESLIB_DISCARD_SCRATCH": "Flag", "FIDESLIB_ENCODE_BARRIER": "Flag",
    "FIDESLIB_MONOMIAL_ASYNC": "Flag", "FIDESLIB_PCMM_ASYNC": "Flag", "FIDESLIB_STC_CACHE_ASYNC": "Flag",
    "FIDESLIB_STC_CACHE_POOLFREE": "Flag", "FIDESLIB_STC_HIGH_ASYNC": "Flag", "FIDESLIB_STC_LAST_CACHE_FREE_GB": "Float",
    "IMPL_WSUM": "Flag", "FIDESLIB_BTS_RAISE_NOSTC": "List", "FIDESLIB_BTS_RAISE_STC_ONUSE": "Flag",
    "FIDESLIB_CTS_ALT_BUDGET": "Int", "FIDESLIB_DUMP_FULL": "Flag", "FIDESLIB_ENCODE_NOSYNC": "Flag",
    "FIDESLIB_ENCODE_PINNED_RING": "Int", "FIDESLIB_PCMM_GEMM_MB": "Int", "FIDESLIB_PCMM_NOSYNC": "Flag",
    "FIDESLIB_PCMM_OUT_CHUNK": "Int", "FIDESLIB_PINNED_ANON": "Flag", "FIDESLIB_STC_CACHE_RELEASE": "Flag",
    "FIDESLIB_STC_POW2_KEY": "Flag", "PF_MASK_HOST_CACHE": "Flag", "PF_WARMUP": "Flag",
    # 5a / 6a readings the scan typed loosely
    "PF_PCMM_RAISE_UP": "Str", "VIT_TAPS": "Int", "GPT2_PF_WARMUP": "Str",
}
# two env names the inventory maps onto one planner field: the profile.py spellings keep their prefix
RENAME = {"PLAN_CACHE_READ_LEVEL": "plan.plan_cache_read_level", "PLAN_MAX_LEVEL": "plan.plan_max_level"}
KIND_OF = {"flag": "Flag", "bool": "Flag", "int": "Int", "float": "Float", "list": "List", "str": "Str",
           "enum": "Str"}
# struct per group path, in output order
STRUCTS = {"chain": "ChainSection", "bts": "BtsSection", "mem": "MemSection", "host": "HostSection",
           "model": "ModelSection", "model.prefill": "PrefillSection", "model.vit": "VitSection", "quirks": "QuirksSection",
           "exec": "ExecSection", "exec.debug": "DebugSection", "plan": "PlanConfig"}
# fields with no legacy env (env name ""): what perseus-core configures that perseus hard-coded; a fifth element is the
# env name of a perseus-core runtime switch (settable from a plan's runtime= line like the legacy ones)
EXTRA = [
    ("quirks", "shadow", "List", "pc-shadow `Quirk` names switched on, replacing the mode's preset (`Quirks::planner()` / "
     "`fake()` / `fakecap()`); retiring a quirk = dropping it here and re-planning (docs/QUIRKS.md)."),
    ("quirks", "fake_bts_out", "Int", "The numpy fake's refresh landing level (perseus `FakeFhe(bts_out=34)` argument; "
     "every route lands there realized, quirk `FakeBootstrapRealized`)."),
    ("bts", "k_max", "Int", "Ciphertexts per batched bootstrap launch (PLAN.md Program: K-batching; 8)."),
    ("bts", "em_fuse_rescale", "Flag", "EvalMod: a Regen relinearization and the rescale after it as one exact conversion "
     "(pc-gpu `BtsSetup::em_fuse_rescale`; default off = relinearize, then rescale).", "PC_EM_FUSE_RESCALE"),
    ("bts", "lt_tails", "Flag", "Resident key `a` tails for the CtS baby keys of packed key sets (pc-gpu "
     "`BtsSetup::lt_tails`; default on, off = those stages regenerate every digit).", "PC_LT_TAILS"),
    ("exec", "use_graphs", "Flag", "Replay a bound block program's static segments as captured CUDA graphs after its "
     "first run (pc-program `ReplayCfg::use_graphs`; default on, off = eager every run)."),
    ("exec", "trace_alloc", "Flag", "Report device free memory and arena use around every planned token (the "
     "'0 allocations after setup' check; default off)."),
    ("host", "n_in_flight", "Int", "Weight-ring frames staged ahead of the running block (pc-program `RingCfg::"
     "n_in_flight`; default 2)."),
]
SUBS = {"model": [("prefill", "PrefillSection"), ("vit", "VitSection")], "exec": [("debug", "DebugSection")]}


def rows():
    """Table-1-format rows (sections 1, 5a, 6a: reads in the port) as sec 1, table-2-format rows (2, 5b, 6b: set only)
    as sec 2."""
    sec = None
    for line in (ROOT / "docs/ENV_INVENTORY.md").read_text().splitlines():
        if line.startswith(("## 1.", "### 5a.", "### 6a.", "### 7a.", "### 8a.")):
            sec = 1
        elif line.startswith(("## 2.", "### 5b.", "### 6b.")):
            sec = 2
        elif line.startswith(("## 3.", "## 4.", "### 5c.", "### 6c.")):
            sec = 3
        if sec in (1, 2) and re.match(r"\| \d+ \|", line):
            c = [x.strip() for x in line.strip().strip("|").split(" | ")]
            if sec == 1:
                _, name, at, default, typ, plans, field = c
            else:
                _, name, at, plans, field = c
                default, typ = "(unset)", ""
            yield dict(name=name.strip("`"), at=at, default=default, type=typ, plans=plans, field=field.strip("`"))


def doc_lines(r):
    if r.get("extra"):
        return [r["doc"].replace('"', "'"), f"Runtime switch `{r['name']}`." if r["name"] else "No legacy env name."]
    if not r["name"]:
        return [r["doc"].replace('"', "'"), "No legacy env name."]
    at = r["at"].replace("<br>", "; ")
    plans = r["plans"] if r["plans"] != "—" else "none"
    out = [f"`{r['name']}`. Default {r['default'].replace('`', '')}. Read at {at}.", f"Shipped plans: {plans}."]
    return [re.sub(r"\s+", " ", x).replace('"', "'") for x in out]


def main():
    groups = collections.OrderedDict((g, []) for g in STRUCTS)
    for r in rows():
        field = RENAME.get(r["name"], r["field"].split(" ")[0])
        if field.startswith("("):
            continue
        path, name = field.rsplit(".", 1)
        kind = MANUAL_KIND.get(r["name"]) or KIND_OF[r["type"].split("/")[0].split(" ")[0]]
        groups[path].append((name, kind, r))
    for path, name, kind, doc, *env in EXTRA:
        groups[path].append((name, kind, dict(name=env[0] if env else "", doc=doc, default="(unset)", extra=True)))
    w = sys.stdout.write
    w("// @generated by crates/pc-config/tools/gen_fields.py from docs/ENV_INVENTORY.md -- edit the inventory or\n"
      "// the generator, not this file.\n")
    w("// The field docs quote the inventory's source locations verbatim (`[contract]`, `<module>`), not Markdown.\n"
      "#![allow(rustdoc::broken_intra_doc_links, rustdoc::invalid_html_tags)]\n")
    w("use crate::section;\n\n")
    for path, fields in groups.items():
        s = STRUCTS[path]
        w(f"section! {{\n    /// `{path}` ({len(fields)} legacy env names).\n    {s} \"{path}\" {{\n")
        for name, kind, r in sorted(fields, key=lambda x: x[0]):
            for d in doc_lines(r):
                w(f"        #[doc = \"{d}\"]\n")
            dflt = r["default"].replace("`", "").replace('"', "'")
            w(f"        {name}: {kind} = \"{r['name']}\", \"{dflt}\";\n")
        w("    }")
        if path in SUBS:
            w(" subs {\n")
            for f, t in SUBS[path]:
                w(f"        {f}: {t};\n")
            w("    }")
        w("\n}\n\n")
    n = {p: len(f) for p, f in groups.items()}
    sys.stderr.write(f"fields per section: {n}; Config {sum(v for k, v in n.items() if k != 'plan')}, "
                     f"PlanConfig {n['plan']}\n")


if __name__ == "__main__":
    main()
