"""Place bootstraps on a captured operation graph.

  perseus-plan --graph-dir graphs/gpt2_decode_python_n32
               --out-dir bootstrap_placements/gpt2_decode_python_n32 [options]

Levels are given in the chain's own units; --level-unit says how many primes one level
costs (2 on the 32-bit composite chain, 1 on the 64-bit reference chain).
"""

from __future__ import annotations

import argparse
import logging

from perseus._log import configure_cli_logging

from .planner import PlanConfig, plan_graph_dir

# `python -m perseus.plan.placer` runs this module as `__main__`, outside the `perseus` logger
# tree configure_cli_logging attaches to, so the final summary line was silently dropped
# there; the `perseus-plan` console script imports it as perseus.plan.placer.__main__. Use the
# same name in both entry modes.
log = logging.getLogger("perseus.plan.placer.__main__" if __name__ == "__main__" else __name__)


def _csv_ints(s: str) -> tuple:
    return tuple(int(x) for x in s.split(",") if x)


def main() -> None:
    configure_cli_logging()
    p = argparse.ArgumentParser(prog="perseus-plan", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph-dir", required=True,
                   help="directory of block_<n>/graph.json captures (STAGE=capture writes it)")
    p.add_argument("--out-dir", required=True,
                   help="directory to write block_<n>_placement.json into")
    p.add_argument("--bootstrap-level", type=int,
                   help="level a refresh lands at")
    p.add_argument("--max-level", type=int,
                   help="top of the chain the plan may use")
    p.add_argument("--source-level", type=int,
                   help="level the block's input arrives at")
    p.add_argument("--cache-read-level", type=int,
                   help="level a KV-cache read arrives at")
    p.add_argument("--level-unit", type=int,
                   help="primes per level (2 on the 32-bit composite chain, 1 on the 64-bit one)")
    p.add_argument("--acc-chain", type=str,
                   help="measured accuracy table to price with, by chain name; '' = the analytic error model")
    p.add_argument("--acc-table", dest="acc_table_path", type=str,
                   help="explicit path to an accuracy table, instead of --acc-chain")
    p.add_argument("--cf-min", type=int,
                   help="smallest correction factor a site may be assigned")
    p.add_argument("--cf-max", type=int,
                   help="largest correction factor a site may be assigned")
    p.add_argument("--err-target", type=float,
                   help="per-site error the correction factor is chosen to meet")
    p.add_argument("--err-hopeless", type=float,
                   help="error above which a site is reported infeasible instead of placed")
    p.add_argument("--prescale-reach", type=float,
                   help="how far a prescale may reach when one is allowed")
    p.add_argument("--mag-safety", type=float,
                   help="margin kappa on every magnitude the graph recorded")
    p.add_argument("--slim-cubic", type=float, dest="slim_cubic",
                   help="StC-first order: price each site's sine cubic (2 pi m 2^-CF)^2/6 x this (1 = measured)")
    p.add_argument("--emit-offset", action="store_true", default=None,
                   help="emit the offset transform alongside each placement")
    p.add_argument("--no-prescale", action="store_false", default=None,
                   dest="allow_prescale",
                   help="refuse the input prescale entirely (band placement only)")
    p.add_argument("--miss-penalty", type=float,
                   help="cut cost charged for a site the accuracy table does not cover")
    p.add_argument("--prescale-bits-max", type=float,
                   help="max bits of restore amplification a prescale may spend")
    p.add_argument("--quality-weight", type=float,
                   help="the pre-score weight in the cut (0 = count-first, default)")
    p.add_argument("--depth-weight", type=float,
                   help="price each cut site by how many levels its refresh restores, blended "
                        "by this weight (0 = count-only, the default and what every shipped "
                        "plan uses)")
    p.add_argument("--depth-form", type=str, choices=["ratio", "linear", "ab", "ms"],
                   help="how --depth-weight prices a site: ratio = budget/restored, "
                        "linear = the levels not regained, ab = landing yield plus wasted runway")
    p.add_argument("--depth-a", type=float, help="ab form: landing-yield coefficient (default 1)")
    p.add_argument("--depth-b", type=float, help="ab form: early-refresh coefficient (default 1)")
    p.add_argument("--bts-ms", type=str,
                   help="measured bootstrap latency per route for --depth-form ms, "
                        "e.g. 0:28.77,512:18.59,1:16.10 (0 = dense; this is the 32-bit "
                        "chain, the 64-bit one is 0:36.69,512:23.59,1:19.95)")
    p.add_argument("--ms-discount", type=float,
                   help="how much of a sparse route's extra restored runway to credit back "
                        "(0 = pure milliseconds, 1 = milliseconds per unit of runway, default)")
    p.add_argument("--level-weight", type=float,
                   help="charge a refresh for the depth of its input: capacity is scaled by "
                        "1 + w x depth/envelope (0 = off, default)")
    p.add_argument("--sparse-slots", type=str,
                   help="runtime sparse precomps, e.g. 512,1")
    p.add_argument("--sparse-bts-out", type=str,
                   help="measured per-s output levels, e.g. 1:24,512:34")
    p.add_argument("--bts-out-deg", type=int, dest="bts_out_deg",
                   help="degree a refresh lands at: 1 realizes the pending rescale (default), 2 leaves it pending")
    p.add_argument("--deliberate-clamp0", action="store_true", default=None,
                   dest="deliberate_clamp0",
                   help="seed deliberate landings clamped at the bootstrap level "
                        "(no signed richer landings)")
    p.add_argument("--hint-veto-steps", dest="hint_veto_steps_csv", type=str,
                   help="comma list of step substrings whose hint sites are pinned "
                        "not to fire (e.g. softmax_v)")
    p.add_argument("--dissolve-hints", action="store_true", default=None,
                   dest="dissolve_hints",
                   help="pin every replaceable hint OFF and let the min-cut place the "
                        "refreshes, so no refresh depth is decided at runtime; hints no "
                        "placement can replace are retained and named in "
                        "summary.hints_retained. Not the default; implies the hard "
                        "refresh envelope (the cut becomes the sole refresher)")
    p.add_argument("--bind-hints", action="store_false", default=None,
                   dest="dissolve_hints",
                   help="explicitly bind each hint's threshold decision into the plan "
                        "(the default; kept so scripts can state it)")
    p.add_argument("--raise-drop-max", dest="raise_drop_max", type=int, default=None,
                   help="level-aware ModRaise: max composite levels a placed site may raise short of the chain top (0 = off)")
    p.add_argument("--raise-drop-landing-max", dest="raise_drop_landing_max", type=int, default=None,
                   help="deepest absolute landing for a raise-dropped site (default 44 = AUTO_BTS_LEVEL - unit)")
    p.add_argument("--raise-drop-routes", dest="raise_drop_routes", type=lambda s: tuple(int(x) for x in s.split(",") if x), default=None,
                   help="routes (slot counts, 0 = dense) that may take a raise drop; default 0,512")
    p.add_argument("--raise-drop-set", dest="raise_drop_set", type=lambda s: tuple(int(x) for x in s.split(",") if x), default=None,
                   help="the drops the runtime builds variants for (FIDESLIB_BTS_RAISE_DROPS), e.g. 1,3,5: a site's drop is rounded down to one of them")
    p.add_argument("--raise-drop-env-rule", dest="raise_drop_env_rule", type=str, default=None,
                   choices=["effective", "nominal"], help="envelope rule for the raise-drop slack")
    p.add_argument("--site-bts-out-file", dest="site_bts_out_file", type=str,
                   help="JSON {block_dir: {var: absolute_out_level}} of measured "
                        "per-site refresh landings (landing feedback)")
    p.add_argument("--rescale-opt", action="store_true", default=None,
                   dest="rescale_opt",
                   help="phase 2: search planted-realize anchors at deg-raising joins, "
                        "keep them when the re-cut plan needs no more bootstraps")
    p.add_argument("--boundary-realize", action="store_true", default=None,
                   dest="boundary_realize",
                   help="realize a deg-2 terminal exit so the next block enters deg-1 "
                        "(changes the plan; gated arms opt in)")
    p.add_argument("--forbid-steps", type=str,
                   help="comma-separated step substrings that may not carry a refresh")
    p.add_argument("--first-entry-level", type=int,
                   help="level the first refresh of a block enters at")
    p.add_argument("--first-entry-deg", type=int,
                   help="pending-rescale degree the first refresh enters at")
    p.add_argument("--placer", type=str,
                   help="min_cut (default), ilp (the same decision solved exactly as a "
                        "mixed-integer program; needs scipy) or a baseline: "
                        "orion|dacapo|fhelipe")
    p.add_argument("--ilp-time-limit", type=float,
                   help="placer ilp: solver time cap per block in seconds (default 300); "
                        "on timeout the best placement found is kept, with its gap")
    p.add_argument("--ilp-gap", type=float,
                   help="placer ilp: relative optimality gap at which the solver may stop "
                        "(default 0 = prove the optimum)")
    p.add_argument("--ilp-objective", type=str, choices=["total", "placed", "ms"],
                   help="placer ilp: minimise placed refreshes plus fired hints (total, "
                        "default), placed refreshes only (placed, the min-cut's objective), "
                        "or bootstrap milliseconds, each refresh at its route's --bts-ms "
                        "latency (ms)")
    p.add_argument("--ilp-free-exit", action="store_false", default=None,
                   dest="ilp_cap_exit",
                   help="placer ilp: let the final block exit deeper than the min-cut's plan "
                        "(off by default: that exit feeds the unplanned encrypted argmax)")
    p.add_argument("--prune", action="store_true", default=None,
                   help="remove redundant refreshes from the final plan, any placer")
    p.add_argument("--baseline-depth-cap", type=float, dest="baseline_depth_cap",
                   help="baseline placers: cap a placed refresh's input depth at this "
                        "absolute prime level (the paper's dense Fhelipe arm uses 48)")
    p.add_argument("--baseline-rescue", action="store_true", default=None,
                   dest="baseline_rescue",
                   help="repair magnitude-infeasible baseline plans with counted rescue "
                        "bootstraps instead of refusing (baselines only)")
    p.add_argument("--latency-table", dest="latency_table_path", type=str,
                   help="JSON per-op latency table for latency-objective baselines (dacapo)")
    args = vars(p.parse_args())

    graph_dir, out_dir = args.pop("graph_dir"), args.pop("out_dir")
    if args.get("sparse_slots") is not None:
        args["sparse_precomps"] = _csv_ints(args.pop("sparse_slots"))
    else:
        args.pop("sparse_slots", None)
    if args.get("sparse_bts_out") is not None:
        args["sparse_out_levels"] = tuple(
            (int(k), int(v)) for k, v in
            (pair.split(":") for pair in args.pop("sparse_bts_out").split(",") if pair))
    else:
        args.pop("sparse_bts_out", None)
    if args.get("bts_ms") is not None:
        args["bts_ms"] = tuple(
            (int(k), float(v)) for k, v in
            (pair.split(":") for pair in args.pop("bts_ms").split(",") if pair))
    else:
        args.pop("bts_ms", None)
    if args.get("hint_veto_steps_csv") is not None:
        args["hint_veto_steps"] = tuple(
            s for s in args.pop("hint_veto_steps_csv").split(",") if s)
    else:
        args.pop("hint_veto_steps_csv", None)
    if args.get("site_bts_out_file") is not None:
        import json
        args["site_bts_out"] = json.load(open(args.pop("site_bts_out_file")))
    else:
        args.pop("site_bts_out_file", None)
    if args.get("forbid_steps") is not None:
        args["forbid_steps"] = tuple(s for s in args.pop("forbid_steps").split(",") if s)
    else:
        args.pop("forbid_steps", None)

    cfg = PlanConfig(**{k: v for k, v in args.items() if v is not None})
    summaries = plan_graph_dir(graph_dir, out_dir, cfg)
    total = sum(s["total_bootstraps"] for s in summaries)
    # the run aggregate of the per-block `[plan] cf clamp:` lines (in memory only)
    reps = [s["cf_clamp"] for s in summaries if s.get("cf_clamp") is not None]
    n_clamped = sum(r.num_clamped for r in reps)
    n_sites = sum(r.num_sites for r in reps)
    log.info(f"[plan] planned {len(summaries)} blocks, {total} bootstraps, "
             f"cf clamp {n_clamped}/{n_sites} -> {out_dir}")


if __name__ == "__main__":
    main()
