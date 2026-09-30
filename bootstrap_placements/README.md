# Bootstrap plans

Every directory holds one plan per transformer block (`block_N_placement.json`) and a
`PLAN_CMD.txt` with the exact recipe (graph, planner flags, error model) that
regenerates it. All plans are for GPT-2 decode, 12 blocks, the LM head and the encrypted argmax,
on the captured graphs under `graphs/`. `bash scripts/make_plans.sh [<dir>...]` replays the
recipes; `tests/test_paper_plans.py` re-plans each directory from its graph and checks the
result is identical.

The Python implementation (the default), from `graphs/gpt2_decode_python_{n32,n64}`. Perseus and
Fhelipe are planned with `examples/gpt2_from_primitives/make_plan.sh` (blocks, then the argmax
stage), pruned of redundant refreshes (`PLAN_PRUNE=1`, the script's default); DaCapo and Orion
are the released tools' site selections on every block (see below), unpruned. `planned`
excludes the 23 deliberate bootstraps per token the code itself runs; `executed` includes them.

| directory | arm | planned | executed / token |
|---|---|---|---|
| `gpt2_decode_python_n32` | Perseus 32-bit | 393 | 416 |
| `gpt2_decode_python_n64` | Perseus 64-bit | 384 | 407 |
| `python/dacapo` | DaCapo (hecate-opt), 48 levels | 595 | 618 |
| `python/orion` | Orion (released solver, seed 0), 48 levels, 68 refreshes repaired | 871 | 894 |
| `python/fhelipe` | Fhelipe (our port), 48 levels, repaired, pruned | 561 | 584 |
| `gpt2_decode_python_n32_dense` | Perseus 32-bit, dense only | 465 | 488 |
| `gpt2_decode_python_n64_dense` | Perseus 64-bit, dense only | 473 | 496 |
| `python/dacapo_dense` | DaCapo, dense only, 48 levels | 609 | 632 |
| `python/orion_dense` | Orion, dense only, 50 levels (infeasible at 48) | 816 | 839 |
| `python/fhelipe_dense` | Fhelipe, dense only, 50 levels with the 48 depth cap | 754 | 777 |
| `python/ablations/cf_fixed_{7,8,9,10}` | Perseus 32-bit, one fixed correction factor | 390–396 | 413–419 |
| `python/ablations/kappa_{1,...,128}` | Perseus 32-bit, margin κ (κ = 2 is the main plan) | 393 each | 416 |

A dense plan (`PLAN_SPARSE_SLOTS=`) runs with `SPARSE_AUTO=0 SPARSE_BTS_SLOTS=0`, which its
contract stamp demands.

Notes.
* The 32-bit plans are priced with the measured accuracy table; the 64-bit plans with the
  analytic error model (`PLAN_ACC_CHAIN=`).
* The 64-bit plans are planned with automatic sparse routing on (`SPARSE_AUTO=2`, which needs
  `CORRECTION_FACTOR=7` on that chain), which is the shipped configuration.
* A dense plan lands the code's own sparse-routed bootstraps (the encrypted argmax's) at the
  bootstrap level, as a dense run does (`PLAN_DELIBERATE_CLAMP0`, set by `make_plan.sh` and the
  upstream tools for a dense plan).

`python/dacapo` comes from the released DaCapo compiler
([corelab-src/dacapo](https://github.com/corelab-src/dacapo) at `4616402`), run on every block by
`scripts/utils/dacapo_upstream/`:

```bash
bash scripts/utils/dacapo_upstream/build_hecate.sh      # hecate-opt into .cache/dacapo_upstream (LLVM/MLIR 18, SEAL 4.0)
PLAN_PRUNE=0 bash scripts/utils/dacapo_upstream/plan.sh graphs/gpt2_decode_python_n32 python/dacapo   # add `dense` for the dense route
```

`dacapo_4616402.patch` only adds a `--dacapo-plan` pipeline (hecate's own RemoveBootstrap,
BypassDetection, CandidateSelection, DaCapoPlanner, BootstrapPlacement and ProactiveRescaling
passes, without lowering) and prints the planner's bootstrap targets. `translate.py` writes each
block as earth-dialect MLIR, a forced (non-multiply) rescale of the planner's IR as a
multiply by a constant so hecate sees the same level structure; `ckks_ml48.json` is hecate's
default config with the bootstrap level bounds set to the deployed chain (5 multiplies between
refreshes). `sites.py` maps the targets back to the graph's variables, `deploy.py` replays them
through the planner (level bookkeeping, the block-exit refresh the tool assumes, repair) and
stamps the capture contract; the argmax stage is planned the way `make_plan.sh` plans it.

`python/orion` comes from the released Orion bootstrap solver
([baahl-nyu/orion](https://github.com/baahl-nyu/orion) at `be8a827`), run on each block's step
graph by `scripts/utils/orion_upstream/`:

```bash
PLAN_PRUNE=0 bash scripts/utils/orion_upstream/plan.sh graphs/gpt2_decode_python_n32 python/orion   # add `dense` for the dense route
```

The solver marks sites in its own 50-level frame (it cannot express a block input with no
level left, which is how a block arrives at 48); the deploy and the argmax stage run at `ML`
(48 unless exported). Deployed at 50, some seeds place a refresh at level 50, past the measured
bootstrap envelope, which the runtime refuses. It clones upstream into `.cache/orion_upstream` (or uses `ORION_SRC`) and applies
`orion_be8a827.patch`, which touches only what stops the solver on transformer graphs: two
enumerations of every simple path that never finish, replaced by exact equivalents (a path count
by dynamic programming, the region subgraph as descendants ∩ ancestors), and four crashes
(forks sharing a join, three-way branches, overlapping regions, which are priced as plain layers,
and nodes off the representative paths, which are left unlevelled and never bootstrapped). `marks.py` feeds the solver our step graph and
records the steps it marks, on every block including the argmax stage; `deploy.py` replays them
through the planner (level bookkeeping, the block-exit refresh the solver assumes, repair) and
stamps the capture contract.
The solver iterates Python sets of node names, so its marks depend on the hash seed; the
recipe pins `PYTHONHASHSEED=0`. Both tools prune the replayed plan unless `PLAN_PRUNE=0`; the
shipped DaCapo and Orion plans are the unpruned ones.
