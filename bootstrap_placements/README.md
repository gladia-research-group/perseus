# Bootstrap plans

A plan says where an encrypted model refreshes its values (bootstraps) and how. Every directory
here holds one plan file per model stage, `block_N_placement.json` (blocks 0-11 are GPT-2's
transformer blocks, 12 is the final LayerNorm and output layer, 13 is the encrypted argmax),
and a `PLAN_CMD.txt` with the exact recipe that regenerates it: the recorded forward pass it was
computed from (under `graphs/`), the planner settings and the error model. `bash
scripts/make_plans.sh [<dir>...]` replays the recipes, and `tests/test_paper_plans.py` replans
every directory and checks that the result is identical to the committed files.

All plans are for encrypted GPT-2 small on the Python implementation, computed from
`graphs/gpt2_decode_python_{n32,n64}` (n32: 32-bit parameters, n64: 64-bit). Perseus and
Fhelipe are planned with `examples/gpt2_from_primitives/make_plan.sh` (the blocks first, then
the encrypted argmax), which also removes bootstraps that turn out redundant (`PLAN_PRUNE=1`,
the default). DaCapo and Orion are planned by their released tools (see below) and left as the
tools output them. *Planned* counts the bootstraps a plan places; *executed* also counts the 23
per token that the model code performs itself, the same for every system.

| directory | system | planned | executed / token |
|---|---|---|---|
| `gpt2_decode_python_n32` | Perseus, 32-bit | 393 | 416 |
| `gpt2_decode_python_n64` | Perseus, 64-bit | 384 | 407 |
| `python/dacapo` | DaCapo (its compiler, `hecate-opt`), level budget 48 | 595 | 618 |
| `python/orion` | Orion (its released solver), level budget 48, 68 bootstraps added by our repair step | 871 | 894 |
| `python/fhelipe` | Fhelipe (our reimplementation), level budget 48, repaired, redundant bootstraps removed | 561 | 584 |
| `gpt2_decode_python_n32_dense` | Perseus, 32-bit, full-size bootstraps only | 465 | 488 |
| `gpt2_decode_python_n64_dense` | Perseus, 64-bit, full-size bootstraps only | 473 | 496 |
| `python/dacapo_dense` | DaCapo, full-size bootstraps only, level budget 48 | 609 | 632 |
| `python/orion_dense` | Orion, full-size bootstraps only, level budget 50 (no plan exists at 48) | 816 | 839 |
| `python/fhelipe_dense` | Fhelipe, full-size bootstraps only, level budget 50 with its depth limit at 48 | 754 | 777 |
| `python/ablations/cf_fixed_{7,8,9,10}` | Perseus, 32-bit, one fixed scaling setting (correction factor) for every bootstrap | 390–396 | 413–419 |
| `python/ablations/kappa_{1,...,128}` | Perseus, 32-bit, safety margin κ on recorded magnitudes (the main plan uses κ = 2) | 393 each | 416 |

Some bootstraps only need to refresh a few distinct values, because the data repeats across the
ciphertext; the runtime has a cheaper bootstrap for those, and every plan uses it except the
*full-size bootstraps only* ones. Those run with `SPARSE_AUTO=0 SPARSE_BTS_SLOTS=0`. Each plan
records the runtime settings it was computed for and refuses to run under different ones.

In generation, the next token re-enters block 0 as the output of a bootstrap rather than as a
fresh encryption, so block 0 needs its own plan for those tokens:
`block_0_feedback_placement.json`, block 0 planned again the same way from that starting point
(`make_plan.sh ... feedback`; the DaCapo and Orion scripts rerun the tool's block-0 choices). Every
plan above has one except the ablations and `python/orion_dense`: on full-size bootstraps,
Orion's choices cannot fit block 0's first LayerNorm in the level budget from that starting
point, even after repair, so generation refuses that plan.

The HEAT GPT-2 ([`gladia/heat-gpt2-small-openwebtext`](https://huggingface.co/gladia/heat-gpt2-small-openwebtext),
approximations in `configs/model/approximation/gpt2_heat_n32`) has two plans, both computed from
`graphs/gpt2_heat_decode_python_n32` on the 32-bit parameters with the cheaper bootstraps and
redundant ones removed, and both with the block-0 plan for generation. They run with the HEAT
weights, config and reference outputs (README, step 1).

| directory | placement algorithm | bootstraps / token |
|---|---|---|
| `gpt2_heat_decode_python_n32_ilp` | exact (`--placer ilp`) | 259 |
| `gpt2_heat_decode_python_n32` | minimum cut (the default) | 267 |

Notes.
* The 32-bit plans estimate each bootstrap's error from measurements; the 64-bit plans use a
  formula (`PLAN_ACC_CHAIN=`).
* The 64-bit plans use the cheaper bootstrap automatically wherever it applies (`SPARSE_AUTO=2`,
  which needs `CORRECTION_FACTOR=7` on the 64-bit parameters), as the default configuration does.
* In a *full-size bootstraps only* plan, the few bootstraps the model code performs itself (in the
  encrypted argmax) are also full-size, as they are in such a run (`PLAN_DELIBERATE_CLAMP0`, set
  by `make_plan.sh` and by the DaCapo and Orion scripts).

## DaCapo

`python/dacapo` comes from the released DaCapo compiler
([corelab-src/dacapo](https://github.com/corelab-src/dacapo) at `4616402`), run on every block by
`scripts/utils/dacapo_upstream/`:

```bash
bash scripts/utils/dacapo_upstream/build_hecate.sh      # hecate-opt into .cache/dacapo_upstream (LLVM/MLIR 18, SEAL 4.0)
PLAN_PRUNE=0 bash scripts/utils/dacapo_upstream/plan.sh graphs/gpt2_decode_python_n32 python/dacapo   # add `dense` for full-size bootstraps only
```

`dacapo_4616402.patch` only adds a `--dacapo-plan` pipeline, which runs DaCapo's own analysis
and placement passes without compiling further, and prints where its planner puts bootstraps.
`translate.py` converts each block into DaCapo's input format; where our planner rescales
without a multiplication, it writes a multiplication by a constant, so that DaCapo sees the same
level structure. `ckks_ml48.json` is DaCapo's default configuration with the bootstrap level
limits set to our parameters (5 multiplications between bootstraps). `sites.py` maps DaCapo's
choices back to our recorded values, and `deploy.py` turns them into a plan: it tracks levels,
adds the bootstrap at each block's end that DaCapo assumes (it plans every block from a fresh
input), repairs anything that would run out of levels, and plans the encrypted argmax the way
`make_plan.sh` does.

## Orion

`python/orion` comes from the released Orion bootstrap solver
([baahl-nyu/orion](https://github.com/baahl-nyu/orion) at `be8a827`), run on each block by
`scripts/utils/orion_upstream/`:

```bash
PLAN_PRUNE=0 bash scripts/utils/orion_upstream/plan.sh graphs/gpt2_decode_python_n32 python/orion   # add `dense` for full-size bootstraps only
```

The solver works with a budget of 50 levels: it cannot represent a block input that has no
levels left, which is how a block starts at 48. The resulting plan is then deployed at the
budget `ML` (48 unless exported); deployed at 50, some solver seeds put a bootstrap past the
range the runtime's bootstrap handles, and the runtime refuses it. The script downloads the
solver into `.cache/orion_upstream` (or uses `ORION_SRC`) and applies `orion_be8a827.patch`,
which changes only what stops the solver on transformer models: two searches that enumerate every
path through the model and never finish, replaced by exact equivalents (counting paths by
dynamic programming, and finding a region as the nodes that are both descendants and
ancestors), and four crashes (branches that rejoin at the same point, three-way branches,
overlapping regions, which are priced as plain layers, and nodes off the paths the solver
analyses, which it leaves without a level and never bootstraps). `marks.py` gives the solver our
recorded forward pass and records where it places bootstraps, for every block including the
encrypted argmax; `deploy.py` turns those into a plan the same way the DaCapo script does.

The solver iterates over Python sets of node names, so its choices depend on Python's hash seed;
the recipe fixes `PYTHONHASHSEED=0`. Both scripts remove redundant bootstraps unless
`PLAN_PRUNE=0`; the shipped DaCapo and Orion plans are the ones without that removal.
