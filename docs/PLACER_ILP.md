# ILP placer vs the min-cut

`--placer ilp` (`perseus/plan/placer/ilp.py`) solves stage 4, where the refreshes go, exactly.
It uses a mixed-integer program transcribed from `sim.simulate` and solves it with HiGHS
(`scipy.optimize.milp`). The min-cut is greedy across passes, so it gives no optimality
guarantee. The ILP returns a proven optimum, or an incumbent with its bound. Every ILP plan
is re-simulated, and it is kept only if the sim agrees with the model on every level,
degree and hint decision.

**Setup.** The graphs are the shipped `graphs/gpt2_decode_{n32,n64}`, planned with the exact
`scripts/make_plans.sh` recipes plus `--placer ilp --ilp-time-limit 600`. There are two
objectives:
- `total`, the default, minimises placed refreshes plus fired hints. This is the real
  bootstrap count.
- `placed` minimises placed refreshes only, which is the min-cut's own objective.

Costs carry the same `1 + miss_penalty * overshoot` per site as `_capacity`. The planner runs
on CPU only; these runs used 16 threads.

**Columns.**
- "shipped" is the committed `bootstrap_placements/` plan.
- "cost ILP / min-cut same entry" compares both placers at the SAME block entry level. The
  min-cut runs inside the ILP placer as its incumbent.
- Block entries chain from the previous block's exit, so the ILP's blocks can see a
  different entry than the shipped ones.

## Totals

| chain | shipped min-cut | ILP `total` | ILP `placed` |
|---|---|---|---|
| n32 (32-bit chain) | 418 | **379 (−39, −9.3%)** | 395 (−23, −5.5%) |
| n64 (64-bit chain) | 373 | **343 (−30, −8.0%)** | 360 (−13, −3.5%) |

The `total` column is the shipping configuration: monotone rungs and the final-block exit
cap. The `placed` column predates both (it is the min-cut's own objective, kept for the
like-for-like comparison); without the cap `total` reads 379 / 341.

- **Optimality.** Every block of the two `total` chains is proven optimal (gap 0). Of the
  `placed` runs, 25 of 26 are; n32 block 12 hit the time limit there at cost 6, bound 4.5.
- **Solve time.** 7–22 s per block, and 58 s for the 4.5k-node final block on n32. See
  "Planning time" for what the monotone rungs changed.
- **`total` is the objective to use.** Minimising placements alone saves fewer bootstraps,
  because the solver then lets more hints fire (e.g. n64 block 1: 18 placed + 6 hints vs
  the min-cut's 19 + 4).
- **Wall time and KL are measured below**, on n32 only. The n64 chain has no GPU arm yet.

## Measured on GPU (n32)

These runs used `cuda_cachemir decode` with 16 tokens on one GPU, pinned to a single NUMA
node. Arms alternated. Walls are means over tokens 1–15; `SUMMARY e2e_s/tok` already excludes
token 0. The host was under other load during these runs, so the min-cut row reads above the
paper's 15.44: compare arms within this table, not against the paper.

| arm | runs | e2e s/token | decode | argmax | median KL | top1 | bts / 16 tok |
|---|---|---|---|---|---|---|---|
| min-cut (shipped) | 7 | 16.40 | 14.62 | 1.77 | 0.042–0.043 | 15/16 | 8833 |
| ILP, before the exit cap | 3 | 15.81 ± 0.09 | 13.92 | 1.91 | 0.049–0.050 | 15/16 | 8225 |
| **ILP, shipping plan** | 2 | **15.42 ± 0.02** | 13.58 | 1.81 | 0.050 | 15/16 | 8209 |

The shipping plan is the one this doc's totals describe: monotone rungs and the capped
final exit.

- **Wall time:** −6.0% end-to-end (16.40 -> 15.42) and −7% decode for the shipping plan,
  with no unplanned bootstraps in any run. The planner predicted −1.1 s/token of bootstrap
  time; the runs measure −0.98.
- **Accuracy:** KL rises about 17% (median 0.050 vs 0.043) in every run; top1 is unchanged
  at 15/16. The mechanism is NOT established. Two candidates are ruled out: refresh depth
  (the min-cut places a larger share at the envelope, 68% vs 62%) and sites missing their
  error target (neither plan has any). The remaining hypothesis is composition -- 39 fewer
  refreshes leave more approximation steps between them, which the planner's per-site
  error model does not price at all. Note that comparing the two plans' predicted p90
  (1.4e-4 vs 6.2e-4) overstates the case: the ILP drops mostly easy CF=2 sites (176 -> 133),
  so the same percentile lands on a harder site. Untried levers: `quality_weight` (linear
  in this model, so it costs nothing to add) and capping refresh input depth below the
  envelope.
- **Capped exit.** Without the cap the final block exits deeper (level 42 vs 40) and the
  unplanned encrypted argmax pays for it: argmax 1.91 s/token against the min-cut's 1.77.
  `ilp_cap_exit` (default on) forbids the final block from exiting deeper than the
  min-cut's plan; the shipping plan exits at 36 and its argmax is back to par (1.81).

## Planning time

Full chains with the shipped recipes, 4 threads (HiGHS is mostly serial). The ILP time
includes its own min-cut incumbent run.

| chain | min-cut | ILP | ratio |
|---|---|---|---|
| n32 | 9.7 s | 164 s | ~17× |
| n64 | 10.4 s | 158 s | ~15× |

**Monotone rungs.** Before the ladders stated `[v >= k+1] => [v >= k]`, n32 took 547 s,
and 433 s of that was block 12 alone (4.5k nodes, the lm_head). HiGHS's own log showed
the cost was presolve, not search: 359 s of presolve, then one branch-and-bound node.
Every integer solution satisfies monotonicity by construction, so the rows change no
answer, but without them the solver works through subtrees whose rung pattern denotes no
level at all. Stating them takes block 12 to 58 s and the chain to 164 s. n64, whose
blocks were never hard, pays ~40% more (113 s -> 158 s) for the extra rows. Disabling
HiGHS presolve instead is worse on both (n32 683 s, n64 52 s).

**Not the cause: the composite chain's odd levels.** n32 carries two primes per level
(`--level-unit 2`), but a ladder holds only the values its var can actually take: across
all 13 n32 blocks every value is even, and the model has no integer level variable at all
(HiGHS sees 110k columns, all binary). Dividing the chain through by the unit would
relabel the value set and produce an identical LP. n32's ladders are one rung longer than
n64's because the chain has one more level of runway (7 levels of budget + 5 below the
bootstrap level from the s1 landing, against 6 + 5), which is 13% more model, not the 33x
that monotonicity removed.

## Dense/sparse pricing: `--ilp-objective ms`

This objective prices every refresh (placed or hint) at its route's measured latency
(`--bts-ms`). It stays linear because a site's route is fixed by its packing.

| chain | objective | bts | dense / s512 / s1 | predicted bts time |
|---|---|---|---|---|
| n32 | shipped | 418 | 243 / 48 / 127 | 9.93 s/token |
| n32 | `total` | 379 | 205 / 48 / 126 | 8.82 s/token |
| n32 | `ms` | 380 | 198 / 53 / 129 | 8.76 s/token |
| n64 | shipped | 373 | 203 / 47 / 123 | 11.01 s/token |
| n64 | `total` | 341 | 192 / 43 / 106 | 10.17 s/token |
| n64 | `ms` | 347 | 183 / 52 / 112 | 10.18 s/token |

`ms` trades dense refreshes for sparse ones, but it gains at most 60 ms/token over
`total`, which is inside the run-to-run noise. The objective cannot choose a refresh's
route; it can only move refreshes to sites whose packing allows sparse routing. So the
lever it adds is small on these graphs.

## Caveats

- **Optimal per block, given its entry level.** The chain of blocks is still planned
  greedily, block after block.
- **Optimal for the planner's model.** That means the count + overshoot pricing and
  `simulate()`'s level rules. It is not optimal for milliseconds, since dense and sparse
  refreshes are priced alike, and it is not optimal for accuracy.
- **The time limit is soft.** HiGHS can overrun `--ilp-time-limit` inside a phase it does
  not check the clock in: measured 1883 s against a 600 s cap on n32 block 12, in the
  configuration before the monotone rungs. Enforcing it would mean solving in a
  subprocess that can be killed.

## Per block

### n32, objective=total

| block | shipped min-cut (placed+hints+delib = total) | ILP (placed+hints+delib = total) | ILP status | cost ILP / min-cut same entry | bound | solve s |
|---|---|---|---|---|---|---|
| 0 | 28+4+1 = **33** | 26+3+1 = **30** | optimal | 29.0 / 32.0 | 29.0 | 9.01 |
| 1 | 22+5+1 = **28** | 25+3+1 = **29** | optimal | 28.0 / 29.0 | 28.0 | 9.98 |
| 2 | 27+5+1 = **33** | 27+3+1 = **31** | optimal | 30.0 / 35.0 | 30.0 | 10.61 |
| 3 | 30+5+1 = **36** | 28+3+1 = **32** | optimal | 31.0 / 35.0 | 31.0 | 10.87 |
| 4 | 42+5+1 = **48** | 38+3+1 = **42** | optimal | 41.0 / 47.0 | 41.0 | 12.34 |
| 5 | 29+5+1 = **35** | 27+3+1 = **31** | optimal | 30.0 / 34.0 | 30.0 | 10.3 |
| 6 | 27+5+1 = **33** | 26+3+1 = **30** | optimal | 29.0 / 32.0 | 29.0 | 8.0 |
| 7 | 26+5+1 = **32** | 24+4+1 = **29** | optimal | 28.0 / 31.0 | 28.0 | 9.66 |
| 8 | 26+5+1 = **32** | 24+4+1 = **29** | optimal | 28.0 / 31.0 | 28.0 | 8.67 |
| 9 | 25+6+1 = **32** | 25+3+1 = **29** | optimal | 28.0 / 31.0 | 28.0 | 8.33 |
| 10 | 26+5+1 = **32** | 25+3+1 = **29** | optimal | 28.0 / 31.0 | 28.0 | 7.74 |
| 11 | 31+5+1 = **37** | 29+3+1 = **33** | optimal | 32.0 / 36.0 | 32.0 | 9.77 |
| 12 | 6+1+0 = **7** | 5+0+0 = **5** | optimal | 5.0 / 6.0 | 5.0 | 386.35 |
| **all** | **418** | **379** (-39, -9.3%) | | | | |

### n32, objective=placed

| block | shipped min-cut (placed+hints+delib = total) | ILP (placed+hints+delib = total) | ILP status | cost ILP / min-cut same entry | bound | solve s |
|---|---|---|---|---|---|---|
| 0 | 28+4+1 = **33** | 25+6+1 = **32** | optimal | 25.0 / 28.0 | 25.0 | 10.64 |
| 1 | 22+5+1 = **28** | 20+6+1 = **27** | optimal | 20.0 / 22.0 | 20.0 | 9.13 |
| 2 | 27+5+1 = **33** | 25+7+1 = **33** | optimal | 25.0 / 30.0 | 25.0 | 15.47 |
| 3 | 30+5+1 = **36** | 26+7+1 = **34** | optimal | 26.0 / 30.0 | 26.0 | 10.74 |
| 4 | 42+5+1 = **48** | 35+7+1 = **43** | optimal | 35.0 / 42.0 | 35.0 | 12.48 |
| 5 | 29+5+1 = **35** | 25+7+1 = **33** | optimal | 25.0 / 29.0 | 25.0 | 10.05 |
| 6 | 27+5+1 = **33** | 24+6+1 = **31** | optimal | 24.0 / 27.0 | 24.0 | 13.82 |
| 7 | 26+5+1 = **32** | 23+6+1 = **30** | optimal | 23.0 / 26.0 | 23.0 | 12.14 |
| 8 | 26+5+1 = **32** | 23+7+1 = **31** | optimal | 23.0 / 26.0 | 23.0 | 10.58 |
| 9 | 25+6+1 = **32** | 22+6+1 = **29** | optimal | 22.0 / 25.0 | 22.0 | 8.67 |
| 10 | 26+5+1 = **32** | 23+7+1 = **31** | optimal | 23.0 / 26.0 | 23.0 | 8.91 |
| 11 | 31+5+1 = **37** | 26+7+1 = **34** | optimal | 26.0 / 31.0 | 26.0 | 8.7 |
| 12 | 6+1+0 = **7** | 6+1+0 = **7** | time-limit | 6.0 / 6.0 | 4.5 | 1883.21 |
| **all** | **418** | **395** (-23, -5.5%) | | | | |

### n64, objective=total

| block | shipped min-cut (placed+hints+delib = total) | ILP (placed+hints+delib = total) | ILP status | cost ILP / min-cut same entry | bound | solve s |
|---|---|---|---|---|---|---|
| 0 | 23+4+1 = **28** | 24+1+1 = **26** | optimal | 25.0 / 27.0 | 25.0 | 8.78 |
| 1 | 19+4+1 = **24** | 21+1+1 = **23** | optimal | 22.0 / 22.0 | 22.0 | 6.27 |
| 2 | 24+4+1 = **29** | 26+1+1 = **28** | optimal | 27.0 / 28.0 | 27.0 | 7.98 |
| 3 | 29+4+1 = **34** | 28+1+1 = **30** | optimal | 29.0 / 33.0 | 29.0 | 9.28 |
| 4 | 39+4+1 = **44** | 37+1+1 = **39** | optimal | 38.0 / 43.0 | 38.0 | 11.39 |
| 5 | 25+4+1 = **30** | 26+1+1 = **28** | optimal | 27.0 / 29.0 | 27.0 | 11.35 |
| 6 | 26+4+1 = **31** | 24+3+1 = **28** | optimal | 27.0 / 30.0 | 27.0 | 9.38 |
| 7 | 26+4+1 = **31** | 24+3+1 = **28** | optimal | 27.0 / 30.0 | 27.0 | 9.94 |
| 8 | 23+4+1 = **28** | 24+1+1 = **26** | optimal | 25.0 / 27.0 | 25.0 | 7.96 |
| 9 | 23+4+1 = **28** | 23+1+1 = **25** | optimal | 24.0 / 27.0 | 24.0 | 6.53 |
| 10 | 23+4+1 = **28** | 23+1+1 = **25** | optimal | 24.0 / 27.0 | 24.0 | 6.76 |
| 11 | 27+5+1 = **33** | 28+1+1 = **30** | optimal | 30.33 / 33.33 | 30.33 | 8.34 |
| 12 | 4+1+0 = **5** | 5+0+0 = **5** | optimal | 5.0 / 5.0 | 5.0 | 13.59 |
| **all** | **373** | **341** (-32, -8.6%) | | | | |

### n64, objective=placed

| block | shipped min-cut (placed+hints+delib = total) | ILP (placed+hints+delib = total) | ILP status | cost ILP / min-cut same entry | bound | solve s |
|---|---|---|---|---|---|---|
| 0 | 23+4+1 = **28** | 21+5+1 = **27** | optimal | 21.0 / 23.0 | 21.0 | 45.66 |
| 1 | 19+4+1 = **24** | 18+5+1 = **24** | optimal | 18.0 / 19.0 | 18.0 | 9.84 |
| 2 | 24+4+1 = **29** | 23+5+1 = **29** | optimal | 23.0 / 24.0 | 23.0 | 8.9 |
| 3 | 29+4+1 = **34** | 25+5+1 = **31** | optimal | 25.0 / 29.0 | 25.0 | 9.27 |
| 4 | 39+4+1 = **44** | 34+5+1 = **40** | optimal | 34.0 / 39.0 | 34.0 | 14.08 |
| 5 | 25+4+1 = **30** | 23+5+1 = **29** | optimal | 23.0 / 25.0 | 23.0 | 11.26 |
| 6 | 26+4+1 = **31** | 23+5+1 = **29** | optimal | 23.0 / 26.0 | 23.0 | 11.99 |
| 7 | 26+4+1 = **31** | 23+5+1 = **29** | optimal | 23.0 / 26.0 | 23.0 | 15.89 |
| 8 | 23+4+1 = **28** | 23+5+1 = **29** | optimal | 23.0 / 23.0 | 23.0 | 9.46 |
| 9 | 23+4+1 = **28** | 22+5+1 = **28** | optimal | 22.0 / 23.0 | 22.0 | 7.08 |
| 10 | 23+4+1 = **28** | 22+5+1 = **28** | optimal | 22.0 / 23.0 | 22.0 | 8.47 |
| 11 | 27+5+1 = **33** | 26+5+1 = **32** | optimal | 26.0 / 30.0 | 26.0 | 12.87 |
| 12 | 4+1+0 = **5** | 4+1+0 = **5** | optimal | 4.0 / 5.0 | 4.0 | 12.82 |
| **all** | **373** | **360** (-13, -3.5%) | | | | |
