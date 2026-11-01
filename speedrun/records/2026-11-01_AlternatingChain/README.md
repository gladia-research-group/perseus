# Decode-8 on a chain whose user levels alternate one prime and two: record 5's plan replanned on it, 99 refreshes per token instead of 112

Record 5's Decode-8 runs every value between its refreshes on n32a4's user levels: 9 levels of two ~27-bit primes,
one pair dropped per rescale, each value at a scale of about 2^54. Here those levels alternate between one prime of
~27.85 bits and a pair of ~2 × 26.15 bits (the chain `n32sp`), so a rescale drops one prime or two in turn and a value's
scale moves between 2^36 and 2^44.15. The user region gets 12 levels in the bits record 5's 9 took, at log QP 1728.55
instead of 1734. Every prime stays below 2^28, so record 5's kernels run unchanged, and the levels the bootstraps run
on (CtS, EvalMod, SparseB, SPRU, StC-first's entry, q_0 and the sparse keys' special primes) keep their primes. Given 3
more levels between refreshes, record 5's configs and plan recipe, recaptured and replanned on the new level grid, need
99 refreshes per token instead of 112. Prefill-128 stays on n32a4: its output frames are byte-identical to record 5's.

## Changes against record #5

- **The chain n32sp.**
  - `ChainSpec` preset `n32sp` (`crates/pc-core/src/params.rs`): n32a4 with its 9 user pairs replaced by 6 single
    primes alternating with 6 pairs, picked top-down so that the scale orbit closes (2^36 at the pair levels, 2^44.15
    at the single ones); every prime an NTT prime below 2^28. 51 Q + 13 special primes, dnum 4.
  - The scale tables and FLEXIBLEAUTO's steps by the primes a level drops (`crates/pc-core/src/scale.rs`).
  - A virtual level grid: two grid levels per CKKS level, a pair level with one prime dropped in between, so that the
    planner, its plans and every route keep counting levels as before; `ChainShape::primes_dropped` / `level_at` and
    `Chain::top_at` map a level to its primes (`crates/pc-shadow/src/{meta,rules,fides,route}.rs`, `crates/pc-core`).
    On n32 and n32a4 the grid is the identity.
  - The routes convert levels through the chain's shape, and the raise takes its composite bottom by it
    (`crates/pc-gpu/src/bts/{raise,stc_first,dense,sparse,sparse_classic,staged,stc_high}.rs`).
  - Key switching: the HPS-exact ModDown conversion on the user region's levels (`crates/pc-gpu/src/ks.rs`).
  - The client's key shapes through the chain (`crates/pc-client/src/client.rs`).
- **The runtime.**
  - The session's plaintext classes, register layouts and StC-first's key-cache entry by the chain's level grid
    (`crates/pc-py/src/{session,meta,shadow,word64}.rs`).
  - Fused relinearization-rescales only where the rescale drops a pair; the one-prime levels take the plain rescale
    (`crates/pc-program/src/compile/rr.rs`, `crates/pc-gpu/src/ops.rs`).
  - `PlannedRun` asserts at setup that the CPU twin's StC-first entry is the plan's (`python/perseus/impl/driver.py`).
- **The compile guard** `[user_region]` (`crates/pc-program/src/compile/mod.rs`): the user region hosts no EvalMod, so
  a refresh landing past the region's top (SparseB: one level past it) or a dense raise drop into it is a compile error.
- **The plan** `bootstrap_placements/gpt2_decode_python_n32sp_kc16r_lh`, made from scratch by
  `speedrun/make_n32sp_decode_plan.sh` (CPU, about 20 minutes): record 5's configs, runtime line and ILP recipe on
  n32sp's level grid, the captures `graphs/gpt2_decode_python_n32sp_kc16r_lhm` and `graphs/gpt2_decode_python_n32sp_kc16r_lh_cm`,
  CutMax's stage 13 on the ILP (`speedrun/make_cutmax_plan.sh`), record 5's raise-drop re-pass
  (`speedrun/make_n32a4_decode_rd.sh`). `speedrun/server.py` serves it.
- **The server** no longer applies position 8's patch ahead, before token 8 is released (`speedrun/server.py`), as
  record 5's review asked.
- **The client:** `speedrun/harness/perseus_spchain_client.py`, record 5's key-set client with the chain n32sp: the
  same roles and secret, n32sp's key shapes, each key's modulus bounded as in record 5's client.
- **Tests:** `crates/pc-core/tests/params_chain.rs` (n32sp), `crates/pc-program/tests/user_region.rs` (the guard, on a
  fixture where the CPU twin's n32 StC-first entry put block 0's K/V push at a dense raise drop of 3, which the guard
  refuses), `crates/pc-program/tests/shipped_plans.rs` (the served n32sp plan: 0 check errors).
- **pc-bench:** SparseB's and SPRU's raise drop and SparseB's EvalMod K from the environment, and `pc-bench ops` (the
  user ops' cost and precision per level); not part of the served build.

## Ablation

The speedrun harness on H100 80GB GPUs (not the reference GPU), the server pinned to its GPU's NUMA node, two dev
prompts per run, pairs in the order AB BA BA AB AB BA, both runs of a pair on one GPU, scored by `score.py`. Each arm with
its own client: record 4 (speedrun `d97a6f7`) with `perseus_lean_client.py`, record 5 with `perseus_keyset_client.py`,
this entry with `perseus_spchain_client.py`.

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| Record 5 (LeanKeySwitch) against record 4 (speedrun `d97a6f7`), dev 0-11 | decode | 2.737 → 2.158 s/token, ×1.269 (p 4.6e-10) |
| n32sp and its plan against record 5, dev 0-11 (`e1_*`) | decode | 2.151 → 2.073 s/token, ×1.037 (p 1.3e-7) |
| The guard, the twin's assert and position 8's patch in its clock | decode | about 2 ms per request (position 8's patch, once per request), below the harness's resolution; the guard and the assert change no served byte |
| **This entry** against record 5, dev 0-11, at the timed commit `ea62bcc` (`dfinal_*`) | decode | **2.151 → 2.077 s/token, ×1.036 (p 3.1e-6)** |
| n32sp and its plan against record 4, dev 0-11 (`r4_*`) | decode | 2.748 → 2.087 s/token, ×1.317 (p 1.1e-10) |

- The same served code before the rebase: dev 16-27, 6 pairs, 2.151 → 2.074 s/token, ×1.037 (p 7.6e-7, an independent
  build, `rt_*`); dev 0-5, 3 pairs, 2.152 → 2.085 s/token, ×1.032 (p 0.0013, `e2_*`). Record 5 against record 4: `r5r4_*`.
- The timed commit `ea62bcc` serves the same files as the commit these runs used (the plan script's provenance lines,
  an empty capture marker and the base's repro and README changes differ; nothing the server builds or reads); the
  smoke runs at `ea62bcc` with its pinned client are `smoke_*`. The `rt_*`, `e2_*`, `e1_*` and `r4_*` runs predate the
  rebase of this work onto speedrun `4e034ba`: their `client_perseus` names the commits before it (`abdfbfc` for the
  final code, `2216d80` for `e1_*` / `r4_*`'s client, whose client code is the same).

## Dev result

The speedrun harness on H100 80GB HBM3 GPUs (not the reference GPU), record 5 (`7640b92`'s served code, which
equals speedrun `4e034ba`'s, with `perseus_keyset_client.py`) against this entry's code (with
`perseus_spchain_client.py`), pairs in the order AB BA BA AB AB BA, two requests per run, both runs of a pair on one GPU,
scored by `score.py`. Logs in `dev_logs/`.

| Track | Score | Record 5, same GPU | KL | Argmax (scored) | Peak GPU memory |
|---|---|---|---|---|---|
| **Decode-8, 6 pairs, dev 0-11, at `ea62bcc` (`dfinal_*`)** | **2.077 s/token** | **2.151 s/token** | 3.7e-6 | 95/96 | 59.5 GiB (record 5 62.7) |
| Decode-8, 6 pairs, dev 16-27 (`rt_*`) | 2.074 s/token | 2.151 s/token | 3.1e-6 | 95/96 | 59.5 GiB |
| Decode-8, 3 pairs, dev 0-5 (`e2_*`) | 2.085 s/token | 2.152 s/token | 3.6e-6 | 48/48 | 59.5 GiB |
| Prefill-128 (not claimed), one run at `ea62bcc` (`smoke_prefill_entry`) | 11.30 s | - | 6.2e-6 | 1/1 | 57.6 GiB |

score.py at `ea62bcc` (`dfinal_*`, this entry's pinned client): quality PASS (mean KL 3.68e-6, p 5.2e-31, argmax 99.0 %),
speed ×1.036 (delta 1.0 %, p 3.1e-6), NEW RECORD; every pair between ×1.031 and ×1.043 (pairs 0-4 on one GPU, pair 5 on
another). The `rt_*` replicate: ×1.037 (p 7.6e-7), every pair between ×1.032 and ×1.042.

- **Prefill-128** stays on record 5's chain and plan: at `ea62bcc`, on one key bundle and one set of input ciphertexts (dev prompts 0-1) its output frames are byte-identical to record 5's, with the same key request (`dev_logs/bit_identity.txt`).
- **KL on C4 validation**, 12 fresh key sets per arm, C4 validation prompts 32-79 of our evaluation slice (4 per run): mean 3.38e-6 (median 2.57e-6,
  p90 6.43e-6, max 2.02e-5) against record 5's 3.77e-6 (2.70e-6, 7.92e-6, 1.40e-5) on the same prompts; paired geometric
  mean ×0.86.
- **Range gate.** Every refresh site's input range against its bootstrap's, with a margin of at least 1.5×, a rule
  declared before the check. On the fitting sets (C4 validation fit slice 0-63 × 16 positions, evaluation slice 0-15 × 16, 4,096 OpenWebText token
  pairs, 8,192 first tokens): 99 sites, minimum 1.83× (block 1's ln_2 SPRU refresh), 6 sites under 2×. On a held-out
  slice, C4 validation evaluation slice 128-175 × 16: minimum 1.85×, no site under 1.5×. The plan was frozen before either; the held-out slice
  was captured while the last fitting-set piece was still running, and read after the fitting-set result was recorded;
  nothing changed after either.

## Disclosures

- **CKKS parameters and keys.** Two chains:
  - Decode-8 moves to `n32sp`: 51 Q primes and 13 special primes, every one an NTT prime below 2^28, dnum 4, log QP
    1728.55 ≤ 1747 at N = 2^16. q_0 q_1 and the three thin primes above them are n32a4's; the special primes are n32's 9
    and 4 others (142344193, 107216897, 106037249, 104857601), which only the dense keys use.
  - Prefill-128 stays on `n32a4`, unchanged, log QP 1733.26 (≤ 1734).
  - The main secret is unchanged (uniform ternary). Every key under it lies within its chain's Q × P; the client bounds
    each at 1734 bits.
  - The sparse-secret keys (the h = 32 encapsulation key, the SPRU block key) stay at the bottom shape: q_0 q_1 × n32's
    9 special primes, 307.6 bits, record 1's modulus. The PCMM ring-switch key (Prefill-128) is unchanged.
  - Record 5's key roles (`relin_level`, `rot` over P'' (6 special primes), `sparse_rot`) are kept, at n32sp's shapes.
- **Scales.** On n32sp values carry 2^36 at the user region's pair levels and 2^44.15 at its single levels (record 5:
  about 2^54). The scales just outside the region change too (log2):

  | levels | n32sp | n32a4 |
  |---|---|---|
  | above the region, tops 24-32 (CtS / EvalMod's bottom) | 48.98, 51.22, 52.43, 52.97, 53.27 | 53.6-53.7 |
  | StC-first's entry (top 4) | 44.13 | 53.84 |
  | top 2 | 35.25 | 54.67 |
  | top 1 | 43.27 | 82.12 |
- **The exact ModDown on the user region.** The fast ModDown conversion leaves about 2^23.9 of error per slot, 12 bits
  at a 2^36 level, so key switches at or below the user region's top use the HPS-exact conversion (an existing,
  self-tested base-conversion path): 19.6 / 27.6 bits at 2^36 / 2^44.15, at about the same cost. Above it (CtS, EvalMod)
  the fast conversion stays.
- **Rescales.** The one-prime levels take the plain rescale: no fused relinearization-rescale there, no folded scalar.
- **The plan.** Regenerated, not reused: record 5's configs, runtime line and ILP recipe on n32sp's level grid. The
  runtime line is record 5's with `CHAIN_PRESET=n32sp`, the reactive ceiling and the input guard 6 grid levels deeper
  (`AUTO_BTS_LEVEL=50`, `BTS_MAX_INPUT_LEVEL=54`), SparseB's raise variants 1-3, no `KV_PUSH_RAISE` or
  `FIDESLIB_CTS_ALT_BUDGET`, and `FAKECAP_STC_ENTRY=52`. Raise drops are limited so that EvalMod stays on its pairs:
  none on dense refreshes, SparseB to a total of 3, SPRU to 6 (record 5's plan has dense and deeper drops). 99
  refreshes per token against record 5's 112. The plan regenerates from scratch with the committed script at the timed
  commit `ea62bcc`, on two machines (Intel and AMD CPUs; clean trees, their own builds, CPU only, 10-20 minutes): `PLAN_CMD.txt`, every graph.json and every
  placement file byte-identical, apart from the solver's wall time in stages 12 and 13 (`summary.placer_meta.solve_s`);
  the captures' capture_config.json differ in host paths only (the run also writes one at the top of `_lh_cm`, which
  is not committed).
- **The CPU twin's StC-first entry.** The plan's captures and its runtime line set `FAKECAP_STC_ENTRY=52`, n32sp's
  StC-first entry (the twin's default is n32's 46). With 46 the twin put block 0's K/V push at a landing the compile
  reached only by a dense raise drop into the user region, which ran EvalMod there. `PlannedRun` asserts at setup that
  the twin's entry is the plan's.
- **The compile guard.** The compile refuses a plan whose refresh would land past the user region's top or raise-drop
  into it (the `[user_region]` check, with a test).
- **CutMax is less precise than record 5's.** CutMax runs between its refreshes on the user region's 2^36 / 2^44
  levels.
  - Its z output carries a noise floor of about 2e-7 per slot (median top-1 mass 0.9952 against record 5's 0.99999).
  - It sometimes returns a token other than the logits' top-1 when the top logits are a few tenths of a nat apart: up
    to 0.40 nats below the top-1 in our measurements.
  - Scored positions where CutMax differs from the decrypted logits' argmax:
    - 2 of 128 on 4 fresh key sets (C4 validation; logit gaps 0.24 and 0.40);
    - 1 of 384 on 12 key sets (C4 validation; a near tie);
    - 1 of 96 on dev prompts 0-11, the same position (prompt 9, position 10) in three A/Bs with different keys (gap
      0.16);
    - 1 of 96 on dev prompts 16-27 (gap 0.20).
  - Record 5 misses none of these. The harness's bar is 95 %; no measured set is below 98 %.
- **Positions 0-7.** No work moves into them. Position 8's patch, which record 5's server applied before token 8 was
  released, is now applied in position 8's clock (about 2 ms, once per request), as the maintainers asked in record 5's
  review.
- **New untimed work.** None. Setup is record 5's in kind and about in time: public placeholder rows, about 21 s of
  position precompute, 188 switching keys (at `ea62bcc`: the plan prepared in 26.8 s, the session in
  9.8 s, the 16 positions' plaintexts in 22.0 s).
- **Memory.** Peak GPU memory 59.5 GiB against record 5's 62.7; server host memory 18.5-18.6 GiB (record 5: 19.2-19.3).
- **Numbers.** Measured on H100s, not on the reference GPU.
