# Decode-8 under a second per token: n32sp with lean configs, raise drops onto the region's top pair and CutMax's masks at a usable scale; Prefill-128 with lighter polynomials and q·Kᵀ as sums of products

Record 6's Decode-8 (AlternatingChain, n32sp) and record 5's Prefill-128 (n32a4), each with fewer and cheaper refreshes
and a leaner runtime under them. Decode-8's configs fold the GELU and exp masks into their series, pack two heads per
lane block so the softmax refreshes on SparseB-256, fold the scores into the Goldschmidt recursion and run lighter GELU
and exp series; replanned on n32sp they need 87 refreshes per token instead of 99, and dense and SPRU refreshes now take
a raise drop that ends EvalMod on the alternating region's top pair. CutMax's centring masks, which n32sp's 2^36 / 2^44
plaintext scales rounded away on the imaginary tile, are moved to a scale where they survive: the three dev prompts on
which record 6's CutMax picks a wrong token in every key draw (prompt 9 at position 10, 19 at 11, 30 at 8) now agree.
Prefill-128 folds its masks, rotates once per attention window, relinearizes each giant step's products once, pairs
more of its real refreshes and runs lighter GELU and exp series. Under both: exact base conversions in their own
kernels, one-prime rescales and SPRU products fused, eager rescales, element-wise chains in one pass and the Siblings
schedule. The keys are record 6's plus four rotation keys of the same kind; the client is record 6's
(`perseus_spchain_client.py`).

## Changes against record 6 (Decode-8)

- **Configs carried from our n32a4 work, replanned on n32sp** (`speedrun/make_n32sp_decode_plan_gn.sh`: record 6's
  `make_n32sp_decode_plan.sh` on these configs and runtime line; ILP, then the raise-drop pass). 87 refreshes per token
  (stages 0-12: 76, CutMax: 11) instead of 99.
  - Masks folded into the series (`GeluCfg.mask_fold`, `SoftmaxCfg.exp_mask_fold`): the GELU's lane mask and the
    softmax's active mask multiply the Chebyshev series' giant steps and leaves where a level is spare, instead of the
    series' output (`poly.eval_chebyshev_masked`); a leaf without slack takes the mask term by term (fold mode 2).
  - Two heads per lane block (`SoftmaxCfg.pair_heads`): with 16 keys at most, head h+8 takes lanes 16-31 of head h's
    block, so every softmax payload repeats every 256 slots and the softmax refreshes run SparseB-256.
  - Scores folded into the Goldschmidt recursion (`gs_num`): each reciprocal round's scores z enter the packed recursion
    (P0 = −i·G·(i·D + z), mu = 1), which returns z·R directly: one product and one level fewer per round.
  - LayerNorm seeds at degree 15 in blocks 0-10; the softmax of layers 4 and 11 in four refreshes instead of five
    (closer to the exact softmax than before).
  - GELU at degree 60 in blocks 0, 1 and 3-10 and 222 in blocks 2 and 11, the exp at degree 6, fitted on the blocks'
    real OpenWebText inputs.
- **Raise drops onto the alternating region's top pair.** Record 6 allows no dense raise drop on n32sp (EvalMod would end in
  the user region). EvalMod ending on the region's top pair (2^44.15) loses at most 0.25 bits against drop 0 (pc-bench
  precision grid, 3 seeds, CF 9/13/17); the compile guard now lets a dense refresh land one level and SPRU seven levels
  below it (`crates/pc-program/src/compile/mod.rs`), and the planner caps drops per route (`PLAN_RAISE_ROUTE_MAX=0:1,1:7`).
  10 dense refreshes (the pre-GELU ones of blocks 2-10) take drop 1 and 23 SPRU refreshes drop 7.
- **CutMax's masks at a usable scale** (`CutMaxCfg.out_scale`, `lm_centre`, `newton_carry`; constants only, the same
  ops and refreshes). CutMax centres each iteration with masks `M·(0.5k/n)` of 2.3e-7 and 1.3e-13 that multiply sums of
  1e7 to 1e9. A plaintext is encoded at its level's scale, 2^36 or 2^44.15 in n32sp's user region against ~2^54 on
  n32a4: tile 1's mask (the imaginary axis) loses its sub-integer coefficients and shifts that axis against the other
  by up to 3% of the top value, deterministically, whatever the keys. Here the LM head's columns are centred (every logit
  shifts by one constant, so the argmax and the KL are unchanged and the entry's sum is ~0), iteration 0's output is
  scaled by 0.25 so the next masks are 2^18 larger, and the folded Newton for 1/√t carries its small operands at a
  larger scale (a one-prime level's rescale leaves 1.6e-7 per slot, `sqrt(N(1 + 2N/3)/24) / 2^36`). Over 4 key draws ×
  32 prompts: 2048/2048 positions agree, against record 6's 2033/2048.

## Changes against record 5 (Prefill-128)

- **Masks folded into the series**, as Decode-8's (the trimmed degree-8 exps mask their leaf term by term).
- **One rotation per window** (`PF_ATTN_WROT`): q·Kᵀ's window masks on the K side, P·V's y rotated once per window and
  each window's complex V pair built once; the window ladders go from 216 to 24 per block.
- **q·Kᵀ as sums of products** (`PF_ATTN_WROT` bit 16): a giant step is Σ_w rot_w(q_g)·rot_w(K̃_w), with K̃_w rotated
  once per chunk pair and q_g once per giant step, so each giant step's products are one relinearized sum (2,157
  relinearizations per request → 309, the same rotation keys).
- **Lazy relinearization** (`PC_MULT_SUM=1`): a chain of products that are only added together is relinearized once
  (the tensor kernel's `acc_d2`).
- **More refresh pairs**: the pair search looks further ahead (`PC_BTS_PAIR_WINDOW`), pairs across input levels and
  raise drops (`PC_BTS_PAIR_LOOSE`) and above StC-first's reference CF (`PC_BTS_PAIR_CF_MAX=15`). 50 pairs per request
  instead of 39.
- **Lighter polynomials**: the exp at degree 6 in every layer, the GELU at degree 120 in blocks 1 and 10 and 60 in the
  others but 2 and 11 (fits weighted by the blocks' real inputs), the up projection's finishes a level lower there;
  layer 4's softmax with one normalising round fewer. The plans keep their refreshes and sites: decisions carried onto
  the new captures (`scripts/utils/map_plan_decisions.py`, `speedrun/make_prefill_carry_plan.sh`).

## Both tracks (exact runtime)

- **Exact base conversions in their own kernels** (`BconvArgs.kind`, ABI 21): a launch of 13-15-source exact
  conversions (n32sp's HPS-exact ModDown, the fused relinearization-rescales) takes its rounding from an exact 128-bit
  fixed-point sum and the FP64 estimate only within 2^-45 of a boundary. Byte-identical.
- **One-prime rescales** out of place with the scalar adjust folded; **one-prime fused relinearization-rescales**
  (record 6 fused only pair drops); **SPRU's product** on the fused relinearization-rescale.
- **Eager rescales** (`rr::early`): a product whose value only flows through rescale-commuting ops (conjugations,
  rotations, negations, products by i, constant adds, sums of such values) into readers that each rescale it is
  rescaled at once: those ops run a level lower and the readers drop their rescale. Exact maths.
- **Element-wise chains as one pass** (both tracks) and Decode-8 on the **Siblings schedule** Prefill-128 already uses
  (independent same-level relinearizations and rotation sums as one K-batch). Byte-identical.

## Keys

Decode-8's request grows from record 6's 188 to 192 switching keys: rotation keys for steps 240 and N − 240 (the head pair's
packing) and its SparseRot keys on the SparseB-256 steps. Every one is a rotation key of record 6's kind: an RLWE sample under
the dense uniform ternary main secret, at record 6's shapes (log2 modulus ≤ 1728.55 ≤ 1734, the bound record 6's client
enforces), generated and checked by record 6's client. Prefill-128's request is unchanged. No sparse key, special prime or client change.

## Dev result

One RTX PRO 6000 Blackwell Server Edition, the harness, 6 pairs (A B B A B A A B A B B A), the entry's server with record 6's
client build (as the entry runs officially); Decode-8 against record 6 with its client, Prefill-128 against record 5 with its
client:

| | entry | baseline | ratio | KL | argmax | peak GPU |
|---|---|---|---|---|---|---|
| Decode-8 | 0.971 s/token | record 6 1.201 s/token | ×1.237 (p 1e-13) | 2.0e-5 | 100% | 65.0 GiB (record 6 59.4) |
| Prefill-128 | 6.382 s | record 5 7.078 s | ×1.109 (p 7.9e-14) | 4.8e-5 | 100% | 57.3 GiB (record 5 56.6) |

All 32 dev prompts: Decode-8 over 4 key draws mean KL 2.2e-5 (max 6.1e-5), argmax 2048/2048; Prefill-128 mean KL 4.7e-5
(max 2.8e-4), argmax 32/32.

## Ablation

Untimed harness A/Bs on GPUs 0-2 (interleaved, 2 requests per run), each against the stack before it:

Decode-8, s/token:

| lever | before → after | Δ | accuracy |
|---|---|---|---|
| element-wise chains in one pass, Siblings schedule (on record 6's plan) | 1.198 → 1.188 | −0.8% | byte-identical |
| configs replanned on n32sp: folded masks, two heads per block, `gs_num`, seeds, layers 4 / 11 (87 refreshes) | 1.190 → 1.020 | −14.2% | KL 3.9e-6 |
| exact conversions in their own kernels, one-prime rescale out of place | 1.008 → 0.987 | −2.1% | byte-identical |
| SPRU's product on the fused relinearization-rescale | 0.988 → 0.983 | −0.6% | byte-identical |
| CutMax's masks at a usable scale | 1.019 → 1.019 | 0 | argmax 2048/2048 (record 6's plan 2033/2048) |
| GELU / exp lighter, dense raise drop 1 | 0.994 → 0.985 | −0.93% | KL 2.2e-5, 2048/2048 |
| eager rescales, one-prime fused relinearization-rescales | 0.984 → 0.977 | −0.69% | KL 2.0e-5 |
| SPRU raise drop 7 | 0.978 → 0.971 | −0.72% | KL 2.2e-5, 2048/2048 |

Prefill-128, s/request:

| lever | before → after | Δ | accuracy |
|---|---|---|---|
| folded masks, one rotation per window, lazy relinearization, more pairs, layer 4, element-wise (timed, GPU 4) | 7.079 → 6.546 | ×1.082 | KL 8.5e-6 |
| exp degree 6, GELU degree 60 in blocks 0 and 4-9, up finishes a level lower | 6.554 → 6.488 | −66 ms | KL 4.5e-5, 32/32 |
| GELU degree 120 / 60 in blocks 1, 3 and 10 | 6.487 → 6.470 | −17 ms | KL 4.9e-5, 32/32 |
| q·Kᵀ as sums of products | 6.472 → 6.424 | −48 ms | KL 4.5e-5, 32/32 |
| exact conversions in their own kernels | 6.408 → 6.393 | −15 ms | byte-identical |
| eager rescales | 6.403 → 6.385 | −18 ms | KL 4.8e-5 |

## Disclosures

- **CKKS parameters and keys.** Unchanged: record 6's chains (n32sp for Decode-8, n32a4 for Prefill-128), main secret,
  special primes and sparse keys. Decode-8 asks for four more rotation keys (steps 240, N − 240 and the SparseRot steps
  256 and 768 in place of 1024 and 1536) at record 6's shapes; record 6's client generates and bounds them.
- **Approximations.** Decode-8's GELU (degree 60 / 222) and exp (degree 6) and Prefill-128's GELU (degree 60 / 120) and
  exp (degree 6) are least-squares fits weighted by each block's real inputs on OpenWebText, so they are GPT-2-specific
  constants; another model would refit them with the same scripts. Decode-8's layers 4 and 11 and Prefill-128's layer 4
  get softmax refits; Decode-8's LayerNorm seeds drop to degree 15. KL above, against float64 GPT-2.
- **Raise drops onto the region's top pair.** Dense drop 1 costs at most 0.25 bits of refresh precision against drop 0;
  SPRU drop 7 costs 1.5-2.5 bits at CF 9 (pc-bench precision grid). Both are inside the KL above.
- **CutMax.** `lm_centre` centres the LM head's columns: every position's logits shift by one constant, so the softmax,
  the KL and the argmax are unchanged, but the logit frames differ from record 6's by that constant.
- **Untimed setup.** The term-by-term mask leaves add plaintexts that the server encodes at setup (Decode-8: 4,669
  per position cache against record 6's 4,407); peak GPU memory rises from 59.4 to 65.0 GiB (Decode-8) and from 56.6 to
  57.3 GiB (Prefill-128).
- **Intermediate plans.** The plans the served ones derive from (n32a4 decode plans whose configs and runtime lines the
  n32sp scripts read, the prefill carry chain) stay in `bootstrap_placements/` so every step regenerates.
- **Build safety.** `scripts/utils/refit_pf_softmax_l4.py` (an offline fitting script, not built or served) reads
  `HF_HOME` to find the GPT-2 weights and tokenizer.

## Reproduce

- Decode-8: the served plan's PLAN_CMD.txt lists its steps: `speedrun/make_n32sp_decode_plan_gn.sh` (record 6's recipe
  and ILP on the n32a4 configs, `_gn`), the CutMax constants on its placements (`_gn_mu2`), its decisions carried onto
  the lighter configs' capture (`scripts/utils/map_plan_decisions.py`) and the raise-drop pass
  (`speedrun/make_n32a4_decode_rd.sh`) with `PLAN_RAISE_ROUTE_MAX=0:1`, then `0:1,1:7`.
- Prefill-128: `speedrun/make_prefill_carry_plan.sh` (capture, decisions carried, raise-drop pass, the window rotations).
- `dev_logs/`: the Dev result's runs (`dfinal_*`: Decode-8 against record 6; `pfinal_*`: Prefill-128 against record 5, in run
  order), the 32-prompt draws (`dev0-31_*`; Decode-8's four draws with the entry's own build as the client) and the
  byte-identity replays (`bit_identity.txt`). The Dev result ran on this branch's code before its rebase onto record 6's
  merge, which changes only the maintainers' README, progress and repro-default files.
