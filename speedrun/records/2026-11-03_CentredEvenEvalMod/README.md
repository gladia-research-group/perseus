# Decode-8 on an even EvalMod series, one level shallower, with a raise centred in q₀q₁

EvalMod's target at K = 16, (2π)^(-1/32) cos(2π(16y − 0.25)/32), is even in w = y − 1/64, so it is a Chebyshev series of
about half the degree in z = 2w² − 1. Record 7's degree-14 series in y becomes a constant add, one square and a degree-5
series in z, followed by the same 5 double angles: 9 CKKS levels instead of 10. Every K = 16 refresh (dense, StC-first,
SparseB; SPRU has no EvalMod) raises and runs CtS one level lower, on fewer limbs, and lands where it did, bit for bit. Placements, landings,
keys and client are record 7's. The series is a least-squares fit weighted by the overflow integer I's distribution, so
it is only accurate on that distribution, and FIDESlib's composite raise gives a wider one. With the series on, the raise
therefore centres in q₀q₁. Dev set, RTX PRO 6000, 6 pairs, **outside the harness's sandbox**: 0.9575 s against 1.005 s per
token, ×1.050, mean KL 2.42e-5, argmax 100 %. Read the Disclosures: this entry has no harness run, and a large tail error
is catastrophic.

## Changes against record 7 (Decode-8)

- **The even series**: `EVALMOD_EVEN_CHEB` in `crates/pc-core/src/bts_setup.rs`; `PsPlan::evalmod_even` and
  `build_even_prestep` in `crates/pc-shadow/src/evalmod.rs`; config `bts.em_even` / `PC_EM_EVEN`
  (`crates/pc-config/src/fields.rs`, `crates/pc-py/src/session.rs`).
- **One level lower**: `BtsSetup::even_levels_saved` in `crates/pc-gpu/src/bts/mod.rs`, used by `dense.rs` (and StC-first,
  which builds on its variants) and `sparse_b.rs`. Tests: `even_lands_as_shipped_one_level_lower`,
  `even_variants_raise_lower_and_land_as_shipped`, `even_variants_need_the_same_rotation_keys`.
- **The centred raise**: `Lift` in `crates/pc-gpu/src/bts/raise.rs`. `Lift::Centred` skips `mod_raise` and has `small_int`
  reconstruct from q₀ and q₁, so one kernel less. It is on only with the even series (`BtsSetup::raise_lift`,
  `SparseBCfg::raise_lift`). No kernel changed. With the series off, the raise uses the same tables and kernels as record 7.
- **The plan**: `bootstrap_placements/gpt2_decode_python_n32sp_kc16r_gnabd7_z5` is `_gnabd7` file for file; only
  `PLAN_CMD.txt` differs (`PC_EM_EVEN=1`). `speedrun/server.py` serves it.

## Why the raise has to change

I = (c₀' + c₁'s − m) / Qd counts the multiples of Qd = q₀q₁ the lift adds. FIDESlib's `mod_raise` centres each Garner term
on its own prime, so every coefficient is a sum of two uniforms. Measured on the real `bts.cuh` kernels compiled for the
CPU (a shim defines the CUDA vocabulary and the launchers are cut;
[`tmodel/z5/kernelhost`](https://github.com/z0cal/perseus-speedrun/tree/z5-evidence/tmodel/z5/kernelhost)), production sparse secret, uniform c₁, 1.31e9
coefficients (`dev_logs/raise_kernels_cpu/census.log`):

| | FIDESlib lift | Centred lift |
|---|---|---|
| σ of I | 2.3456 | 1.6588 |
| P(\|I\| ≥ 8) | 1.19e-3 | 2.14e-6 |
| P(\|I\| ≥ 10) | 3.69e-5 | 7.6e-10 (1 case) |
| P(\|I\| ≥ 12) | 5.3e-7 | none |
| Largest \|I\| | 14 | 10 |

The centred column agrees with the density of 33 uniforms the series was fitted to. On both lifts the raised ciphertext
decrypts to c(m + I·Qd) within the rounding bound (`raise_kernels_cpu/scaled.log`).

## Ablation

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| Even series at degree 7, no level saved (not in this PR, commit `fafd3b8`, measured on another pod) | decode | ×1.022, mean KL 2.1e-5 |
| Even series at degree 5, one level less, FIDESlib's raise | decode | Fails: request KL 1.5e-3, 257, 601, 1.7e-3 (one run, no pair; `dev_logs/perprime_raise/`) |
| The same with the centred raise (this entry) | decode | ×1.050, mean KL 2.42e-5 |
| This entry, series off on this track | prefill | ×1.000 (6.517 s against 6.518 s, 2 pairs), mean KL 4.89e-5 |

## Dev result

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, one RunPod container, the record and the entry alternated. Commit
`224c286` (this PR's head differs only by this folder).

| Track | Entry | Record 7, same GPU | Mean KL, entry | Mean KL, record | Argmax |
|---|---|---|---|---|---|
| Decode-8, 6 pairs, prompts 0-23 | 0.9575 s/token | 1.005 s/token | 2.42e-5 | 2.23e-5 | 100 % |
| Prefill-128, 2 pairs, prompts 0-7 | 6.517 s | 6.518 s | 4.89e-5 | 4.45e-5 | 100 % |

`score.py` on the `run.json` copies here: ×1.050, p 1.5e-9. Largest KL at one position: entry 2.2e-4, record 2.0e-4.

Refresh precision at A = 1, in bits (`pc-bench`; StC-first over its four `own_cf` / `real` settings), logs in `dev_logs/`:

| Route | Record 7 | Even series, FIDESlib raise | This entry |
|---|---|---|---|
| Dense, CF 6 | 12.87 | 7.28 | 13.00 |
| StC-first, CF 8 | 13.27-13.28 | 6.31-9.77 | 13.16-13.27 |
| StC-first, CF 13 | 11.57-14.40 | 1.31-4.90 | 10.87-11.62 |

Refresh device time, ms per ciphertext, graph replay, same pod (`bts_all_n32sp_*.log`):

| Route | K = 1, record → entry | K = 2, record → entry |
|---|---|---|
| StC-first | 10.69 → 9.74 | 9.33 → 8.41 |
| SparseB-256 | 9.19 → 8.22 | 8.40 → 7.43 |
| SparseB-512 | 10.09 → 9.09 | 9.06 → 7.91 |

## Disclosures

**No harness run.** The entrant has no machine that can run the harness's sandbox (`bwrap`, a systemd user session and
48 GB of GPU together). The dev runs ran in a RunPod container on the harness's own runner, clock, protocol, KL and
`score.py`, copied from record 7's `speedrun/`, with `speedrun/runner/sandbox.sh` replaced by a stand-in with no isolation
(`dev_logs/harness_stand_in/`). Their `run.json` has `harness: null` and `score.py --official` refuses them. Treat every
number here as indicative: the maintainers' timing decides. The build has not run in the real sandbox.

**Changed approximation, with a worse tail.** The series is not minimax on |I| ≤ 15. Its error grows where I's distribution
thins out. Worst coefficient error in bits, over |δ| ≤ 2⁻¹³ (`dev_logs/raise_kernels_cpu/tail.log`):

| \|I\| | 0 | 4 | 6 | 8 | 9 | 10 | 11 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|---|
| Record 7 | 36.2 | 30.9 | 31.2 | 30.1 | 31.1 | 30.3 | 29.6 | 31.8 | 29.8 |
| This entry | 31.4 | 27.4 | 24.6 | 20.6 | 17.9 | 15.9 | 14.3 | 12.9 | 11.1 |

At StC-first CF 13 this entry's worst slot error is up to 3.5 bits worse than record 7 with `real=false` (10.87-10.92 bits
against 14.06-14.40), level with it with `real=true` (11.48-11.62 against 11.57-11.58), and its RMS is 1.42-2.22e-5 against
1.07-1.48e-5.

**A large tail error is catastrophic, and we do not know the mechanism.** With FIDESlib's raise, two of four requests went
to KL ≈ 687 at one position and stayed there for every later position. One such event in an official run would fail the
mean. The centred raise makes the event rare; it does not make it impossible. No catastrophe in 28 entry requests (24 plus
a gate pair), which alone only bounds the rate at about 10 % per request.

To measure it we injected faults on a throwaway branch (`dev_logs/fault_injection/`, patch included, **not in this PR**):
`small_int` shifts I by ±k on about one coefficient per refresh, far above the natural rate. The patch was checked on the
CPU kernels (only the chosen coefficients move, by exactly ±k). Runs on prompts 0-3:

| k | Result |
|---|---|
| 14 | All 4 requests at KL ≈ 687 from position 0 |
| 10 | All 4 requests at KL ≈ 687 from position 0 |
| 8 | 3 of 4 requests reach KL ≈ 687 (positions 9, 9, 5); the other stays below 1, mean 0.067 |

Crossing the k = 8 rate with the two catastrophes of FIDESlib's raise suggests that errors become catastrophic from
|I| ≈ 12-13 (11 is unlikely, 10 is excluded), which would put this entry's rate at about 1e-8 per request, and at about
6e-5 per request in the worst case we considered (threshold 11, every coefficient sensitive). This is our estimate, not a
measurement: it assumes one sharp threshold and about 50 events per token, and rests on 5 events. The run without faults
on that pod was cut for lack of credit, so the clean reference is the A/B above.

**The raise is no longer FIDESlib-bit-exact when the series is on.** `check_bts` and the oracle checks use `Lift::PerPrime`.

**Keys and parameters: none changed.** Same secret, sparse keys, chain and client as record 7. The key request was generated
on the CPU and is identical (170 keys).

**Dev prompts.** The fit used only the analytic distribution of I: no prompt, weight or capture. Scripts:
[z0cal/perseus-speedrun, `tmodel/z5/`](https://github.com/z0cal/perseus-speedrun/tree/z5-evidence/tmodel/z5) (`em_rms.py`,
`profile_by_I.py`); result in `dev_logs/raise_kernels_cpu/even_ls_k16_z5_r5.json`.

**New untimed work:** none.

**Open:** record 7 itself has a peak of about 3.3e-4 at StC-first CF 13 with `real=true`, independent of the amplitude and
unexplained by EvalMod's error; it may add to this entry's tail. In the entrant's environment `shipped_stamps_round_trip`,
the `oracle_*` tests (their dumps are missing) and 15 Python tests fail, the same ones with and without this change.

**AI assistance.** Written with Claude Code, which wrote essentially all of the changed lines.
