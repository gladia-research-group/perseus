# Decode-8 with its position plaintexts encoded at setup and its SparseB on the degree-14 EvalMod; both tracks on a leaner key switch

Record 4's code and plans (LeanChainDecode, LeanChainPrefill), with the served decode's host work trimmed, Decode-8's
SparseB refreshes on Prefill-128's EvalMod series, and fewer instructions and shorter waits in the key switch's kernels,
which both tracks run. The host-side and kernel changes leave what the device computes as it was: on one key bundle and
one set of input ciphertexts, the host-side changes give outputs byte-identical to record 4's, and the kernel changes
outputs byte-identical to the state before them (Prefill-128's, whose plan was already on K = 16: byte-identical to
record 4's).

- **The position plaintexts, encoded once at setup.** Record 4's server already records every position's patch at
  setup (the plaintexts whose values change from one position to the next: masks and position constants, 94 to 97 per
  position after aliasing), but encodes them in the clock, one device encode and one host-device sync each. Here the
  1,550 distinct values the 16 positions stage are encoded once at setup into a device cache (7.4 GB), and a token copies
  its position's encodings into their slots, device to device, in place (the slots keep their addresses, so the
  captured graphs that read them stay valid).
- **CutMax issued behind the forward.** The server exported the logits (which waits for the device) before issuing
  CutMax, so the device idled while the host downloaded the logits and launched CutMax. Now CutMax is issued first,
  and both outputs are exported after it.
- **Ciphertext frames read and written in place.** Importing the 12.8 MB input frame took about 10 ms: a slice copy
  in Python, a copy into a word vector, then 25 row uploads per polynomial, each followed by a stream sync. Now the
  frame's words are checked and uploaded where they lie (one copy per polynomial), and an exported frame is downloaded
  straight into its bytes. The server reads its input without the slice copy and writes the packed outputs without
  joining them.
- **The next position's patch, ahead of its input.** Once a position is answered, the next position's dynamic steps and
  plaintext copies are applied while the client sends the next token.
- **SparseB refreshes on the degree-14 EvalMod series (K = 16).** Record 4's Decode-8 plan runs its 40 SparseB refreshes
  per token with K = 24, FIDESlib's degree-20 table; its Prefill-128 plan already sets K = 16, the dense route's degree-14
  table (`FIDESLIB_BTS_SPARSE_B_K=16`). The plan compiles to the same levels and landings; each SparseB refresh is about
  0.7 ms cheaper on the RTX PRO 6000.
- **Base conversion's pre factors in the INTT's last pass.** A fast base conversion first multiplies each source
  coefficient by its pre factor `(B / b_i)^-1 mod b_i`, a constant of the source row. Every conversion in a key switch
  reads the output of an INTT, whose last pass already multiplies each row by `N^-1`, so the INTT now multiplies by
  `N^-1 pre_i` instead and the conversion reads `x_i pre_i` as written: the ModUp (every live digit, per level), the
  ModDown (P, and P' at the bottom), and the ModDowns of the linear-transform stages and EvalMod.
- **Double-word sums reduced in 32-bit products.** The sums of 28-bit products (base conversion, the inner products,
  CtS, the tensor product, the keys' regeneration) were reduced with a 64-bit Barrett. Each is now `hi 2^32 + lo`,
  with `hi (2^32 mod p)` and `lo` reduced as 32-bit Shoup products (exact for every 64-bit sum when p < 2^31): the same
  residue in about half the instructions. Base conversion keeps its targets' constants in shared memory.
- **L2 prefetch of the next key reads.** CtS (K = 1) reads its key `b`, a resident `a` digit, a ModUp digit and the
  diagonals digit by digit, and each load waits on DRAM; a baby step now first asks L2 for the next baby step's lines.
  The K = 1 inner product asks L2 for the next digit's segment the same way, at three blocks per SM.
- **The NTT's row passes at five blocks per SM.** Their loads wait on L2 (the passes are L2-resident); at 48 registers,
  without spills, more warps hide the wait.

The kernel changes, per refresh on pc-bench (`bts --all --keys packed`, the lean chain, the RTX PRO 6000): StC-first
13.00 → 12.43 ms, SparseB-512 13.06 → 12.08 ms, SPRU 7.34 → 6.79 ms, with outputs bit-identical.

On the reference GPU (RTX PRO 6000 Blackwell Server Edition), Decode-8 runs at 1.645 s/token against record 4's 1.793
on the same GPU (×1.090, 6 pairs, p 2.4e-12), with the KL at record 4's level (8.4e-6 mean on all 32 dev prompts).
Prefill-128 runs at 9.842 s against record 4's 10.13 (×1.029, 6 pairs, p 2.3e-10), with outputs byte-identical to record
4's: the kernels' gain (before them, its prompt's import through the same `import_ct` took about 0.3% off, unpaired).

## Changes against record #4

- **The position cache.**
  - `PtStore::copy_out` / `copy_in` (a resident plaintext's rows to a buffer and back, in place) and `words_at`:
    `crates/pc-gpu/src/pt_store.rs`.
  - The session's cache, in its own device allocation, and `PtStore.cache` / `PtStore.restore`:
    `crates/pc-py/src/{session,store}.rs`.
  - `PlannedRun(pt_cache=True)`: `precompute` encodes every position's staged values into the cache, stages position
    0's plaintexts whole at setup, and `token` restores instead of encoding: `python/perseus/impl/driver.py`.
  - The served decode takes `pt_cache` (and `--pt-cache` on its own CLI): `python/perseus/serve/gpt2_decode.py`;
    `speedrun/server.py` turns it on.
- **CutMax issued before the exports:** `Gpt2Handler.run` in `python/perseus/serve/gpt2_decode.py`.
- **Frames in place.**
  - `DBuf::upload_bytes_at` / `download_bytes_into` (whole elements from or into bytes at any host alignment):
    `crates/pc-cuda/src/mem.rs`.
  - `import_ct` reads a frame (bytes or a memoryview) in place, with `CtWire::from_bytes`'s checks; `export_ct` writes
    `CtWire::to_bytes`'s frame directly: `crates/pc-py/src/client.rs`. The client crate (`crates/pc-client`) is
    unchanged.
  - `speedrun/server.py`: `read` without the slice copy, `write_packed` and `unpacked` (record 1's list framing without
    joining or slicing copies).
- **The patch ahead:** `PlannedRun.patch_ahead` (`python/perseus/impl/driver.py`), `Gpt2Handler.ahead`, and the call in
  `speedrun/server.py`'s decode loop.
- **SparseB K = 16:** the runtime lines of `bootstrap_placements/gpt2_decode_python_n32a4_kc16r{,_cm}/PLAN_CMD.txt`, and
  `speedrun/make_n32a4_decode_plan.sh`, which writes them (its placements are unchanged).
- **Base conversion's pre factors in the INTT** (kernel ABI 16 → 17).
  - The inverse column pass takes a per-row scale (`NttArgs.es` / `ess`, offset per chunk): `ntt_inv_cols{,8}` in
    `crates/pc-kernels/kernels/ntt.cuh`; `ntt::inverse_rows_prefix_scaled` / `inverse_k_scaled` in
    `crates/pc-gpu/src/ntt.rs`.
  - `BconvJob.prescaled` (the conversion reads its input as the pre-multiplied value): `bconv.cuh`, `args.cuh`,
    `crates/pc-gpu/src/bconv.rs` (`pre_factors`).
  - The tables (`N^-1 pre_i` and its Shoup word per row): the ModUp's per level and the ModDown's for P and P'
    (`crates/pc-gpu/src/ks.rs`), the linear transforms' ModDown (`MdPlan` in `crates/pc-gpu/src/bts/cts.rs`).
- **The split reduction.**
  - `PrimeConst.split` (`2^32 mod p` and its Shoup word, in the last word of a prime's constants):
    `crates/pc-core/src/modmul.rs`, `args.cuh`; the 32-bit `reduce_dw` in `arith.cuh` uses it.
  - Base conversion's own path, with its targets' constants in shared memory (`BconvJob.split`, set when every target is
    below 2^30): `bconv.cuh`, `crates/pc-gpu/src/bconv.rs`.
- **The prefetch:** `cts_fused_rq` in `crates/pc-kernels/kernels/cts.cuh`, `ks_dot_seg` in `ks.cuh`.
- **The row passes' occupancy:** `ntt_fwd_rows` / `ntt_inv_rows` in `ntt.cuh`.
- **Tests:** `python/tests/test_speedrun_server_framing.py` (the server's framing writes and reads `record1.pack`'s
  bytes); `crates/pc-kernels/src/bin/selftest.rs` gains prescaled and split base conversions and the row-scaled INTT
  (lazy and not, chunk offsets 0 and 5), against the CPU, and its prime table now passes the new word.

## Ablation

The speedrun harness on the RTX PRO 6000 of the Dev result, two prompts per run (dev 0-1). Each row adds one change to
the row above (this PR's commits in order). The first four rows' five trees were timed in one job: 4 rounds,
alternately forward and backward, each row scored by `score.py` against the row above it (logs `abl_*`); those rows give
byte-identical outputs. K = 16 was timed against the row above it in its own job (4 pairs AB BA BA AB, `ab4_*`). The
kernel rows were timed in a second job of the first kind, from record 4 through K = 16 and each kernel commit (six trees,
4 rounds, `fabl_*`; K = 16 against record 4 there: 1.794 → 1.733, ×1.035, p 1e-6); they give outputs byte-identical to
K = 16's.

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| Position plaintexts encoded at setup, copied per token | decode | 1.792 → 1.772 s/token, ×1.011 (p 3.4e-5) |
| CutMax issued before the exports | decode | 1.772 → 1.771 s/token, ×1.000 (p 0.11, not resolved) |
| Frames imported and exported in place | decode | 1.771 → 1.762 s/token, ×1.005 (p 9.8e-6) |
| The next position's patch ahead of its input | decode | 1.762 → 1.760 s/token, ×1.001 (p 0.003) |
| SparseB refreshes on the degree-14 EvalMod series (K = 16) | decode | 1.762 → 1.733 s/token, ×1.017 (p 6e-5) |
| Base conversion's pre factors in the INTT's last pass | decode | 1.733 → 1.699 s/token, ×1.020 (p 1.1e-5) |
| Double-word sums reduced in 32-bit products | decode | 1.699 → 1.664 s/token, ×1.021 (p 3.3e-6) |
| L2 prefetch of the next key reads | decode | 1.664 → 1.650 s/token, ×1.009 (p 2.7e-4) |
| The NTT's row passes at five blocks per SM | decode | 1.650 → 1.646 s/token, ×1.003 (p 0.011) |
| **This entry** vs record 4 (the Dev result) | decode | **1.793 → 1.645 s/token, ×1.090 (p 2.4e-12)** |
| **This entry** vs record 4 (the Dev result) | prefill | **10.13 → 9.842 s, ×1.029 (p 2.3e-10)** |

- CutMax's order is not measurable here: the gap it closes (the logits' download and CutMax's launch) is a few
  milliseconds against the harness's noise. It stays because it removes an idle device gap at no cost and its outputs are
  byte-identical.
- Prefill-128 is not ablated per commit: the host-side commits leave its device work as record 4's (byte-identical
  outputs; about 0.3% from the in-place import, unpaired), its plan was already on K = 16, and the kernel commits are
  what it gains.

## Dev result

The speedrun harness on one RTX PRO 6000 Blackwell Server Edition (96 GB; the server pinned to its NUMA node, `NUMA=3:3,4`),
record 4 (`d97a6f7` with its lean client `24d4b3f`) against this entry, 6 pairs in the order AB BA BA AB AB BA, dev
prompts 0-1, two requests per run, scored by `score.py`, one job per track. Logs in `dev_logs/` (Decode-8 `ffinal_*`,
Prefill-128 `fpre_*`). Another user's jobs kept GPU 6 busy in Decode-8's runs 1-10 and GPU 5 (on GPU 4's PCIe switch) in
its runs 8-12, whose scores sit within the others' range in both arms, and GPU 6 busy through Prefill-128's job, GPU 5
idle (`*_contention.tsv`).

| Track | Score | Record 4, same GPU | KL | Argmax (scored) | Peak GPU memory |
|---|---|---|---|---|---|
| Decode-8 | 1.645 s/token | 1.793 s/token | 4.3e-6 (positions 8-15) | 96/96 | 56.3 GiB (record 4 49.0) |
| Prefill-128 | 9.842 s | 10.13 s | 8.6e-6 (position 127) | 12/12 | 59.3 GiB (record 4 59.3) |

score.py, Decode-8: quality PASS (mean KL 4.28e-6, p 1e-32), speed ×1.090 (p 2.4e-12), NEW RECORD. Record 4's arm took
1.791-1.795 s/token, against its published 1.798; this entry's 1.640-1.647. Prefill-128: quality PASS (mean KL 8.64e-6,
p 5.5e-40), speed ×1.029 (p 2.3e-10), NEW RECORD. Record 4's arm took 10.10-10.15 s per request, against its published
10.12; this entry's 9.83-9.86.

- All 32 dev prompts through this entry, one session per track: Decode-8 (`dev0-31_entry.json`) mean KL 8.4e-6 (median
  7.1e-6, max 2.3e-5), argmax 256/256; Prefill-128 (`dev0-31_prefill_entry.json`) mean KL 1.2e-5 (median 8.9e-6, max
  1.3e-4), argmax 32/32; every request within its bar.
- Earlier states, same protocol: the commits up to K = 16 ×1.036 (1.796 → 1.734 s/token, p 3.7e-9, `ab6_*`, GPUs 5-7
  idle); with another user's jobs keeping GPU 5 (on GPU 4's PCIe switch) 97-99 % busy:
  the host-side changes alone ×1.018 (1.793 → 1.760 s/token, p 2.1e-8, `ab2_*`), and an earlier state of them (the
  cache, CutMax's order and the in-place import) ×1.016 (1.795 → 1.767, p 2e-7, `ab1_*`).

## Disclosures

- **Outputs.** The host-side changes (every commit before K = 16) give outputs byte-identical to record 4's on one key
  bundle and one set of input ciphertexts (the lean client's, for dev prompts 0-1), both requests and all 16 positions:
  32 of 32 output frames equal, with record 4's own replay equal to itself (`dev_logs/bit_identity.txt`). The cached
  encodings are the encoder's own output for the same values, copied. The kernel commits give outputs byte-identical
  to K = 16's, same protocol (32 of 32): each is exact modular arithmetic in another order (the pre factor multiplied in
  with `N^-1`, the same residue reduced in two halves), a prefetch hint or a launch bound. Prefill-128 through this entry,
  and through the commits up to K = 16, gives output frames byte-identical to record 4's, same protocol (2 of 2).
- **SparseB at K = 16.** The EvalMod series of the SparseB refreshes changes, as on Prefill-128 since record 4: the dense
  route's degree-14 table over (-16, 16) instead of the degree-20 one over (-24, 24) (the integer after the fold has the
  dense route's distribution). Same levels, landings and plan. Refresh precision, pc-bench's grid on the lean chain (4
  seeds, packed keys), at each amplitude's best CF: 17.98 → 17.77 bits (A 0.5, CF 5), 17.32 → 16.08 (A 4, CF 10),
  17.88 → 17.61 (A 16, CF 10); at most 1.2 bits less at the CFs the plan uses. The plan's accuracy tables are record
  4's (measured at K = 24). KL in the Dev result.
- **New untimed work.** At setup, the 1,550 distinct position plaintexts are encoded into the cache (2.7 s), and
  position 0's plaintexts are staged. Both depend on the position only: the values are the ones record 4's `precompute`
  already records at setup from the CPU twin on public placeholder rows. Nothing depends on the prompt.
- **Unscored positions.** No work moves into positions 0-7. Position 0 of a session's first request gets faster (2.37-2.47
  s against record 4's 3.67-3.93 in the A/B), because staging every plaintext of position 0 moves from its clock to
  setup; a later request's position 0 is a steady token in both.
- **Memory.** The cache takes 7.4 GB of device memory in its own allocation (sized to the values it holds): peak GPU
  memory 56.3 GiB against 49.0. Host memory is unchanged (server peak 20.7-21.4 GB in both arms). The kernel commits'
  scale tables (two words per row of each INTT they fold into) leave the peak where K = 16 had it. Prefill-128's peak
  is record 4's (59.3 GiB in both arms).
- **Keys, parameters, plans, approximations, client:** record 4's, unchanged (the plans' placements too; only their
  runtime lines gain `FIDESLIB_BTS_SPARSE_B_K=16`). The key request is byte-identical.
- **Scope.** The cache is on for the served 16-position decode only (`pt_cache`, off by default in `PlannedRun`): its
  size grows with the positions a run stages (about 470 MB per position here).
