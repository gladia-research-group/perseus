# Layout of this directory (2026-08-14)

Three lanes, and the same names appear in all three trees — `configs/model/approximation/`,
`.cache/graph_*`, `bootstrap_placements/planned_*`. A config, its graph and its plan for the same
arm always live under the same-named lane. That is not cosmetic: a plan built from one arm's graph
is meaningless against another arm's config, and the failure mode is a **silent wrong answer**, not
an error. Name alignment is what makes a bad pairing visible.

```
paper/     the artifacts behind every PUBLISHED number — live, not archived, because the
           paper cites them and they must stay reproducible until a new campaign replaces them
atlas/     the ATLAS search lane (vit_atlas, bert_atlas): its counts come from its own search
           method, not from any calibration criterion, so it is never in the base/squeeze/heat set
<bare>     THIS burst — the current best artifact for each arm
_backup/   superseded and retired, never deleted, with a MANIFEST
```

Every model has exactly three arms: **`base`**, **`squeeze`**, **`heat`**.

**Out of scope for the 2026-08-14 recount, by ruling:** `bert_*` (BERT is not part of this port —
its *counts* are untouched) and `vit_atlas` (its counts come from the ATLAS search method, not from
any calibration criterion, so a calibration target does not apply). `vit_base_112` is the 112-pixel
tier and stays live because `scripts/run_task.sh`'s `vit112` task resolves it.

### BERT naming, consolidated 2026-08-15

BERT now uses the same three names as everything else: **`bert_base`, `bert_squeeze`, `bert_heat`**
(ledgers 616 / 556 / 201). `bert_base_128_v3` was **promoted into `bert_base`** — it is the shipping
baseline per `docs/backup/HANDOFF_bert_atlas_campaign_2026-08-10.md:65` and the bert-heat-lane
memory. `_128` was the SST-2 sequence length; `v2`/`v3` were successive fixes, `v3` carrying the h.9
poisoned-fit repair. The old `bert_base` (ledger 593, a structurally different calibration),
`bert_base_128` and `bert_base_128_v2` are archived under `_backup/_bert_naming_*` with a manifest.

Two things this rename settled:

- **The existing plan stays valid.** All three `_128` revisions have *identical per-site iteration
  counts* — they differ only in constants (v3 vs `_128` differ in exactly
  `softmax.…h.9.attn.refine_{alpha,beta}`, the poisoned fit). Topology and level accounting are
  therefore unchanged, so `.cache/graph_bert_base` + `planned_bert_base` (captured 2026-08-09,
  before the repair) still bind. Verified by field-level diff, not assumed.
- **It fixed a latent mismatch.** `scripts/run_notebooks.sh` already resolved `bert_base` +
  `planned_bert_base`; before the promotion that paired the *593-count* config with a plan built for
  a *616-count* circuit.

⚠️ **`bert_base` FAILS `lint_approx_config.py`**, and the reason is real, not cosmetic: it carries
**fixed** LayerNorm Goldschmidt (12) and **fixed** softmax init (8) rather than adaptive per-site
counts. That is the same defect class the 1e-4 recount corrected for GPT-2 and ViT, and BERT was
excluded from that recount by ruling — so it remains open. `bert_squeeze` *is* adaptive (4/5/6/8);
the base is the fixed one. If a BERT baseline number is ever published, it is **not comparable in
kind** to the GPT-2/ViT baselines until it is refitted under an explicit criterion.

## What the old suffixes meant (so nobody has to reverse-engineer them again)

| retired name | meaning |
|---|---|
| `gpt2_squeeze_tg` | the thor-GELU twin of `gpt2_squeeze`, and the source of the recount, because the shipped squeeze arm carried an *iterative* softsign GELU (gs 14 + Newton 1 × 12 sites = 180 iterations that no ledger counts) while base and heat use the non-iterative `thor_composite`. **Not retired — it ships in `_sources/`**, because a provenance pointer that resolves to nothing is not provenance. |
| `*_refine`, `*_refine2`, `*_cm`, `*_cmv*`, `*_kcr`, `*_deep` | the free-generation / e33 debugging ladder from the behemoth AR campaign. Archived as `_prerefit`. |
| `*_gen`, `*_prefill_T128*` | autoregressive-generation lane (distinct from the paper's teacher-forced decode). Archived as `_prerefit`. |
| `*_SHIPPED_ab` | same-machine A/B reference: the *shipped* config captured on behemoth, so new-vs-old comparisons are not cross-machine. Archived as `_ab_reference` — **not** pre-refit artifacts. |

## The burst (2026-08-14 recount)

Counts re-derived against a flat **1e-4** target on each approximant's own output, minimising
multiplicative levels (1 Goldschmidt = 1 level, 1 Newton = 3), Newton capped at 4. Data-free: the
tool reads each config's own stored bands, so it reproduces bit-for-bit anywhere. HEAT arms are
untouched — their counts are learned halting modes and the criterion does not apply. CutMax is
frozen and excluded. See `scripts/recount_1e4.py` and each config's `_provenance` block.

| arm | published | burst | recounted from |
|---|---|---|---|
| gpt2_base | 684 | **712** | `paper/gpt2_base` |
| gpt2_squeeze | 454 | 455 | **`_sources/gpt2_squeeze_tg`** — *not* the published arm |
| gpt2_heat | 228 | 228 (untouched) | — |
| vit_base | 573 | **617** | `paper/vit_base` |
| vit_squeeze | 508 | **529** | `paper/vit_squeeze` |
| vit_heat | 250 | 250 (untouched) | — |

⚠️ **The squeeze row is not a "+1".** 454 and 455 are not two counts of the same circuit: the
published squeeze arm carries an *iterative softsign* GELU (gs 14 + Newton 1 across 12 sites =
180 iterations that no ledger counts), while base and heat use the non-iterative
`thor_composite`. The recount was therefore run on `gpt2_squeeze_tg`, the thor-GELU twin, so the
ledger delta reflects the recount alone. The **GELU form change is the dominant effect** and it
shows up where the ledger cannot: the plan drops from 309 to 225 placements (−27%). Quoting
"454 → 455" as the cost of recounting the shipped squeeze arm would be wrong twice over.

## Trap

`scripts/extract_paper_numbers.py` must read published rows through `PAPER_CFG` (= `paper/`).
The canonical names now hold the **burst**, so a stale path resolves to the new circuit silently.
There is a second, older trap in the same family: this tree (`src/perseus/configs/model/approximation/`)
and the training tree (`configs/model/approximation/`) share directory names but hold different
artifacts — deployed circuits vs trainer calibrations. Always say which tree you mean.
