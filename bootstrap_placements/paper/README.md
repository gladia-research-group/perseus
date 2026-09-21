# paper/ — plans behind the published numbers

One directory per published arm, including `planned_gpt2_heat` and `planned_vit_heat`. The HEAT
copies are byte-identical to the burst ones today: HEAT was untouched by the 2026-08-14 recount
(its counts are learned halting modes, not calibrated ones), so its *config* did not change.

**A config that did not change does not imply a plan that still binds.** Plans are bound to the
FIDESlib binary as well as to the config, and the pin moved after the HEAT artifacts were made:

| artifact | captured | pin at capture |
|---|---|---|
| `planned_gpt2_heat` | 2026-07-28 | `4619590` (then `main`) |
| `planned_vit_heat` | 2026-07-30 | `4619590` |
| every base / squeeze plan here and in the burst | 2026-08-14 | **`509990c`** (`behemoth-port`) |

`509990c` = `4619590` + two commits (S7 host-race guards + CUDA-13 teardown; BERT multi-chunk
staging). Both read as behaviour-neutral for GPT-2 decode and single-chunk ViT forward, which is
why the HEAT plans were carried forward rather than regenerated — but "reads as neutral" is not
the standard this repo uses, because a stale plan fails as a **silent wrong answer**, not an
error.

The decisive test is cheap and is the one the top-level `CLAUDE.md` prescribes: captures are
deterministic, so recapture HEAT on the current pin and diff the graph against the archived one.
Node-identical proves the carried-forward plan is valid; any diff means recapture + replan.
See `docs/CAMPAIGN_recalib_1e-4_2026-08-14.md` §8 for the result of that diff.

Until that diff is on the record, treat the HEAT rows as measured on a **different library pin**
than the base/squeeze rows they are compared against.
