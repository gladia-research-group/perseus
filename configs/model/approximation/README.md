# Approximation configs — canonical naming (2026-08-13)

One name per arm, matching the GPT-2/BERT scheme (`gpt2_{base,squeeze,heat}`,
`bert_{squeeze,heat}`):

| config | is | ex-name |
|---|---|---|
| `vit_base` | THE deployed ViT-80 baseline: adaptive cert on the FP EuroSAT ft80 teacher (573 it/fwd raw). Search source of `vit_atlas`. | `vit_base_80_ft` |
| `vit_squeeze` | range-only squeeze arm (508) | `vit_squeeze_80` |
| `vit_heat` | THE deployed heat arm (user-ratified frozen deploy: mode counts + trained-final recalib constants; 230 quoted / 250 executed — matches paper tab:plcs-vit) | `vit_heat_80` (ex `_mode`) |
| `vit_atlas` | ATLAS-searched counts-only circuit on `vit_base` (366) | — (new 08-12) |
| `vit_base_112` | the 112/complex tier (separate lane; ⚠ still fixed-count, violates the adaptive-baselines ruling — recalib when that thread reopens) | — |

Plans/graphs follow: `planned_vit_{base,heat,squeeze,atlas}` /
`.cache/graph_vit_{base,heat,squeeze,atlas}`.

Retired to `_backup/` (evidence, do not deploy): `vit_base_80`
(tiny-imagenet-calibrated, never deployed), `vit_heat_80_nokd` (95%-mass
counts bug, the paper's "371"), `vit_heat_80_trained{,_lnr}` (faithful-builder
variants, 257 executed — constants contradict the published table),
`vit_nokd_adaptive`, `vit_heat_80_ntc1d`.

GPT-2 and BERT dirs are already canonical; the historical campaign scripts
(`scripts/_backup/`, chain_*.sh in the training repo) reference the old names
on purpose — they are records of what ran, not live tooling.
