# `_sources/` — recount inputs that are not live anywhere else

A `_provenance.source_config` must resolve to a file that actually ships, otherwise the
recount is unauditable at the destination: you can read the claim "only counts changed"
but you cannot check it.

Three of the four recounted arms were derived from their published configs, which stay
live in `paper/`. The fourth was not:

| burst arm | derived from | lives in |
|---|---|---|
| gpt2_base | the published baseline | `paper/gpt2_base` |
| vit_base | the published baseline | `paper/vit_base` |
| vit_squeeze | the published L_range arm | `paper/vit_squeeze` |
| **gpt2_squeeze** | **`gpt2_squeeze_tg`** | **here** |

`gpt2_squeeze_tg` is the thor-GELU twin of the published squeeze arm. It is the recount
source because the *published* squeeze arm carries an iterative softsign GELU (gs 14 +
Newton 1 across 12 sites) that no ledger counts, while base and heat use the
non-iterative `thor_composite`. Recounting from the published arm would have compared
two different GELU forms and attributed the difference to the recount.

Not a lane you deploy from — it exists so the provenance chain closes.
