# Contributing

## Setup

```bash
git clone --recurse-submodules https://github.com/gladia-research-group/perseus
cd perseus
uv sync                     # the pure-Python package + the dev group (torch, pytest, ruff, ...)
```

The CUDA extension `perseus._core` is built separately (README → Install). Everything under
`perseus.nn` needs it at run time; `perseus.plan`, `perseus.calibrate`, `perseus.hub` and
`perseus.export` do not, and `python -c "import perseus.nn"` says what to run when it is missing.

## Tests

```bash
pytest                                  # the CPU tier: seconds, no GPU, no network
PERSEUS_ALL_PLANS=1 pytest tests/test_paper_plans.py   # every shipped plan regenerates (minutes)
pytest -m gpu tests/gpu                 # the GPU tier: a real session on the visible GPU
ruff check perseus tests scripts        # the lint gate
bash scripts/utils/lint_no_narrative.sh # no development-log residue in the tree
```

The CPU tier is what CI runs: the planner and its accuracy model, the ported baseline placers
against the vendored upstream outputs, the plan/env contracts, the module surface, the
client/server manifest, the generation loop over a fake session, and the no-extension import
path. Tests that need `perseus._core` skip themselves when it is not built. Every test gets a
private copy of `os.environ` (`tests/conftest.py`).

The runtime gate is `TASK=decode bash scripts/run_task.sh` (README → Reproduce a row): a run
passes on its own `PASS` marker with `unplanned_bts=0`; the exit code alone is not evidence.

## Conventions

- Library code never prints and never assigns into `os.environ`; log through
  `logging.getLogger(__name__)`. Console scripts configure logging themselves.
- Validate at the Python boundary: a wrong shape is a `ValueError` naming the expected one.
- `perseus/_core.pyi` and `perseus/_client.pyi` are generated: run
  `.venv/bin/python scripts/utils/gen_core_stub.py [perseus._client]` after a binding change.
- Plans are bound to the runtime and the calibration: a change to an op's math, a config
  value, or a default that alters the executed graph needs a recapture (`STAGE=capture`) and
  a replan (`scripts/make_plans.sh`), and the decode gate must still read 552 bootstraps per
  token with `unplanned_bts=0`.
- The two option structs in `include/client_context.h` mirror `include/fideslib_wrapper.h` /
  `include/inference.h` field for field (`tests/test_client_layout.py`).
