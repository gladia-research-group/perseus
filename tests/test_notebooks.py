"""CPU validation of the demo notebooks (they are never executed in-tree: they need a GPU).

Three notebooks are generated (`scripts/utils/make_*_notebook.py`): each tracked file must
be exactly what its generator builds, the generator must be deterministic, every code cell
must parse. Every notebook under `notebooks/` — generated or hand-written — must ship
without saved outputs and without references to a machine, a scheduler or an artifact that
is not in the checkout.
"""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
GENERATED = {   # notebook -> generator (scripts/utils is not a package: loaded by path)
    "setup_artifacts.ipynb": "make_setup_notebook.py",
    "gpt2_torch_forward.ipynb": "make_gpt2_notebook.py",
    "gpt2_perseus_nn.ipynb": "make_gpt2_nn_notebook.py",
}
MODEL_NOTEBOOKS = ("gpt2_torch_forward.ipynb", "gpt2_perseus_nn.ipynb")   # carry the prereq cell
FORBIDDEN = ("/data02", "/leonardo", "sbatch", "SCRATCH", "planned_", "vit", "bert")


def _load_generator(name):
    path = REPO / "scripts" / "utils" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def generators():
    return {nb: _load_generator(g) for nb, g in GENERATED.items()}


def _sources(nb, kind):
    return "\n".join(c.source for c in nb.cells if c.cell_type == kind)


@pytest.mark.parametrize("notebook", sorted(GENERATED))
def test_generated_notebook_is_valid_and_every_code_cell_parses(generators, notebook):
    nb = generators[notebook].build()
    nbformat.validate(nb)
    assert nb.cells, "empty notebook"
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            ast.parse(c.source, filename=f"cell[{i}]")   # no %magics / !shell in code cells
    assert len({c.id for c in nb.cells}) == len(nb.cells), "cell ids must be unique"


@pytest.mark.parametrize("notebook", sorted(GENERATED))
def test_generator_is_deterministic_and_matches_the_tracked_file(generators, notebook):
    gen = generators[notebook]
    a, b = nbformat.writes(gen.build()), nbformat.writes(gen.build())
    assert a == b, "build() is not deterministic (unpinned cell ids?)"
    tracked = NOTEBOOKS / notebook
    cmd = f"python scripts/utils/{GENERATED[notebook]}"
    assert tracked.is_file(), f"{tracked} is missing: {cmd}"
    assert tracked.read_text(encoding="utf-8") == a + "\n", (
        f"notebooks/{notebook} drifted from its generator; regenerate with {cmd}")
    nbformat.validate(nbformat.read(str(tracked), as_version=4))


@pytest.mark.parametrize("notebook", MODEL_NOTEBOOKS)
def test_model_notebooks_carry_the_prerequisites_cell(generators, notebook):
    gen = generators[notebook]
    nb = gen.build()
    assert nb.cells[1].cell_type == "markdown" and gen.PREREQ_MARK in nb.cells[1].source, (
        "the prerequisites cell sits at index 1 in every model notebook")
    assert "setup_artifacts.ipynb" in nb.cells[1].source


def test_gpt2_nn_notebook_walks_the_perseus_nn_path(generators):
    nb = generators["gpt2_perseus_nn.ipynb"].build()
    code, md = _sources(nb, "code"), _sources(nb, "markdown")
    for name in ("EncGPT2.from_pretrained(", "EncGenerationServer(", "EncGenerationClient(",
                 "EncClient(", "EncServer(", ".generate(", "SessionProfile"):
        assert name in code, f"{name} not used by any code cell"
    assert "_core.run_generate" not in code, "the C++ driver path belongs to gpt2_torch_forward"
    assert "from perseus import _core" not in code
    assert "gpt2_torch_forward.ipynb" in md, "missing the pointer cell to the C++-driver notebook"


def test_gpt2_torch_notebook_runs_the_driver_eager(generators):
    nb = generators["gpt2_torch_forward.ipynb"].build()
    code = _sources(nb, "code")
    assert "_core.run_generate(" in code
    assert 'cfg.plan_dir = ""' in code, "the generation demo runs eager"


def test_setup_notebook_covers_every_artifact(generators):
    code = _sources(generators["setup_artifacts.ipynb"].build(), "code")
    for name in ("perseus.export", "load_token_pool(", "perseus.calibrate",
                 "gen_gpt2_oracle.py", "examples.gpt2_from_primitives.run_decode",
                 "scripts/make_plans.sh",
                 "PERSEUS_DATA"):
        assert name in code, f"{name} not used by any code cell"


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.name)
def test_every_notebook_ships_clean(path):
    text = path.read_text(encoding="utf-8")
    nb = json.loads(text)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            assert c.get("outputs") == [], f"{path.name} cell[{i}] carries saved outputs"
            assert c.get("execution_count") is None, f"{path.name} cell[{i}] was executed"
            assert "execution" not in c.get("metadata", {}), (
                f"{path.name} cell[{i}] kept its execution timestamps")
    low = text.lower()
    hits = [w for w in FORBIDDEN if w.lower() in low]
    assert not hits, f"{path.name} mentions {hits}"
