import json
import os

import numpy as np

from .. import __version__
from .container import EncSequential
from .module import EncModule

FORMAT = 1
STRUCTURE = "model.json"
WEIGHTS = "weights.npz"


def _registry():
    from .. import nn
    return {name: getattr(nn, name) for name in nn.__all__
            if isinstance(getattr(nn, name), type) and issubclass(getattr(nn, name), EncModule)}


def to_tree(module):
    """Nested JSON description of a module tree (no parameters)."""
    node = {"class": type(module).__name__, "config": module.to_config()}
    if isinstance(module, EncSequential):
        node["children"] = [to_tree(m) for m in module]
    return node


def from_tree(node, params, prefix=""):
    cls = _registry().get(node["class"])
    if cls is None:
        raise ValueError(f"unknown module class {node['class']!r} in {STRUCTURE}")
    own = {p.split(".")[-1]: v for p, v in params.items()
           if p.rpartition(".")[0] == prefix}
    if "children" in node:
        children = [from_tree(c, params, f"{prefix}.{i}" if prefix else str(i))
                    for i, c in enumerate(node["children"])]
        return cls.from_config(node["config"], own, children=children)
    return cls.from_config(node["config"], own)


def save(module: EncModule, path) -> str:
    """Write `path/model.json` + `path/weights.npz`. Returns path."""
    os.makedirs(path, exist_ok=True)
    tree = to_tree(module)
    with open(os.path.join(path, STRUCTURE), "w", encoding="utf-8") as f:
        json.dump({"format": FORMAT, "perseus_version": __version__, "model": tree}, f, indent=1)
    np.savez(os.path.join(path, WEIGHTS), **module.state_dict())
    return path


def load(path) -> EncModule:
    """Rebuild the module tree saved by save(); bind() it to a session afterwards."""
    with open(os.path.join(path, STRUCTURE), encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{STRUCTURE}: format {doc.get('format')!r} is not the supported {FORMAT}")
    with np.load(os.path.join(path, WEIGHTS)) as z:
        params = {k: z[k] for k in z.files}
    return from_tree(doc["model"], params)
