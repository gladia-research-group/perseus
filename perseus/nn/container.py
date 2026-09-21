import json
import os
import warnings

from .. import _core
from .module import EncModule
from .pipeline import resolve_overlap, run_stages, stage_of


class EncSequential(EncModule):
    def __init__(self, *modules, overlap=None):
        super().__init__()
        self._items = list(modules)
        for i, mod in enumerate(self._items):
            setattr(self, str(i), mod)
        self.overlap = resolve_overlap(overlap)
        self._plans = None
        self._warned_no_residency = False

    def to_config(self):
        return {"overlap": None if self.overlap is None else self.overlap.name.lower()}

    @classmethod
    def from_config(cls, cfg, params, children=()):
        return cls(*children, overlap=cfg.get("overlap"))

    def extra_repr(self):
        return "" if self.overlap is None else f"overlap={self.overlap.name.lower()}"

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def torch_mirror(self):
        import torch.nn as nn
        mirrors = [m.torch_mirror() for m in self._items]
        return None if any(m is None for m in mirrors) else nn.Sequential(*mirrors)

    def export_graphs(self, x, graph_root):
        from ..plan import contract as _contract
        inf = self.inf
        os.makedirs(graph_root, exist_ok=True)
        with open(os.path.join(graph_root, "capture_env.json"), "w", encoding="utf-8") as f:
            json.dump(_contract.capture_contract(inf), f, indent=2)
        for i, mod in enumerate(self._items):
            state = mod.residency()
            if isinstance(state, _core.EncodedBlock):
                # eager capture has no pipeline install; do it here, run_block-style
                _core.load_block_to_device(inf, state)
                _core.install_block_state(inf, state)
            d = os.path.join(graph_root, f"block_{i}")
            os.makedirs(d, exist_ok=True)
            _core.reset_graph_runtime(inf)
            inf.enable_graph_capture()
            x = mod(x)
            inf.export_graph_json(os.path.join(d, "graph.json"))
            inf.disable_graph_capture()
            if isinstance(state, _core.EncodedBlock):
                _core.evict_block_from_device(inf, state)
        return x

    def load_plans(self, plan_dir, validate=True):
        """Attach block_{i}_placement.json per depth-1 child; a missing file leaves
        that stage unplanned. Plans install live per stage on every forward (the
        EncGPT2 pattern: ct-var namespace reset + plan install ahead of compute).

        """
        from ..plan import contract as _contract
        self._plans = []
        for i in range(len(self._items)):
            p = os.path.join(plan_dir, f"block_{i}_placement.json")
            if not os.path.exists(p):
                self._plans.append(None)
                continue
            if validate:
                _contract.validate_contract(_contract.read_stamp(p),
                                            getattr(self, "inf", None), source=p)
            self._plans.append(_core.parse_bootstrap_plan_file(p))
        return self

    def _apply_plan(self, i, stage):
        plan = self._plans[i] if self._plans else None
        if plan is None:
            return stage
        if stage.state is not None:
            stage.state.plan = plan          # installs with the state, pipeline-side
            inner, install = stage.compute, None
        else:
            inner, install = stage.compute, plan

        def compute(x, _inner=inner, _plan=install, _inf=self.inf):
            _core.reset_graph_runtime(_inf)  # restart var naming so the plan re-binds
            if _plan is not None:
                _core.install_plan_live(_inf, _plan)
            return _inner(x)

        stage.compute = compute
        return stage

    def forward(self, x):
        if self.overlap is None:
            for i, mod in enumerate(self._items):
                plan = self._plans[i] if self._plans else None
                if plan is not None:
                    _core.reset_graph_runtime(self.inf)
                    _core.install_plan_live(self.inf, plan)
                x = mod(x)
            if self._plans and any(p is not None for p in self._plans):
                self.inf.clear_bootstrap_plan()
            return x
        stages = [self._apply_plan(i, stage_of(mod, label=f"{i}:{type(mod).__name__.lower()}"))
                  for i, mod in enumerate(self._items)]
        if (not self._warned_no_residency
                and self.overlap != _core.InferenceMode.Sync
                and all(s.state is None and not s.weights and s.loader is None
                        for s in stages)):
            warnings.warn("EncSequential(overlap=...): no child declared residency() "
                          "state, weights, or loader, so the pipeline has nothing to overlap",
                          stacklevel=2)
            self._warned_no_residency = True
        x = run_stages(self.inf, x, stages, self.overlap)
        if self._plans and any(p is not None for p in self._plans):
            self.inf.clear_bootstrap_plan()
        return x


class EncModuleList(EncSequential):
    def forward(self, x):
        raise NotImplementedError("EncModuleList is a container; iterate it explicitly")
