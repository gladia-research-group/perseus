from .. import _core
from .block import EncBlock
from .head import EncCutMax, EncLMHead
from .module import EncModule


class EncGPT2(EncModule):
    """GPT-2 forward from Enc modules; per-block state via the production loader.

    forward(x) -> encrypted logit tiles for the token's next-token distribution.

    Precomputed plans. bind(inf, plan_dir=...) loads a frozen per-block bootstrap
    plan (block_{b}_placement.json for blocks 0..n_layers-1; block_{n_layers}_
    placement.json is the shared final-LN + lm_head tail, mirroring the C++
    session's decode_plans_.at(b) / .at(n_blocks) mapping). When planned, run_block
    resets the ct-var namespace per block exactly as the C++ decode loop does
    (gpt2_decode.cu) so every block's v_* placements bind; decode masks stay online
    (strict_masks off), so no capture or mask priming is needed. bind(inf) with no
    plan stays eager (the forward-gate path).
    """

    def __init__(self, store, configs, n_layers=None):
        super().__init__()
        self.store = store
        self.configs = configs
        self.n_layers = n_layers if n_layers is not None else configs.model.n_layers
        self.block = EncBlock()
        self.lm_head = EncLMHead(store)
        self.cutmax = EncCutMax.from_configs(self.lm_head.vocab, configs)
        self._states = None
        self._lnf = None
        self._planned = False

    def bind(self, inf, plan=None, plan_dir=None):
        super().bind(inf)
        if plan_dir is not None:
            # frozen per-block plan: blocks 0..n_layers-1 + the tail (final LN + head).
            block_plans = [_core.parse_bootstrap_plan_file(
                f"{plan_dir}/block_{b}_placement.json") for b in range(self.n_layers)]
            tail_plan = _core.parse_bootstrap_plan_file(
                f"{plan_dir}/block_{self.n_layers}_placement.json")
            self._planned = True
        else:
            empty = _core.BootstrapPlan()
            block_plans = [plan if plan is not None else empty] * self.n_layers
            tail_plan = plan if plan is not None else empty
            self._planned = plan is not None and plan.valid
        self._states = []
        for b in range(self.n_layers):
            state = _core.load_block_state(inf, self.store, self.configs, block_plans[b], b)
            _core.evict_block_from_device(inf, state)
            self._states.append(state)
        self._lnf = _core.load_final_ln_state(inf, self.store, self.configs, tail_plan)
        self.lm_head.plan = tail_plan
        return self

    def start(self):
        _core.reset_kv_cache(self.inf, self.n_layers)

    def run_block(self, b, x):
        state = self._states[b]
        _core.load_block_to_device(self.inf, state)
        _core.install_block_state(self.inf, state)
        if self._planned:
            _core.reset_graph_runtime(self.inf)
        self.inf.block_prefix = _core.block_scope(b)
        self.inf.capture_b = b
        with self.inf.step("kv_reload"):
            _core.kv_block_prologue(self.inf, b, self.n_layers)
        x = self.block(x)
        _core.block_release(self.inf, b)
        return x

    def forward(self, x):
        _core.kv_prefetch_first(self.inf, self.n_layers)
        for b in range(self.n_layers):
            x = self.run_block(b, x)
        _core.kv_finalize_last(self.inf, self.n_layers)
        if self._planned:
            _core.reset_graph_runtime(self.inf)
        x = _core.apply_final_ln(self.inf, x, self._lnf)
        return self.lm_head(x)

    def decode_logits(self, tiles):
        return self.lm_head.decode_logits(tiles)
