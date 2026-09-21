import os

from .. import _core
from .._env import scoped_env
from ._sampling import _make_sampler, _stop_ids  # noqa: F401  (shared with perseus.nn.serve)
from .block import EncBlock
from .head import EncCutMax, EncLMHead
from .module import EncModule
from .pipeline import Stage, resolve_overlap, run_stages


class EncGPT2(EncModule):

    def __init__(self, store, configs, n_layers=None):
        super().__init__()
        self.store = store
        self.configs = configs
        self.n_layers = n_layers if n_layers is not None else configs.model.n_layers
        self.block = EncBlock()
        self.lm_head = EncLMHead(store)
        self.cutmax = EncCutMax.from_configs(self.lm_head.vocab, configs)
        self._states = None
        self._block_plans = None
        self._lnf = None
        self._planned = False
        self._plan13 = None
        self._plan14 = None
        self._fb = None
        self.overlap = None
        self.cache_states = True
        self._ran_decode = False
        self._bind_kwargs = {}

    @classmethod
    def from_pretrained(cls, name, *, tag="classic", weights=None, configs=None, n_layers=None,
                        strict=True):
        """Build from the artifacts `perseus-export` / `perseus-calibrate` produced for
        `name` (perseus cache or explicit paths; see perseus.nn.pretrained)."""
        from .pretrained import from_pretrained
        return from_pretrained(cls, name, tag=tag, weights=weights, configs=configs,
                               n_layers=n_layers, strict=strict, model_type="gpt2")

    def configure(self, **kwargs):
        """Store bind-time settings (plan, plan_dir, overlap, cache_states, artifact_dir,
        coeff_encode) so a plain `bind(inf)` — the uniform call a container makes —
        applies them. Returns self."""
        allowed = {"plan", "plan_dir", "overlap", "cache_states", "artifact_dir", "coeff_encode"}
        bad = set(kwargs) - allowed
        if bad:
            raise TypeError(f"configure(): unknown setting(s) {sorted(bad)}; expected {sorted(allowed)}")
        self.__dict__.setdefault("_bind_kwargs", {}).update(kwargs)
        return self

    def extra_repr(self):
        s = f"n_layers={self.n_layers}"
        return s + (", planned" if self._planned else "")

    def bind(self, inf, plan=None, plan_dir=None, overlap=None, cache_states=None,
             artifact_dir=None, coeff_encode=None):
        c = self.__dict__.get("_bind_kwargs", {})
        plan = c.get("plan") if plan is None else plan
        plan_dir = c.get("plan_dir") if plan_dir is None else plan_dir
        overlap = c.get("overlap") if overlap is None else overlap
        cache_states = c.get("cache_states", True) if cache_states is None else cache_states
        artifact_dir = c.get("artifact_dir") if artifact_dir is None else artifact_dir
        coeff_encode = c.get("coeff_encode") if coeff_encode is None else coeff_encode
        inf = getattr(inf, "inf", inf)
        super().bind(inf)
        self.overlap = resolve_overlap(overlap)
        self.cache_states = cache_states
        self._ran_decode = False
        if plan_dir is not None:
            from ..plan import contract as _contract
            for b in range(self.n_layers + 1):
                _contract.validate_contract(
                    _contract.read_stamp(f"{plan_dir}/block_{b}_placement.json"),
                    inf, source=f"{plan_dir}/block_{b}_placement.json")
            block_plans = [_core.parse_bootstrap_plan_file(
                f"{plan_dir}/block_{b}_placement.json") for b in range(self.n_layers)]
            tail_plan = _core.parse_bootstrap_plan_file(
                f"{plan_dir}/block_{self.n_layers}_placement.json")
            for attr, b in (("_plan13", self.n_layers + 1), ("_plan14", self.n_layers + 2)):
                p = f"{plan_dir}/block_{b}_placement.json"
                setattr(self, attr,
                        _core.parse_bootstrap_plan_file(p) if os.path.exists(p) else None)
            self._planned = True
        else:
            empty = _core.BootstrapPlan()
            block_plans = [plan if plan is not None else empty] * self.n_layers
            tail_plan = plan if plan is not None else empty
            self._planned = plan is not None and plan.valid
        self._block_plans = block_plans
        self._states = None
        if coeff_encode is None and artifact_dir:
            coeff_encode = True
        env = {} if coeff_encode is None else {"FHE_PT_COEFF_ENCODE": "1" if coeff_encode else "0"}
        with scoped_env(**env):
            self._bind_states(inf, block_plans, cache_states, artifact_dir, tail_plan)
        return self

    def _bind_states(self, inf, block_plans, cache_states, artifact_dir, tail_plan):
        if cache_states:
            if artifact_dir:
                os.makedirs(artifact_dir, exist_ok=True)
            self._states = []
            for b in range(self.n_layers):
                art = artifact_dir and os.path.join(artifact_dir, f"block_{b}.enc")
                if art and os.path.exists(art):
                    state = _core.load_block_state_file(inf, self.configs,
                                                        block_plans[b], b, art)
                elif art:
                    state = _core.encode_block_state_coeff(
                        inf, self.store, self.configs, block_plans[b], b)
                    _core.save_block_state(inf, state, art)
                else:
                    state = _core.load_block_state(inf, self.store, self.configs,
                                                   block_plans[b], b)
                _core.evict_block_from_device(inf, state)
                self._states.append(state)
        self._lnf = _core.load_final_ln_state(inf, self.store, self.configs, tail_plan)
        self.lm_head.plan = tail_plan

    def start(self):
        _core.reset_kv_cache(self.inf, self.n_layers)

    def run_block(self, b, x):
        if self._states is None:
            raise RuntimeError("run_block needs cache_states=True; the loader-shaped "
                               "bind holds no host states (use forward)")
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

    def _block_stage(self, b):
        def compute(x):
            if self._planned or os.environ.get("FHE_GRAPH_DIR"):
                _core.reset_graph_runtime(self.inf)
            self.inf.block_prefix = _core.block_scope(b)
            self.inf.capture_b = b
            cap = _core.begin_subgraph_capture(self.inf, b)
            with self.inf.step("kv_reload"):
                _core.kv_block_prologue(self.inf, b, self.n_layers)
            x = self.block(x)
            if cap:
                _core.end_subgraph_capture(self.inf, b)
            return x

        def release():
            _core.block_release(self.inf, b)

        if self.cache_states:
            return Stage(compute=compute, state=self._states[b], release=release,
                         label=f"blk{b}")
        return Stage(compute=compute, release=release, label=f"blk{b}",
                     loader=(self.store, self.configs, self._block_plans[b], b))

    def forward(self, x, head=True):
        """Blocks + final LN + lm_head -> logit tiles. head=False stops after the
        final LN (prompt tokens whose logits nobody reads)."""
        _core.kv_prefetch_first(self.inf, self.n_layers)
        x = run_stages(self.inf, x,
                       [self._block_stage(b) for b in range(self.n_layers)],
                       self.overlap)
        _core.kv_finalize_last(self.inf, self.n_layers)
        if self._planned or os.environ.get("FHE_GRAPH_DIR"):
            _core.reset_graph_runtime(self.inf)
        self.inf.capture_b = self.n_layers
        cap = _core.begin_subgraph_capture(self.inf, self.n_layers)   # tail = block n_layers
        x = _core.apply_final_ln(self.inf, x, self._lnf)
        if not head:
            if cap:
                _core.end_subgraph_capture(self.inf, self.n_layers)
            return x
        tiles = self.lm_head(x)
        if cap:
            _core.end_subgraph_capture(self.inf, self.n_layers)
        return tiles

    def decode_logits(self, tiles):
        return self.lm_head.decode_logits(tiles)

    def prepare_feedback(self):
        """Encode the CutMax->embedding codebook once (host-cached; loaded per token)."""
        if self._fb is None:
            self._fb = _core.LMHeadCache()
            _core.prepare_feedback_weights(self.inf, self.store, self.lm_head.vocab,
                                           self.inf.fhe.complex_payload, self._fb,
                                           self._plan14)
        return self._fb

    def prefill(self, prompt):
        """Causal-chunked prefill (the gpt2_prefill.cu schedule): the whole prompt is
        consumed through the filling packing, then the KV caches convert to
        the decode layout (kv_handoff_filling_to_cachemir). Returns the encrypted
        logit tiles of the last prompt position.
        """
        if self._planned:
            raise RuntimeError("prefill is eager-only for now (per-chunk plan "
                               "templates are not wired); bind without plan_dir")

        if self._ran_decode:
            raise RuntimeError(
                "prefill after decode-mode forwards in the same session is not "
                "state-clean (divergent trajectory); use a fresh process/bind, "
                "or run prompt_mode='prefill' first")
        inf = self.inf
        m = len(prompt)
        t = inf.slots // inf.size.hidDim
        chunks = self._prefill_chunks(m, t, bool(inf.fhe.complex_payload))
        if chunks[-1][1]:
            raise NotImplementedError(
                "token-pair tail readout needs pair_unpack bindings: the last prefill "
                f"chunk would carry {chunks[-1][0]} tokens on Re/Im lanes; use a prompt "
                f"whose last chunk holds <= {t} tokens (slots/hidDim) for now")
        complex_decode = inf.complex
        inf.fhe.complete_setup()
        mag_prev = inf.fhe.magnitude_suppressed
        inf.fhe.magnitude_suppressed = True
        mj, tp = chunks[0]
        inf.token_pair = tp
        _core.configure_prefill_phase(inf, mj)
        _core.reset_kv_cache(inf, self.n_layers)
        h, off = None, 0
        for mj, tp in chunks:
            inf.token_pair = tp
            _core.configure_prefill_phase(inf, mj)
            x = _core.encode_prefill_input(inf, [list(p) for p in prompt[off:off + mj]])
            h = _core.gpt2_prefill(inf, x, self.store, self.configs, self.n_layers,
                                   True)
            off += mj
        # final LN rides the last chunk's packing; self._lnf is decode-packed
        lnf = _core.load_final_ln_state(inf, self.store, self.configs,
                                        _core.BootstrapPlan())
        hidden = _core.apply_final_ln(inf, h, lnf)
        inf.fhe.magnitude_suppressed = mag_prev
        _core.free_rotation_steps(inf, _core.filling_rot_steps(inf))
        _core.configure_decode_phase(inf, complex_decode)
        _core.kv_handoff_filling_to_cachemir(inf, self.n_layers, m)
        lane = (m - 1) - (off - mj)   # last position within its chunk
        tok = _core.extract_token_i_cachemir(inf, hidden, lane)
        return self.lm_head(tok)

    @staticmethod
    def _prefill_chunks(m, t, complex_payload):
        """[(tokens in chunk, token-pair?)] for an m-token prompt: the gpt2_prefill.cu
        chunk schedule (a chunk carries up to 2t tokens on Re/Im lanes when the payload
        is complex and more than t remain, else up to t)."""
        out, off = [], 0
        while off < m:
            r = m - off
            tp = complex_payload and r > t
            mj = min(2 * t, r) if tp else min(t, r)
            out.append((mj, tp))
            off += mj
        return out

    def encode_input(self, values):
        """One token's input embedding (d_real floats) -> fresh ciphertext."""
        return _core.encode_token_input(self.inf, list(values))

    def embed_token(self, token_id, position):
        """Client-side re-embedding for feedback="client": wte+wpe row -> fresh
        ciphertext. Override to supply embeddings from the client's own tables."""
        return self.encode_input(_core.token_embedding(self.store, int(token_id), int(position)))

    def generate(self, prompt, n_tokens=None, feedback="encrypted", prompt_mode="decode", *,
                 max_new_tokens=None, eos_token_id=None, on_token=None, do_sample=False,
                 temperature=1.0, top_k=None, seed=None, entry_bootstrap=None,
                 realize_entry=None):
        """Autoregressive generation; returns the emitted token ids.

        prompt: [P][d_real] input embeddings.
        max_new_tokens (or the positional n_tokens) bounds the run;
        eos_token_id (an id, or a set of ids) stops it early.
        on_token(token_id, step) runs after every
        emitted token; stream() is the generator form.

        prompt_mode="decode" consumes prompt tokens as decode steps (full forward
        each, lm_head skipped); "prefill" runs the causal-chunked filling prefill
        (one pass for up to slots/hidDim tokens) and hands the KV caches to the
        decode layout.

        feedback="encrypted": the CutMax one-hot re-embeds on the server via the
        codebook (gpt2_cutmax_feedback);
        "client": decrypt logits, pick the next token, re-encode wte+wpe, the trust-boundary
        round-trip, and the only path where sampling (do_sample / temperature /
        top_k / seed) applies: the encrypted path takes the argmax inside the
        ciphertext.
        """
        if n_tokens is not None and max_new_tokens is not None:
            raise ValueError("pass max_new_tokens or the positional n_tokens, not both")
        if max_new_tokens is None:
            max_new_tokens = n_tokens
        return list(self.stream(prompt, max_new_tokens, feedback, prompt_mode,
                                eos_token_id=eos_token_id, on_token=on_token,
                                do_sample=do_sample, temperature=temperature, top_k=top_k,
                                seed=seed, entry_bootstrap=entry_bootstrap,
                                realize_entry=realize_entry))

    def stream(self, prompt, max_new_tokens, feedback="encrypted", prompt_mode="decode", *,
               eos_token_id=None, on_token=None, do_sample=False, temperature=1.0,
               top_k=None, seed=None, entry_bootstrap=None, realize_entry=None):
        """Generator form of generate(): yields each token id as soon as it is known."""
        if feedback not in ("encrypted", "client"):
            raise ValueError(f"feedback must be 'encrypted' or 'client', got {feedback!r}")
        if prompt_mode not in ("decode", "prefill"):
            raise ValueError(f"prompt_mode must be 'decode' or 'prefill', got {prompt_mode!r}")
        if max_new_tokens is None or int(max_new_tokens) < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens!r}")
        if do_sample and feedback != "client":
            raise ValueError("sampling needs feedback='client': the encrypted path picks "
                             "the argmax inside the ciphertext (CutMax)")
        if feedback == "encrypted" and self._planned and self._plan14 is None:
            raise RuntimeError(
                "generate(feedback='encrypted') needs an eager bind or a gen-shaped "
                "plan dir (replanned deg-2 block-0 entry; block_13/14 placements, or "
                "set _plan14 = BootstrapPlan() for an eager tail); this model is bound "
                "to a decode-shaped plan dir")
        P = len(prompt)
        if P < 1:
            raise ValueError("generate needs at least one prompt embedding")
        stops = _stop_ids(eos_token_id)
        sampler = _make_sampler(do_sample, temperature, top_k, seed)
        if prompt_mode == "prefill":
            tiles = self.prefill(prompt)
        else:
            if entry_bootstrap is None:
                entry_bootstrap = True
            if realize_entry is None:
                realize_entry = False
            self._ran_decode = True
            self.start()
            tiles = None
            capturing = self._planned or bool(os.environ.get("FHE_GRAPH_DIR"))
            for p in range(P):
                self.inf.capture_t = p
                x = self.encode_input(prompt[p])
                if capturing and entry_bootstrap:
                    self.inf.fhe.bootstrap(x)
                    if realize_entry:
                        _core.realize_pending_rescale(self.inf, x)
                tiles = self.forward(x, head=(p == P - 1))
        yield from self._feedback_steps(tiles, P, int(max_new_tokens), feedback, stops,
                                        on_token, sampler)

    def _feedback_steps(self, tiles, P, max_new_tokens, feedback, stops, on_token, sampler):
        for j in range(max_new_tokens):
            pos = P - 1 + j
            logits = self.decode_logits(tiles)
            if sampler is not None:
                tok = sampler(logits)
            else:
                tok = int(max(range(len(logits)), key=logits.__getitem__))
            if on_token is not None:
                on_token(tok, j)
            yield tok
            if j + 1 == max_new_tokens or tok in stops:
                break
            self.inf.capture_t = P + j
            if feedback == "encrypted":
                self.prepare_feedback()
                x, _z, _am_s = _core.cutmax_feedback(
                    self.inf, tiles, self.store, self.lm_head.vocab,
                    self.cutmax.config, self._fb, pos + 1, self._plan13, self._plan14)
            else:
                x = self.embed_token(tok, pos + 1)
            tiles = self.forward(x)
