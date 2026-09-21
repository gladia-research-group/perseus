"""
perseus CUDA core: FIDESlib/OpenFHE CKKS runtime bindings.

Threading: one session (Inference) per process; every GPU-heavy call releases the GIL, but the runtime's state is process-global, so drive one session from one thread at a time (the residency pipeline owns its own worker). CUDA errors inside FIDESlib are fatal (the library exits the process); parameter and shape errors raise ValueError / FHEError before any device work. List-valued option fields (CKKSOptions.level_budget, extra_rot_steps, ...) are copied on access: assign a whole list, in-place mutation of the returned list does nothing.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['BootstrapPlan', 'CKKSOptions', 'Context', 'CutMaxCalib', 'CutMaxConfig', 'DecodeSession', 'EncodedBlock', 'FHEError', 'GSInitMethod', 'GeLUConfig', 'GeLUMethod', 'GtSteps', 'Inference', 'InferenceMode', 'InferenceOptions', 'LMHeadCache', 'LayoutError', 'MaskError', 'ModelConfig', 'ModelSize', 'NRInitMethod', 'NormConfig', 'PackedCtx', 'PackingKind', 'ParsedConfigs', 'PlanError', 'RunConfig', 'RunResult', 'SoftmaxConfig', 'Stage', 'StepScope', 'WeightGranularity', 'WeightStore', 'apply_final_ln', 'attention_softmax_thor', 'begin_subgraph_capture', 'block_release', 'block_scope', 'block_state_diff', 'build_info', 'cache_k_push', 'cache_kv_push', 'cache_kv_push_packed', 'cache_v_push', 'chain', 'close_session', 'configure_decode_phase', 'configure_prefill_phase', 'cutmax_argmax', 'cutmax_config_from_calib', 'cutmax_feedback', 'decode_linear_output', 'decode_lm_head_logits', 'decode_token_output', 'decode_tokens_output', 'decrypt_slots', 'default_cutmax_config', 'deserialize_ct', 'device_free_gb', 'encode_block_state_coeff', 'encode_prefill_input', 'encode_token_input', 'end_subgraph_capture', 'evict_block_from_device', 'exp_approx', 'extract_token_i_cachemir', 'filling_rot_steps', 'fold_ln_affine', 'free_rotation_steps', 'gelu_approx', 'gpt2_prefill', 'hard_exit', 'head_reduce_sum', 'install_block_state', 'install_fatal_exit_handler', 'install_plan_live', 'kv_block_prologue', 'kv_finalize_last', 'kv_handoff_filling_to_cachemir', 'kv_prefetch_first', 'layer_norm', 'layout_of', 'linear', 'linear_multi', 'linear_outputpack', 'lm_head', 'lm_head_tile_width', 'lm_head_vocab', 'ln_affine', 'load_block_state', 'load_block_state_file', 'load_block_to_device', 'load_configs', 'load_final_ln_state', 'make_context', 'make_gpt2_inference', 'make_inference', 'mha_block', 'mlp_block', 'native_int_bits', 'norm', 'pack_tokens', 'parse_bootstrap_plan_file', 'prepare_feedback_weights', 'prepare_mha_masks', 'prepare_vcache', 'qkt', 'read_lm_head_steps', 'read_teacher_forced_inputs', 'realize_pending_rescale', 'reset_graph_runtime', 'reset_kv_cache', 'run_decode', 'run_generate', 'run_prefill', 'run_stages', 'save_block_state', 'save_keys', 'save_secret_key', 'serialize_ct', 'set_ln_affine', 'set_strict_layout', 'softmax_v', 'throw_test', 'token_embedding', 'transformer_block', 'unpack_tokens']
class BootstrapPlan:
    def __init__(self) -> None:
        """
        An empty plan (valid=False).
        """
    def __repr__(self) -> str:
        ...
    @property
    def expected_levels(self) -> dict[str, int]:
        """
        var -> level the strict runtime checks (a copy).
        """
    @property
    def hint_fire(self) -> list[str]:
        ...
    @property
    def num_placements(self) -> int:
        ...
    @property
    def placements(self) -> list[str]:
        """
        Sorted ct-variable names a bootstrap is planted after.
        """
    @property
    def valid(self) -> bool:
        ...
    @property
    def weight_levels(self) -> dict[str, int]:
        ...
class CKKSOptions:
    """
    CKKS parameters for a context. List fields are copied on read: set them whole (o.level_budget = [4, 3]).
    """
    ckks_complex_payload: bool
    defer_heavy_setup: bool
    enable_bootstrap: bool
    keys_dir: str
    skip_gpu_load: bool
    @staticmethod
    def from_env() -> CKKSOptions:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, other: CKKSOptions) -> None:
        """
        Copy constructor: CKKSOptions(other) is an exact copy, hidden fields included.
        """
    @property
    def auto_bts_level_override(self) -> int:
        ...
    @auto_bts_level_override.setter
    def auto_bts_level_override(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def batch_size(self) -> int:
        ...
    @batch_size.setter
    def batch_size(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def bootstrap_slots(self) -> int:
        ...
    @bootstrap_slots.setter
    def bootstrap_slots(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def btp_depth_overhead(self) -> int:
        ...
    @btp_depth_overhead.setter
    def btp_depth_overhead(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def btp_scale_bits(self) -> int:
        ...
    @btp_scale_bits.setter
    def btp_scale_bits(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def bts_iterations(self) -> int:
        ...
    @bts_iterations.setter
    def bts_iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def bts_precision(self) -> int:
        ...
    @bts_precision.setter
    def bts_precision(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def composite_degree(self) -> int:
        ...
    @composite_degree.setter
    def composite_degree(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def correction_factor(self) -> int:
        ...
    @correction_factor.setter
    def correction_factor(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def deferred_rot_steps(self) -> list[int]:
        ...
    @deferred_rot_steps.setter
    def deferred_rot_steps(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @property
    def depth(self) -> int:
        ...
    @depth.setter
    def depth(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def extra_rot_steps(self) -> list[int]:
        ...
    @extra_rot_steps.setter
    def extra_rot_steps(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @property
    def first_mod_bits(self) -> int:
        ...
    @first_mod_bits.setter
    def first_mod_bits(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def h_weight(self) -> int:
        ...
    @h_weight.setter
    def h_weight(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def level_budget(self) -> list[int]:
        ...
    @level_budget.setter
    def level_budget(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @property
    def logN(self) -> int:
        ...
    @logN.setter
    def logN(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def num_large_digits(self) -> int:
        ...
    @num_large_digits.setter
    def num_large_digits(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def scale_bits(self) -> int:
        ...
    @scale_bits.setter
    def scale_bits(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def sparse_bts_slots(self) -> int:
        ...
    @sparse_bts_slots.setter
    def sparse_bts_slots(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def sparse_bts_slots_list(self) -> list[int]:
        ...
    @sparse_bts_slots_list.setter
    def sparse_bts_slots_list(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @property
    def sparse_level_budget(self) -> list[int]:
        ...
    @sparse_level_budget.setter
    def sparse_level_budget(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
class Context:
    magnitude_suppressed: bool
    @typing.overload
    def add(self, a: PackedCtx, b: PackedCtx) -> PackedCtx:
        """
        a + b (ciphertext + ciphertext).
        """
    @typing.overload
    def add(self, a: PackedCtx, scalar: typing.SupportsFloat | typing.SupportsIndex) -> PackedCtx:
        """
        a + scalar.
        """
    def bootstrap(self, ct: PackedCtx) -> None:
        """
        Refresh `ct` to the bootstrap output level (in place).
        """
    def bootstrap_hint(self, ct: PackedCtx, level_threshold: typing.SupportsInt | typing.SupportsIndex, account_pending_rescale: bool = False) -> None:
        """
        Bootstrap `ct` if it has fewer than `level_threshold` levels left.
        """
    def bootstrap_output_level(self) -> int:
        ...
    def complete_setup(self) -> None:
        """
        Run a deferred heavy setup (rot keygen/upload, bts precomps, LoadContext) now; idempotent no-op when nothing is pending. The C++ drivers call this defensively at every phase entry.
        """
    def conjugate(self, ct: PackedCtx) -> PackedCtx:
        """
        Complex conjugate of every slot (identity on real payloads).
        """
    def inplace_add(self, a: PackedCtx, b: PackedCtx) -> None:
        """
        a += b.
        """
    def level_hint(self, ct: PackedCtx, level: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Drop `ct` to `level` if it is above it.
        """
    def level_limit(self) -> int:
        """
        The highest level a fresh ciphertext can carry in this context.
        """
    def maybe_bootstrap(self, ct: PackedCtx) -> None:
        """
        Bootstrap `ct` only if its level is below the auto threshold.
        """
    @typing.overload
    def mult(self, a: PackedCtx, b: PackedCtx) -> PackedCtx:
        """
        a * b (ciphertext * ciphertext, relinearized).
        """
    @typing.overload
    def mult(self, a: PackedCtx, scalar: typing.SupportsFloat | typing.SupportsIndex) -> PackedCtx:
        """
        a * scalar.
        """
    def negate(self, ct: PackedCtx) -> PackedCtx:
        """
        -ct.
        """
    def rotate(self, ct: PackedCtx, steps: typing.SupportsInt | typing.SupportsIndex) -> PackedCtx:
        """
        Cyclic slot rotation by `steps` (out[i] = in[i + steps]); the rotation key for `steps` must exist in the session's band.
        """
    def roundtrip(self, values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> list[float]:
        """
        encode -> encrypt -> decrypt -> decode `values` (a context self-test).
        """
    def square(self, a: PackedCtx) -> PackedCtx:
        """
        a * a.
        """
    @typing.overload
    def sub(self, a: PackedCtx, b: PackedCtx) -> PackedCtx:
        """
        a - b (ciphertext - ciphertext).
        """
    @typing.overload
    def sub(self, a: PackedCtx, scalar: typing.SupportsFloat | typing.SupportsIndex) -> PackedCtx:
        """
        a - scalar.
        """
    @property
    def complex_payload(self) -> bool:
        ...
    @property
    def has_secret_key(self) -> bool:
        ...
    @property
    def loaded_rot_steps(self) -> list[int]:
        """
        Rotation steps this session loaded (keygen band + load_rotation_steps - free_rotation_steps); empty after close_session. The ones shared with the bootstrap precomputation stay resident with the context.
        """
class CutMaxCalib:
    """
    Parsed 'cutmax' calibration section (opaque).
    """
class CutMaxConfig:
    """
    The encrypted-argmax schedule: per-iteration amplification powers, range-reduction passes and Newton settings.
    """
    def __repr__(self) -> str:
        ...
    @property
    def iters(self) -> list:
        """
        One dict per CutMax iteration (p, c, m, s2_hi, passes, ex2, ca, cb, casc_iters).
        """
    @property
    def n_iters(self) -> int:
        ...
    @property
    def newton_per_pass(self) -> int:
        ...
    @property
    def newton_polish(self) -> int:
        ...
class DecodeSession:
    def __init__(self, config: RunConfig) -> None:
        """
        Build the model once (CKKS context, keys, encoded weights) from config; decode() reuses it.
        """
    def decode(self, inputs: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], raise_on_error: bool = True) -> RunResult:
        """
        Teacher-forced decode of config.tokens tokens on a fresh KV cache; returns a RunResult, or raises on a token error unless raise_on_error=False.
        """
class EncodedBlock:
    plan: BootstrapPlan
    prefix: str
    def __init__(self) -> None:
        """
        An empty block state; fill it via set_weight/set_bias/set_*_cfg.
        """
    @typing.overload
    def set_bias(self, inf: Inference, name: str, b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, fill: bool = True) -> None:
        """
        numpy fast path for EncodedBlock.set_bias.
        """
    @typing.overload
    def set_bias(self, inf: Inference, name: str, b: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, fill: bool = True) -> None:
        ...
    def set_gelu_cfg(self, name: str, cfg: GeLUConfig) -> None:
        """
        Store the GeLUConfig for GeLU site `name` (copied into inf.gelu_cfg at install).
        """
    def set_norm_cfg(self, name: str, cfg: NormConfig) -> None:
        """
        Store the NormConfig for norm site `name` (copied into inf.norm_cfg at install).
        """
    def set_softmax_cfg(self, name: str, cfg: SoftmaxConfig) -> None:
        """
        Store the SoftmaxConfig for softmax site `name` (copied into inf.sm_cfg at install).
        """
    @typing.overload
    def set_weight(self, inf: Inference, name: str, W: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, level: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        numpy fast path: a (d_in, d_out) float array.
        """
    @typing.overload
    def set_weight(self, inf: Inference, name: str, W: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, level: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def set_weight_complex(self, inf: Inference, name: str, W_re: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], W_im: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, level: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
class FHEError(RuntimeError):
    pass
class GSInitMethod:
    """
    Members:
    
      LINEAR
    
      CHEBYSHEV
    """
    CHEBYSHEV: typing.ClassVar[GSInitMethod]  # value = <GSInitMethod.CHEBYSHEV: 1>
    LINEAR: typing.ClassVar[GSInitMethod]  # value = <GSInitMethod.LINEAR: 0>
    __members__: typing.ClassVar[dict[str, GSInitMethod]]  # value = {'LINEAR': <GSInitMethod.LINEAR: 0>, 'CHEBYSHEV': <GSInitMethod.CHEBYSHEV: 1>}
    @typing.overload
    def __eq__(self, other: GSInitMethod) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: GSInitMethod) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GeLUConfig:
    gate: bool
    method: GeLUMethod
    def __init__(self) -> None:
        """
        Default GELU config: SOFTSIGN_INV_SQRT with the gate on, exp_iters 12, newton_iters 2, gs_iters 14; fits zero/empty.
        """
    def __repr__(self) -> str:
        ...
    @property
    def Dcoeffs(self) -> list[float]:
        ...
    @Dcoeffs.setter
    def Dcoeffs(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def Ncoeffs(self) -> list[float]:
        ...
    @Ncoeffs.setter
    def Ncoeffs(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def a(self) -> float:
        ...
    @a.setter
    def a(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def b(self) -> float:
        ...
    @b.setter
    def b(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c(self) -> float:
        ...
    @c.setter
    def c(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cheb_a(self) -> float:
        ...
    @cheb_a.setter
    def cheb_a(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cheb_b(self) -> float:
        ...
    @cheb_b.setter
    def cheb_b(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cheb_coeffs(self) -> list[float]:
        ...
    @cheb_coeffs.setter
    def cheb_coeffs(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def exp_iters(self) -> int:
        ...
    @exp_iters.setter
    def exp_iters(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def gate_cheb_a(self) -> float:
        ...
    @gate_cheb_a.setter
    def gate_cheb_a(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gate_cheb_b(self) -> float:
        ...
    @gate_cheb_b.setter
    def gate_cheb_b(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gate_cheb_coeffs(self) -> list[float]:
        ...
    @gate_cheb_coeffs.setter
    def gate_cheb_coeffs(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def gs_hi(self) -> float:
        ...
    @gs_hi.setter
    def gs_hi(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gs_iters(self) -> int:
        ...
    @gs_iters.setter
    def gs_iters(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def gs_lo(self) -> float:
        ...
    @gs_lo.setter
    def gs_lo(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def inv_out_scale(self) -> float:
        ...
    @inv_out_scale.setter
    def inv_out_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lin_alpha(self) -> float:
        ...
    @lin_alpha.setter
    def lin_alpha(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lin_beta(self) -> float:
        ...
    @lin_beta.setter
    def lin_beta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def newton_iters(self) -> int:
        ...
    @newton_iters.setter
    def newton_iters(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def thor_p1(self) -> list[float]:
        ...
    @thor_p1.setter
    def thor_p1(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def thor_p2(self) -> list[float]:
        ...
    @thor_p2.setter
    def thor_p2(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def xmax(self) -> float:
        ...
    @xmax.setter
    def xmax(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def z_max(self) -> float:
        ...
    @z_max.setter
    def z_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def z_min(self) -> float:
        ...
    @z_min.setter
    def z_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class GeLUMethod:
    """
    Members:
    
      SOFTSIGN_INV_SQRT
    
      CHEBYSHEV
    
      THOR_COMPOSITE
    """
    CHEBYSHEV: typing.ClassVar[GeLUMethod]  # value = <GeLUMethod.CHEBYSHEV: 1>
    SOFTSIGN_INV_SQRT: typing.ClassVar[GeLUMethod]  # value = <GeLUMethod.SOFTSIGN_INV_SQRT: 0>
    THOR_COMPOSITE: typing.ClassVar[GeLUMethod]  # value = <GeLUMethod.THOR_COMPOSITE: 2>
    __members__: typing.ClassVar[dict[str, GeLUMethod]]  # value = {'SOFTSIGN_INV_SQRT': <GeLUMethod.SOFTSIGN_INV_SQRT: 0>, 'CHEBYSHEV': <GeLUMethod.CHEBYSHEV: 1>, 'THOR_COMPOSITE': <GeLUMethod.THOR_COMPOSITE: 2>}
    @typing.overload
    def __eq__(self, other: GeLUMethod) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: GeLUMethod) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GtSteps:
    @property
    def T(self) -> int:
        """
        Number of oracle steps (rows in logits); 0 when no oracle file exists.
        """
    @property
    def logits(self) -> list[list[float]]:
        """
        Ground-truth lm_head logits per step, [T][vocab].
        """
class Inference:
    block_prefix: str
    cache_weights: bool
    complex: bool
    mode: InferenceMode
    size: ModelSize
    strict_masks: bool
    token_pair: bool
    use_cache: bool
    weight_granularity: WeightGranularity
    def add_affine_term(self, ct: PackedCtx, name: str) -> None:
        """
        ct += the stored per-feature affine term `name` (mirrored into the Im lane under token-pair packing).
        """
    def add_pt(self, ct: PackedCtx, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> PackedCtx:
        """
        ct + values (slot-wise plaintext add).
        """
    def clear_bootstrap_plan(self) -> None:
        """
        Drop the installed bootstrap placement plan.
        """
    def clear_enc_cache(self) -> int:
        """
        Evict and forget every cached encoded plaintext; returns how many were dropped.
        """
    def disable_graph_capture(self) -> None:
        """
        Detach the graph builder from the context and drop it.
        """
    def enable_graph_capture(self) -> None:
        """
        Start graph capture: a fresh (or cleared) GraphBuilder attached to the context.
        """
    def eval_chebyshev(self, ct: PackedCtx, coeffs: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], a: typing.SupportsFloat | typing.SupportsIndex = -1.0, b: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> PackedCtx:
        """
        Evaluate sum_k coeffs[k] * T_k(x) slot-wise for x in [a, b] (the runtime's Chebyshev evaluator — what the GELU/softmax composites use). Levels consumed grow with the degree; the argument must lie inside [a, b].
        """
    def evict_weights(self, key: str) -> None:
        """
        Erase weight `key` (its device plaintexts freed unless weights are resident); no-op if absent.
        """
    def export_graph_json(self, path: str) -> None:
        """
        Write the captured graph to `path` (plus capture_env.json beside it); no-op when not capturing.
        """
    def graph_capture_enabled(self) -> bool:
        """
        True while a graph builder is attached and enabled.
        """
    def load_bootstrap_plan_json(self, path: str) -> bool:
        """
        Install the bootstrap placement plan at `path`; returns whether planned bootstraps are enabled.
        """
    def mult_pt(self, ct: PackedCtx, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> PackedCtx:
        """
        ct * values (slot-wise): `values` is encoded as a plaintext at ct's level; shorter than the slot count is zero-filled.
        """
    def name_ct(self, ct: PackedCtx, name: str) -> None:
        """
        Name `ct` in the captured graph, overwriting any existing name.
        """
    def name_ct_if_absent(self, ct: PackedCtx, name: str) -> None:
        """
        Name `ct` in the captured graph only if it has no name yet.
        """
    def scoped(self, name: str) -> str:
        """
        block_prefix + name: the key a block-scoped tag resolves to.
        """
    @typing.overload
    def set_bias(self, name: str, b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, fill: bool = True) -> None:
        """
        numpy fast path for set_bias.
        """
    @typing.overload
    def set_bias(self, name: str, b: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, fill: bool = True) -> None:
        """
        Install a bias of up to d_out entries under `name` (shorter is zero-filled; longer raises ValueError).
        """
    def set_gelu_cfg(self, name: str, cfg: GeLUConfig) -> None:
        """
        Store `cfg` as the GeLUConfig gelu_approx looks up by `name`.
        """
    def set_norm_cfg(self, name: str, cfg: NormConfig) -> None:
        """
        Store `cfg` as the NormConfig norm/layer_norm look up by `name`.
        """
    def set_softmax_cfg(self, name: str, cfg: SoftmaxConfig) -> None:
        """
        Store `cfg` as the SoftmaxConfig attention_softmax_thor looks up by `name`.
        """
    @typing.overload
    def set_weight(self, name: str, W: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, level: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        numpy fast path: a (d_in, d_out) float array is copied once from its buffer.
        """
    @typing.overload
    def set_weight(self, name: str, W: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, level: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        Install a (d_in, d_out) weight matrix under `name` as CKKS plaintexts (y = x @ W). A wrong shape raises ValueError.
        """
    def set_weight_complex(self, name: str, W_re: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], W_im: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, level: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def step(self, label: str) -> StepScope:
        """
        `with inf.step(label):` scopes the ops inside under a step label, as the C++ WithStep does.
        """
    def sum_slots(self, ct: PackedCtx, width: typing.SupportsInt | typing.SupportsIndex) -> PackedCtx:
        """
        Rotate-and-add: slot i receives the sum of slots i .. i+width-1 (width a power of two; needs rotation keys for 1, 2, 4, ... width/2). The lane-0 slot of each width-aligned group holds that group's total.
        """
    @property
    def capture_b(self) -> int:
        ...
    @capture_b.setter
    def capture_b(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def capture_t(self) -> int:
        ...
    @capture_t.setter
    def capture_t(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def fhe(self) -> Context:
        ...
    @property
    def installed_weights(self) -> list[str]:
        """
        Names of the weights currently installed (bind order not preserved).
        """
    @property
    def logN(self) -> int:
        ...
    @property
    def mlp_tile_dim(self) -> int:
        ...
    @mlp_tile_dim.setter
    def mlp_tile_dim(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_tok(self) -> int:
        ...
    @n_tok.setter
    def n_tok(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_tok_imag(self) -> int:
        ...
    @n_tok_imag.setter
    def n_tok_imag(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def slots(self) -> int:
        ...
class InferenceMode:
    """
    Members:
    
      Sync
    
      Threaded
    
      Prefetch
    """
    Prefetch: typing.ClassVar[InferenceMode]  # value = <InferenceMode.Prefetch: 2>
    Sync: typing.ClassVar[InferenceMode]  # value = <InferenceMode.Sync: 0>
    Threaded: typing.ClassVar[InferenceMode]  # value = <InferenceMode.Threaded: 1>
    __members__: typing.ClassVar[dict[str, InferenceMode]]  # value = {'Sync': <InferenceMode.Sync: 0>, 'Threaded': <InferenceMode.Threaded: 1>, 'Prefetch': <InferenceMode.Prefetch: 2>}
    @typing.overload
    def __eq__(self, other: InferenceMode) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: InferenceMode) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class InferenceOptions:
    bench_mode: bool
    ckks: CKKSOptions
    mode: InferenceMode
    packing_kind: PackingKind
    parallel: bool
    @typing.overload
    def __init__(self) -> None:
        """
        Default options: GPT-2-small sizes, Cachemir packing, Threaded mode, parallel on, bench_mode off, default CKKS options.
        """
    @typing.overload
    def __init__(self, other: InferenceOptions) -> None:
        """
        Copy constructor: InferenceOptions(other) is an exact copy (ckks included).
        """
    def __repr__(self) -> str:
        ...
    @property
    def aux_packing_kinds(self) -> list[PackingKind]:
        ...
    @aux_packing_kinds.setter
    def aux_packing_kinds(self, arg0: collections.abc.Sequence[PackingKind]) -> None:
        ...
    @property
    def dim(self) -> int:
        ...
    @dim.setter
    def dim(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def expDim(self) -> int:
        ...
    @expDim.setter
    def expDim(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def expanded(self) -> int:
        ...
    @expanded.setter
    def expanded(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def hidDim(self) -> int:
        ...
    @hidDim.setter
    def hidDim(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def numHeads(self) -> int:
        ...
    @numHeads.setter
    def numHeads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def numHeadsReal(self) -> int:
        ...
    @numHeadsReal.setter
    def numHeadsReal(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def seqLen(self) -> int:
        ...
    @seqLen.setter
    def seqLen(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class LMHeadCache:
    def __init__(self) -> None:
        """
        An empty lm_head tile store; filled by lm_head / prepare_feedback_weights.
        """
class LayoutError(FHEError):
    pass
class MaskError(FHEError):
    pass
class ModelConfig:
    @property
    def n_embd(self) -> int:
        ...
    @property
    def n_head(self) -> int:
        ...
    @property
    def n_inner(self) -> int:
        ...
    @property
    def n_layers(self) -> int:
        ...
class ModelSize:
    def __init__(self) -> None:
        """
        Default model size: GPT-2 small (dim 768, expanded 3072) padded to hidDim 1024 / expDim 4096, 12 real heads in 16, seqLen 1024.
        """
    def __repr__(self) -> str:
        ...
    @property
    def dim(self) -> int:
        ...
    @dim.setter
    def dim(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def expDim(self) -> int:
        ...
    @expDim.setter
    def expDim(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def expanded(self) -> int:
        ...
    @expanded.setter
    def expanded(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def hidDim(self) -> int:
        ...
    @hidDim.setter
    def hidDim(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def numHeads(self) -> int:
        ...
    @numHeads.setter
    def numHeads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def numHeadsReal(self) -> int:
        ...
    @numHeadsReal.setter
    def numHeadsReal(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def seqLen(self) -> int:
        ...
    @seqLen.setter
    def seqLen(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class NRInitMethod:
    """
    Members:
    
      TAYLOR
    
      REMEZ
    """
    REMEZ: typing.ClassVar[NRInitMethod]  # value = <NRInitMethod.REMEZ: 1>
    TAYLOR: typing.ClassVar[NRInitMethod]  # value = <NRInitMethod.TAYLOR: 0>
    __members__: typing.ClassVar[dict[str, NRInitMethod]]  # value = {'TAYLOR': <NRInitMethod.TAYLOR: 0>, 'REMEZ': <NRInitMethod.REMEZ: 1>}
    @typing.overload
    def __eq__(self, other: NRInitMethod) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: NRInitMethod) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NormConfig:
    nr_init_method: NRInitMethod
    def __init__(self) -> None:
        """
        Default LayerNorm config: TAYLOR NR init, 16 NR iterations, unit center/output scales, empty polynomials.
        """
    def __repr__(self) -> str:
        ...
    @property
    def Dcoeffs(self) -> list[float]:
        ...
    @Dcoeffs.setter
    def Dcoeffs(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def Ncoeffs(self) -> list[float]:
        ...
    @Ncoeffs.setter
    def Ncoeffs(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def center_scale(self) -> float:
        ...
    @center_scale.setter
    def center_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def center_scale_sq(self) -> list[float]:
        ...
    @center_scale_sq.setter
    def center_scale_sq(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gs_hi(self) -> float:
        ...
    @gs_hi.setter
    def gs_hi(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gs_iters(self) -> int:
        ...
    @gs_iters.setter
    def gs_iters(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def gs_lo(self) -> float:
        ...
    @gs_lo.setter
    def gs_lo(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def inv_out_scale(self) -> float:
        ...
    @inv_out_scale.setter
    def inv_out_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lin_alpha(self) -> float:
        ...
    @lin_alpha.setter
    def lin_alpha(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lin_beta(self) -> float:
        ...
    @lin_beta.setter
    def lin_beta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def nr_iters(self) -> int:
        ...
    @nr_iters.setter
    def nr_iters(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def taylor_z0(self) -> float:
        ...
    @taylor_z0.setter
    def taylor_z0(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PackedCtx:
    def __repr__(self) -> str:
        ...
    @property
    def level(self) -> int:
        ...
    @property
    def noise_deg(self) -> int:
        ...
    @property
    def packing(self) -> str:
        ...
class PackingKind:
    """
    Members:
    
      Cachemir
    
      Diagonal
    
      CachemirFilling
    
      CachemirComplex
    """
    Cachemir: typing.ClassVar[PackingKind]  # value = <PackingKind.Cachemir: 0>
    CachemirComplex: typing.ClassVar[PackingKind]  # value = <PackingKind.CachemirComplex: 3>
    CachemirFilling: typing.ClassVar[PackingKind]  # value = <PackingKind.CachemirFilling: 2>
    Diagonal: typing.ClassVar[PackingKind]  # value = <PackingKind.Diagonal: 1>
    __members__: typing.ClassVar[dict[str, PackingKind]]  # value = {'Cachemir': <PackingKind.Cachemir: 0>, 'Diagonal': <PackingKind.Diagonal: 1>, 'CachemirFilling': <PackingKind.CachemirFilling: 2>, 'CachemirComplex': <PackingKind.CachemirComplex: 3>}
    @typing.overload
    def __eq__(self, other: PackingKind) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: PackingKind) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ParsedConfigs:
    @property
    def cutmax(self) -> CutMaxCalib:
        ...
    @property
    def has_cutmax(self) -> bool:
        ...
    @property
    def model(self) -> ModelConfig:
        ...
    @property
    def norm(self) -> dict[str, NormConfig]:
        ...
    @property
    def softgelu(self) -> dict[str, GeLUConfig]:
        ...
    @property
    def softmax(self) -> dict[str, SoftmaxConfig]:
        ...
class PlanError(FHEError):
    pass
class RunConfig:
    @staticmethod
    def from_env() -> RunConfig:
        """
        Build a RunConfig from the environment (MULTI_T, STEPS_T, CONFIGS_PATH, WEIGHTS_PATH, ALL_BLOCKS_IO_DIR, FHE_*_PLACEMENTS_DIR, FHE_GRAPH_DIR, GPT2_INFERENCE_MODE, GPT2_CACHE, TEACHER_FORCED).
        """
    def __init__(self) -> None:
        """
        Construct a RunConfig with the compiled-in defaults (app/pipeline.h).
        """
    def __repr__(self) -> str:
        """
        Repr listing every bound field.
        """
    @property
    def cache_weights(self) -> bool:
        """
        Cache encoded weights across blocks (GPT2_CACHE).
        """
    @cache_weights.setter
    def cache_weights(self, arg0: bool) -> None:
        ...
    @property
    def configs_path(self) -> str:
        """
        Approximation configs.json path (CONFIGS_PATH).
        """
    @configs_path.setter
    def configs_path(self, arg0: str) -> None:
        ...
    @property
    def decode_plan_dir(self) -> str:
        """
        FHE_DECODE_PLACEMENTS_DIR: post-handoff decode-phase plan (run_prefill).
        """
    @decode_plan_dir.setter
    def decode_plan_dir(self, arg0: str) -> None:
        ...
    @property
    def decode_tokens(self) -> int:
        """
        run_prefill only: tail decode token count (DECODE_TOKENS); -1 = 1.
        """
    @decode_tokens.setter
    def decode_tokens(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def gen_prompt(self) -> int:
        """
        run_generate: teacher-forced prompt rows (GEN_PROMPT).
        """
    @gen_prompt.setter
    def gen_prompt(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def gen_tokens(self) -> int:
        """
        run_generate: tokens generated under encryption (GEN_TOKENS).
        """
    @gen_tokens.setter
    def gen_tokens(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def graph_dir(self) -> str:
        """
        Graph capture output directory (FHE_GRAPH_DIR).
        """
    @graph_dir.setter
    def graph_dir(self, arg0: str) -> None:
        ...
    @property
    def io_dir(self) -> str:
        """
        ALL_BLOCKS_IO_DIR: teacher-forced input / oracle directory.
        """
    @io_dir.setter
    def io_dir(self, arg0: str) -> None:
        ...
    @property
    def mode(self) -> InferenceMode:
        """
        InferenceMode passed to GPT2Model::load (GPT2_INFERENCE_MODE: sync|threaded|prefetch).
        """
    @mode.setter
    def mode(self, arg0: InferenceMode) -> None:
        ...
    @property
    def plan_dir(self) -> str:
        """
        Bootstrap placement plan directory (FHE_BOOTSTRAP_PLACEMENTS_DIR); empty = eager.
        """
    @plan_dir.setter
    def plan_dir(self, arg0: str) -> None:
        ...
    @property
    def prefill_tokens(self) -> int:
        """
        run_prefill only: prefill row count (PREFILL_TOKENS); -1 = tokens - 1.
        """
    @prefill_tokens.setter
    def prefill_tokens(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def prime_dummy_cache(self) -> bool:
        """
        Not read by the C++ pipelines; from_env leaves it False.
        """
    @prime_dummy_cache.setter
    def prime_dummy_cache(self, arg0: bool) -> None:
        ...
    @property
    def steps_t(self) -> int:
        """
        Oracle step count T of the all_blocks IO files in io_dir (STEPS_T).
        """
    @steps_t.setter
    def steps_t(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def teacher_forced(self) -> bool:
        """
        run_generate: advance on the GT token instead of the CutMax feedback (TEACHER_FORCED).
        """
    @teacher_forced.setter
    def teacher_forced(self, arg0: bool) -> None:
        ...
    @property
    def tokens(self) -> int:
        """
        Tokens to run (MULTI_T); run_prefill derives prefill_tokens from it when unset.
        """
    @tokens.setter
    def tokens(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def weights_path(self) -> str:
        """
        Model weight archive: a .zip file or a directory (WEIGHTS_PATH).
        """
    @weights_path.setter
    def weights_path(self, arg0: str) -> None:
        ...
class RunResult:
    def __repr__(self) -> str:
        """
        Repr with the scalar fields and the sizes of logits/top1/positions.
        """
    @property
    def avg_argmax_s(self) -> float:
        """
        Average CutMax argmax stage seconds per token (within e2e).
        """
    @property
    def avg_s_per_tok(self) -> float:
        """
        Average seconds per token; token 0 (cold start) excluded.
        """
    @property
    def bootstraps(self) -> int:
        """
        Total bootstraps performed during the run.
        """
    @property
    def completed(self) -> int:
        """
        Tokens completed (rows in logits).
        """
    @property
    def error(self) -> str:
        """
        Error text naming the throwing token; empty unless threw.
        """
    @property
    def logits(self) -> list[list[float]]:
        """
        Decrypted logits per completed token, [completed][vocab].
        """
    @property
    def positions(self) -> list[int]:
        """
        Absolute token position of each logits row.
        """
    @property
    def requested(self) -> int:
        """
        Tokens requested by the config.
        """
    @property
    def threw(self) -> bool:
        """
        True if a token threw and the run stopped there (see error).
        """
    @property
    def top1(self) -> list[int]:
        """
        Argmax token per completed row (run_generate: the CutMax argmax).
        """
    @property
    def unplanned_bts(self) -> int:
        """
        Auto-bootstraps fired at a level ceiling the plan did not predict.
        """
    @property
    def weight_relevels(self) -> int:
        """
        Weight re-encodes at a new level (diagnostic; 0 = weight levels perfect).
        """
class SoftmaxConfig:
    def __init__(self) -> None:
        """
        Default softmax config: LINEAR GS init with every fit zero/empty (calibration fills them).
        """
    def __repr__(self) -> str:
        ...
    @property
    def cheb_a(self) -> float:
        ...
    @cheb_a.setter
    def cheb_a(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cheb_b(self) -> float:
        ...
    @cheb_b.setter
    def cheb_b(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cheb_coeffs(self) -> list[float]:
        ...
    @cheb_coeffs.setter
    def cheb_coeffs(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def clip_hi(self) -> float:
        ...
    @clip_hi.setter
    def clip_hi(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def clip_lo(self) -> float:
        ...
    @clip_lo.setter
    def clip_lo(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gs_init_method(self) -> GSInitMethod:
        """
        Which formula the calibration used for the Goldschmidt seed. A record of how init_alpha/init_beta were fitted; the evaluation reads those two, not this, so changing it on a live config has no effect.
        """
    @gs_init_method.setter
    def gs_init_method(self, arg0: GSInitMethod) -> None:
        ...
    @property
    def gs_iters_refine_scaled(self) -> int:
        ...
    @gs_iters_refine_scaled.setter
    def gs_iters_refine_scaled(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def gs_iters_scaled(self) -> int:
        ...
    @gs_iters_scaled.setter
    def gs_iters_scaled(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def init_alpha(self) -> float:
        ...
    @init_alpha.setter
    def init_alpha(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def init_beta(self) -> float:
        ...
    @init_beta.setter
    def init_beta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def log2delta1(self) -> int:
        ...
    @log2delta1.setter
    def log2delta1(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def log2delta2(self) -> int:
        ...
    @log2delta2.setter
    def log2delta2(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def per_step_refine_iters(self) -> list[float]:
        ...
    @per_step_refine_iters.setter
    def per_step_refine_iters(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def poly_coeffs(self) -> list[float]:
        ...
    @poly_coeffs.setter
    def poly_coeffs(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def refine_alpha(self) -> list[float]:
        ...
    @refine_alpha.setter
    def refine_alpha(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def refine_beta(self) -> list[float]:
        ...
    @refine_beta.setter
    def refine_beta(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def sm_kc_r(self) -> list[float]:
        ...
    @sm_kc_r.setter
    def sm_kc_r(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
class Stage:
    """
    One residency-pipeline stage: compute(x) -> x on the main thread while the runner streams the next stage's weights. Exactly one of state= (an EncodedBlock), weights= (inf.w keys) or loader= describes what to stream. release=: a callable run on the main thread after compute; it REPLACES the stage's default release (device sync + evict), so a stage that passes release= must evict what it acquired itself.
    """
    def __init__(self, compute: typing.Any, state: typing.Any = None, weights: collections.abc.Sequence[str] = [], loader: typing.Any = None, release: typing.Any = None, label: str = 'stage') -> None:
        """
        One residency-pipeline stage: `compute` (x -> x) is the Python forward; pick at most one of `state` (EncodedBlock), `weights` (inf.w keys) or `loader` ((WeightStore, ParsedConfigs, BootstrapPlan, block_idx): encode-on-the-fly); `release` replaces the stage's default C++ release.
        """
    @property
    def compute(self) -> typing.Any:
        """
        Callable x -> x: the stage's Python forward (main thread, GIL held).
        """
    @compute.setter
    def compute(self, arg0: typing.Any) -> None:
        ...
    @property
    def label(self) -> str:
        """
        Stage name: its compute's WithStep profiler step and the run_stages error messages.
        """
    @label.setter
    def label(self, arg0: str) -> None:
        ...
    @property
    def loader(self) -> typing.Any:
        """
        None | (WeightStore, ParsedConfigs, BootstrapPlan, block_idx): encode-on-the-fly stage, the canonical loader run per pass.
        """
    @loader.setter
    def loader(self, arg0: typing.Any) -> None:
        ...
    @property
    def release(self) -> typing.Any:
        """
        None | callable(): replaces the stage's default C++ release (main thread, GIL held).
        """
    @release.setter
    def release(self, arg0: typing.Any) -> None:
        ...
    @property
    def state(self) -> typing.Any:
        """
        None | _core.EncodedBlock: pre-encoded state of a cached-block stage.
        """
    @state.setter
    def state(self, arg0: typing.Any) -> None:
        ...
    @property
    def weights(self) -> list[str]:
        """
        inf.w keys loaded before / evicted after compute (streamed-key stage).
        """
    @weights.setter
    def weights(self, arg0: collections.abc.Sequence[str]) -> None:
        ...
class StepScope:
    def __enter__(self) -> None:
        """
        Push the scope's label onto the context's step stack.
        """
    def __exit__(self, exc_type: typing.Any, exc_value: typing.Any, traceback: typing.Any) -> bool:
        """
        Pop the step label; returns False so exceptions propagate.
        """
class WeightGranularity:
    """
    Members:
    
      Block
    
      Sublayer
    
      Linear
    
      Plaintext
    """
    Block: typing.ClassVar[WeightGranularity]  # value = <WeightGranularity.Block: 0>
    Linear: typing.ClassVar[WeightGranularity]  # value = <WeightGranularity.Linear: 2>
    Plaintext: typing.ClassVar[WeightGranularity]  # value = <WeightGranularity.Plaintext: 3>
    Sublayer: typing.ClassVar[WeightGranularity]  # value = <WeightGranularity.Sublayer: 1>
    __members__: typing.ClassVar[dict[str, WeightGranularity]]  # value = {'Block': <WeightGranularity.Block: 0>, 'Sublayer': <WeightGranularity.Sublayer: 1>, 'Linear': <WeightGranularity.Linear: 2>, 'Plaintext': <WeightGranularity.Plaintext: 3>}
    @typing.overload
    def __eq__(self, other: WeightGranularity) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: WeightGranularity) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class WeightStore:
    @staticmethod
    def from_dir(dir_path: str) -> WeightStore:
        """
        Load a WeightStore from an export directory (manifest.json + tensor files).
        """
    @staticmethod
    def from_zip(zip_path: str) -> WeightStore:
        """
        Load a WeightStore from a zip export (manifest.json + tensor entries).
        """
    def __contains__(self, arg0: str) -> bool:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def has(self, name: str) -> bool:
        """
        Whether `name` is in the store.
        """
    def names(self) -> list[str]:
        """
        Sorted tensor names in the store.
        """
    def shape(self, name: str) -> list[int]:
        """
        Shape of tensor `name` (KeyError when absent).
        """
    def tensor(self, name: str) -> numpy.typing.NDArray[numpy.float64]:
        """
        Tensor `name` as a float64 array in its manifest shape (KeyError when absent).
        """
def apply_final_ln(inf: Inference, x: PackedCtx, lnf: EncodedBlock) -> PackedCtx:
    """
    Install lnf, apply the final LayerNorm (ln_f) to x, evict lnf; returns the normed ct.
    """
def attention_softmax_thor(inf: Inference, scores: collections.abc.Sequence[PackedCtx], cfg_name: str) -> list[PackedCtx]:
    """
    Softmax over the score ciphertexts under SoftmaxConfig `cfg_name` (THOR approximation).
    """
def begin_subgraph_capture(inf: Inference, block: typing.SupportsInt | typing.SupportsIndex) -> bool:
    """
    Start capturing block `block` (FHE_GRAPH_DIR set, capture wanted, no graph.json there yet); returns whether capture began.
    """
def block_release(inf: Inference, block_idx: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Decode-arm block release: device sync, evict the block weights, offload its KV.
    """
def block_scope(block_idx: typing.SupportsInt | typing.SupportsIndex) -> str:
    """
    Per-block cache/weight scope prefix ("transformer.h.{b}.").
    """
def block_state_diff(a: EncodedBlock, b: EncodedBlock) -> str:
    """
    Compare two EncodedBlocks pt-by-pt: DCRT element equality for coeff-staged, slot values for full. Returns a summary string (all-zero mismatches = equal).
    """
def build_info() -> dict:
    """
    Version, chain (n32/n64), native integer width, pybind11 and CUDA runtime versions, and the compile timestamp of this extension.
    """
def cache_k_push(inf: Inference, key: PackedCtx) -> None:
    """
    Append `key` to this block's K cache.
    """
def cache_kv_push(inf: Inference, key: PackedCtx, value: PackedCtx) -> None:
    """
    Push `key` and `value` into this block's K and V caches.
    """
def cache_kv_push_packed(inf: Inference, kv_packed: PackedCtx) -> None:
    """
    Cachemir-only: push a pre-packed K + i*V ciphertext into this block's K and V caches.
    """
def cache_v_push(inf: Inference, value: PackedCtx) -> None:
    """
    Append `value` to this block's V cache.
    """
def close_session(inf: Inference) -> int:
    """
    Release what the session holds: every installed weight, the KV and mask caches, the encode cache and the loaded rotation keys. Returns the number of rotation keys freed (the ones shared with the bootstrap precomputation are protected and stay with the context; on the GPT-2 n32 band 87 of 133). The memory goes back to the runtime's device pool (FIDESlib keeps freed limbs in per-size free lists and the CUDA pool's release threshold is unbounded), so cudaMemGetInfo does not move: it is reused by the next session in this process and returned to the device at process exit. The CKKS context itself (keys, bootstrap precomputation) stays alive — ciphertexts still reference it — so a closed session cannot compute, but a new one can be created.
    """
def configure_decode_phase(inf: Inference, complex_decode: bool) -> None:
    """
    Set inf's decode phase fields: Cachemir packing, n_tok=1, block weights, complex flag.
    """
def configure_prefill_phase(inf: Inference, n_tok: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Set inf's prefill phase fields: CachemirFilling packing, n_tok, plaintext weights.
    """
def cutmax_argmax(inf: Inference, tiles: collections.abc.Sequence[PackedCtx], vocab: typing.SupportsInt | typing.SupportsIndex, config: CutMaxConfig) -> list[PackedCtx]:
    """
    Encrypted CutMax argmax over the logit tiles; returns one-hot Z tiles (eager only).
    """
def cutmax_config_from_calib(calib: CutMaxCalib) -> CutMaxConfig:
    """
    Build the runtime CutMaxConfig from a configs.json 'cutmax' calibration.
    """
def cutmax_feedback(inf: Inference, tiles: collections.abc.Sequence[PackedCtx], store: WeightStore, vocab: typing.SupportsInt | typing.SupportsIndex, config: CutMaxConfig, feedback: LMHeadCache = None, position: typing.SupportsInt | typing.SupportsIndex = -1, plan13: BootstrapPlan = None, plan14: BootstrapPlan = None) -> tuple[typing.Any, typing.Any, float]:
    """
    Generation tail: CutMax argmax + codebook feedback + wpe + entry bootstrap. Returns (next_x | None, z_tiles, argmax_seconds); position < 0 skips the feedback half.
    """
def decode_linear_output(inf: Inference, ct: PackedCtx, d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Decrypt `ct` and decode a (d_in, d_out) linear output to its d_out values (packing-aware).
    """
def decode_lm_head_logits(inf: Inference, tiles: collections.abc.Sequence[PackedCtx], vocab: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Decrypt lm_head logit tiles into a [vocab] float array (uses inf's secret key).
    """
def decode_token_output(inf: Inference, ct: PackedCtx) -> numpy.typing.NDArray[numpy.float64]:
    """
    Decrypt one token's real features from `ct` (unpack_tokens with T=1).
    """
def decode_tokens_output(inf: Inference, ct: PackedCtx, n_tok: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Decode the n_tok tokens packed in `ct` (slot[i*t + tok]) to [n_tok][d_real]; n_tok=1 equals decode_token_output.
    """
def decrypt_slots(inf: Inference, x: PackedCtx) -> list[float]:
    ...
def default_cutmax_config() -> CutMaxConfig:
    """
    The oracle-locked default GPT-2 CutMax schedule.
    """
def deserialize_ct(inf: Inference, data: bytes) -> PackedCtx:
    """
    bytes -> PackedCtx in this session's context; host-resident until first use.
    """
def device_free_gb() -> float:
    """
    Free memory on the current CUDA device, in GiB (cudaMemGetInfo).
    """
def encode_block_state_coeff(inf: Inference, store: WeightStore, configs: ParsedConfigs, plan: BootstrapPlan, block_idx: typing.SupportsInt | typing.SupportsIndex) -> EncodedBlock:
    """
    load_block_state with the coeff-encode gate armed (needs FHE_PT_COEFF_ENCODE=1): eligible weight plaintexts come back as 1-limb coeff-staged forms.
    """
@typing.overload
def encode_prefill_input(inf: Inference, embeddings: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> PackedCtx:
    """
    numpy fast path: a [T][d] float array.
    """
@typing.overload
def encode_prefill_input(inf: Inference, embeddings: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]]) -> PackedCtx:
    """
    Pack T token embeddings into one CachemirFilling input ciphertext for prefill (bootstrap output level; token-pair aware).
    """
@typing.overload
def encode_token_input(inf: Inference, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> PackedCtx:
    """
    numpy fast path for encode_token_input.
    """
@typing.overload
def encode_token_input(inf: Inference, x: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> PackedCtx:
    """
    One token's real features (<= size.dim; shorter is zero-padded) -> fresh ciphertext. Longer raises ValueError instead of silently dropping the tail.
    """
def end_subgraph_capture(inf: Inference, block: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Finish block `block`'s capture: write its graph.json and detach; no-op when not capturing.
    """
def evict_block_from_device(inf: Inference, state: EncodedBlock) -> None:
    """
    Evict every weight plaintext of `state` from the GPU.
    """
def exp_approx(inf: Inference, x: PackedCtx, r: typing.SupportsInt | typing.SupportsIndex) -> PackedCtx:
    """
    exp(x) ~= (1 + x / 2^r)^(2^r).
    """
def extract_token_i_cachemir(inf: Inference, filling_ct: PackedCtx, i: typing.SupportsInt | typing.SupportsIndex) -> PackedCtx:
    """
    Extract token i of a cachemir_filling group ct into a cachemir single-token ct.
    """
def filling_rot_steps(inf: Inference) -> list[int]:
    """
    The filling-exclusive rotation steps (freeable post-prefill).
    """
def fold_ln_affine(cfg_name: str) -> bool:
    """
    Whether this LN's affine is folded into its consumer (GPT2_FOLD_LN_AFFINE, per-tag GPT2_FOLD_LN1/LN2/LNF).
    """
def free_rotation_steps(inf: Inference, steps: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> int:
    """
    Free the loaded rotation keys for `steps` to reclaim GPU memory; returns the count freed.
    """
def gelu_approx(inf: Inference, x: PackedCtx, cfg_name: str) -> PackedCtx:
    """
    GELU(x) under GeLUConfig `cfg_name` (softsign, Chebyshev or THOR composite per cfg.method).
    """
def gpt2_prefill(inf: Inference, x: PackedCtx, store: WeightStore, configs: ParsedConfigs, n_blocks: typing.SupportsInt | typing.SupportsIndex, chunk: bool = True) -> PackedCtx:
    """
    One EAGER prefill chunk over the filling packing (weights encoded per pass, worker-overlapped). chunk=True skips the final LN (caller LNs after the last chunk). KV lands in the filling cache: kv_handoff_filling_to_cachemir after the last chunk.
    """
def head_reduce_sum(inf: Inference, x: PackedCtx) -> PackedCtx:
    """
    Cachemir-only: sum `x` over the t slots of each head lane and broadcast the sum back to all t.
    """
def install_block_state(inf: Inference, state: EncodedBlock) -> None:
    """
    Copy state's weights, configs and plan into inf; inf.weight_store then points at it.
    """
def install_plan_live(inf: Inference, plan: BootstrapPlan) -> None:
    """
    Install a bootstrap plan on the live context (clears it when plan.valid is False).
    """
def kv_block_prologue(inf: Inference, block_idx: typing.SupportsInt | typing.SupportsIndex, n_blocks: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Per-block KV prologue: prefetch the next block's KV reload.
    """
def kv_finalize_last(inf: Inference, n_blocks: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Post-loop: drain + evict the last block's deferred KV offload.
    """
def kv_handoff_filling_to_cachemir(inf: Inference, n_blocks: typing.SupportsInt | typing.SupportsIndex, m: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Convert a filling-packing prefill's KV caches to the cachemir decode layout.
    """
def kv_prefetch_first(inf: Inference, n_blocks: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Overlapped KV pipeline: seed block 0's KV reload before the block loop.
    """
def layer_norm(inf: Inference, x: PackedCtx, cfg_name: str) -> PackedCtx:
    """
    norm(x, cfg_name) plus the LN affine: the folded shift when fold_ln_affine(cfg_name), else ln_affine.
    """
def layout_of(x: PackedCtx) -> str:
    """
    Debug: the ciphertext's tracked slot-layout basis.
    """
def linear(inf: Inference, x: PackedCtx, wname: str, d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, stream_pt: bool = False) -> PackedCtx:
    """
    y = x @ W for weight `wname` (d_in, d_out), dispatched on the packing; `stream_pt` loads/evicts each weight plaintext around its use.
    """
def linear_multi(inf: Inference, x: PackedCtx, wnames: collections.abc.Sequence[str], d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex, stream_pt: bool = False) -> list[PackedCtx]:
    """
    Prepare `x` once, then apply every weight in `wnames` (d_in, d_out) to it; one output per name.
    """
def linear_outputpack(inf: Inference, x: PackedCtx, wname: str, d_in: typing.SupportsInt | typing.SupportsIndex, d_out: typing.SupportsInt | typing.SupportsIndex) -> PackedCtx:
    """
    Cachemir-only linear whose output blocks are paired into complex slots (output-row pack, S4).
    """
def lm_head(inf: Inference, x: PackedCtx, store: WeightStore, vocab: typing.SupportsInt | typing.SupportsIndex, plan: BootstrapPlan, cache: LMHeadCache = None) -> list[PackedCtx]:
    """
    lm_head logits of x as ciphertext tiles (tiles encoded once into `cache` when given).
    """
def lm_head_tile_width(inf: Inference, vocab: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    The lm_head tile width: hidDim when vocab <= hidDim (small head), else slots.
    """
def lm_head_vocab(store: WeightStore) -> int:
    """
    Vocab size: rows of the lm_head (wte) weight in the store.
    """
def ln_affine(inf: Inference, normed: PackedCtx, tag: str) -> PackedCtx:
    """
    normed * <tag>.weight + <tag>.bias: the LayerNorm gamma/beta stored under `tag`.
    """
def load_block_state(inf: Inference, store: WeightStore, configs: ParsedConfigs, plan: BootstrapPlan, block_idx: typing.SupportsInt | typing.SupportsIndex) -> EncodedBlock:
    """
    Encode transformer block `block_idx` (weights, configs, plan) into an EncodedBlock.
    """
def load_block_state_file(inf: Inference, configs: ParsedConfigs, plan: BootstrapPlan, block_idx: typing.SupportsInt | typing.SupportsIndex, path: str) -> EncodedBlock:
    """
    Rebuild an EncodedBlock from a save_block_state artifact: weights from disk, configs/prefix/plan from the canonical parse.
    """
def load_block_to_device(inf: Inference, state: EncodedBlock) -> None:
    """
    Upload every weight plaintext of `state` to the GPU (default stream).
    """
def load_configs(path: str) -> ParsedConfigs:
    """
    Parse configs.json (path = the file or its directory).
    """
def load_final_ln_state(inf: Inference, store: WeightStore, configs: ParsedConfigs, plan: BootstrapPlan) -> EncodedBlock:
    """
    Encode the final LayerNorm (ln_f) weights + config into an EncodedBlock.
    """
def make_context(options: CKKSOptions = ...) -> Context:
    """
    Create a bare CKKS context (keygen) from CKKSOptions; model sessions use make_gpt2_inference / make_inference.
    """
def make_gpt2_inference(options: ... = ...) -> Inference:
    """
    make_inference plus the GPT-2 rotation keys for `options`' packing (and aux packings).
    """
def make_inference(options: ... = ...) -> Inference:
    """
    Build a generic Inference session from `options`: CKKS context, model sizes and packing.
    """
def mha_block(inf: Inference, x: PackedCtx) -> PackedCtx:
    """
    Multi-head attention sublayer on `x` (qkv, attention core, out-proj) run as an op sequence.
    """
def mlp_block(inf: Inference, x: PackedCtx) -> PackedCtx:
    """
    MLP sublayer on `x` (up-linear, GELU, down-linear); tiled when inf.tiled_mlp().
    """
def norm(inf: Inference, x: PackedCtx, cfg_name: str) -> PackedCtx:
    """
    Normalize `x` under NormConfig `cfg_name`: mean-centred times inverse-sqrt variance, no gamma/beta.
    """
@typing.overload
def pack_tokens(inf: Inference, embeddings: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], target_level: typing.SupportsInt | typing.SupportsIndex = 0) -> PackedCtx:
    """
    numpy fast path: a [T][d] float array.
    """
@typing.overload
def pack_tokens(inf: Inference, embeddings: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], target_level: typing.SupportsInt | typing.SupportsIndex = 0) -> PackedCtx:
    """
    Encode T token embeddings (each <= size.dim, zero-padded to hidDim) into one ciphertext at `target_level`.
    """
def parse_bootstrap_plan_file(path: str) -> BootstrapPlan:
    """
    Parse a bootstrap placement JSON file into a BootstrapPlan (valid=False when missing).
    """
def prepare_feedback_weights(inf: Inference, store: WeightStore, vocab: typing.SupportsInt | typing.SupportsIndex, packed_z: bool, cache: LMHeadCache, plan14: BootstrapPlan = None) -> None:
    """
    Encode the CutMax one-hot codebook (wte tiles) into cache once; plan14 = strict-tail encode level.
    """
def prepare_mha_masks(inf: Inference) -> None:
    """
    Reset this block's K cache before the first push.
    """
def prepare_vcache(inf: Inference) -> None:
    """
    Reset this block's V cache before the first push.
    """
def qkt(inf: Inference, query: PackedCtx) -> list[PackedCtx]:
    """
    Attention scores query . K^T against this block's K cache (a list of ciphertexts, packing-dependent).
    """
def read_lm_head_steps(config: RunConfig) -> GtSteps:
    """
    Read the ground-truth lm_head logits per step from config.io_dir (all_blocks_lm_head_steps_T{steps_t}.json); T=0 when the file is missing.
    """
def read_teacher_forced_inputs(config: RunConfig) -> list[list[float]]:
    """
    Read max(1, config.tokens) block-0 input rows from config.io_dir/all_blocks_L00_T*.json.
    """
def realize_pending_rescale(inf: Inference, x: PackedCtx) -> None:
    """
    Realize a deg-2 ciphertext's pending rescale in place (deg 1, level + d). FIDESlib multPt otherwise re-realizes a COPY inside every product.
    """
def reset_graph_runtime(inf: Inference) -> None:
    """
    Reset the runtime-graph naming state (ct/pt vars + counters) for capture/planned runs.
    """
def reset_kv_cache(inf: Inference, n_blocks: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Reset every block's K/V caches for a fresh sequence (also prewarms the pinned arenas).
    """
def run_decode(config: RunConfig, inputs: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], raise_on_error: bool = True) -> RunResult:
    """
    Teacher-forced decode of config.tokens tokens; equals DecodeSession(config).decode(inputs). Raises on a token error unless raise_on_error=False (then see RunResult.threw / error).
    """
def run_generate(config: RunConfig, inputs: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], raise_on_error: bool = True) -> RunResult:
    """
    Prompt with gen_prompt rows, then generate gen_tokens tokens feeding back the encrypted CutMax argmax (GT rows instead when teacher_forced). Raises on a token error unless raise_on_error=False.
    """
def run_prefill(config: RunConfig, inputs: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], raise_on_error: bool = True) -> RunResult:
    """
    Prefill config.prefill_tokens rows, then a teacher-forced decode of config.decode_tokens tail tokens. Raises on a token error unless raise_on_error=False.
    """
def run_stages(inf: Inference, x: typing.Any, stages: collections.abc.Sequence[Stage], mode: typing.Any = None) -> typing.Any:
    """
    Run stages through the residency pipeline: acquire/install/evict each stage's weight state around its Python compute. Cached states follow the decode shape (worker CPU extraction + streamed uploads); loader stages the prefill/encoder shape (per-pass canonical encode on the worker). mode None = inf.mode.
    """
def save_block_state(inf: Inference, state: EncodedBlock, path: str) -> None:
    """
    Serialize an EncodedBlock's plaintexts (coeff-staged ~0.5 MB each; gate-failed ones full-size) to one artifact file.
    """
def save_keys(inf: Inference, dir: str) -> None:
    """
    Write the SERVER bundle (context/public/eval keys — no secret material).
    """
def save_secret_key(inf: Inference, path: str) -> None:
    """
    Write the CLIENT's secret key. This file never leaves the client.
    """
def serialize_ct(inf: Inference, x: PackedCtx) -> bytes:
    """
    Ciphertext -> bytes (OpenFHE binary), syncing a device-computed value to the host first. The packing rides out-of-band: deserialize_ct stamps the session's.
    """
def set_ln_affine(inf: Inference, tag: str, weight: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], bias: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], level: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
    """
    Install LayerNorm gamma/beta as <tag>.weight / <tag>.bias for ln_affine(tag) (level <= 0: encode at the bootstrap output level).
    """
def set_strict_layout(on: bool) -> None:
    """
    Upgrade slot-layout mismatches from a once-per-weight warning to a throw ([layout_error]).
    """
def softmax_v(inf: Inference, softmax_scores: collections.abc.Sequence[PackedCtx]) -> PackedCtx:
    """
    Multiply the softmax probabilities by this block's V cache; one ciphertext out.
    """
def token_embedding(store: WeightStore, token: typing.SupportsInt | typing.SupportsIndex, position: typing.SupportsInt | typing.SupportsIndex) -> list[float]:
    """
    wte[token] + wpe[position] read from the plaintext store (client-side embedding).
    """
def transformer_block(inf: Inference, x: PackedCtx) -> PackedCtx:
    """
    One GPT-2 block on `x`: ln_1, MHA, residual, ln_2, MLP, residual; writes the block's graph.json under FHE_GRAPH_DIR.
    """
def unpack_tokens(inf: Inference, ct: PackedCtx, T: typing.SupportsInt | typing.SupportsIndex) -> list[list[float]]:
    """
    Decrypt `ct` and decode its T packed tokens to [T][d_real].
    """
__version__: str = '0.1.0'
chain: str = 'n32'
native_int_bits: int = 32
class _debug:
    """Harness and research taps: for probes, tests and diagnostics."""
    decrypt_slots = decrypt_slots
    @staticmethod
    def hard_exit(code: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        Flush and std::_Exit(code), skipping Python and C++ teardown. A harness escape hatch for a crash during cross-library static destruction, not an API for applications.
        """
    @staticmethod
    def install_fatal_exit_handler() -> None:
        """
        Install the std::terminate handler that _Exit(134)s the process after printing the escaped exception (done at import unless PERSEUS_FATAL_EXIT=0).
        """
    @staticmethod
    def throw_test(message: str = '') -> None:
        """
        Raise a runtime error carrying `message` (default: an [fhe_error] marker) so the Python-side exception translation can be tested without a context.
        """
    @staticmethod
    def throw_typed(kind: str, message: str = 'typed test throw') -> None:
        """
        Raise the typed C++ error `kind` ('plan' | 'mask' | 'layout' | 'openfhe' | other = FHEError) so the translation is testable without a context.
        """
hard_exit = _debug.hard_exit
install_fatal_exit_handler = _debug.install_fatal_exit_handler
throw_test = _debug.throw_test
