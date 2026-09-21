"""
perseus client core: the CKKS client role (keygen, bundle, encode/encrypt, decrypt, ciphertext bytes) on OpenFHE alone — no CUDA driver, no FIDESlib. Keys and ciphertexts interchange with a perseus._core server. List-valued option fields are copied on access: assign a whole list.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['CKKSOptions', 'Context', 'FHEError', 'Inference', 'InferenceMode', 'InferenceOptions', 'LayoutError', 'MaskError', 'ModelSize', 'PackedCtx', 'PackingKind', 'PlanError', 'build_info', 'chain', 'decode_linear_output', 'decode_lm_head_logits', 'decode_token_output', 'decode_tokens_output', 'decrypt_slots', 'deserialize_ct', 'encode_token_input', 'lm_head_tile_width', 'load_secret_key', 'make_context', 'make_gpt2_inference', 'make_inference', 'native_int_bits', 'pack_tokens', 'save_keys', 'save_secret_key', 'serialize_ct', 'throw_test', 'unpack_tokens']
class CKKSOptions:
    """
    CKKS parameters for a context (same fields as perseus._core.CKKSOptions). List fields are copied on read: set them whole.
    """
    ckks_complex_payload: bool
    defer_heavy_setup: bool
    enable_bootstrap: bool
    keys_dir: str
    skip_gpu_load: bool
    @staticmethod
    def from_env() -> CKKSOptions:
        """
        The same 17 environment knobs perseus._core.CKKSOptions.from_env reads (LOGN, CKKS_DEPTH, ..., SPARSE_BTS_SLOTS, LEVEL_BUDGET, CKKS_COMPLEX); silent.
        """
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
    """
    The client's CKKS context: lbcrypto context + key pair (secret key present after keygen or load_secret_key).
    """
    def bootstrap_output_level(self) -> int:
        """
        The parameter formula composite_degree * (btp_depth_overhead + [bts_iterations >= 2]): what a GPU-less perseus._core session reports (its probe needs a bootstrap).
        """
    @property
    def automorphism_key_indexes(self) -> list[int]:
        """
        Sorted automorphism indices OpenFHE's store holds for this key tag (band + bootstrap rotations + conj + ENCAPS pair after keygen).
        """
    @property
    def complex_payload(self) -> bool:
        ...
    @property
    def from_keys(self) -> bool:
        ...
    @property
    def has_secret_key(self) -> bool:
        ...
    @property
    def key_dist(self) -> int:
        ...
    @property
    def key_tag(self) -> str:
        """
        OpenFHE's key tag of this session's key pair (the rotkeys.bin map key).
        """
    @property
    def loaded_rot_steps(self) -> list[int]:
        """
        The sorted-unique rotation band the keys were generated for (or the bundle sidecar's RotationIndexes when opened with keys_dir).
        """
class FHEError(RuntimeError):
    pass
class Inference:
    complex: bool
    mode: InferenceMode
    size: ModelSize
    def __repr__(self) -> str:
        ...
    @property
    def fhe(self) -> Context:
        ...
    @property
    def logN(self) -> int:
        ...
    @property
    def n_tok(self) -> int:
        ...
    @n_tok.setter
    def n_tok(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def packing(self) -> str:
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
class LayoutError(FHEError):
    pass
class MaskError(FHEError):
    pass
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
class PlanError(FHEError):
    pass
def build_info() -> dict:
    """
    Version, chain (n32/n64), native integer width, pybind11 version, backend 'client' (cuda_runtime None) and the compile timestamp of this extension.
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
    Decode the n_tok tokens packed in `ct` (slot[i*t + tok]) to [n_tok][d_real].
    """
def decrypt_slots(inf: Inference, x: PackedCtx) -> list[float]:
    """
    Every slot of a ciphertext (the layout research tool).
    """
def deserialize_ct(inf: Inference, data: bytes) -> PackedCtx:
    """
    bytes -> PackedCtx in this session's context (the session's packing is stamped).
    """
@typing.overload
def encode_token_input(inf: Inference, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> PackedCtx:
    """
    numpy fast path for encode_token_input.
    """
@typing.overload
def encode_token_input(inf: Inference, x: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> PackedCtx:
    """
    One token's real features (<= size.dim; shorter is zero-padded) -> fresh ciphertext at the formula bootstrap output level. Longer raises ValueError.
    """
def lm_head_tile_width(inf: Inference, vocab: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    The lm_head tile width: hidDim when vocab <= hidDim, else slots.
    """
def load_secret_key(inf: Inference, path: str) -> None:
    """
    Load a secret key written by save_secret_key (either extension) into a session opened with keys_dir, so it can decrypt.
    """
def make_context(options: CKKSOptions = ...) -> Context:
    """
    Create a bare CKKS context (keygen, or a bundle's context when keys_dir is set).
    """
def make_gpt2_inference(options: ... = ...) -> Inference:
    """
    make_inference plus the GPT-2 rotation keys for `options`' packing (and aux packings).
    """
def make_inference(options: ... = ...) -> Inference:
    """
    Build a generic client session from `options`: CKKS context + keys, model sizes and packing.
    """
@typing.overload
def pack_tokens(inf: Inference, embeddings: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], target_level: typing.SupportsInt | typing.SupportsIndex = 0) -> PackedCtx:
    """
    numpy fast path: a [T][d] float array.
    """
@typing.overload
def pack_tokens(inf: Inference, embeddings: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]], target_level: typing.SupportsInt | typing.SupportsIndex = 0) -> PackedCtx:
    """
    Encode T token embeddings (each <= size.dim, zero-padded to hidDim) into one ciphertext at `target_level` (same slot layout as perseus._core.pack_tokens).
    """
def save_keys(inf: Inference, dir: str) -> None:
    """
    Write the SERVER bundle: context.bin, context.bin.dev, public.key, multkeys.bin, rotkeys.bin — the layout perseus._core's keys_dir loads; no secret material.
    """
def save_secret_key(inf: Inference, path: str) -> None:
    """
    Write the CLIENT's secret key. This file never leaves the client.
    """
def serialize_ct(inf: Inference, x: PackedCtx) -> bytes:
    """
    Ciphertext -> bytes (OpenFHE binary; the same bytes perseus._core.serialize_ct writes).
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
    def automorphism_indexes_in_file(path: str) -> dict[str, list[int]]:
        """
        {key tag: sorted automorphism indices} of a rotkeys.bin (reads the whole file).
        """
    @staticmethod
    def bootstrap_indexes(options: perseus._client.InferenceOptions, family: str = 'gpt2', slots: typing.SupportsInt | typing.SupportsIndex = 0) -> list[int]:
        """
        FIDESlib's GetBootstrapIndexes for the `slots` precomp (0 = the dense one).
        """
    @staticmethod
    def bundle_meta(options: perseus._client.InferenceOptions, family: str = 'gpt2') -> tuple[bytes, str]:
        """
        (context.bin bytes, context.bin.dev text) a bundle for `options` / `family` carries: context + Enable set + bootstrap setups + band, no keygen.
        """
    @staticmethod
    def dev_sidecar(inf: perseus._client.Inference) -> str:
        """
        The context.bin.dev text save_keys writes for this session.
        """
    @staticmethod
    def expected_automorphism_indexes(options: perseus._client.InferenceOptions, family: str = 'gpt2') -> list[int]:
        """
        Sorted automorphism indices the keygen for `options` / `family` writes into rotkeys.bin (band + every bootstrap precomp's rotations + M-1 conj + M-2/M-4 ENCAPS pair).
        """
    @staticmethod
    def family_rot_band(options: perseus._client.InferenceOptions, family: str = 'gpt2') -> list[int]:
        """
        The sorted-unique band make_<family>_inference generates keys for.
        """
    @staticmethod
    def formula_level(options: perseus._client.CKKSOptions) -> int:
        """
        Context.bootstrap_output_level() for `options` without building a context.
        """
    @staticmethod
    def rot_band(kind: str, slots: typing.SupportsInt | typing.SupportsIndex, hidDim: typing.SupportsInt | typing.SupportsIndex, ffDim: typing.SupportsInt | typing.SupportsIndex, numHeads: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
        """
        The GPT-2 rotation band of one packing ('cachemir' | 'diagonal' | 'cachemir_filling').
        """
    @staticmethod
    def throw_test(message: str = '') -> None:
        """
        Raise a runtime error carrying `message` (default: an [fhe_error] marker).
        """
    @staticmethod
    def throw_typed(kind: str, message: str = 'typed test throw') -> None:
        """
        Raise the typed C++ error `kind` ('plan' | 'mask' | 'layout' | 'openfhe' | other = FHEError).
        """
throw_test = _debug.throw_test
