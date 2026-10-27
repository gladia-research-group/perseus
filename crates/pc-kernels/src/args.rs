//! `#[repr(C)]` mirrors of `kernels/args.cuh` (field for field; sizes asserted at compile time on both sides).
//! Data pointers are word-generic (`*const c_void`): the instantiation reads them as its word W. Strides and offsets
//! are in W words.
use std::ffi::c_void;

/// `cudaStream_t`
pub type CudaStream = *mut c_void;

pub const ABI: u32 = 11;
pub const LOG_N: u32 = 16;
pub const N: usize = 1 << LOG_N;

/// Per-prime constants, one table per word size (layout table in args.cuh). The Rust definition lives in
/// pc-core (`DevPrimeConst`, size/offset-asserted there) so host setup code builds device tables without CUDA.
pub use pc_core::modmul::DevPrimeConst as PrimeConst;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RowSet {
    pub nq: i32,
    pub np: i32,
    pub qmax: i32,
    pub pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PolyRef {
    pub base: *mut c_void,
    pub ct_stride: u64,
    pub part_stride: u64,
}
impl Default for PolyRef {
    fn default() -> Self {
        PolyRef { base: std::ptr::null_mut(), ct_stride: 0, part_stride: 0 }
    }
}

macro_rules! zeroed_default {
    ($($t:ty),*) => {$(
        impl Default for $t {
            fn default() -> Self {
                // SAFETY: plain-old-data of integers, floats and raw pointers: all-zero is a valid value (null pointers)
                unsafe { std::mem::zeroed() }
            }
        }
    )*};
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NttArgs {
    pub base: *mut c_void,
    pub roff: *const u32,
    pub rprime: *const i32,
    pub fw: *const c_void,
    pub fws: *const c_void,
    pub iw: *const c_void,
    pub iws: *const c_void,
    pub ninv: *const c_void,
    pub ninvs: *const c_void,
    pub pr: *const PrimeConst,
    pub nrows: i32,
    pub chunk: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BconvJob {
    pub src_off: u32,
    pub dst_off: u32,
    pub nsrc: u32,
    pub ntgt: u32,
    pub pool_off: u32,
    pub exact: u32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BconvArgs {
    pub src: *const c_void,
    pub dst: *mut c_void,
    pub jobs: *const BconvJob,
    pub pool: *const c_void,
    pub pr: *const PrimeConst,
    pub njobs: i32,
    pub max_tgt: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EwArgs {
    pub out: PolyRef,
    pub a: PolyRef,
    pub b: PolyRef,
    pub rs: RowSet,
    pub sc: *const c_void,
    pub scs: *const c_void,
    pub pr: *const PrimeConst,
    pub k: i32,
    pub nparts: i32,
    pub sub: i32,
    pub zero_p: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PtDotArgs {
    pub rot: PolyRef,
    pub res: PolyRef,
    pub rot_stride: u64,
    pub res_stride: u64,
    pub pts: *const *const c_void,
    pub pr: *const PrimeConst,
    pub rs: RowSet,
    pub npts: i32,
    pub bstep: i32,
    pub gstep: i32,
    pub k: i32,
    pub ptw: i32,
    pub pt_prow: i32,
    pub enc: u32,
    pub pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KsArgs {
    pub c1: *const c_void,
    pub modup: *const c_void,
    pub c0: *const c_void,
    pub out0: *mut c_void,
    pub out1: *mut c_void,
    pub kb: *const *const c_void,
    pub ka: *const *const c_void,
    pub autok: *const u32,
    pub rows: *const u32,
    pub digit_of: *const i32,
    pub pmod: *const c_void,
    pub pmods: *const c_void,
    pub pr: *const PrimeConst,
    pub ct_stride: u64,
    pub mu_ct_stride: u64,
    pub mu_digit_stride: u64,
    pub out_ct_stride: u64,
    pub out_rot_stride: u64,
    pub key_digit_stride: u64,
    pub k: i32,
    pub r: i32,
    pub ndig: i32,
    pub c0_mode: i32,
    pub nq: i32,
    pub nrows: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EpiArgs {
    pub dst: *mut c_void,
    pub conv: *const c_void,
    pub binv: *const c_void,
    pub binvs: *const c_void,
    pub k: *const c_void,
    pub part_stride: u64,
    pub dbl: i32,
    pub pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KsRegenArgs {
    pub ks: KsArgs,
    pub seed: *const u32,
    pub epi: EpiArgs,
    pub use_epi: i32,
    pub pad: i32,
    /// ABI 7: key 0's `b` read in place when `key.kb` is non-null (else `ks.kb[0]`, dense prime-id rows).
    pub key: KeyTab,
    /// ABI 7: epilogue `+ add` on both parts (laid out as `epi.dst`); null = none.
    pub add: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KskExpandArgs {
    pub out: *mut c_void,
    pub seed: *const u32,
    pub prime_ids: *const i32,
    pub pr: *const PrimeConst,
    pub nmod: i32,
    pub dnum: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CtsArgs {
    pub x: *const c_void,
    pub modup: *const c_void,
    pub res: *mut c_void,
    pub kb: *const *const c_void,
    pub ka: *const *const c_void,
    pub kinv: *const u32,
    pub pts: *const *const c_void,
    pub rows: *const u32,
    pub digit_of: *const i32,
    pub pmod: *const c_void,
    pub pmods: *const c_void,
    pub pr: *const PrimeConst,
    pub ct_stride: u64,
    pub part_stride: u64,
    pub mu_ct_stride: u64,
    pub mu_digit_stride: u64,
    pub key_digit_stride: u64,
    pub k: i32,
    pub b: i32,
    pub g: i32,
    pub npts: i32,
    pub ndig: i32,
    pub nq: i32,
    pub nlive: i32,
    pub ptw: i32,
    pub pt_prow: i32,
    pub enc: u32,
    pub ngrp: i32,
    pub nrows: i32,
    pub kmax: i32,
    /// > 0: giant pair l of a launch at res + l * pair_cts cts (one K=1 launch of up to 4 giant steps); 0: G <= 2
    pub pair_cts: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TensorArgs {
    pub out: PolyRef,
    pub d2: PolyRef,
    pub a0: PolyRef,
    pub b0: PolyRef,
    pub a1: PolyRef,
    pub b1: PolyRef,
    pub coef: *mut c_void,
    pub coef_ct_stride: u64,
    pub pr: *const PrimeConst,
    pub npairs: i32,
    pub acc: i32,
    pub n: i32,
    pub k: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MdEpiArgs {
    pub dst: PolyRef,
    pub x: PolyRef,
    pub t: PolyRef,
    pub binv: *const c_void,
    pub binvs: *const c_void,
    pub k: *const c_void,
    pub pr: *const PrimeConst,
    pub dbl: i32,
    pub n: i32,
    pub kk: i32,
    pub pad: i32,
}

/// dst = a +- (x - t) binv over Q rows (ABI 7; `m.k` / `m.dbl` unused).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MdEpiToArgs {
    pub m: MdEpiArgs,
    pub a: PolyRef,
    pub sub: i32,
    pub pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WsumArgs {
    pub out: PolyRef,
    pub src: [PolyRef; 8],
    pub w: *const c_void,
    pub ws: *const c_void,
    pub pr: *const PrimeConst,
    pub nsrc: i32,
    pub rowstride: i32,
    pub n: i32,
    pub k: i32,
}

/// wsum then part 0 += k\[row\] (ABI 7; = wsum + add_const).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WsumKArgs {
    pub w: WsumArgs,
    pub k: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AddConstArgs {
    pub x: PolyRef,
    pub k: *const c_void,
    pub pr: *const PrimeConst,
    pub n: i32,
    pub kk: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MulTableArgs {
    pub out: PolyRef,
    pub inp: PolyRef,
    pub t: *const c_void,
    pub pr: *const PrimeConst,
    pub n: i32,
    pub k: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RescaleArgs {
    pub x: PolyRef,
    pub tmp: PolyRef,
    pub a: *const c_void,
    pub b: *const c_void,
    pub a_s: *const c_void,
    pub b_s: *const c_void,
    pub qam: *const c_void,
    pub qbm: *const c_void,
    pub pr: *const PrimeConst,
    pub qinv_ab: u64,
    pub c_ab: u64,
    pub ca: u64,
    pub cb: u64,
    pub top: i32,
    pub nrows: i32,
    pub k: i32,
    pub ndrop: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RescaleCombineArgs {
    pub x: PolyRef,
    pub tmp: PolyRef,
    pub alpha: *const c_void,
    pub alphas: *const c_void,
    pub pr: *const PrimeConst,
    pub n: i32,
    pub k: i32,
}

/// out = alpha x + tmp, x read only (ABI 7; EvalMod's out-of-place rescale).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RescaleCombineToArgs {
    pub a: RescaleCombineArgs,
    pub out: PolyRef,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AutomorphArgs {
    pub out: PolyRef,
    pub inp: PolyRef,
    pub add: PolyRef,
    pub rs: RowSet,
    pub pr: *const PrimeConst,
    pub k: u32,
    pub kk: i32,
    pub nparts: i32,
    pub has_add: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EncodeArgs {
    pub v: *const c_void,
    pub a: *mut c_void,
    pub b: *mut c_void,
    pub c: *mut c_void,
    pub cmax: *mut u64,
    pub out: *mut c_void,
    pub deg_factor: *const c_void,
    pub pr: *const PrimeConst,
    pub scale: f64,
    pub nl: i32,
    pub pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EncryptArgs {
    pub c0: *mut c_void,
    pub c1: *mut c_void,
    pub s: *const c_void,
    pub e: *const c_void,
    pub m: *const c_void,
    pub seed: *const u32,
    pub pr: *const PrimeConst,
    pub nonce: u32,
    pub nl: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KskGenArgs {
    pub b: *mut c_void,
    pub a: *mut c_void,
    pub e: *const c_void,
    pub s_new: *const c_void,
    pub s_old: *const c_void,
    pub seed: *const u32,
    pub ids: *const i32,
    pub digit_of: *const i32,
    pub pmod: *const c_void,
    pub pmods: *const c_void,
    pub pr: *const PrimeConst,
    pub row_words: u64,
    pub nrows: i32,
    pub ndig: i32,
    pub nq: i32,
    pub bits: i32,
    pub sq: i32,
    pub pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KskUnpackArgs {
    pub out: *mut c_void,
    pub inp: *const u32,
    pub row_words: u64,
    pub nrows: i32,
    pub bits: i32,
}

pub const SI_MAXK: usize = 4;
pub const SI_MAXW: usize = 8;
pub const SI_MAXR: usize = 8;
pub const SI_SHIFT: u32 = 192;
pub const SI_MAXOUT: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ModRaiseArgs {
    pub x: *mut c_void,
    pub pr: *const PrimeConst,
    pub ct_stride: u64,
    pub part_stride: u64,
    pub qhatinv: [u64; 4],
    pub qhat: [[u64; 64]; 4],
    pub k: i32,
    pub d: i32,
    pub r0: i32,
    pub r1: i32,
    pub nparts: i32,
    pub pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SmallIntTab {
    pub k: i32,
    pub nout: i32,
    pub nw: i32,
    pub nr: i32,
    pub divide: i32,
    pub pad: i32,
    pub q: [u32; SI_MAXK],
    pub qinv: [[u32; SI_MAXK]; SI_MAXK],
    pub qk: [u32; SI_MAXW],
    pub qkh: [u32; SI_MAXW],
    pub r: [u32; SI_MAXR],
    pub orow: [i32; SI_MAXOUT],
    pub pow32: [[u32; SI_MAXW + SI_MAXR]; SI_MAXOUT],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SmallIntArgs {
    pub src: *const c_void,
    pub out: *mut c_void,
    pub tab: *const SmallIntTab,
    pub pr: *const PrimeConst,
    pub src_ct_stride: u64,
    pub src_part_stride: u64,
    pub ct_stride: u64,
    pub part_stride: u64,
    pub k: i32,
    pub nparts: i32,
}

/// The exact post-raise scaling from one source prime (args.cuh SmallInt1Args, ABI 8).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SmallInt1Args {
    pub x: *mut c_void,
    pub pr: *const PrimeConst,
    pub ct_stride: u64,
    pub part_stride: u64,
    pub mc: u64,
    pub sh: i32,
    pub nrows: i32,
    pub k: i32,
    pub nparts: i32,
}

/// Terms per `coeff_sum` launch and limbs it covers (args.cuh PC_COEFF_MAXT / PC_COEFF_MAXL).
pub const COEFF_MAXT: usize = 32;
pub const COEFF_MAXL: usize = 64;

/// Streamed-weight inner sum from float32 coefficient rows (args.cuh CoeffArgs, ABI 9).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CoeffArgs {
    pub scratch: *mut c_void,
    pub out: *mut c_void,
    pub fw: *const c_void,
    pub fws: *const c_void,
    pub pr: *const PrimeConst,
    pub overflow: *mut u64,
    pub part_stride: u64,
    pub scale: f64,
    pub n: i32,
    pub l: i32,
    pub accumulate: i32,
    pub pad: i32,
    pub rows: [*const f32; COEFF_MAXT],
    pub ct: [*const c_void; COEFF_MAXT],
}

/// SPRU extraction (args.cuh SpruExtractArgs).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SpruExtractArgs {
    pub c0: *const c_void,
    pub c1: *const c_void,
    pub out: *mut c_void,
    pub p0: u64,
    pub p1: u64,
    pub inv: u64,
    pub delta: f64,
    pub ct_stride: u64,
    pub h: i32,
    pub n: i32,
    pub b: i32,
    pub k: i32,
}

/// Batched Stockham FFT pass (args.cuh SpruFftArgs).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SpruFftArgs {
    pub inp: *const c_void,
    pub out: *mut c_void,
    pub tw: *const c_void,
    pub ns: i32,
    pub batch: i32,
}

/// FFT output -> rounded coefficient limbs (args.cuh SpruScatterArgs).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SpruScatterArgs {
    pub y: *const c_void,
    pub out: *mut c_void,
    pub zt: *const c_void,
    pub pr: *const PrimeConst,
    pub scale: f64,
    pub nl: i32,
    pub batch: i32,
    pub out_stride: u64,
}

/// Seeded key set read in-kernel (ABI 4): `a` from each key's KSKA seed, `b` from its storage (dense W rows
/// or packed u32 rows) in slot order (`KskShape::slot_of`). See args.cuh.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KeyTab {
    pub kb: *const *const c_void,
    pub seed: *const *const u32,
    pub row_words: u64,
    pub bits: i32,
    pub q_top: i32,
    pub srows: i32,
    /// digits below regenerate `a`, the rest read the resident dense `a` (`cts.ka` / `ks.ka`)
    pub nregen: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CtsRegenArgs {
    pub cts: CtsArgs,
    pub key: KeyTab,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KsKeyArgs {
    pub ks: KsArgs,
    pub key: KeyTab,
}

/// PCMM op codes (args.cuh PcmmOp, ABI 5).
pub mod pcmm_op {
    pub const GATHER: i32 = 0;
    pub const SCATTER: i32 = 1;
    pub const IN_RESCALE: i32 = 2;
    pub const LIFT: i32 = 3;
    pub const KS_DIGITS: i32 = 4;
    pub const KS_MAC: i32 = 5;
    pub const MOD_DOWN: i32 = 6;
    pub const ADD: i32 = 7;
    pub const NTT: i32 = 8;
    pub const INTT: i32 = 9;
    pub const MLWE_A: i32 = 10;
    pub const MLWE_B: i32 = 11;
    pub const DIGITS: i32 = 12;
    pub const RECOMBINE: i32 = 13;
    pub const PACK_C: i32 = 14;
    pub const ADD_BPK: i32 = 15;
    pub const COMPOSE: i32 = 16;
}

/// One PCMM kernel launch (args.cuh PcmmArgs: per-op field meanings there).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PcmmArgs {
    pub op: i32,
    pub n: i32,
    pub count: i32,
    pub i0: i32,
    pub i1: i32,
    pub i2: i32,
    pub i3: i32,
    pub i4: i32,
    pub x: *mut c_void,
    pub y: *const c_void,
    pub z: *const c_void,
    pub tw: *const c_void,
    pub ct: PolyRef,
    pub p: [u32; 4],
    pub mu: [u64; 4],
    pub c: [u32; 8],
}

/// CCMM op codes (args.cuh CcmmOp, ABI 6).
pub mod ccmm_op {
    pub const GATHER: i32 = 0;
    pub const SCATTER: i32 = 1;
    pub const NTT: i32 = 2;
    pub const INTT: i32 = 3;
    pub const KS_DIGITS: i32 = 4;
    pub const KS_MAC: i32 = 5;
    pub const MOD_DOWN: i32 = 6;
    pub const RING_IN: i32 = 7;
    pub const COMPOSE_A: i32 = 8;
    pub const RING_OUT: i32 = 9;
    pub const MONO: i32 = 10;
    pub const BITREV: i32 = 11;
    pub const TWEAK: i32 = 12;
    pub const SCRAMBLE: i32 = 13;
    pub const KS_APPLY: i32 = 14;
    pub const ADD: i32 = 15;
    pub const GATHER_U: i32 = 16;
    pub const DIGITS: i32 = 17;
    pub const RECOMBINE: i32 = 18;
    pub const SCATTER_ROWS: i32 = 19;
}

/// One CCMM kernel launch (args.cuh CcmmArgs: per-op field meanings there).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CcmmArgs {
    pub op: i32,
    pub n: i32,
    pub count: i32,
    pub i0: i32,
    pub i1: i32,
    pub i2: i32,
    pub i3: i32,
    pub i4: i32,
    pub x: *mut c_void,
    pub y: *const c_void,
    pub z: *const c_void,
    pub tw: *const c_void,
    pub ct: PolyRef,
    pub p: [u32; 8],
    pub mu: [u64; 8],
    pub c: [u32; 16],
    pub s: [i64; 4],
}

zeroed_default!(
    SpruFftArgs, SpruScatterArgs, SpruExtractArgs, NttArgs, BconvArgs, EwArgs, PtDotArgs, KsArgs, EpiArgs, KsRegenArgs, KskExpandArgs, CtsArgs, TensorArgs, MdEpiArgs,
    WsumArgs, AddConstArgs, MulTableArgs, RescaleArgs, RescaleCombineArgs, AutomorphArgs, EncodeArgs, EncryptArgs, KskGenArgs, KskUnpackArgs, ModRaiseArgs, SmallIntTab, SmallIntArgs,
    KeyTab, CtsRegenArgs, KsKeyArgs, PcmmArgs, CcmmArgs, WsumKArgs, RescaleCombineToArgs, MdEpiToArgs, SmallInt1Args, CoeffArgs, TfheArgs
);

/// TFHE op codes (args.cuh PC_TFHE_*, ABI 11).
pub mod tfhe_op {
    pub const PBS_MS: i32 = 0;
    pub const PBS_INIT: i32 = 1;
    pub const PBS_STEP: i32 = 2;
    pub const PBS_EXTRACT: i32 = 3;
    pub const KS_DIGITS: i32 = 4;
    pub const KS_RECOMBINE: i32 = 5;
}

/// One TFHE kernel launch (args.cuh TfheArgs: per-op field meanings there).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TfheArgs {
    pub op: i32,
    pub n: i32,
    pub count: i32,
    pub i0: i32,
    pub i1: i32,
    pub i2: i32,
    pub i3: i32,
    pub i4: i32,
    pub x: *mut c_void,
    pub y: *const c_void,
    pub z: *const c_void,
    pub w: *const c_void,
    pub tw: *const c_void,
    pub tws: *const c_void,
    pub q: u64,
    pub m: u64,
    pub c: [u64; 8],
}

/// A launcher: (args, stream) -> cudaError_t of the launch (0 ok; negative = argument outside the kernel's range).
pub type Launch<A> = unsafe extern "C" fn(*const A, CudaStream) -> i32;

#[repr(C)]
#[derive(Debug)]
pub struct KernelTable {
    pub abi: u32,
    pub word_bytes: u32,
    pub mm: u32,
    pub log_n: u32,
    pub ntt_fwd: Launch<NttArgs>,
    pub ntt_inv: Launch<NttArgs>,
    pub bconv: Launch<BconvArgs>,
    pub ew_add: Launch<EwArgs>,
    pub ew_scale: Launch<EwArgs>,
    pub ew_sub_mul: Launch<EwArgs>,
    pub ew_copy: Launch<EwArgs>,
    pub pt_dot: Launch<PtDotArgs>,
    pub ks_dot: Launch<KsArgs>,
    pub ks_dot_regen: Launch<KsRegenArgs>,
    pub ksk_expand: Launch<KskExpandArgs>,
    pub cts_fused: Launch<CtsArgs>,
    pub tensor: Launch<TensorArgs>,
    pub md_epilogue: Launch<MdEpiArgs>,
    pub wsum: Launch<WsumArgs>,
    pub add_const: Launch<AddConstArgs>,
    pub mul_table: Launch<MulTableArgs>,
    pub rescale_coef: Launch<RescaleArgs>,
    pub rescale_combine: Launch<RescaleCombineArgs>,
    pub automorph: Launch<AutomorphArgs>,
    pub encode: Launch<EncodeArgs>,
    pub encrypt: Launch<EncryptArgs>,
    pub ksk_gen: Launch<KskGenArgs>,
    pub ksk_unpack: Launch<KskUnpackArgs>,
    pub mod_raise: Launch<ModRaiseArgs>,
    pub small_int: Launch<SmallIntArgs>,
    pub spru_extract: Launch<SpruExtractArgs>,
    pub spru_fft_pass: Launch<SpruFftArgs>,
    pub spru_scatter: Launch<SpruScatterArgs>,
    pub cts_fused_regen: Launch<CtsRegenArgs>,
    pub ks_dot_key: Launch<KsKeyArgs>,
    pub pcmm: Launch<PcmmArgs>,
    pub ccmm: Launch<CcmmArgs>,
    pub wsum_k: Launch<WsumKArgs>,
    pub rescale_combine_to: Launch<RescaleCombineToArgs>,
    pub md_epilogue_to: Launch<MdEpiToArgs>,
    pub small_int1: Launch<SmallInt1Args>,
    pub coeff_sum: Launch<CoeffArgs>,
    pub tfhe: Launch<TfheArgs>,
}

// Sizes must equal the static_asserts in args.cuh.
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<PrimeConst>() == 64);
    assert!(size_of::<RowSet>() == 16);
    assert!(size_of::<PolyRef>() == 24);
    assert!(size_of::<NttArgs>() == 88);
    assert!(size_of::<BconvJob>() == 32);
    assert!(size_of::<BconvArgs>() == 48);
    assert!(size_of::<EwArgs>() == 128);
    assert!(size_of::<PtDotArgs>() == 128);
    assert!(size_of::<KsArgs>() == 176);
    assert!(size_of::<EpiArgs>() == 56);
    assert!(size_of::<KsRegenArgs>() == 296);
    assert!(size_of::<KskExpandArgs>() == 40);
    assert!(size_of::<CtsArgs>() == 192);
    assert!(size_of::<TensorArgs>() == 184);
    assert!(size_of::<MdEpiArgs>() == 120);
    assert!(size_of::<WsumArgs>() == 256);
    assert!(size_of::<WsumKArgs>() == 264);
    assert!(size_of::<AddConstArgs>() == 48);
    assert!(size_of::<MulTableArgs>() == 72);
    assert!(size_of::<RescaleArgs>() == 152);
    assert!(size_of::<RescaleCombineArgs>() == 80);
    assert!(size_of::<AutomorphArgs>() == 112);
    assert!(size_of::<EncodeArgs>() == 80);
    assert!(size_of::<EncryptArgs>() == 64);
    assert!(size_of::<KskGenArgs>() == 120);
    assert!(size_of::<KskUnpackArgs>() == 32);
    assert!(size_of::<ModRaiseArgs>() == 2136);
    assert!(size_of::<SmallIntTab>() == 4552);
    assert!(size_of::<SmallIntArgs>() == 72);
    assert!(size_of::<SpruExtractArgs>() == 80);
    assert!(size_of::<SpruFftArgs>() == 32);
    assert!(size_of::<SpruScatterArgs>() == 56);
    assert!(size_of::<KeyTab>() == 40);
    assert!(size_of::<CtsRegenArgs>() == 232);
    assert!(size_of::<KsKeyArgs>() == 216);
    assert!(size_of::<PcmmArgs>() == 168);
    assert!(size_of::<CcmmArgs>() == 280);
    assert!(size_of::<SmallInt1Args>() == 56);
    assert!(size_of::<KernelTable>() == 16 + 39 * 8);
    assert!(size_of::<TfheArgs>() == 32 + 6 * 8 + 16 + 64);
    assert!(size_of::<CoeffArgs>() == 80 + 16 * COEFF_MAXT);
    assert!(size_of::<MdEpiToArgs>() == 152);
    assert!(size_of::<RescaleCombineToArgs>() == 104);
};

#[cfg(test)]
mod tests {
    use super::PrimeConst;
    #[test]
    fn prime_const_fields() {
        let p = 268_042_241u64; // 28-bit NTT prime
        let c = PrimeConst::new(p, 32);
        assert_eq!(c.bits, 28);
        assert_eq!(c.mu_lo, ((1u128 << 64) / p as u128) as u64);
        assert_eq!((p.wrapping_mul(c.mont_pinv.wrapping_neg())) & 0xFFFF_FFFF, 1);
        assert_eq!(c.mont_r2, ((1u128 << 64) % p as u128) as u64);
        let q = (1u64 << 60) - 93; // any odd < 2^62
        let d = PrimeConst::new(q, 64);
        let mu = ((d.mu_hi as u128) << 64) | d.mu_lo as u128;
        assert_eq!(mu, u128::MAX / q as u128);
        assert_eq!(q.wrapping_mul(d.mont_pinv.wrapping_neg()), 1);
    }
}
