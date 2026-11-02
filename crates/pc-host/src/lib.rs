//! pc-host: the kernel tables of the host device. Every `KernelTable` entry is a Rust launcher over the same
//! `args.cuh` struct, run on a pc-hostrt stream (inline, or recorded under a capture) and on the pc-hostrt pool, with
//! the device's results word for word: the kernels' contract is canonical residues under every ModMul strategy, so
//! one host table per word size stands for every (word, ModMul) instantiation of that word.
mod arith;
mod bconv;
mod elem;
mod encode;
mod expand;
mod keys;
mod ks;
mod ntt;
mod raw;
mod rescale;

use pc_kernels::args::*;
use pc_kernels::{Instance, TableError};

pub use arith::Hw;

type S = CudaStream;

/// Run `f` on a copy of the arguments when the stream passes this point (0, or the argument check's code).
fn launch<A: Copy + 'static>(a: *const A, s: S, check: impl FnOnce(&A) -> i32, f: fn(&A)) -> i32 {
    // SAFETY: the launcher ABI hands a valid argument struct; launches copy it (a CUDA launch takes it by value)
    let a = unsafe { *a };
    let rc = check(&a);
    if rc != 0 {
        return rc;
    }
    if std::env::var_os("PC_HOST_TRACE").is_some() {
        eprintln!("[pc-host] {}", std::any::type_name::<A>().rsplit("::").next().unwrap_or("?"));
    }
    // SAFETY: `s` is a host stream (the host device only resolves host tables); the pointers in `a` follow the
    // CUDA launch's lifetime rules, which the callers keep
    unsafe { pc_hostrt::submit(s, move || f(&a)) };
    0
}

/// A table entry the host does not implement: refuses the launch, naming the kernel.
macro_rules! missing {
    ($($name:ident : $args:ty),* $(,)?) => {$(
        extern "C" fn $name(_a: *const $args, _s: S) -> i32 {
            eprintln!("[pc-host] kernel `{}` has no host implementation", stringify!($name));
            -100
        }
    )*};
}
missing!(
    cts_fused: CtsArgs,
    mod_raise: ModRaiseArgs,
    small_int: SmallIntArgs,
    spru_extract: SpruExtractArgs,
    spru_fft_pass: SpruFftArgs,
    spru_scatter: SpruScatterArgs,
    cts_fused_regen: CtsRegenArgs,
    pcmm: PcmmArgs,
    ccmm: CcmmArgs,
    small_int1: SmallInt1Args,
    coeff_sum: CoeffArgs,
    tfhe: TfheArgs,
);

const fn build<W: Hw>(inst: Instance) -> KernelTable {
    KernelTable {
        abi: ABI,
        word_bytes: std::mem::size_of::<W>() as u32,
        mm: match inst {
            Instance::U32Barrett64 | Instance::U64Barrett64 => 0,
            Instance::U32Barrett32 | Instance::U16Barrett32 => 1,
            Instance::U32Mont => 2,
        },
        log_n: LOG_N,
        ntt_fwd: ntt::fwd::<W>,
        ntt_inv: ntt::inv::<W>,
        bconv: bconv::bconv::<W>,
        ew_add: elem::ew_add::<W>,
        ew_scale: elem::ew_scale::<W>,
        ew_sub_mul: elem::ew_sub_mul::<W>,
        ew_copy: elem::ew_copy::<W>,
        pt_dot: elem::pt_dot::<W>,
        ks_dot: ks::ks_dot::<W>,
        ks_dot_regen: ks::ks_dot_regen::<W>,
        ksk_expand: ks::ksk_expand::<W>,
        cts_fused,
        tensor: elem::tensor::<W>,
        md_epilogue: elem::md_epilogue::<W>,
        wsum: elem::wsum::<W>,
        add_const: elem::add_const::<W>,
        mul_table: elem::mul_table::<W>,
        rescale_coef: rescale::rescale_coef::<W>,
        rescale_combine: elem::rescale_combine::<W>,
        automorph: elem::automorph::<W>,
        encode: encode::encode::<W>,
        encrypt: keys::encrypt::<W>,
        ksk_gen: keys::ksk_gen::<W>,
        ksk_unpack: keys::ksk_unpack::<W>,
        mod_raise,
        small_int,
        spru_extract,
        spru_fft_pass,
        spru_scatter,
        cts_fused_regen,
        ks_dot_key: ks::ks_dot_key::<W>,
        pcmm,
        ccmm,
        wsum_k: elem::wsum_k::<W>,
        rescale_combine_to: elem::rescale_combine_to::<W>,
        md_epilogue_to: elem::md_epilogue_to::<W>,
        small_int1,
        coeff_sum,
        mul_add: elem::mul_add::<W>,
        rescale_fwd: rescale::rescale_fwd::<W>,
        tfhe,
    }
}

static U32_B64: KernelTable = build::<u32>(Instance::U32Barrett64);
static U32_B32: KernelTable = build::<u32>(Instance::U32Barrett32);
static U32_MONT: KernelTable = build::<u32>(Instance::U32Mont);
static U64_B64: KernelTable = build::<u64>(Instance::U64Barrett64);
static U16_B32: KernelTable = build::<u16>(Instance::U16Barrett32);

/// The host kernel table standing for `inst` (every instantiation exists on the host).
pub fn table(inst: Instance) -> Result<&'static KernelTable, TableError> {
    Ok(match inst {
        Instance::U32Barrett64 => &U32_B64,
        Instance::U32Barrett32 => &U32_B32,
        Instance::U32Mont => &U32_MONT,
        Instance::U64Barrett64 => &U64_B64,
        Instance::U16Barrett32 => &U16_B32,
    })
}
