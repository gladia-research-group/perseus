//! pc-host: the kernel tables of the host device. Every `KernelTable` entry is a Rust launcher over the same
//! `args.cuh` struct, run on a pc-hostrt stream (inline, or recorded under a capture) and on the pc-hostrt pool, with
//! the device's results word for word: the kernels' contract is canonical residues under every ModMul strategy, so
//! one host table per word size stands for every (word, ModMul) instantiation of that word.
mod arith;
mod bconv;
mod ccmm;
mod chacha;
mod coeffrows;
mod cts;
mod elem;
mod encode;
mod expand;
mod keys;
mod ks;
mod mm;
mod ntt;
mod ntt_avx;
mod ntt_g;
mod pbs_v;
mod pcmm;
mod raise;
mod raw;
mod rescale;
mod simd;
mod spru;
mod tfhe;

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
    unsafe {
        if prof_on() {
            pc_hostrt::submit(s, move || {
                let t = std::time::Instant::now();
                f(&a);
                prof_add(std::any::type_name::<A>(), t.elapsed().as_secs_f64());
            })
        } else {
            pc_hostrt::submit(s, move || f(&a))
        }
    };
    0
}

fn prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PC_HOST_PROF").is_some())
}

static PROF: std::sync::Mutex<Vec<(&'static str, f64, u64)>> = std::sync::Mutex::new(Vec::new());

fn prof_add(name: &'static str, secs: f64) {
    let mut p = PROF.lock().unwrap();
    match p.iter_mut().find(|e| e.0 == name) {
        Some(e) => {
            e.1 += secs;
            e.2 += 1;
        }
        None => p.push((name, secs, 1)),
    }
}

/// Seconds and launches per kernel argument type since the process started (PC_HOST_PROF=1), largest first.
pub fn profile() -> Vec<(String, f64, u64)> {
    let mut v: Vec<_> = PROF.lock().unwrap().iter().map(|(n, s, c)| (n.rsplit("::").next().unwrap_or(n).to_string(), *s, *c)).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v
}

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
        cts_fused: cts::cts_fused::<W>,
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
        mod_raise: raise::mod_raise::<W>,
        small_int: raise::small_int::<W>,
        spru_extract: spru::spru_extract::<W>,
        spru_fft_pass: spru::spru_fft_pass,
        spru_scatter: spru::spru_scatter::<W>,
        cts_fused_regen: cts::cts_fused_regen::<W>,
        ks_dot_key: ks::ks_dot_key::<W>,
        pcmm: pcmm::pcmm::<W>,
        ccmm: ccmm::ccmm::<W>,
        wsum_k: elem::wsum_k::<W>,
        rescale_combine_to: elem::rescale_combine_to::<W>,
        md_epilogue_to: elem::md_epilogue_to::<W>,
        small_int1: raise::small_int1::<W>,
        coeff_sum: coeffrows::coeff_sum::<W>,
        mul_add: elem::mul_add::<W>,
        rescale_fwd: rescale::rescale_fwd::<W>,
        tfhe: tfhe::tfhe,
        tensor_sum: elem::tensor_sum::<W>,
        tensor_multi: elem::tensor_multi::<W>,
        tensor_seeded: elem::tensor_seeded::<W>,
        ew_many: elem::ew_many::<W>,
        submul_many: elem::submul_many::<W>,
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

