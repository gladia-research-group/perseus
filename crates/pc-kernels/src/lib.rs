//! pc-kernels: the CUDA kernels of perseus as C++ templates over (Word, ModMul), one instantiation per
//! translation unit, each exporting a `KernelTable` of `extern "C"` launchers (see `kernels/args.cuh`).
//! Rust side: `args` mirrors the ABI structs with size asserts, `table` resolves an instantiation's table.
pub mod args;
pub mod ring;
pub mod table;

pub use args::*;
pub use ring::{ring_table, RingTable, RingWord};
pub use table::{table, Instance, TableError};

/// The value of `#define <name> <digits>` in `src` (a C header), evaluated at compile time; 0 when absent.
pub(crate) const fn cuh_define(src: &str, name: &str) -> u32 {
    let (s, n) = (src.as_bytes(), name.as_bytes());
    let key = b"#define ";
    let mut i = 0;
    while i + key.len() + n.len() < s.len() {
        let mut j = 0;
        while j < key.len() && s[i + j] == key[j] {
            j += 1;
        }
        let mut k = 0;
        while j == key.len() && k < n.len() && s[i + j + k] == n[k] {
            k += 1;
        }
        if k == n.len() && s[i + j + k] == b' ' {
            let (mut p, mut v) = (i + j + k + 1, 0u32);
            while p < s.len() && s[p].is_ascii_digit() {
                v = v * 10 + (s[p] - b'0') as u32;
                p += 1;
            }
            return v;
        }
        i += 1;
    }
    0
}

/// The kernels' header, for the ABI asserts of `args` and `ring`.
pub(crate) const ARGS_CUH: &str = include_str!("../kernels/args.cuh");
