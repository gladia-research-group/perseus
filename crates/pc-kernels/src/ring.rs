//! Ring-size tables: the NTT and row kernels compiled per (word, log n)
//! (Barrett64; Cargo feature `ring`, u64 additionally with `w64`). Mirrors of the `args.cuh` ring structs.
use crate::args::{BconvArgs, EwArgs, Launch, NttArgs, PrimeConst};
use crate::table::TableError;
use std::ffi::c_void;

pub const RING_ABI: u32 = 7;
/// Ring sizes with a table when the feature is on.
pub const RING_LOG_N: std::ops::RangeInclusive<u32> = 10..=17;

#[repr(C)]
#[derive(Debug)]
pub struct RingTable {
    pub abi: u32,
    pub word_bytes: u32,
    pub mm: u32,
    pub log_n: u32,
    pub ntt_fwd: Launch<NttArgs>,
    pub ntt_inv: Launch<NttArgs>,
    pub ntt_fwd_g: Launch<NttArgs>,
    pub ntt_inv_g: Launch<NttArgs>,
    pub bconv: Launch<BconvArgs>,
    pub ew_add: Launch<EwArgs>,
    pub ew_scale: Launch<EwArgs>,
    pub ew_sub_mul: Launch<EwArgs>,
    pub ew_copy: Launch<EwArgs>,
}

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<RingTable>() == 16 + 9 * 8);
};

/// Word of a ring table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RingWord {
    U32,
    U64,
}
impl RingWord {
    pub fn bytes(self) -> u32 {
        match self {
            RingWord::U32 => 4,
            RingWord::U64 => 8,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            RingWord::U32 => "u32",
            RingWord::U64 => "u64",
        }
    }
}

macro_rules! ring_fns {
    ($feat:meta, $w:ident, $($f:ident = $l:expr),*) => {
        extern "C" {
            $( #[$feat] fn $f() -> *const RingTable; )*
        }
        fn $w(log_n: u32) -> Option<*const RingTable> {
            match log_n {
                // SAFETY: each export returns the address of a constant-initialized static table
                $( #[$feat] $l => Some(unsafe { $f() }), )*
                _ => None,
            }
        }
    };
}
ring_fns!(cfg(feature = "ring"), raw_u32, pc_ring_u32_10 = 10, pc_ring_u32_11 = 11, pc_ring_u32_12 = 12, pc_ring_u32_13 = 13,
    pc_ring_u32_14 = 14, pc_ring_u32_15 = 15, pc_ring_u32_16 = 16, pc_ring_u32_17 = 17);
ring_fns!(cfg(all(feature = "ring", feature = "w64")), raw_u64, pc_ring_u64_10 = 10, pc_ring_u64_11 = 11, pc_ring_u64_12 = 12,
    pc_ring_u64_13 = 13, pc_ring_u64_14 = 14, pc_ring_u64_15 = 15, pc_ring_u64_16 = 16, pc_ring_u64_17 = 17);

/// The ring table of (`word`, `log_n`), checked against the ring ABI, word size and ring size.
pub fn ring_table(word: RingWord, log_n: u32) -> Result<&'static RingTable, TableError> {
    let name = match word {
        RingWord::U32 => "ring u32",
        RingWord::U64 => "ring u64",
    };
    let p = match word {
        RingWord::U32 => raw_u32(log_n),
        RingWord::U64 => raw_u64(log_n),
    }
    .ok_or(TableError::NotBuilt(name, if word == RingWord::U64 { "ring + w64" } else { "ring" }))?;
    // SAFETY: non-null pointer to an immutable static table
    let t: &'static RingTable = unsafe { &*p };
    let chk = |field, got, want| if got == want { Ok(()) } else { Err(TableError::Mismatch { name, field, got, want }) };
    chk("abi", t.abi, RING_ABI)?;
    chk("word_bytes", t.word_bytes, word.bytes())?;
    chk("mm", t.mm, 0)?;
    chk("log_n", t.log_n, log_n)?;
    Ok(t)
}
