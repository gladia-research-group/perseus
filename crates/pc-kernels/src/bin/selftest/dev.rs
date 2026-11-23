//! Word-generic device buffers on one arena (u32-word `DBuf`s): host data travels as `Vec<u64>` and is
//! packed to the instantiation's word size on upload.
#![allow(dead_code)]
use pc_cuda::{Arena, Compute, DBuf, Device, Stream};
use std::ffi::c_void;

pub struct Ctx<'a> {
    pub dev: &'a Device,
    pub arena: &'a Arena,
    pub s: Stream<Compute>,
    /// word bytes of the instantiation under test
    pub wb: usize,
}

pub struct Dev<'a> {
    buf: DBuf<'a, Compute, u32>,
    ptr: *mut c_void,
    pub wb: usize,
    pub len: usize,
}

pub fn pack(data: &[u64], wb: usize) -> Vec<u32> {
    match wb {
        4 => data.iter().map(|&x| x as u32).collect(),
        8 => data.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect(),
        _ => data.chunks(2).map(|c| (c[0] as u32 & 0xFFFF) | ((c.get(1).copied().unwrap_or(0) as u32) << 16)).collect(),
    }
}
pub fn unpack(w: &[u32], wb: usize, len: usize) -> Vec<u64> {
    match wb {
        4 => w[..len].iter().map(|&x| x as u64).collect(),
        8 => (0..len).map(|i| w[2 * i] as u64 | ((w[2 * i + 1] as u64) << 32)).collect(),
        _ => (0..len).map(|i| ((w[i / 2] >> (16 * (i & 1))) & 0xFFFF) as u64).collect(),
    }
}
fn u32_words(len: usize, wb: usize) -> usize {
    (len * wb).div_ceil(4).max(1)
}

/// The arena of a group of checks, which never frees: `need_mib` MiB (the group's peak plus headroom; the
/// selftest prints each group's use), or `PC_SELFTEST_ARENA_MIB` when set.
pub fn arena(dev: &Device, need_mib: usize) -> Arena {
    let mib = std::env::var("PC_SELFTEST_ARENA_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(need_mib);
    Arena::new(dev, mib << 20).unwrap_or_else(|e| panic!("selftest arena of {mib} MiB: {e}"))
}

/// An arena of exactly the buffers `bufs` = (words, word bytes) (alignment included).
pub fn exact_arena(dev: &Device, bufs: &[(usize, usize)]) -> Arena {
    let bytes: usize = bufs.iter().map(|&(len, wb)| Arena::bytes_for::<u32>(u32_words(len, wb))).sum();
    Arena::new(dev, bytes).unwrap_or_else(|e| panic!("selftest arena of {bytes} bytes: {e}"))
}

impl<'a> Ctx<'a> {
    /// zero-filled buffer of `len` words of the instantiation
    pub fn zeros(&self, len: usize) -> Dev<'a> {
        self.zeros_wb(len, self.wb)
    }
    pub fn zeros_wb(&self, len: usize, wb: usize) -> Dev<'a> {
        self.zeros_in(self.arena, len, wb)
    }
    /// zero-filled buffer carved from another arena (filled on this context's stream)
    pub fn zeros_in<'b>(&self, arena: &'b Arena, len: usize, wb: usize) -> Dev<'b> {
        let mut buf = arena.alloc::<Compute, u32>("selftest", u32_words(len, wb)).expect("arena");
        buf.zero(&self.s).expect("zero");
        let ptr = buf.dptr_mut() as *mut c_void;
        Dev { buf, ptr, wb, len }
    }
    pub fn up(&self, data: &[u64]) -> Dev<'a> {
        self.up_wb(data, self.wb)
    }
    pub fn up_wb(&self, data: &[u64], wb: usize) -> Dev<'a> {
        self.up_in(self.arena, data, wb)
    }
    pub fn up_in<'b>(&self, arena: &'b Arena, data: &[u64], wb: usize) -> Dev<'b> {
        let mut d = self.zeros_in(arena, data.len(), wb);
        d.write(self, data);
        d
    }
    /// u32 index table
    pub fn up32(&self, data: &[u32]) -> Dev<'a> {
        let v: Vec<u64> = data.iter().map(|&x| x as u64).collect();
        self.up_wb(&v, 4)
    }
    pub fn up_i32(&self, data: &[i32]) -> Dev<'a> {
        let v: Vec<u64> = data.iter().map(|&x| x as u32 as u64).collect();
        self.up_wb(&v, 4)
    }
    /// table of device addresses (for `const void* const*` arguments)
    pub fn up_ptrs(&self, ptrs: &[*const c_void]) -> Dev<'a> {
        let v: Vec<u64> = ptrs.iter().map(|&p| p as u64).collect();
        self.up_wb(&v, 8)
    }
    pub fn up_f64(&self, data: &[f64]) -> Dev<'a> {
        let v: Vec<u64> = data.iter().map(|x| x.to_bits()).collect();
        self.up_wb(&v, 8)
    }
    /// raw bytes (pool with mixed content), as u32 words
    pub fn up_raw32(&self, words: &[u32]) -> Dev<'a> {
        self.up32(words)
    }
}

impl<'a> Dev<'a> {
    pub fn p(&self) -> *mut c_void {
        self.ptr
    }
    pub fn c(&self) -> *const c_void {
        self.ptr as *const c_void
    }
    /// pointer `off` words in
    pub fn at(&self, off: usize) -> *mut c_void {
        // SAFETY: offset inside the buffer (callers index within len)
        unsafe { (self.ptr as *mut u8).add(off * self.wb) as *mut c_void }
    }
    pub fn write(&mut self, ctx: &Ctx, data: &[u64]) {
        assert!(data.len() <= self.len);
        self.buf.upload(&ctx.s, &pack(data, self.wb)).expect("upload");
    }
    pub fn read(&self, ctx: &Ctx) -> Vec<u64> {
        let w = self.buf.download_range(&ctx.s, 0, u32_words(self.len, self.wb)).expect("download");
        unpack(&w, self.wb, self.len)
    }
    pub fn read_f64(&self, ctx: &Ctx) -> Vec<f64> {
        self.read(ctx).into_iter().map(f64::from_bits).collect()
    }
    pub fn copy_from(&mut self, ctx: &Ctx, src: &Dev) {
        let n = u32_words(self.len.min(src.len), self.wb);
        self.buf.copy_from(&ctx.s, &src.buf, n).expect("copy");
    }
}
