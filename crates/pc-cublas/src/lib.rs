//! pc-cublas: hand-written cuBLASLt FFI for the exact int8 x int8 -> int32 GEMMs of PCMM/CCMM, with safe owners for
//! the handle and descriptors and one typed `GemmI8I32` plan per shape.
//!
//! Why our own FFI: the GEMM path needs ~15 cuBLASLt entry points; hand-written declarations against the CUDA 13.2
//! header keep the struct sizes checked here (const asserts in `ffi`) and avoid a bindgen/cudarc dependency.
//! Streams and device buffers are raw pointers at this layer (pc-cuda/pc-gpu wrap them); everything that dereferences
//! them is `unsafe fn` with the contract written on it.
//!
//! Exactness: int8 operands with |x| <= 128 give |a b| <= 2^14 per product, so a dot of length k accumulated over
//! `pairs` beta=1 calls stays inside i32 iff pairs * k * 2^14 < 2^31 (`exact_k_limit`). cuBLASLt's int32 compute
//! paths are integer MMA (no rounding): within that bound the result is the exact integer. A host `Lt` (`Lt::host`)
//! plans nothing and runs the same contract on the CPU (`host.rs`), on pc-hostrt streams.
pub mod ffi;
mod host;

use std::ffi::{c_void, CStr};
use std::marker::PhantomData;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{call} failed: cublas status {status} ({msg})")]
    Status { call: &'static str, status: i32, msg: String },
    #[error("no cuBLASLt algorithm for {spec}")]
    NoAlgo { spec: String },
    #[error("invalid GEMM spec ({field} = {value}): {why}")]
    Spec { field: &'static str, value: i64, why: &'static str },
    #[error("workspace of {have} bytes, the selected algorithm needs {need}")]
    Workspace { have: usize, need: usize },
    #[error("algorithm index {index} out of {count} candidates")]
    AlgoIndex { index: usize, count: usize },
    #[error("operand {operand} at {addr:#x} is not {need}-byte aligned as planned (Prefs::align)")]
    Align { operand: &'static str, addr: usize, need: u32 },
}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn status_string(s: i32) -> String {
    // cublasLtGetStatusString returns a static string (or null for unknown codes in old releases)
    let p = unsafe { ffi::cublasLtGetStatusString(s) };
    if p.is_null() {
        format!("status {s}")
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

pub(crate) fn check(call: &'static str, s: i32) -> Result<()> {
    if s == ffi::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Status { call, status: s, msg: status_string(s) })
    }
}

/// Largest dot length k that is exact in i32 when `pairs` int8 GEMMs (|x| <= 128) accumulate into one output.
pub fn exact_k_limit(pairs: u64) -> u64 {
    // pairs * k * 2^14 <= 2^31 - 1
    ((1u64 << 31) - 1) / (pairs.max(1) << 14)
}

/// cuBLASLt runtime version (`major * 10000 + minor * 100 + patch`).
pub fn version() -> usize {
    unsafe { ffi::cublasLtGetVersion() }
}
pub fn version_string() -> String {
    let v = version();
    format!("{}.{}.{}", v / 10000, (v / 100) % 100, v % 100)
}

/// A cuBLASLt handle. Not `Send`/`Sync`: one host thread issues work through it.
pub struct Lt {
    h: ffi::cublasLtHandle_t,
    host: bool,
    _nosend: PhantomData<*const ()>,
}
impl Lt {
    pub fn new() -> Result<Self> {
        let mut h = std::ptr::null_mut();
        check("cublasLtCreate", unsafe { ffi::cublasLtCreate(&mut h) })?;
        Ok(Lt { h, host: false, _nosend: PhantomData })
    }
    /// The host GEMM (no library handle); its plans run on pc-hostrt streams.
    pub fn host() -> Self {
        Lt { h: std::ptr::null_mut(), host: true, _nosend: PhantomData }
    }
    pub fn is_host(&self) -> bool {
        self.host
    }
    pub fn raw(&self) -> ffi::cublasLtHandle_t {
        self.h
    }
}
impl Drop for Lt {
    fn drop(&mut self) {
        if !self.host {
            unsafe { ffi::cublasLtDestroy(self.h) };
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Order {
    Col,
    Row,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    N,
    T,
}
/// `Pedantic` forbids the reduced-precision/alternate paths cuBLAS may otherwise pick; for int32 it is a control.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Compute {
    I32,
    I32Pedantic,
}
impl Order {
    fn raw(self) -> i32 {
        match self {
            Order::Col => ffi::CUBLASLT_ORDER_COL,
            Order::Row => ffi::CUBLASLT_ORDER_ROW,
        }
    }
}
impl Op {
    fn raw(self) -> i32 {
        match self {
            Op::N => ffi::CUBLAS_OP_N,
            Op::T => ffi::CUBLAS_OP_T,
        }
    }
}
impl Compute {
    fn raw(self) -> i32 {
        match self {
            Compute::I32 => ffi::CUBLAS_COMPUTE_32I,
            Compute::I32Pedantic => ffi::CUBLAS_COMPUTE_32I_PEDANTIC,
        }
    }
}

/// D[m x n] (+)= op(A) op(B) with op(A) m x k, op(B) k x n; int8 A/B, int32 C (= D, in place); `batch` strided
/// batches. Leading dimensions and strides are in elements of the stored matrix (A stored m x k for `Op::N`,
/// k x m for `Op::T`; likewise B).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmSpec {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub op_a: Op,
    pub op_b: Op,
    pub order_a: Order,
    pub order_b: Order,
    pub order_c: Order,
    pub lda: i64,
    pub ldb: i64,
    pub ldc: i64,
    pub batch: i32,
    pub stride_a: i64,
    pub stride_b: i64,
    pub stride_c: i64,
    pub compute: Compute,
}

impl GemmSpec {
    /// Densely packed operands (minimal leading dimensions, batches back to back) in the given ops/orders.
    pub fn packed(m: u64, n: u64, k: u64, op_a: Op, op_b: Op, order: Order, batch: i32) -> Self {
        let mut s = GemmSpec {
            m,
            n,
            k,
            op_a,
            op_b,
            order_a: order,
            order_b: order,
            order_c: order,
            lda: 0,
            ldb: 0,
            ldc: 0,
            batch,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            compute: Compute::I32,
        };
        s.lda = s.min_ld(s.a_dims(), order) as i64;
        s.ldb = s.min_ld(s.b_dims(), order) as i64;
        s.ldc = s.min_ld((m, n), order) as i64;
        s.stride_a = (m * k) as i64;
        s.stride_b = (k * n) as i64;
        s.stride_c = (m * n) as i64;
        s
    }
    fn min_ld(&self, (rows, cols): (u64, u64), order: Order) -> u64 {
        match order {
            Order::Col => rows,
            Order::Row => cols,
        }
    }
    /// Stored (rows, cols) of A.
    pub fn a_dims(&self) -> (u64, u64) {
        match self.op_a {
            Op::N => (self.m, self.k),
            Op::T => (self.k, self.m),
        }
    }
    pub fn b_dims(&self) -> (u64, u64) {
        match self.op_b {
            Op::N => (self.k, self.n),
            Op::T => (self.n, self.k),
        }
    }
    /// Element offset of stored (r, c) in a matrix with this order and ld.
    pub fn offset(order: Order, ld: i64, r: u64, c: u64) -> usize {
        match order {
            Order::Col => (c as i64 * ld) as usize + r as usize,
            Order::Row => (r as i64 * ld) as usize + c as usize,
        }
    }
    fn extent((rows, cols): (u64, u64), order: Order, ld: i64, batch: i32, stride: i64) -> usize {
        if rows == 0 || cols == 0 {
            return 0;
        }
        let last = Self::offset(order, ld, rows - 1, cols - 1) + 1;
        (batch as usize - 1) * stride as usize + last
    }
    /// Elements the A buffer must hold (all batches).
    pub fn len_a(&self) -> usize {
        Self::extent(self.a_dims(), self.order_a, self.lda, self.batch, self.stride_a)
    }
    pub fn len_b(&self) -> usize {
        Self::extent(self.b_dims(), self.order_b, self.ldb, self.batch, self.stride_b)
    }
    pub fn len_c(&self) -> usize {
        Self::extent((self.m, self.n), self.order_c, self.ldc, self.batch, self.stride_c)
    }
    pub fn validate(&self) -> Result<()> {
        let bad = |field, value: i64, why| Err(Error::Spec { field, value, why });
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return bad("m*n*k", (self.m * self.n * self.k) as i64, "empty GEMM");
        }
        if self.batch < 1 {
            return bad("batch", self.batch as i64, "must be >= 1");
        }
        for (field, ld, dims, order) in [
            ("lda", self.lda, self.a_dims(), self.order_a),
            ("ldb", self.ldb, self.b_dims(), self.order_b),
            ("ldc", self.ldc, (self.m, self.n), self.order_c),
        ] {
            if (ld as u64) < self.min_ld(dims, order) {
                return bad(field, ld, "below the stored matrix's minor dimension");
            }
        }
        if self.batch > 1 {
            for (field, st, len) in [
                ("stride_a", self.stride_a, Self::extent(self.a_dims(), self.order_a, self.lda, 1, 0)),
                ("stride_b", self.stride_b, Self::extent(self.b_dims(), self.order_b, self.ldb, 1, 0)),
                ("stride_c", self.stride_c, Self::extent((self.m, self.n), self.order_c, self.ldc, 1, 0)),
            ] {
                // overlapping inputs are legal for cuBLASLt, overlapping outputs are not
                if field == "stride_c" && (st as usize) < len {
                    return bad(field, st, "output batches overlap");
                }
                if st < 0 {
                    return bad(field, st, "negative stride");
                }
            }
        }
        Ok(())
    }
}

struct Desc(ffi::cublasLtMatmulDesc_t);
impl Drop for Desc {
    fn drop(&mut self) {
        unsafe { ffi::cublasLtMatmulDescDestroy(self.0) };
    }
}
struct Layout(ffi::cublasLtMatrixLayout_t);
impl Drop for Layout {
    fn drop(&mut self) {
        unsafe { ffi::cublasLtMatrixLayoutDestroy(self.0) };
    }
}
struct Pref(ffi::cublasLtMatmulPreference_t);
impl Drop for Pref {
    fn drop(&mut self) {
        unsafe { ffi::cublasLtMatmulPreferenceDestroy(self.0) };
    }
}

fn set<T>(f: unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> i32, call: &'static str, obj: *mut c_void, attr: i32, v: &T) -> Result<()> {
    check(call, unsafe { f(obj, attr, v as *const T as *const c_void, std::mem::size_of::<T>()) })
}

fn layout(ty: i32, (rows, cols): (u64, u64), ld: i64, order: Order, batch: i32, stride: i64) -> Result<Layout> {
    let mut l = std::ptr::null_mut();
    check("cublasLtMatrixLayoutCreate", unsafe { ffi::cublasLtMatrixLayoutCreate(&mut l, ty, rows, cols, ld) })?;
    let l = Layout(l);
    let f = ffi::cublasLtMatrixLayoutSetAttribute;
    set(f, "cublasLtMatrixLayoutSetAttribute(ORDER)", l.0, ffi::CUBLASLT_MATRIX_LAYOUT_ORDER, &order.raw())?;
    if batch > 1 {
        set(f, "cublasLtMatrixLayoutSetAttribute(BATCH_COUNT)", l.0, ffi::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch)?;
        set(f, "cublasLtMatrixLayoutSetAttribute(STRIDED_BATCH_OFFSET)", l.0, ffi::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride)?;
    }
    Ok(l)
}

/// One heuristic candidate with its decoded configuration (for tables and for pinning an algorithm).
#[derive(Clone, Copy, Debug)]
pub struct Algo {
    pub raw: ffi::cublasLtMatmulAlgo_t,
    pub workspace: usize,
    pub waves: f32,
    pub id: i32,
    pub tile: u32,
    pub stages: u32,
    pub splitk: i32,
}
impl Algo {
    fn decode(r: &ffi::cublasLtMatmulHeuristicResult_t) -> Self {
        fn get<T: Default>(a: &ffi::cublasLtMatmulAlgo_t, attr: i32) -> T {
            let mut v = T::default();
            let mut w = 0usize;
            // unknown attributes leave the default (0): these fields are informational only
            unsafe {
                ffi::cublasLtMatmulAlgoConfigGetAttribute(a, attr, &mut v as *mut T as *mut c_void, std::mem::size_of::<T>(), &mut w)
            };
            v
        }
        Algo {
            raw: r.algo,
            workspace: r.workspace_size,
            waves: r.waves_count,
            id: get(&r.algo, ffi::CUBLASLT_ALGO_CONFIG_ID),
            tile: get(&r.algo, ffi::CUBLASLT_ALGO_CONFIG_TILE_ID),
            stages: get(&r.algo, ffi::CUBLASLT_ALGO_CONFIG_STAGES_ID),
            splitk: get(&r.algo, ffi::CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
        }
    }
    /// Compact label: `a<id>t<tile>s<stages>k<splitk>`.
    pub fn label(&self) -> String {
        format!("a{}t{}s{}k{}", self.id, self.tile, self.stages, self.splitk)
    }
}

/// Minimum byte alignment the caller guarantees for each operand pointer (every batch included). cuBLASLt's
/// heuristic assumes 256 unless told otherwise, and an algorithm picked for 256 may fault on a less aligned pointer,
/// so a plan records what it was made for and `run` refuses pointers that break it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Align {
    pub a: u32,
    pub b: u32,
    pub c: u32,
}
impl Default for Align {
    fn default() -> Self {
        Align { a: 256, b: 256, c: 256 }
    }
}
impl Align {
    /// Alignment of an address (or byte offset from a 256-aligned base), capped at 256.
    pub fn of(addr: usize) -> u32 {
        if addr == 0 {
            256
        } else {
            1u32 << addr.trailing_zeros().min(8)
        }
    }
}

/// Planning preferences.
#[derive(Clone, Copy, Debug)]
pub struct Prefs {
    pub max_workspace: usize,
    /// heuristic candidates to keep (>= 1)
    pub max_algos: usize,
    pub align: Align,
}
impl Default for Prefs {
    fn default() -> Self {
        Prefs { max_workspace: 32 << 20, max_algos: 1, align: Align::default() }
    }
}

/// Device scratch for cuBLASLt (raw at this layer).
#[derive(Clone, Copy, Debug)]
pub struct Workspace {
    pub ptr: *mut c_void,
    pub bytes: usize,
}

/// A planned int8 -> int32 GEMM: descriptors built once, heuristic candidates queried once, one selected.
pub struct GemmI8I32<'lt> {
    lt: &'lt Lt,
    spec: GemmSpec,
    /// The cuBLASLt descriptors (None for a host plan).
    plan: Option<Plan>,
    algos: Vec<Algo>,
    sel: usize,
    align: Align,
}

struct Plan {
    desc: Desc,
    a: Layout,
    b: Layout,
    c: Layout,
}

impl<'lt> GemmI8I32<'lt> {
    /// Plans `spec`, asking the heuristic for up to `prefs.max_algos` candidates that fit `prefs.max_workspace`
    /// bytes and the operand alignment; selects the first. `Error::NoAlgo` when cuBLASLt offers none (the
    /// configuration is not supported).
    pub fn new(lt: &'lt Lt, spec: GemmSpec, prefs: Prefs) -> Result<Self> {
        spec.validate()?;
        // every batch's pointer must meet the alignment too
        let eff = |al: u32, stride: i64, elem: usize| {
            if spec.batch > 1 {
                al.min(Align::of(stride as usize * elem))
            } else {
                al
            }
        };
        let align = Align {
            a: eff(prefs.align.a, spec.stride_a, 1),
            b: eff(prefs.align.b, spec.stride_b, 1),
            c: eff(prefs.align.c, spec.stride_c, 4),
        };
        if lt.host {
            let algo = Algo { raw: Default::default(), workspace: 0, waves: 0.0, id: -1, tile: 0, stages: 0, splitk: 1 };
            return Ok(GemmI8I32 { lt, spec, plan: None, algos: vec![algo], sel: 0, align });
        }
        let mut d = std::ptr::null_mut();
        // int32 scale: alpha/beta are exact integers (1 and 0/1)
        check("cublasLtMatmulDescCreate", unsafe {
            ffi::cublasLtMatmulDescCreate(&mut d, spec.compute.raw(), ffi::CUDA_R_32I)
        })?;
        let desc = Desc(d);
        let f = ffi::cublasLtMatmulDescSetAttribute;
        set(f, "cublasLtMatmulDescSetAttribute(TRANSA)", desc.0, ffi::CUBLASLT_MATMUL_DESC_TRANSA, &spec.op_a.raw())?;
        set(f, "cublasLtMatmulDescSetAttribute(TRANSB)", desc.0, ffi::CUBLASLT_MATMUL_DESC_TRANSB, &spec.op_b.raw())?;
        let a = layout(ffi::CUDA_R_8I, spec.a_dims(), spec.lda, spec.order_a, spec.batch, spec.stride_a)?;
        let b = layout(ffi::CUDA_R_8I, spec.b_dims(), spec.ldb, spec.order_b, spec.batch, spec.stride_b)?;
        let c = layout(ffi::CUDA_R_32I, (spec.m, spec.n), spec.ldc, spec.order_c, spec.batch, spec.stride_c)?;

        let mut p = std::ptr::null_mut();
        check("cublasLtMatmulPreferenceCreate", unsafe { ffi::cublasLtMatmulPreferenceCreate(&mut p) })?;
        let pref = Pref(p);
        let ws = prefs.max_workspace as u64;
        let pf = ffi::cublasLtMatmulPreferenceSetAttribute;
        set(pf, "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)", pref.0, ffi::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws)?;
        for (attr, v) in [
            (ffi::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, align.a),
            (ffi::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, align.b),
            (ffi::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, align.c),
            (ffi::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, align.c),
        ] {
            set(pf, "cublasLtMatmulPreferenceSetAttribute(MIN_ALIGNMENT)", pref.0, attr, &v)?;
        }
        let want = prefs.max_algos.max(1);
        let mut res = vec![ffi::cublasLtMatmulHeuristicResult_t::default(); want];
        let mut got = 0i32;
        let st = unsafe {
            ffi::cublasLtMatmulAlgoGetHeuristic(lt.h, desc.0, a.0, b.0, c.0, c.0, pref.0, want as i32, res.as_mut_ptr(), &mut got)
        };
        // NOT_SUPPORTED here means "no algorithm for this configuration", which callers treat as a table entry
        if st != ffi::CUBLAS_STATUS_SUCCESS && st != 15 {
            check("cublasLtMatmulAlgoGetHeuristic", st)?;
        }
        let algos: Vec<Algo> = res[..got.max(0) as usize]
            .iter()
            .filter(|r| r.state == ffi::CUBLAS_STATUS_SUCCESS)
            .map(Algo::decode)
            .collect();
        if algos.is_empty() {
            return Err(Error::NoAlgo { spec: format!("{spec:?}") });
        }
        Ok(GemmI8I32 { lt, spec, plan: Some(Plan { desc, a, b, c }), algos, sel: 0, align })
    }
    pub fn spec(&self) -> &GemmSpec {
        &self.spec
    }
    pub fn algos(&self) -> &[Algo] {
        &self.algos
    }
    pub fn align(&self) -> Align {
        self.align
    }
    pub fn algo(&self) -> &Algo {
        &self.algos[self.sel]
    }
    pub fn select(&mut self, index: usize) -> Result<()> {
        if index >= self.algos.len() {
            return Err(Error::AlgoIndex { index, count: self.algos.len() });
        }
        self.sel = index;
        Ok(())
    }

    /// Enqueue D = op(A) op(B) (+ D when `accumulate`) on `stream`.
    ///
    /// # Safety
    /// `a`, `b`, `c` are device pointers to at least `spec.len_a()` / `len_b()` int8 and `len_c()` int32 elements
    /// that stay valid until the GEMM completed on `stream`; `c` aliases no input; `ws` is device memory owned by
    /// this stream for the same span; `stream` is a live CUDA stream on the current device.
    pub unsafe fn run(
        &self,
        stream: ffi::cudaStream_t,
        a: *const i8,
        b: *const i8,
        c: *mut i32,
        accumulate: bool,
        ws: Workspace,
    ) -> Result<()> {
        let algo = self.algo();
        for (operand, addr, need) in [("A", a as usize, self.align.a), ("B", b as usize, self.align.b), ("C", c as usize, self.align.c)] {
            if addr % need as usize != 0 {
                return Err(Error::Align { operand, addr, need });
            }
        }
        if ws.bytes < algo.workspace {
            return Err(Error::Workspace { have: ws.bytes, need: algo.workspace });
        }
        let Some(p) = &self.plan else {
            let spec = self.spec;
            pc_hostrt::submit(stream, move || host::run(&spec, a, b, c, accumulate));
            return Ok(());
        };
        let one = 1i32;
        let beta = accumulate as i32;
        check(
            "cublasLtMatmul",
            ffi::cublasLtMatmul(
                self.lt.h,
                p.desc.0,
                &one as *const i32 as *const c_void,
                a as *const c_void,
                p.a.0,
                b as *const c_void,
                p.b.0,
                &beta as *const i32 as *const c_void,
                c as *const c_void,
                p.c.0,
                c as *mut c_void,
                p.c.0,
                &algo.raw,
                ws.ptr,
                ws.bytes,
                stream,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bound() {
        // PCMM: up to 4 digit pairs per exponent; FIDESlib's guard is min(4, nW) * 2^14 * rowsIn < 2^31
        assert_eq!(exact_k_limit(1), 131071);
        assert_eq!(exact_k_limit(3), 43690);
        assert_eq!(exact_k_limit(4), 32767);
        assert!(3 * 16384 * (1u64 << 14) < 1 << 31);
    }

    #[test]
    fn spec_extents() {
        let s = GemmSpec::packed(8, 5, 3, Op::N, Op::N, Order::Col, 1);
        assert_eq!((s.lda, s.ldb, s.ldc), (8, 3, 8));
        assert_eq!((s.len_a(), s.len_b(), s.len_c()), (24, 15, 40));
        let t = GemmSpec::packed(8, 5, 3, Op::T, Op::N, Order::Row, 4);
        assert_eq!(t.a_dims(), (3, 8));
        assert_eq!(t.lda, 8);
        assert_eq!(t.len_a(), 4 * 24);
        assert!(t.validate().is_ok());
        let mut bad = s;
        bad.ldc = 4;
        assert!(matches!(bad.validate(), Err(Error::Spec { field: "ldc", .. })));
    }

    #[test]
    fn align_of() {
        assert_eq!(Align::of(0), 256);
        assert_eq!(Align::of(4), 4);
        assert_eq!(Align::of(1), 1);
        assert_eq!(Align::of(4096 + 16), 16);
        assert_eq!(Align::of(1 << 20), 256);
    }

    #[test]
    fn version_is_13() {
        // no device needed: cublasLtGetVersion only reads the library's constant
        assert_eq!(version() / 10000, 13, "{}", version_string());
    }
}
