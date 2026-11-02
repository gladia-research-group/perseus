//! Bootstrap constants (host, CUDA-free): the EvalMod Chebyshev tables, the collapsed-FFT CoeffsToSlots /
//! SlotsToCoeffs factor matrices with their BSGS parameters and rotation sets (OpenFHE `EvalBootstrapSetup` ->
//! `GetCollapsedFFTParams`, `CoeffEncodingCollapse`, `CoeffDecodingCollapse`, `EvalCoeffsToSlotsPrecompute`,
//! `EvalSlotsToCoeffsPrecompute`, `Find*RotationIndices`), FIDESlib's "LT-friendly" stage layout (rotIn/rotOut and the
//! accumulated plaintext pre-rotation of `RawCiphertext.cu`), the SparseB depth-1 diagonals (`buildSparseB`) and the
//! StC-first stage factors. Values only: encoding levels / scales come from pc-shadow.
//!
//! Every port keeps OpenFHE's operation order so values agree to the last bits with OpenFHE/FIDESlib (the gate
//! compares against the decoded plaintexts of the d3 dump at <= 1e-9 relative).
use crate::error::{Error, Result};
use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

/// Minimal complex double (std::complex\<double> arithmetic: (ac - bd, ad + bc)).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    pub const fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }
    pub fn conj(self) -> Self {
        C64 { re: self.re, im: -self.im }
    }
    pub fn scale(self, s: f64) -> Self {
        C64 { re: self.re * s, im: self.im * s }
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    /// e^(i theta)
    pub fn cis(theta: f64) -> Self {
        C64 { re: theta.cos(), im: theta.sin() }
    }
}
impl Add for C64 {
    type Output = C64;
    fn add(self, o: C64) -> C64 {
        C64 { re: self.re + o.re, im: self.im + o.im }
    }
}
impl AddAssign for C64 {
    fn add_assign(&mut self, o: C64) {
        self.re += o.re;
        self.im += o.im;
    }
}
impl Sub for C64 {
    type Output = C64;
    fn sub(self, o: C64) -> C64 {
        C64 { re: self.re - o.re, im: self.im - o.im }
    }
}
impl Mul for C64 {
    type Output = C64;
    fn mul(self, o: C64) -> C64 {
        C64 { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}
impl MulAssign for C64 {
    fn mul_assign(&mut self, o: C64) {
        *self = *self * o;
    }
}
impl Neg for C64 {
    type Output = C64;
    fn neg(self) -> C64 {
        C64 { re: -self.re, im: -self.im }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// EvalMod tables
// ------------------------------------------------------------------------------------------------------------------

/// The shipped EvalMod series (dense/ENCAPS route): degree-14 Chebyshev refit of (2 pi)^(-1/32) cos(2 pi (16 y -
/// 0.25) / 32), followed by 5 double-angle iterations, K = 16 (sparse-encapsulated uniform secret). OpenFHE
/// evaluator convention (c0 not halved). Provenance: FIDESlib `openfhe-interface/RawCiphertext.cu`, ENCAPS branch of
/// `GetRawParams` (FIDESlib f31017d) — not the OpenFHE patch series, whose g_coefficients tables are the deg-32/44/88
/// upstream ones; the d3 dump's `params.json::cheb` is this table bit for bit.
pub const EVALMOD_CHEB: [f64; 15] = [
    -5.73829476916553172e-01,
    2.63718536771466207e-02,
    -9.15574236933522245e-01,
    -3.08975417541565676e-02,
    2.85601052580403802e-01,
    4.83129148738224799e-03,
    -2.74350673185783482e-02,
    -3.16919293037672828e-04,
    1.31295181914509542e-03,
    1.15825536926191591e-05,
    -3.79010152666429525e-05,
    -2.71035793019274615e-07,
    7.33952552563662122e-07,
    4.41686895092293209e-09,
    -1.03169742989480305e-08,
];
/// EvalMod K of the shipped table (bootK = 16, K_SPARSE_ENCAPSULATED).
pub const EVALMOD_K: f64 = 16.0;
/// Double-angle iterations after the shipped series.
pub const EVALMOD_DA_ITS: u32 = 5;

/// The frozen EvalMod coefficient table (see `EVALMOD_CHEB`).
pub fn evalmod_coeffs() -> &'static [f64] {
    &EVALMOD_CHEB
}

/// SparseB's EvalMod series for K = 24: (2 pi)^(-1/32) cos(2 pi (24 y - 0.25) / 32), Chebyshev degree 20 (same PS
/// depth as degree 14), 5 double angles. Provenance: FIDESlib `SparseB.cu::sparseBChebyshevK24`.
pub const SPARSE_B_CHEB_K24: [f64; 21] = [
    -5.01431878575762102e-01,
    -2.60978240793176830e-02,
    -2.75969415548868335e-01,
    -3.76057943042461193e-02,
    6.98674483745098551e-01,
    2.06638885145157200e-02,
    -1.93915485831718332e-01,
    -3.59503488102385325e-03,
    2.34902992390395096e-02,
    3.23162556614403976e-04,
    -1.63628724411827371e-03,
    -1.80046855470203313e-05,
    7.47055232583680419e-05,
    6.86710430736672347e-07,
    -2.41801471221991641e-06,
    -1.91108603715104241e-08,
    5.85030407613337840e-08,
    4.05831352269635484e-10,
    -1.09946346874119116e-09,
    -6.79891244700694627e-12,
    1.65253168377321969e-11,
];
pub const SPARSE_B_K: f64 = 24.0;
pub const SPARSE_B_DA_ITS: u32 = 5;

/// SparseB's K = 16 series refitted for the overflow integer it meets: the degree-14 least-squares fit of
/// `EVALMOD_CHEB`'s function on t / 16, t within 1/32 of each integer I, weighted by sqrt(P(I)) (I a rounded sum of the
/// h = 32 encapsulation secret's 33 uniforms). Same degree, products, depth and double angles as `EVALMOD_CHEB`, so it
/// replaces it with no other change; its EvalMod error (5 double angles, noise-free) is 2^-35 or less for |I| <= 6
/// (`EVALMOD_CHEB`: 2^-29.6), 2^-33.8 or less up to |I| = 7, no worse than `EVALMOD_CHEB` at |I| = 8, worse from |I| = 9
/// (P(|I| >= 9) ~ 5e-8).
pub const SPARSE_B_K16_W14: [f64; 15] = [
    -0.57382947630327,
    0.02637185350745373,
    -0.9155742365679239,
    -0.0308975418423564,
    0.28560105273647235,
    0.004831291368967446,
    -0.0274350669956903,
    -0.00031691938216668613,
    0.0013129519873082535,
    1.1582715628454975e-05,
    -3.7901496763804674e-05,
    -2.707038127090425e-07,
    7.331598959140216e-07,
    4.63808824848783e-09,
    -1.068560591081154e-08,
];

/// FNV-1a 64 over the little-endian f64 bit patterns: the checksum recorded for a frozen table.
pub fn table_checksum(t: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in t {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

// ------------------------------------------------------------------------------------------------------------------
// Collapsed FFT parameters
// ------------------------------------------------------------------------------------------------------------------

/// OpenFHE `ReduceRotation(index, slots)`, verbatim: for a power-of-two `slots` a negative exact multiple of `slots`
/// maps to `slots` (not 0) — kept, callers reduce again where it matters.
pub fn reduce_rotation(index: i64, slots: u32) -> u32 {
    let islots = slots as i64;
    if slots.is_power_of_two() {
        let n = slots.trailing_zeros();
        if index >= 0 {
            return (index - ((index >> n) << n)) as u32;
        }
        return (index + islots + ((index.abs() >> n) << n)) as u32;
    }
    ((islots + index % islots) % islots) as u32
}

/// OpenFHE `SelectLayers(logSlots, budget)` -> (layers, rows, rem).
pub fn select_layers(log_slots: u32, budget: u32) -> (u32, u32, u32) {
    let mut layers = (log_slots as f64 / budget as f64).ceil() as u32;
    let mut rows = log_slots / layers;
    let mut rem = log_slots % layers;
    let mut dim = rows + (rem != 0) as u32;
    if dim < budget {
        layers -= 1;
        rows = log_slots / layers;
        rem = log_slots - rows * layers;
        dim = rows + (rem != 0) as u32;
        if dim > budget {
            while dim != budget {
                rows -= 1;
                rem = log_slots - rows * layers;
                dim = rows + (rem != 0) as u32;
            }
        }
    }
    (layers, rows, rem)
}

/// OpenFHE `GetCollapsedFFTParams` (the CKKS_BOOT_PARAMS vector). `b` = number of giant steps (outer loop),
/// `g` = baby steps (inner loop), as OpenFHE names them; FIDESlib's bStep/gStep are g/b.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollapsedFftParams {
    pub level_budget: u32,
    pub layers_collapse: u32,
    pub rem_collapse: u32,
    pub num_rotations: u32,
    pub b: u32,
    pub g: u32,
    pub num_rotations_rem: u32,
    pub b_rem: u32,
    pub g_rem: u32,
}

pub fn collapsed_fft_params(slots: u32, level_budget: u32, dim1: u32) -> Result<CollapsedFftParams> {
    if slots == 0 || level_budget == 0 {
        return Err(Error::BtsSetup(format!("collapsed_fft_params: slots={slots} level_budget={level_budget}")));
    }
    let log_slots = if slots < 3 { 1 } else { slots.ilog2() };
    let (layers, _rows, rem) = select_layers(log_slots, level_budget);
    let num_rotations = (1u32 << (layers + 1)) - 1;
    let num_rotations_rem = (1u32 << (rem + 1)) - 1;
    let g = if dim1 == 0 || dim1 > num_rotations {
        1u32 << (layers / 2 + 1 + (num_rotations > 7) as u32)
    } else {
        dim1
    };
    let b = (num_rotations + 1) / g;
    let g_rem = if rem != 0 { 1u32 << (rem / 2 + 1 + (num_rotations_rem > 7) as u32) } else { 0 };
    let b_rem = if rem != 0 { (num_rotations_rem + 1) / g_rem } else { 0 };
    Ok(CollapsedFftParams {
        level_budget,
        layers_collapse: layers,
        rem_collapse: rem,
        num_rotations,
        b,
        g,
        num_rotations_rem,
        b_rem,
        g_rem,
    })
}

/// rotGroup of `EvalBootstrapSetup`: 5^i mod m, i < slots (m = 4 slots).
pub fn rot_group(slots: u32, m: u32) -> Vec<u32> {
    let mask = m - 1;
    let mut v = Vec::with_capacity(slots as usize);
    let mut f: u32 = 1;
    for _ in 0..slots {
        v.push(f);
        f = f.wrapping_mul(5) & mask;
    }
    v
}

/// ksiPows of `EvalBootstrapSetup`: exp(2 pi i j / m), j <= m (last = first).
pub fn ksi_pows(m: u32) -> Vec<C64> {
    let ak = 2.0 * PI / m as f64;
    let mut v: Vec<C64> = (0..m).map(|j| C64::new((ak * j as f64).cos(), (ak * j as f64).sin())).collect();
    v.push(v[0]);
    v
}

fn rot_idx(k: u32, delta: i64, slots: u32) -> usize {
    // OpenFHE computes `k - x` / `k + x` in uint32 and reduces as int32: (k +- x) mod slots
    let v = (k as i64 + delta) as u32 as i32 as i64;
    reduce_rotation(v, slots) as usize % slots as usize
}

/// OpenFHE `CoeffEncodingOneLevel`: 3 log2(slots) vectors (shifted right / unshifted / shifted left per layer).
pub fn coeff_encoding_one_level(pows: &[C64], rot_group: &[u32], flag_i: bool) -> Vec<Vec<C64>> {
    let dim = (pows.len() - 1) as u32;
    let slots = rot_group.len() as u32;
    let log2slots = slots.ilog2();
    let neg_i = C64::new((-PI / 2.0).cos(), (-PI / 2.0).sin()); // std::exp(-M_PI/2 * I)
    let mut coeff = vec![vec![C64::ZERO; slots as usize]; 3 * log2slots as usize];
    let mut m = slots;
    while m > 1 {
        let s0 = (m.ilog2() - 1) as usize;
        let (s1, s2) = (s0 + log2slots as usize, s0 + 2 * log2slots as usize);
        let b = if flag_i && m == 2 { neg_i } else { C64::ONE };
        let mut k = 0;
        while k < slots {
            let lenq = m << 2;
            let lenh = m >> 1;
            let klenh = k + lenh;
            for x in k..klenh {
                coeff[s2][x as usize] = b;
                coeff[s1][x as usize] = b;
            }
            for j in 0..lenh {
                let jk = (klenh + j) as usize;
                let w = b * pows[((lenq - (rot_group[j as usize] % lenq)) * (dim / lenq)) as usize];
                coeff[s1][jk] = -w;
                coeff[s0][jk] = w;
            }
            k += m;
        }
        m >>= 1;
    }
    coeff
}

/// OpenFHE `CoeffDecodingOneLevel`.
pub fn coeff_decoding_one_level(pows: &[C64], rot_group: &[u32], flag_i: bool) -> Vec<Vec<C64>> {
    let dim = (pows.len() - 1) as u32;
    let slots = rot_group.len() as u32;
    let log2slots = slots.ilog2();
    let pos_i = C64::new((PI / 2.0).cos(), (PI / 2.0).sin());
    let mut coeff = vec![vec![C64::ZERO; slots as usize]; 3 * log2slots as usize];
    let mut m = 2;
    while m <= slots {
        let s0 = (m.ilog2() - 1) as usize;
        let (s1, s2) = (s0 + log2slots as usize, s0 + 2 * log2slots as usize);
        let b = if flag_i && m == 2 { pos_i } else { C64::ONE };
        let mut k = 0;
        while k < slots {
            let lenq = m << 2;
            let lenh = m >> 1;
            let klenh = k + lenh;
            for x in klenh..klenh + lenh {
                coeff[s0][x as usize] = b;
            }
            for x in k..klenh {
                coeff[s1][x as usize] = b;
            }
            for j in 0..lenh {
                let jk = (k + j) as usize;
                let w = b * pows[((rot_group[j as usize] % lenq) * (dim / lenq)) as usize];
                coeff[s2][jk] = w;
                coeff[s1][jk + lenh as usize] = -w;
            }
            k += m;
        }
        m <<= 1;
    }
    coeff
}

/// OpenFHE `CoeffEncodingCollapse` -> coeff\[s\]\[d\]\[k\], s = 0..levelBudget (index 0 = the remainder stage when
/// layers do not divide log2 slots).
pub fn coeff_encoding_collapse(
    pows: &[C64],
    rot_group: &[u32],
    level_budget: u32,
    flag_i: bool,
) -> Result<Vec<Vec<Vec<C64>>>> {
    let slots = rot_group.len() as u32;
    if slots == 0 || level_budget == 0 {
        return Err(Error::BtsSetup("coeff_encoding_collapse: empty rotGroup or zero budget".into()));
    }
    let log2slots = slots.ilog2();
    let (layers, _rows, rem) = select_layers(log2slots, level_budget);
    let dim_collapse = level_budget as i64;
    let flag_rem = (rem != 0) as i64;
    let num_rot = ((1u32 << (layers + 1)) - 1) as usize;
    let num_rot_rem = ((1u32 << (rem + 1)) - 1) as usize;
    let c1 = coeff_encoding_one_level(pows, rot_group, flag_i);
    let su = slots as usize;
    let ls = log2slots as usize;
    let mut coeff = vec![vec![vec![C64::ZERO; su]; num_rot]; level_budget as usize];
    if flag_rem == 1 {
        coeff[0] = vec![vec![C64::ZERO; su]; num_rot_rem];
    }
    let collapse = |cs: &mut Vec<Vec<C64>>, top: usize, nl: u32, nz: usize| {
        cs[0] = c1[top].clone();
        cs[1] = c1[top + ls].clone();
        cs[2] = c1[top + 2 * ls].clone();
        for l in 1..nl {
            let temp = std::mem::replace(cs, vec![vec![C64::ZERO; su]; nz]);
            let sh = 1i64 << (top - l as usize);
            for u in 0..((1usize << (l + 1)) - 1) {
                for k in 0..su {
                    let a = c1[top - l as usize][k] * temp[u][rot_idx(k as u32, -sh, slots)];
                    cs[2 * u][k] += a;
                    let b = c1[top - l as usize + ls][k] * temp[u][k];
                    cs[2 * u + 1][k] += b;
                    let c = c1[top - l as usize + 2 * ls][k] * temp[u][rot_idx(k as u32, sh, slots)];
                    cs[2 * u + 2][k] += c;
                }
            }
        }
    };
    if layers > 0 {
        let mut s = dim_collapse - 1;
        while s >= flag_rem {
            let back = ((dim_collapse - 1 - s) as u32) * layers + 1;
            if log2slots < back {
                return Err(Error::BtsSetup("coeff_encoding_collapse: top can not be negative".into()));
            }
            let top = (log2slots - ((dim_collapse - 1 - s) as u32) * layers - 1) as usize;
            collapse(&mut coeff[s as usize], top, layers, num_rot);
            s -= 1;
        }
    }
    if flag_rem == 1 {
        let top = (log2slots - ((dim_collapse - 1) as u32) * layers - 1) as usize;
        collapse(&mut coeff[0], top, rem, num_rot_rem);
    }
    Ok(coeff)
}

/// OpenFHE `CoeffDecodingCollapse` -> coeff\[s\]\[d\]\[k\], s = 0..levelBudget (the last index = remainder stage).
pub fn coeff_decoding_collapse(
    pows: &[C64],
    rot_group: &[u32],
    level_budget: u32,
    flag_i: bool,
) -> Result<Vec<Vec<Vec<C64>>>> {
    let slots = rot_group.len() as u32;
    if slots == 0 || level_budget == 0 {
        return Err(Error::BtsSetup("coeff_decoding_collapse: empty rotGroup or zero budget".into()));
    }
    let log2slots = slots.ilog2();
    let (layers, rows, rem) = select_layers(log2slots, level_budget);
    let flag_rem = rem != 0;
    let num_rot = ((1u32 << (layers + 1)) - 1) as usize;
    let num_rot_rem = ((1u32 << (rem + 1)) - 1) as usize;
    let c1 = coeff_decoding_one_level(pows, rot_group, flag_i);
    let su = slots as usize;
    let ls = log2slots as usize;
    let mut coeff = vec![vec![vec![C64::ZERO; su]; num_rot]; level_budget as usize];
    if flag_rem {
        coeff[level_budget as usize - 1] = vec![vec![C64::ZERO; su]; num_rot_rem];
    }
    let collapse = |cs: &mut Vec<Vec<C64>>, s: usize, nl: u32, nz: usize| {
        let base = s * layers as usize;
        cs[0] = c1[base].clone();
        cs[1] = c1[ls + base].clone();
        cs[2] = c1[2 * ls + base].clone();
        for l in 1..nl {
            let temp = std::mem::replace(cs, vec![vec![C64::ZERO; su]; nz]);
            for t in 0..3usize {
                let shift = match t {
                    0 => 0usize,
                    1 => 1usize << l,
                    _ => 1usize << (l + 1),
                };
                for u in 0..((1usize << (l + 1)) - 1) {
                    for k in 0..su {
                        let a = c1[base + l as usize + t * ls][k] * temp[u][k];
                        cs[u + shift][k] += a;
                    }
                }
            }
        }
    };
    if layers > 0 {
        for s in 0..rows as usize {
            collapse(&mut coeff[s], s, layers, num_rot);
        }
    }
    if flag_rem {
        collapse(&mut coeff[rows as usize], rows as usize, rem, num_rot_rem);
    }
    Ok(coeff)
}

// ------------------------------------------------------------------------------------------------------------------
// Stages
// ------------------------------------------------------------------------------------------------------------------

/// One collapsed-FFT stage (OpenFHE stage index `s`). `diags[d]` is the d-th factor-matrix diagonal (scale folded in
/// on the stage OpenFHE scales), length `len` = slots (full packing) or 2 slots (sparse: [coeff | coeff_i]).
/// Semantics on a length-`len` slot vector x: y\[k\] = sum_d diags\[d\]\[k\] * x[(k + (d - center) * stride) mod len],
/// center = (num_diags - 1) / 2. OpenFHE evaluates it in BSGS form with the diagonals of giant step i pre-rotated
/// by -g i stride (`bsgs_diag`).
#[derive(Clone, Debug)]
pub struct FftStage {
    pub s: u32,
    pub stride: u32,
    /// OpenFHE giant steps (outer)
    pub b: u32,
    /// OpenFHE baby steps (inner)
    pub g: u32,
    pub diags: Vec<Vec<C64>>,
}

impl FftStage {
    pub fn num_diags(&self) -> usize {
        self.diags.len()
    }
    pub fn len(&self) -> usize {
        self.diags.first().map_or(0, |d| d.len())
    }
    pub fn center(&self) -> i64 {
        (self.diags.len() as i64 - 1) / 2
    }
    /// The plaintext vector OpenFHE encodes for diagonal `idx` (= g i + j): Rotate(diag, -g i stride).
    pub fn bsgs_diag(&self, idx: usize) -> Vec<C64> {
        let i = (idx / self.g as usize) as i64;
        rotate_left(&self.diags[idx], -(self.g as i64) * i * self.stride as i64)
    }
    /// OpenFHE rot_in\[j\] = ReduceRotation((j - (numRot + 1) / 2 + 1) stride, red), j < g.
    pub fn rot_in(&self, red: u32) -> Vec<u32> {
        let h = (self.diags.len() as i64 + 1) / 2;
        (0..self.g as i64).map(|j| reduce_rotation((j - h + 1) * self.stride as i64, red)).collect()
    }
    /// OpenFHE rot_out\[i\] = ReduceRotation(g i stride, M/4), i < b.
    pub fn rot_out(&self, m4: u32) -> Vec<u32> {
        (0..self.b as i64).map(|i| reduce_rotation(self.g as i64 * i * self.stride as i64, m4)).collect()
    }
    /// Apply the stage to a vector (reference semantics, O(len * diags)).
    pub fn apply(&self, x: &[C64]) -> Vec<C64> {
        let len = x.len() as i64;
        let c = self.center();
        (0..x.len())
            .map(|k| {
                let mut acc = C64::ZERO;
                for (d, dv) in self.diags.iter().enumerate() {
                    let src = (k as i64 + (d as i64 - c) * self.stride as i64).rem_euclid(len) as usize;
                    acc += dv[k] * x[src];
                }
                acc
            })
            .collect()
    }
}

/// OpenFHE `Rotate(a, r)`: left rotation, out\[k\] = a[(k + r) mod len].
pub fn rotate_left(a: &[C64], r: i64) -> Vec<C64> {
    let n = a.len() as i64;
    if n == 0 {
        return Vec::new();
    }
    let r = r.rem_euclid(n) as usize;
    let mut v = Vec::with_capacity(a.len());
    v.extend_from_slice(&a[r..]);
    v.extend_from_slice(&a[..r]);
    v
}

/// All CtS + StC factors of one slot count.
#[derive(Clone, Debug)]
pub struct BtsFactors {
    pub n: usize,
    pub slots: u32,
    pub enc: CollapsedFftParams,
    pub dec: CollapsedFftParams,
    /// CtS stages by OpenFHE index s (evaluation runs s = budget-1 down to 0; stage 0 carries scale_enc)
    pub cts: Vec<FftStage>,
    /// StC stages by OpenFHE index s (evaluation runs s = 0 up; the last carries scale_dec)
    pub stc: Vec<FftStage>,
}

/// The collapsed-FFT factor values of `EvalBootstrapSetup(levelBudget, dim1, slots)` with `precompute` on, for ring
/// dimension `n` (M = 2n). `scale_enc` / `scale_dec` are the constants OpenFHE folds into the CtS stage 0 / last StC
/// stage (composite chains: scaleEnc = 1/K, scaleDec = q0 / sf(0); see `EvalBootstrapSetup`). Level budgets are
/// clamped to [1, log2 slots] as OpenFHE does. The single-LT (budget {1,1}) route is not built here.
pub fn cts_stc_factors(
    n: usize,
    slots: u32,
    level_budget: [u32; 2],
    dim1: [u32; 2],
    scale_enc: f64,
    scale_dec: f64,
) -> Result<BtsFactors> {
    if !n.is_power_of_two() || n < 8 {
        return Err(Error::BtsSetup(format!("cts_stc_factors: ring dimension n={n}")));
    }
    let m_big = 2 * n as u32;
    let slots = if slots == 0 { m_big / 4 } else { slots };
    if !slots.is_power_of_two() || slots > m_big / 4 || slots < 2 {
        return Err(Error::BtsSetup(format!("cts_stc_factors: slots={slots} for n={n}")));
    }
    let log_slots = if slots < 3 { 1 } else { slots.ilog2() };
    let budget = level_budget.map(|b| b.clamp(1, log_slots));
    if budget == [1, 1] {
        return Err(Error::BtsSetup("cts_stc_factors: level budget {1,1} is the single-LT route (not built here)".into()));
    }
    let enc = collapsed_fft_params(slots, budget[0], dim1[0])?;
    let dec = collapsed_fft_params(slots, budget[1], dim1[1])?;
    let m = 4 * slots;
    let rg = rot_group(slots, m);
    let pows = ksi_pows(m);
    let full = slots == m_big / 4;

    // CtS
    let flag_rem = (enc.rem_collapse != 0) as u32;
    let ce = coeff_encoding_collapse(&pows, &rg, enc.level_budget, false)?;
    let cei = if full { None } else { Some(coeff_encoding_collapse(&pows, &rg, enc.level_budget, true)?) };
    let mut cts = Vec::with_capacity(enc.level_budget as usize);
    for s in 0..enc.level_budget {
        let rem_stage = flag_rem == 1 && s == 0;
        let (b, g, nr) =
            if rem_stage { (enc.b_rem, enc.g_rem, enc.num_rotations_rem) } else { (enc.b, enc.g, enc.num_rotations) };
        let stride = if rem_stage { 1 } else { 1u32 << ((s - flag_rem) * enc.layers_collapse + enc.rem_collapse) };
        let scaled = s == 0; // "do the scaling only at the last set of coefficients" (flagRem: the remainder stage)
        let diags = (0..nr as usize)
            .map(|d| {
                let mut v = ce[s as usize][d].clone();
                if let Some(ci) = &cei {
                    v.extend_from_slice(&ci[s as usize][d]);
                }
                if scaled {
                    for x in v.iter_mut() {
                        *x = x.scale(scale_enc);
                    }
                }
                v
            })
            .collect();
        cts.push(FftStage { s, stride, b, g, diags });
    }

    // StC
    let flag_rem = (dec.rem_collapse != 0) as u32;
    let cd = coeff_decoding_collapse(&pows, &rg, dec.level_budget, false)?;
    let cdi = if full { None } else { Some(coeff_decoding_collapse(&pows, &rg, dec.level_budget, true)?) };
    let mut stc = Vec::with_capacity(dec.level_budget as usize);
    for s in 0..dec.level_budget {
        let rem_stage = flag_rem == 1 && s == dec.level_budget - 1;
        let (b, g, nr) =
            if rem_stage { (dec.b_rem, dec.g_rem, dec.num_rotations_rem) } else { (dec.b, dec.g, dec.num_rotations) };
        let stride = 1u32 << (s * dec.layers_collapse);
        let scaled = s == dec.level_budget - 1;
        let diags = (0..nr as usize)
            .map(|d| {
                let mut v = cd[s as usize][d].clone();
                if let Some(ci) = &cdi {
                    v.extend_from_slice(&ci[s as usize][d]);
                }
                if scaled {
                    for x in v.iter_mut() {
                        *x = x.scale(scale_dec);
                    }
                }
                v
            })
            .collect();
        stc.push(FftStage { s, stride, b, g, diags });
    }
    Ok(BtsFactors { n, slots, enc, dec, cts, stc })
}

impl BtsFactors {
    fn m4(&self) -> u32 {
        self.n as u32 / 2
    }

    /// OpenFHE `FindCoeffsToSlotsRotationIndices` (with duplicates, in OpenFHE's order).
    pub fn cts_rotation_indices(&self) -> Vec<u32> {
        let p = &self.enc;
        let flag_rem = (p.rem_collapse != 0) as i64;
        let mut v = Vec::new();
        let mut s = p.level_budget as i64 - 1;
        while s >= flag_rem {
            let sf = 1i64 << ((s - flag_rem) * p.layers_collapse as i64 + p.rem_collapse as i64);
            let half = 1 - (p.num_rotations as i64 + 1) / 2;
            for j in half..p.g as i64 + half {
                v.push(reduce_rotation(j * sf, self.slots));
            }
            for i in 0..p.b as i64 {
                v.push(reduce_rotation(p.g as i64 * i * sf, self.m4()));
            }
            s -= 1;
        }
        if flag_rem == 1 {
            let half = 1 - (p.num_rotations_rem as i64 + 1) / 2;
            for j in half..p.g_rem as i64 + half {
                v.push(reduce_rotation(j, self.slots));
            }
            for i in 0..p.b_rem as i64 {
                v.push(reduce_rotation(p.g_rem as i64 * i, self.m4()));
            }
        }
        self.push_sparse(&mut v);
        v
    }

    /// OpenFHE `FindSlotsToCoeffsRotationIndices` (with duplicates, in OpenFHE's order).
    pub fn stc_rotation_indices(&self) -> Vec<u32> {
        let p = &self.dec;
        let flag_rem = (p.rem_collapse != 0) as u32;
        let mut v = Vec::new();
        for s in 0..(p.level_budget - flag_rem) as i64 {
            let sf = 1i64 << (s * p.layers_collapse as i64);
            let half = 1 - (p.num_rotations as i64 + 1) / 2;
            for j in half..p.g as i64 + half {
                v.push(reduce_rotation(j * sf, self.m4()));
            }
            for i in 0..p.b as i64 {
                v.push(reduce_rotation(p.g as i64 * i * sf, self.m4()));
            }
        }
        if flag_rem == 1 {
            let s = (p.level_budget - 1) as i64;
            let sf = 1i64 << (s * p.layers_collapse as i64);
            let half = 1 - (p.num_rotations_rem as i64 + 1) / 2;
            for j in half..p.g_rem as i64 + half {
                v.push(reduce_rotation(j * sf, self.m4()));
            }
            for i in 0..p.b_rem as i64 {
                v.push(reduce_rotation(p.g_rem as i64 * i * sf, self.m4()));
            }
        }
        self.push_sparse(&mut v);
        v
    }

    fn push_sparse(&self, v: &mut Vec<u32>) {
        let m = self.slots * 4;
        let mm = 2 * self.n as u32;
        if m != mm {
            let mut j = 1;
            while j < mm / m {
                v.push(j * self.slots);
                j <<= 1;
            }
        }
    }

    /// OpenFHE `FindBootstrapRotationIndices`: the sorted set of CtS + StC rotations without 0 and M/4.
    pub fn bootstrap_rotation_indices(&self) -> Vec<u32> {
        let mut s: std::collections::BTreeSet<u32> = self.cts_rotation_indices().into_iter().collect();
        s.extend(self.stc_rotation_indices());
        s.remove(&0);
        s.remove(&self.m4());
        s.into_iter().collect()
    }
}

// ------------------------------------------------------------------------------------------------------------------
// FIDESlib stage layout
// ------------------------------------------------------------------------------------------------------------------

/// One LT stage as FIDESlib holds it (`BootstrapPrecomputation::LTstep` after `GetBootstrapIndexes`
/// and the loader's `ltFriendly`): `pts[i b_step + j]` multiplies rot_{rot_in\[j\]}(x) inside giant step i, which
/// is rotated by rot_out\[i\]. `b_step` = OpenFHE g, `g_step` = OpenFHE b.
#[derive(Clone, Debug)]
pub struct FidesStage {
    pub num_diags: usize,
    pub b_step: u32,
    pub g_step: u32,
    pub rot_in: Vec<i64>,
    pub rot_out: Vec<i64>,
    /// left rotation applied to every OpenFHE BSGS plaintext of this stage (`Plaintext::automorph(-acc_offset)`)
    pub pt_rotation: u32,
    /// slot vectors of the plaintexts: rotate_left(bsgs_diag(idx), pt_rotation)
    pub pts: Vec<Vec<C64>>,
}

/// CtS (FIDESlib order = OpenFHE s descending) and StC (s ascending) in FIDESlib's layout.
#[derive(Clone, Debug)]
pub struct FidesLayout {
    pub cts: Vec<FidesStage>,
    pub stc: Vec<FidesStage>,
    /// the accumulated offset after all stages (the last StC rotOut\[0\] before the modulo)
    pub acc_offset: i64,
}

/// C++ `%` on int (sign of the dividend).
fn cmod(a: i64, m: i64) -> i64 {
    a % m
}

impl BtsFactors {
    /// FIDESlib `GetBootstrapIndexes` + `ltFriendly` (MAKE_CTS_LT_FRIENDLY = MAKE_STC_LT_FRIENDLY = AFFINE_LT = true):
    /// rot_in shifted to start at 0 with its first-step spacing, rot_out re-based on the giant step 0 offset, and every
    /// plaintext pre-rotated by the running sum of those offsets so no extra rotation is needed between stages; the
    /// accumulated offset is paid once by the last StC stage's rot_out.
    pub fn fides_layout(&self) -> FidesLayout {
        let m4 = self.m4();
        let modv = std::cmp::min(2 * self.slots as i64, self.n as i64 / 2);
        let build = |st: &FftStage, red_in: u32| -> (Vec<i64>, Vec<i64>) {
            let mut ri: Vec<i64> = st.rot_in(red_in).into_iter().map(|x| x as i64).collect();
            let mut ro: Vec<i64> = st.rot_out(m4).into_iter().map(|x| x as i64).collect();
            let off = ri[0];
            for x in ri.iter_mut() {
                *x -= off;
            }
            for x in ro.iter_mut() {
                *x += off;
            }
            (ri, ro)
        };
        let fix_in = |ri: &mut Vec<i64>, b_step: usize, g_step: usize, nd: usize| {
            for i in 0..g_step {
                for j in 1..b_step {
                    if i * b_step + j < nd && ri.len() > 1 && ri[j] - ri[j - 1] != ri[1] - ri[0] {
                        ri[j] = ri[j - 1] + ri[1] - ri[0];
                    }
                }
            }
        };
        let mut acc: i64 = 0;
        let mut acc_load: i64 = 0;
        let mut out_cts = Vec::new();
        for st in self.cts.iter().rev() {
            let (mut ri, mut ro) = build(st, self.slots);
            // GetBootstrapIndexes: re-base rot_out[1..] on rot_out[0], repair rot_in spacing
            let off = ro[0];
            acc += ro[0];
            for x in ro.iter_mut().skip(1) {
                *x = cmod(*x - off, modv);
            }
            fix_in(&mut ri, st.g as usize, st.b as usize, st.num_diags());
            // loader ltFriendly: rot_out[0] -> 0, plaintexts automorph(-acc)
            acc_load += ro[0];
            ro[0] = 0;
            let rot = reduce_rotation(-acc_load, m4);
            let pts = (0..st.num_diags()).map(|d| rotate_left(&st.bsgs_diag(d), rot as i64)).collect();
            out_cts.push(FidesStage {
                num_diags: st.num_diags(),
                b_step: st.g,
                g_step: st.b,
                rot_in: ri,
                rot_out: ro,
                pt_rotation: rot,
                pts,
            });
        }
        let mut out_stc = Vec::new();
        let nstc = self.stc.len();
        for (si, st) in self.stc.iter().enumerate() {
            let (mut ri, mut ro) = build(st, m4);
            let off = ro[0];
            acc += ro[0];
            for x in ro.iter_mut().skip(1) {
                *x = cmod(*x - off, modv);
            }
            fix_in(&mut ri, st.g as usize, st.b as usize, st.num_diags());
            acc_load += ro[0];
            ro[0] = 0;
            let rot = reduce_rotation(-acc_load, m4);
            let pts = (0..st.num_diags()).map(|d| rotate_left(&st.bsgs_diag(d), rot as i64)).collect();
            if si + 1 == nstc {
                ro[0] = cmod(acc_load, modv);
                for x in ro.iter_mut().skip(1) {
                    *x = cmod(*x + acc_load, modv);
                }
            }
            out_stc.push(FidesStage {
                num_diags: st.num_diags(),
                b_step: st.g,
                g_step: st.b,
                rot_in: ri,
                rot_out: ro,
                pt_rotation: rot,
                pts,
            });
        }
        debug_assert_eq!(acc, acc_load);
        FidesLayout { cts: out_cts, stc: out_stc, acc_offset: acc_load }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// StC-first
// ------------------------------------------------------------------------------------------------------------------

/// Value factors of the StC-first route (FIDESlib FIDESLIB_BTS_STC_FIRST, `RawCiphertext.cu` loader): the shipped StC
/// stages run on the input side; mode 1 multiplies the LAST stage by 2^-deg (deg = log2 of the shipped scaleDec), mode 2
/// keeps stages 0..n-2 and builds the last per call with `stc_first_mode2_factor`. Returns the per-stage multiplier,
/// None for the mode-2 last stage (runtime-built).
pub fn stc_first_stage_factors(n_stages: usize, deg: i32, mode: u32) -> Result<Vec<Option<f64>>> {
    if n_stages == 0 || !(1..=2).contains(&mode) {
        return Err(Error::BtsSetup(format!("stc_first: n_stages={n_stages} mode={mode}")));
    }
    Ok((0..n_stages)
        .map(|si| {
            if si + 1 < n_stages {
                Some(1.0)
            } else if mode == 1 {
                Some((-deg as f64).exp2())
            } else {
                None
            }
        })
        .collect())
}

/// Mode-2 last-stage factor (FIDESlib `Bootstrap.cu`, StC-first prescaled path):
/// 2^-cf_stage * (target_sf / sf(d - 1)) * (sf(entry) / nf_in).
pub fn stc_first_mode2_factor(cf_stage: i32, target_sf: f64, sf_d_minus_1: f64, sf_entry: f64, nf_in: f64) -> f64 {
    (-cf_stage as f64).exp2() * (target_sf / sf_d_minus_1) * (sf_entry / nf_in)
}

// ------------------------------------------------------------------------------------------------------------------
// SparseB (eprint 2026/1023, Alg. 4/5)
// ------------------------------------------------------------------------------------------------------------------

/// SparseB geometry: n = 2 slots effective coefficients repeated r = N / n times, s = 2n / r diagonals, BSGS baby
/// step b_step = min(s, 16) (override with `b_step_override` > 0), g_step = ceil(s / b_step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseBGeom {
    pub n: u32,
    pub r: u32,
    pub s: u32,
    pub b_step: u32,
    pub g_step: u32,
}

/// FIDESlib `sparseBGeometry`: None when the route does not qualify (dense, or n <= r / 2).
pub fn sparse_b_geometry(ring_n: u32, slots: u32, b_step_override: u32) -> Option<SparseBGeom> {
    if slots == ring_n / 2 || slots < 1 {
        return None;
    }
    let n = 2 * slots;
    let r = ring_n / n;
    if n <= r / 2 {
        return None;
    }
    let s = 2 * n / r;
    let b = if b_step_override > 0 { s.min(b_step_override) } else { s.min(16) };
    Some(SparseBGeom { n, r, s, b_step: b, g_step: s.div_ceil(b) })
}

/// FIDESlib `buildSparseB` host vectors (length N/2 each): CtS diagonals P_k and StC diagonals Q_k, already BSGS
/// pre-rotated by -(k / b_step) b_step (out\[l\] = in[l - shift]) so that sum_k d_k (.) rot_k(ct) is the transform.
/// U(j, l) = ksi^(l 5^j), ksi = exp(2 pi i / (4 slots)); P uses gc conj(U)^T, Q uses gd U (gc = K_raise / 24 *
/// 2^-t shift tag, gd = q0 / sf(top) — supplied by the caller).
pub fn sparse_b_diagonals(ring_n: u32, slots: u32, geom: &SparseBGeom, gc: f64, gd: f64) -> (Vec<Vec<C64>>, Vec<Vec<C64>>) {
    let n2 = (ring_n / 2) as usize;
    let m = 4 * slots;
    let mmask = m - 1;
    let rg = rot_group(slots, m);
    let ksi: Vec<C64> =
        (0..m).map(|k| C64::new((2.0 * PI * k as f64 / m as f64).cos(), (2.0 * PI * k as f64 / m as f64).sin())).collect();
    let u = |j: usize, l: usize| ksi[((l as u32).wrapping_mul(rg[j]) & mmask) as usize];
    let (n, r, s, b) = (geom.n as usize, geom.r as usize, geom.s as usize, geom.b_step as usize);
    let half = n / 2;
    let mut p_out = Vec::with_capacity(s);
    let mut q_out = Vec::with_capacity(s);
    for k in 0..s {
        let shift = (k / b) * b;
        let mut pk = vec![C64::ZERO; n2];
        let mut qk = vec![C64::ZERO; n2];
        for a in 0..r / 2 {
            for j in 0..half {
                let v = u((k + j) % half, a * s + (j % s)).conj().scale(gc);
                pk[a * n + j] = v;
                pk[a * n + half + j] = v;
                qk[a * n + j] = u(j, a * s + ((j + k) % s)).scale(gd);
                qk[a * n + half + j] = C64::ZERO;
            }
        }
        let rot = |src: &[C64]| -> Vec<C64> {
            (0..n2).map(|l| src[((l as i64 - shift as i64).rem_euclid(n2 as i64)) as usize]).collect()
        };
        p_out.push(rot(&pk));
        q_out.push(rot(&qk));
    }
    (p_out, q_out)
}
