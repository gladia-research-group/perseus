//! CKKS encoding on the host (seam "encode", host-FFT implementation).
//!
//! `encode_slots` follows OpenFHE `CKKSPackedEncoding::Encode`: the special inverse FFT
//! (`DiscreteFourierTransform::FFTSpecialInv`, ported with its rotation-group / ksi tables), scale, round half away
//! from zero, real parts at coefficient `gap*i` and imaginary parts at `gap*i + N/2`. The integer step is a redesign:
//! the rounded coefficient is held exactly as an i128 and reduced per prime (no `2^61` approx split, no
//! FitToNativeVector sign juggling of patch 0005), so any `|x * scale| < 2^126` is encoded exactly.
//! `decode_slots` recovers the integer coefficients by exact CRT over the bottom composite pair (`q_0..q_{d-1}`), so
//! it is valid while the message stays below half the bottom modulus. Rows are coefficient-domain (no NTT), one row
//! of N residues per prime `0..=limb_top`; primes are processed in parallel with rayon.
use crate::error::{Error, Result};
use crate::modmul::{invmod, mulmod};
use crate::params::Chain;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}
impl Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}
impl AddAssign for Complex {
    fn add_assign(&mut self, o: Complex) {
        *self = *self + o;
    }
}
impl Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl MulAssign for Complex {
    fn mul_assign(&mut self, o: Complex) {
        *self = *self * o;
    }
}
impl Mul<f64> for Complex {
    type Output = Complex;
    fn mul(self, s: f64) -> Complex {
        Complex::new(self.re * s, self.im * s)
    }
}
impl Div<f64> for Complex {
    type Output = Complex;
    fn div(self, s: f64) -> Complex {
        Complex::new(self.re / s, self.im / s)
    }
}

/// What the slots carry: `Real` drops imaginary parts on encode and returns zero imaginary parts on decode
/// (OpenFHE REAL); `Complex` keeps both (OpenFHE COMPLEX, the `CKKS_COMPLEX` payload).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Payload {
    Real,
    Complex,
}

/// Largest |coefficient| the exact path accepts (rounded coefficients live in i128; CRT sums need one spare bit).
const MAX_COEFF_BITS: i32 = 126;

pub struct Encoder {
    n: usize,
    /// `FFTSpecial`'s twiddles by stage: the stage of half-length h at [h - 1, 2h - 1), entry j the power
    /// (5^j mod 8h) M / 8h of exp(2 pi i / M) (PrecomputedValues' m_ksiPows at m_rotGroup; M = 2N)
    tw: Vec<Complex>,
    /// `FFTSpecialInv`'s: the powers (8h - 5^j mod 8h) M / 8h
    tw_inv: Vec<Complex>,
    q: Vec<u64>,
    /// primes in the bottom composite level (decode basis)
    d: usize,
}

impl Encoder {
    pub fn new(chain: &Chain) -> Encoder {
        let n = chain.n;
        let m = 2 * n;
        let nh = n / 2;
        let mut rot_group = Vec::with_capacity(nh);
        let mut five = 1usize;
        for _ in 0..nh {
            rot_group.push(five);
            five = five * 5 % m;
        }
        let mut ksi: Vec<Complex> = (0..m)
            .map(|j| {
                let angle = 2.0 * PI * j as f64 / m as f64;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();
        ksi.push(ksi[0]);
        let (mut tw, mut tw_inv) = (Vec::with_capacity(nh), Vec::with_capacity(nh));
        let mut lenh = 1;
        while lenh < nh {
            let lenq = 8 * lenh;
            let gap = m / lenq;
            tw.extend(rot_group[..lenh].iter().map(|&r| ksi[(r % lenq) * gap]));
            tw_inv.extend(rot_group[..lenh].iter().map(|&r| ksi[(lenq - r % lenq) * gap]));
            lenh *= 2;
        }
        Encoder { n, tw, tw_inv, q: chain.q.clone(), d: chain.level_unit }
    }

    pub fn ring_dim(&self) -> usize {
        self.n
    }

    fn check_slots(&self, slots: usize) -> Result<()> {
        if slots == 0 || !slots.is_power_of_two() || slots > self.n / 2 {
            return Err(Error::Encode(format!("slots = {slots} must be a power of two in [1, N/2 = {}]", self.n / 2)));
        }
        Ok(())
    }

    /// DiscreteFourierTransform::FFTSpecialInv (includes the 1/len normalisation).
    pub fn fft_special_inv(&self, vals: &mut [Complex]) {
        let size = vals.len();
        let mut len = size;
        while len >= 2 {
            let lenh = len >> 1;
            let tw = &self.tw_inv[lenh - 1..2 * lenh - 1];
            for block in vals.chunks_exact_mut(len) {
                let (lo, hi) = block.split_at_mut(lenh);
                for ((a, b), &w) in lo.iter_mut().zip(hi).zip(tw) {
                    let u = *a + *b;
                    let mut v = *a - *b;
                    v *= w;
                    *a = u;
                    *b = v;
                }
            }
            len >>= 1;
        }
        bit_reverse(vals);
        for v in vals.iter_mut() {
            *v = *v / size as f64;
        }
    }

    /// DiscreteFourierTransform::FFTSpecial.
    pub fn fft_special(&self, vals: &mut [Complex]) {
        bit_reverse(vals);
        let size = vals.len();
        let mut len = 2;
        while len <= size {
            let lenh = len >> 1;
            let tw = &self.tw[lenh - 1..2 * lenh - 1];
            for block in vals.chunks_exact_mut(len) {
                let (lo, hi) = block.split_at_mut(lenh);
                for ((a, b), &w) in lo.iter_mut().zip(hi).zip(tw) {
                    let u = *a;
                    let v = *b * w;
                    *a = u + v;
                    *b = u - v;
                }
            }
            len <<= 1;
        }
    }

    fn round_exact(x: f64, what: &str) -> Result<i128> {
        let r = x.round();
        if !r.is_finite() || r.abs() >= 2f64.powi(MAX_COEFF_BITS) {
            return Err(Error::Encode(format!(
                "{what}: scaled value {x:e} exceeds 2^{MAX_COEFF_BITS}; decrease the scaling factor"
            )));
        }
        Ok(r as i128)
    }

    fn rows_from_coeffs(&self, c: &[i128], limb_top: usize) -> Result<Vec<Vec<u64>>> {
        if limb_top >= self.q.len() {
            return Err(Error::Encode(format!("limb_top {limb_top} beyond the chain's top limb {}", self.q.len() - 1)));
        }
        Ok(self.q[..=limb_top]
            .par_iter()
            // 64-bit remainders where the coefficient fits (the i128 division is the encode's cost)
            .map(|&qi| c.iter().map(|&x| i64::try_from(x).map_or_else(|_| x.rem_euclid(qi as i128) as u64, |v| v.rem_euclid(qi as i64) as u64)).collect())
            .collect())
    }

    /// Integer coefficients (length N) of the encoding of `values` padded to `slots`, scaled by `scale`.
    pub fn slot_coeffs(&self, values: &[Complex], slots: usize, payload: Payload, scale: f64) -> Result<Vec<i128>> {
        self.check_slots(slots)?;
        if values.len() > slots {
            return Err(Error::Encode(format!("{} values do not fit {slots} slots", values.len())));
        }
        let mut inv = vec![Complex::default(); slots];
        for (o, v) in inv.iter_mut().zip(values) {
            *o = match payload {
                Payload::Real => Complex::new(v.re, 0.0),
                Payload::Complex => *v,
            };
        }
        self.fft_special_inv(&mut inv);
        let gap = self.n / (2 * slots);
        let mut c = vec![0i128; self.n];
        for (i, v) in inv.iter().enumerate() {
            c[gap * i] = Self::round_exact(v.re * scale, "encode")?;
            c[gap * i + self.n / 2] = Self::round_exact(v.im * scale, "encode")?;
        }
        Ok(c)
    }

    /// Encode `values` (at most `slots`, zero-padded) at scale `scale` into coefficient-domain rows over the primes
    /// `0..=limb_top`.
    pub fn encode_slots(
        &self,
        values: &[Complex],
        slots: usize,
        limb_top: usize,
        payload: Payload,
        scale: f64,
    ) -> Result<Vec<Vec<u64>>> {
        let c = self.slot_coeffs(values, slots, payload, scale)?;
        self.rows_from_coeffs(&c, limb_top)
    }

    /// Exact centered integer coefficients from the bottom composite pair of `rows` (rows of primes `0..d`).
    pub fn crt_bottom(&self, rows: &[Vec<u64>]) -> Result<Vec<i128>> {
        if rows.len() < self.d {
            return Err(Error::Encode(format!("decode needs the bottom {} rows, got {}", self.d, rows.len())));
        }
        let qs = &self.q[..self.d];
        let big: u128 = qs.iter().fold(1u128, |a, &x| a.saturating_mul(x as u128));
        if big >= 1u128 << MAX_COEFF_BITS {
            return Err(Error::Encode(format!("bottom modulus of {} primes exceeds 2^{MAX_COEFF_BITS}", self.d)));
        }
        let hats: Vec<(u128, u64)> = (0..self.d)
            .map(|i| {
                let hat = big / qs[i] as u128;
                (hat, invmod((hat % qs[i] as u128) as u64, qs[i]))
            })
            .collect();
        let half = big / 2;
        let n = rows[0].len();
        Ok((0..n)
            .into_par_iter()
            .map(|j| {
                let mut x: u128 = 0;
                for (i, &(hat, hinv)) in hats.iter().enumerate() {
                    let t = mulmod(rows[i][j] % qs[i], hinv, qs[i]) as u128;
                    x = (x + t * hat) % big;
                }
                if x > half {
                    -((big - x) as i128)
                } else {
                    x as i128
                }
            })
            .collect())
    }

    /// Decode `slots` values from coefficient-domain rows (only the bottom composite pair is read).
    pub fn decode_slots(&self, rows: &[Vec<u64>], slots: usize, payload: Payload, scale: f64) -> Result<Vec<Complex>> {
        self.decode_centered(&self.crt_bottom(rows)?, slots, payload, scale)
    }

    /// `decode_slots` from the plaintext's centred coefficients.
    pub fn decode_centered(&self, c: &[i128], slots: usize, payload: Payload, scale: f64) -> Result<Vec<Complex>> {
        self.check_slots(slots)?;
        if c.len() != self.n {
            return Err(Error::Encode(format!("rows have {} coefficients, ring dimension is {}", c.len(), self.n)));
        }
        let gap = self.n / (2 * slots);
        // through i64 where the coefficient fits (the same double, without the 128-bit conversion)
        let f = |x: i128| i64::try_from(x).map_or(x as f64, |v| v as f64) / scale;
        let mut v: Vec<Complex> = (0..slots).map(|i| Complex::new(f(c[gap * i]), f(c[gap * i + self.n / 2]))).collect();
        self.fft_special(&mut v);
        if payload == Payload::Real {
            v.iter_mut().for_each(|x| x.im = 0.0);
        }
        Ok(v)
    }

    /// Coefficient encoding (no FFT): coefficient `j` is `round(coeffs[j] * scale)`, zero-padded to N.
    pub fn encode_coeff_rows(&self, coeffs: &[f64], limb_top: usize, scale: f64) -> Result<Vec<Vec<u64>>> {
        if coeffs.len() > self.n {
            return Err(Error::Encode(format!("{} coefficients exceed ring dimension {}", coeffs.len(), self.n)));
        }
        let mut c = vec![0i128; self.n];
        for (o, &x) in c.iter_mut().zip(coeffs) {
            *o = Self::round_exact(x * scale, "encode_coeff_rows")?;
        }
        self.rows_from_coeffs(&c, limb_top)
    }

    /// Inverse of `encode_coeff_rows` over the bottom composite pair.
    pub fn decode_coeff_rows(&self, rows: &[Vec<u64>], scale: f64) -> Result<Vec<f64>> {
        Ok(self.crt_bottom(rows)?.into_iter().map(|x| x as f64 / scale).collect())
    }
}

/// DiscreteFourierTransform::BitReverse.
fn bit_reverse(vals: &mut [Complex]) {
    let size = vals.len();
    let mut j = 0usize;
    for i in 1..size {
        let mut bit = size >> 1;
        while j >= bit {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if i < j {
            vals.swap(i, j);
        }
    }
}
