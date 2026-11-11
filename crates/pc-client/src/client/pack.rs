//! Packed frames: a bottom-level ciphertext switched to the pack secret `t` so that its c1 lies in the subring
//! Z[X^k] (k = `sub`), n / k coefficients on the wire instead of n.
//!
//! The switch splits c1 (raised to the bottom key's modulus Q P') by coefficient class: d_j = the coefficients of c1
//! at the exponents = j mod k, so c1 = sum_j d_j. Key j is (b_j, a'_j) with a'_j = X^-j a_j(X^k), a_j uniform over
//! the n / k coefficients of the subring, and b_j = e_j - a'_j t + P' s on the Q rows (e_j - a'_j t on the special
//! ones). Then sum_j d_j a'_j has exponents = 0 mod k only, and
//!   P' c0 + sum_j d_j b_j + (sum_j d_j a'_j) t = P' (c0 + c1 s) + sum_j d_j e_j   (mod Q P'),
//! a ciphertext of the same plaintext under t after the division by P'. Each b_j's coefficient class i is an RLWE
//! sample of the ring of n / k coefficients at Q P' under t's class i (t = sum_i X^i t_i(X^k), dense ternary).
//!
//! The client decrypts on the small ring: c1(X^k) t = sum_i X^i (c1 t_i)(X^k).
use super::{mm, Client};
use crate::error::{Error, Result};
use crate::ntt::Ntt;
use crate::wire::CtWire;
use pc_core::keys::Prng;
use pc_core::params::Chain;
use rayon::prelude::*;

/// Row `r` (the `r`-th of the key's rows, prime `p`) of a'_j = X^-j a_j(X^k) in coefficient form: a_j's n / k
/// coefficients drawn from `seed` (stream `j << 8 | r`), coefficient i at exponent k i - j (i = 0: -a at n - j).
pub fn pack_a_row(seed: &[u32; 8], j: usize, r: usize, p: u64, k: usize, out: &mut [u32]) {
    let n = out.len();
    let mut a = vec![0u32; n / k];
    Prng::from_seed(*seed, (j << 8 | r) as u32).fill_below(p, &mut a);
    out.fill(0);
    for (i, &v) in a.iter().enumerate() {
        if i == 0 && j > 0 {
            out[n - j] = if v == 0 { 0 } else { (p - v as u64) as u32 };
        } else {
            out[k * i - j] = v;
        }
    }
}

/// Prime ids of a pack key's rows: the bottom level's Q ids, then the bottom key's special ids.
pub fn pack_row_ids(chain: &Chain) -> Vec<usize> {
    (0..chain.level_unit).chain(chain.q.len()..chain.q.len() + chain.p_sparse).collect()
}

/// The `a` half of a pack key of stride `k` from its seed, `[k][row][N]` words in eval form over `pack_row_ids`
/// (the server's side of `pack_keygen`).
pub fn pack_a_eval(chain: &Chain, k: usize, seed: &[u32; 8]) -> Result<Vec<u32>> {
    let n = chain.n;
    if k < 2 || !k.is_power_of_two() || n % k != 0 || k > 256 {
        return Err(Error::Params(format!("pack: subring stride {k} in a ring of {n}")));
    }
    let (primes, roots) = (chain.primes(), chain.roots());
    let ids = pack_row_ids(chain);
    let rows = ids.len();
    let ntt = ids.iter().map(|&id| Ntt::new(primes[id], roots[id], n)).collect::<Result<Vec<_>>>()?;
    let mut out = vec![0u32; k * rows * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, a)| {
        let (j, r) = (i / rows, i % rows);
        pack_a_row(seed, j, r, primes[ids[r]], k, a);
        ntt[r].fwd(a);
    });
    Ok(out)
}

/// Negacyclic products of real polynomials of 2 n coefficients in double precision: a polynomial is folded to the n
/// complex points (a_j + i a_{j + n}) zeta^j (zeta = e^{i pi / 2n}), whose DFT is its values at zeta^(4 k + 1). Real
/// and imaginary parts in separate arrays, so that the loops vectorize.
struct NegFft {
    /// zeta^j, j < n.
    tw: [Vec<f64>; 2],
    /// e^{i pi j / h}, j < h, at h - 1 + j, for the half block lengths h = 1, 2, .., n / 2.
    w: [Vec<f64>; 2],
}

/// 1.5 * 2^52: x + MAGIC holds round(x) in its low mantissa bits (two's complement) for |x| < 2^51.
const MAGIC: f64 = 6755399441055744.0;

impl NegFft {
    fn new(n: usize) -> Self {
        let angle = |j: usize, m: usize| std::f64::consts::PI * j as f64 / m as f64;
        let hs = || (0..n.trailing_zeros()).flat_map(|l| (0..1usize << l).map(move |j| (j, 1usize << l)));
        NegFft {
            tw: [(0..n).map(|j| angle(j, 2 * n).cos()).collect(), (0..n).map(|j| angle(j, 2 * n).sin()).collect()],
            w: [hs().map(|(j, h)| angle(j, h).cos()).collect(), hs().map(|(j, h)| angle(j, h).sin()).collect()],
        }
    }

    /// In place, the values (bit-reversed order) of the polynomial with coefficients `re` (0..n) and `im` (n..2 n).
    fn fwd(&self, re: &mut [f64], im: &mut [f64]) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 is present (checked at run time)
                unsafe { fwd_avx2(self, re, im) };
                return;
            }
        }
        fwd_plain(self, re, im);
    }

    /// The coefficients, rounded and masked, of the product of the polynomials with the values `f` and `t` (`t`
    /// scaled by 1 / n).
    fn prod(&self, f: &[Vec<f64>; 2], t: &[Vec<f64>; 2], mask: u64, out: &mut [u64]) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 is present (checked at run time)
                unsafe { prod_avx2(self, f, t, mask, out) };
                return;
            }
        }
        prod_plain(self, f, t, mask, out);
    }
}

#[inline(always)]
fn fwd_plain(t: &NegFft, re: &mut [f64], im: &mut [f64]) {
    let n = t.tw[0].len();
    for ((a, b), (&c, &s)) in re.iter_mut().zip(im.iter_mut()).zip(t.tw[0].iter().zip(&t.tw[1])) {
        (*a, *b) = (*a * c - *b * s, *a * s + *b * c);
    }
    let mut h = n / 2;
    while h > 0 {
        let (wr, wi) = (&t.w[0][h - 1..2 * h - 1], &t.w[1][h - 1..2 * h - 1]);
        match h {
            1 => stage::<1>(re, im, wr, wi, false),
            2 => stage::<2>(re, im, wr, wi, false),
            4 => stage::<4>(re, im, wr, wi, false),
            8 => stage::<8>(re, im, wr, wi, false),
            _ => {
                for (br, bi) in re.chunks_exact_mut(2 * h).zip(im.chunks_exact_mut(2 * h)) {
                    let ((ur, vr), (ui, vi)) = (br.split_at_mut(h), bi.split_at_mut(h));
                    for ((((ur, vr), ui), vi), (&c, &s)) in ur.iter_mut().zip(vr).zip(ui).zip(vi).zip(wr.iter().zip(wi)) {
                        let (xr, xi) = (*ur - *vr, *ui - *vi);
                        (*ur, *ui) = (*ur + *vr, *ui + *vi);
                        (*vr, *vi) = (xr * c - xi * s, xr * s + xi * c);
                    }
                }
            }
        }
        h /= 2;
    }
}

/// One stage over blocks of 2 H points, H small (the block loaded, its butterflies unrolled): `inv` the inverse
/// transform's (v times the conjugate root, then the sum and the difference), else the forward one's.
#[inline(always)]
fn stage<const H: usize>(re: &mut [f64], im: &mut [f64], wr: &[f64], wi: &[f64], inv: bool) {
    let (wr, wi): ([f64; H], [f64; H]) = (wr.try_into().unwrap(), wi.try_into().unwrap());
    for (br, bi) in re.chunks_exact_mut(2 * H).zip(im.chunks_exact_mut(2 * H)) {
        let ((ur, vr), (ui, vi)) = (br.split_at_mut(H), bi.split_at_mut(H));
        let (a, b, c, d): ([f64; H], [f64; H], [f64; H], [f64; H]) = ((&*ur).try_into().unwrap(), (&*ui).try_into().unwrap(), (&*vr).try_into().unwrap(), (&*vi).try_into().unwrap());
        for j in 0..H {
            if inv {
                let (yr, yi) = (c[j] * wr[j] + d[j] * wi[j], d[j] * wr[j] - c[j] * wi[j]);
                (ur[j], ui[j], vr[j], vi[j]) = (a[j] + yr, b[j] + yi, a[j] - yr, b[j] - yi);
            } else {
                let (xr, xi) = (a[j] - c[j], b[j] - d[j]);
                (ur[j], ui[j], vr[j], vi[j]) = (a[j] + c[j], b[j] + d[j], xr * wr[j] - xi * wi[j], xr * wi[j] + xi * wr[j]);
            }
        }
    }
}

#[inline(always)]
fn prod_plain(t: &NegFft, f: &[Vec<f64>; 2], g: &[Vec<f64>; 2], mask: u64, out: &mut [u64]) {
    let n = t.tw[0].len();
    let mut re: Vec<f64> = f[0].iter().zip(&f[1]).zip(g[0].iter().zip(&g[1])).map(|((&a, &b), (&c, &d))| a * c - b * d).collect();
    let mut im: Vec<f64> = f[0].iter().zip(&f[1]).zip(g[0].iter().zip(&g[1])).map(|((&a, &b), (&c, &d))| a * d + b * c).collect();
    let mut h = 1;
    while h < n {
        let (wr, wi) = (&t.w[0][h - 1..2 * h - 1], &t.w[1][h - 1..2 * h - 1]);
        match h {
            1 => stage::<1>(&mut re, &mut im, wr, wi, true),
            2 => stage::<2>(&mut re, &mut im, wr, wi, true),
            4 => stage::<4>(&mut re, &mut im, wr, wi, true),
            8 => stage::<8>(&mut re, &mut im, wr, wi, true),
            _ => {
                for (br, bi) in re.chunks_exact_mut(2 * h).zip(im.chunks_exact_mut(2 * h)) {
                    let ((ur, vr), (ui, vi)) = (br.split_at_mut(h), bi.split_at_mut(h));
                    for ((((ur, vr), ui), vi), (&c, &s)) in ur.iter_mut().zip(vr).zip(ui).zip(vi).zip(wr.iter().zip(wi)) {
                        let (yr, yi) = (*vr * c + *vi * s, *vi * c - *vr * s);
                        (*vr, *vi) = (*ur - yr, *ui - yi);
                        (*ur, *ui) = (*ur + yr, *ui + yi);
                    }
                }
            }
        }
        h *= 2;
    }
    let (lo, hi) = out.split_at_mut(n);
    for (((lo, hi), (&a, &b)), (&c, &s)) in lo.iter_mut().zip(hi).zip(re.iter().zip(&im)).zip(t.tw[0].iter().zip(&t.tw[1])) {
        *lo = (a * c + b * s + MAGIC).to_bits() & mask;
        *hi = (b * c - a * s + MAGIC).to_bits() & mask;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fwd_avx2(t: &NegFft, re: &mut [f64], im: &mut [f64]) {
    fwd_plain(t, re, im)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prod_avx2(t: &NegFft, f: &[Vec<f64>; 2], g: &[Vec<f64>; 2], mask: u64, out: &mut [u64]) {
    prod_plain(t, f, g, mask, out)
}

/// A packed frame's plaintext: coefficient k x + i from `p[i][x]` (c1 t_i as a word of `b1` bits) and c0's word
/// `y[k x + i]` of `b0` bits, round((p / 2^b1 + y / 2^b0) q) mod q, centred.
fn pack_combine(p: &[&[u64]], y: &[u64], out: &mut [i128], q: u64, b0: u32, b1: u32) {
    let k = p.len();
    // the sum in units of 2^-63 (below 2^64): its product with q is the quotient from bit 63 up
    let (s0, s1, h) = (63 - b0, 63 - b1, (q / 2) as i64);
    for (x, (o, y)) in out.chunks_exact_mut(k).zip(y.chunks_exact(k)).enumerate() {
        for ((o, &y), p) in o.iter_mut().zip(y).zip(p) {
            let w = ((p[x] << s1) + (y << s0)) as u128 * q as u128;
            // at most 2 q: less q once past q / 2, twice past 3 q / 2 (counted: the compares are coin flips)
            let v = ((w >> 63) as u64 + (w as u64 >> 62 & 1)) as i64;
            *o = (v - q as i64 * ((v > h) as i64 + (v > h + q as i64) as i64)) as i128;
        }
    }
}

/// The pack secret on the small ring: the values (`NegFft`) of its `k` coefficient classes t_i.
pub(super) struct PackTabs {
    k: usize,
    fft: NegFft,
    t: Vec<[Vec<f64>; 2]>,
}

/// The smallest ring degree N / k a pack key's coefficient classes may have (`Client::pack_keygen`).
pub const PACK_MIN_DEGREE: usize = 8192;

impl Client {
    /// Prime ids of a pack key's rows: the bottom level's Q ids, then the bottom key's special ids.
    pub fn pack_ids(&self) -> Vec<usize> {
        pack_row_ids(&self.chain)
    }

    fn pack_secret(&self, k: usize) -> Result<&[i8]> {
        let n = self.chain.n;
        if k < 2 || !k.is_power_of_two() || n % k != 0 || k > 256 {
            return Err(Error::Params(format!("pack: subring stride {k} in a ring of {n}")));
        }
        self.secret.pack.as_deref().ok_or_else(|| Error::Params("pack: the secret has no pack secret (ClientSecret::with_pack)".into()))
    }

    /// The pack key of stride `k` (module doc): `b`, `[k][row][N]` u32 words in eval form over `pack_ids`, its `a`
    /// rows `pack_a_row(seed, ..)`.
    /// Refused when the coefficient classes would fall below `PACK_MIN_DEGREE` (each class is an RLWE sample in
    /// degree N / k under a dense ternary secret: ~262 bits at the 112-bit bottom key modulus for N / k = 8192;
    /// smaller degrees are not analysed).
    pub fn pack_keygen(&self, k: usize, seed: &[u32; 8], prng: &mut Prng) -> Result<Vec<u32>> {
        if self.chain.n / k.max(1) < PACK_MIN_DEGREE {
            return Err(Error::Params(format!("pack: stride {k} leaves degree {} below {PACK_MIN_DEGREE}", self.chain.n / k.max(1))));
        }
        let t: Vec<i64> = self.pack_secret(k)?.iter().map(|&x| x as i64).collect();
        let (n, nq) = (self.chain.n, self.ring.nq);
        let ids = self.pack_ids();
        let rows = ids.len();
        let te: Vec<Vec<u32>> = ids.par_iter().map(|&id| self.ring.eval_row(&t, id)).collect();
        let errs: Vec<Vec<i64>> = (0..k).map(|_| self.gauss.sample_vec(prng, n)).collect();
        let pm = self.ring.gadget(self.ring.kps);
        let mut out = vec![0u32; k * rows * n];
        out.par_chunks_mut(n).enumerate().for_each(|(i, b)| {
            let (j, r) = (i / rows, i % rows);
            let id = ids[r];
            let p = self.ring.primes[id];
            let mut a = vec![0u32; n];
            pack_a_row(seed, j, r, p, k, &mut a);
            self.ring.ntt[id].fwd(&mut a);
            let e = self.ring.eval_row(&errs[j], id);
            let s = &self.s_eval[id];
            for x in 0..n {
                let mut v = e[x] as u64 + p - mm(a[x], te[r][x], p);
                if id < nq {
                    v += mm(pm[id], s[x], p);
                }
                b[x] = (v % p) as u32;
            }
        });
        Ok(out)
    }

    /// Max |e| of a pack key's relation b_j + a'_j t - P' s (checks).
    pub fn pack_relation_max_err(&self, k: usize, seed: &[u32; 8], words: &[u32]) -> Result<i64> {
        let t: Vec<i64> = self.pack_secret(k)?.iter().map(|&x| x as i64).collect();
        let (n, nq) = (self.chain.n, self.ring.nq);
        let ids = self.pack_ids();
        let rows = ids.len();
        if words.len() != k * rows * n {
            return Err(Error::Params(format!("pack key of {} words, stride {k} over {rows} rows", words.len())));
        }
        let te: Vec<Vec<u32>> = ids.par_iter().map(|&id| self.ring.eval_row(&t, id)).collect();
        let pm = self.ring.gadget(self.ring.kps);
        Ok(words
            .par_chunks(n)
            .enumerate()
            .map(|(i, b)| {
                let (j, r) = (i / rows, i % rows);
                let id = ids[r];
                let p = self.ring.primes[id];
                let mut a = vec![0u32; n];
                pack_a_row(seed, j, r, p, k, &mut a);
                self.ring.ntt[id].fwd(&mut a);
                let s = &self.s_eval[id];
                let mut e: Vec<u32> = (0..n).map(|x| ((b[x] as u64 + mm(a[x], te[r][x], p) + if id < nq { p - mm(pm[id], s[x], p) } else { 0 }) % p) as u32).collect();
                self.ring.ntt[id].inv(&mut e);
                e.iter().map(|&x| (x as i64).min(p as i64 - x as i64)).max().unwrap_or(0)
            })
            .max()
            .unwrap_or(0))
    }

    fn pack_tabs(&self, k: usize) -> Result<&PackTabs> {
        let t = self.pack_secret(k)?;
        let tabs = self.pack_tabs.get_or_init(|| {
            let h = self.chain.n / k / 2;
            let fft = NegFft::new(h);
            let te = (0..k)
                .map(|i| {
                    let half = |o: usize| (0..h).map(|m| t[k * (m + o) + i] as f64 / h as f64).collect::<Vec<f64>>();
                    let (mut re, mut im) = (half(0), half(h));
                    fft.fwd(&mut re, &mut im);
                    [re, im]
                })
                .collect();
            PackTabs { k, fft, t: te }
        });
        if tabs.k != k {
            return Err(Error::Params(format!("pack: frames of stride {k} after frames of stride {}", tabs.k)));
        }
        Ok(tabs)
    }

    /// A packed frame (`CtHeader::sub1`) at q0, centred: c0 + c1(X^k) t mod q0. c1 is y q0 / 2^b1 up to its rounding
    /// (y its frame words), so c1 t_i = (y t_i mod 2^b1) q0 / 2^b1 mod q0: the products of y with t's k coefficient
    /// classes on the small ring as exact integers (below 2^45: double-precision transforms), reduced mod 2^b1.
    pub(super) fn pack_centred(&self, ct: &CtWire, c: usize, b0: u32, b1: u32) -> Result<Vec<i128>> {
        let n = ct.header.n;
        let k = ct.header.sub1.unwrap_or(1) as usize;
        let tabs = self.pack_tabs(k)?;
        let ns = n / k;
        let q = self.ring.primes[0] as u128 * self.ring.primes[1] as u128;
        if b0 > 32 || b1 > 32 || q >> 62 != 0 {
            return Err(Error::Params(format!("pack: frame words of {b0} / {b1} bits at a modulus of {} bits", 128 - q.leading_zeros())));
        }
        let (w0, w1) = (crate::wire::packed_words(n, b0), crate::wire::packed_words(ns, b1));
        let base = c * (w0 + w1);
        let y1 = crate::wire::unpack_bits(&ct.data[base + w0..base + w0 + w1], ns, b1);
        let y0 = crate::wire::unpack_bits(&ct.data[base..base + w0], n, b0);
        let (half, wrap) = (1u64 << (b1 - 1), 2f64.powi(b1 as i32));
        let centred = |y: &[u64]| y.iter().map(|&y| if y >= half { y as f64 - wrap } else { y as f64 }).collect::<Vec<f64>>();
        let mut f = [centred(&y1[..ns / 2]), centred(&y1[ns / 2..])];
        let [re, im] = &mut f;
        tabs.fft.fwd(re, im);
        let mask = u64::MAX >> (64 - b1);
        // serial: a product is 40 us, a pool's wake-up several times that
        let prods: Vec<Vec<u64>> = (0..k)
            .map(|i| {
                let mut v = vec![0u64; ns];
                tabs.fft.prod(&f, &tabs.t[i], mask, &mut v);
                v
            })
            .collect();
        let p: Vec<&[u64]> = prods.iter().map(|p| &p[..]).collect();
        let mut out = vec![0i128; n];
        pack_combine(&p, &y0, &mut out, q as u64, b0, b1);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::ClientSecret;
    use crate::wire::{pack_frame_part, CtHeader};
    use pc_core::encode::{Complex, Payload};
    use pc_core::modmul::{invmod, mulmod};

    /// x in [0, prod q) from its residues (mixed radix).
    fn crt(q: &[u64], inv: &[u64], v: &[u64]) -> u128 {
        let (mut x, mut m) = (0u128, 1u128);
        for (i, &p) in q.iter().enumerate() {
            let t = mulmod((v[i] + p - (x % p as u128) as u64) % p, inv[i], p);
            x += m * t as u128;
            m *= p as u128;
        }
        x
    }

    /// The switch in exact host arithmetic (exact raise, exact rounded division by P'), framed with only c1's
    /// subring coefficients: the frame decrypts under the pack secret to the plaintext, up to the frame's rounding.
    #[test]
    fn pack_switch_on_the_host() {
        let mut g = Prng::from_seed([21, 1, 2, 3, 4, 5, 6, 7], 0);
        let sec = ClientSecret::generate_h(&mut g, 1 << 16, 72).unwrap().with_pack(&mut g);
        let sec = ClientSecret::from_bytes(&sec.to_bytes()).unwrap();
        let c = Client::new("n32/lp4b3/lp2b1", sec).unwrap();
        let (k, n, unit) = (8usize, c.chain.n, c.chain.level_unit);
        let seed = [9u32, 8, 7, 6, 5, 4, 3, 2];
        let key = c.pack_keygen(k, &seed, &mut g).unwrap();
        assert!(c.pack_relation_max_err(k, &seed, &key).unwrap() < 64);
        let ids = c.pack_ids();
        let rows = ids.len();
        let q: Vec<u64> = ids.iter().map(|&i| c.ring.primes[i]).collect();
        let inv: Vec<u64> = (0..rows).map(|i| invmod(q[..i].iter().fold(1u64, |a, &p| mulmod(a, p % q[i], q[i])), q[i])).collect();
        let level = (0..=64u32).rev().find(|&l| c.top_of(l).ok() == Some(unit - 1)).unwrap();
        let meta = c.fresh_meta(level, 1).unwrap();
        let v: Vec<Complex> = (0..n / 2).map(|i| Complex::new(((i * 37) % 101) as f64 / 50.0 - 1.0, ((i * 11) % 7) as f64 / 7.0)).collect();
        let ct = c.encrypt(&[v], &meta, Payload::Complex, &mut g).unwrap();
        let want = c.decrypt_coeffs(&ct).unwrap();
        // c1 raised to every row, coefficient form
        let c1: Vec<Vec<u32>> = (0..unit)
            .map(|r| {
                let mut x = ct.row(0, 1, r).to_vec();
                c.ring.ntt[r].inv(&mut x);
                x
            })
            .collect();
        let up: Vec<u128> = (0..n).map(|i| crt(&q[..unit], &inv[..unit], &(0..unit).map(|r| c1[r][i] as u64).collect::<Vec<_>>())).collect();
        let pm = c.ring.gadget(c.ring.kps);
        // the two dots, eval form per row, then coefficient form
        let acc: Vec<[Vec<u32>; 2]> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let (id, p) = (ids[r], q[r]);
                let mut s = [vec![0u64; n], vec![0u64; n]];
                for j in 0..k {
                    let mut d: Vec<u32> = (0..n).map(|i| if i % k == j { (up[i] % p as u128) as u32 } else { 0 }).collect();
                    c.ring.ntt[id].fwd(&mut d);
                    let mut a = vec![0u32; n];
                    pack_a_row(&seed, j, r, p, k, &mut a);
                    c.ring.ntt[id].fwd(&mut a);
                    let b = &key[(j * rows + r) * n..][..n];
                    for i in 0..n {
                        s[0][i] = (s[0][i] + mm(d[i], b[i], p)) % p;
                        s[1][i] = (s[1][i] + mm(d[i], a[i], p)) % p;
                    }
                }
                if r < unit {
                    let c0 = ct.row(0, 0, r);
                    for i in 0..n {
                        s[0][i] = (s[0][i] + mm(pm[id], c0[i], p)) % p;
                    }
                }
                s.map(|x| {
                    let mut x: Vec<u32> = x.into_iter().map(|v| v as u32).collect();
                    c.ring.ntt[id].inv(&mut x);
                    x
                })
            })
            .collect();
        // exact division by P', rounded, mod Q
        let big: u128 = q[..unit].iter().map(|&p| p as u128).product();
        let ps: u128 = q[unit..].iter().map(|&p| p as u128).product();
        let down = |part: usize| -> Vec<Vec<u32>> {
            let y: Vec<u128> = (0..n).map(|i| ((crt(&q, &inv, &(0..rows).map(|r| acc[r][part][i] as u64).collect::<Vec<_>>()) + ps / 2) / ps) % big).collect();
            (0..unit).map(|r| y.iter().map(|&x| (x % q[r] as u128) as u32).collect()).collect()
        };
        let (o0, o1) = (down(0), down(1));
        assert!(o1.iter().all(|row| row.iter().enumerate().all(|(i, &x)| i % k == 0 || x == 0)), "c1 off the subring");
        let sub: Vec<Vec<u32>> = o1.iter().map(|row| row.iter().step_by(k).copied().collect()).collect();
        for (b0, b1, bound) in [(32u32, 32u32, 34), (16, 28, 40), (16, 26, 40)] {
            let mut data = Vec::new();
            pack_frame_part(&q[..unit], &o0.iter().map(|r| r.as_slice()).collect::<Vec<_>>(), b0, &mut data).unwrap();
            pack_frame_part(&q[..unit], &sub.iter().map(|r| r.as_slice()).collect::<Vec<_>>(), b1, &mut data).unwrap();
            let header = CtHeader { bits: Some(b0), bits1: Some(b1), sub1: Some(k as u32), ..ct.header.clone() };
            let frame = CtWire::from_bytes(&CtWire { header, data }.to_bytes().unwrap()).unwrap();
            assert_eq!(frame.data.len(), crate::wire::packed_words(n, b0) + crate::wire::packed_words(n / k, b1));
            let got = c.decrypt_coeffs(&frame).unwrap();
            let err = got[0].iter().zip(&want[0]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) * meta.nf;
            eprintln!("bits {b0}/{b1}: {} words, error 2^{:.1}", frame.data.len(), err.log2());
            assert!(err < 2f64.powi(bound), "bits {b0}/{b1}: error 2^{:.1}", err.log2());
        }
    }

    /// Client cost of one packed frame (16 / 28 bits, stride 8) next to a sparse one (16 / 22, h = 72), random words:
    /// The double-precision products are the exact integer ones at the widest frame words (32 bits, dense ternary).
    #[test]
    fn neg_fft_products_are_exact() {
        let mut g = Prng::from_seed([5; 8], 0);
        let (n, b) = (1usize << 13, 32u32);
        let fft = NegFft::new(n / 2);
        let y: Vec<i64> = (0..n).map(|_| g.uniform_below(1 << b) as i64 - (1 << (b - 1))).collect();
        let t: Vec<i64> = (0..n).map(|_| g.uniform_below(3) as i64 - 1).collect();
        let vals = |a: &[i64], s: f64| {
            let (mut re, mut im): (Vec<f64>, Vec<f64>) = (a[..n / 2].iter().map(|&x| x as f64 * s).collect(), a[n / 2..].iter().map(|&x| x as f64 * s).collect());
            fft.fwd(&mut re, &mut im);
            [re, im]
        };
        let mut got = vec![0u64; n];
        fft.prod(&vals(&y, 1.0), &vals(&t, 2.0 / n as f64), u64::MAX >> (64 - b), &mut got);
        let mut want = vec![0i64; n];
        for (i, &a) in y.iter().enumerate() {
            for (j, &c) in t.iter().enumerate() {
                if i + j < n {
                    want[i + j] += a * c;
                } else {
                    want[i + j - n] -= a * c;
                }
            }
        }
        assert!(got.iter().zip(&want).all(|(&g, &w)| g == w as u64 & (u64::MAX >> (64 - b))));
    }

    /// `cargo test --release -p pc-client pack_frame_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn pack_frame_cost() {
        let mut g = Prng::from_seed([3; 8], 0);
        let (n, k) = (1usize << 16, 8usize);
        let c = Client::new("n32/lp4b3/lp2b1", ClientSecret::generate_h(&mut g, n, 72).unwrap().with_pack(&mut g)).unwrap();
        let meta = c.fresh_meta(52, 1).unwrap();
        let ct = c.encrypt(&[vec![Complex::new(0.5, 0.0); n / 2]], &meta, Payload::Complex, &mut g).unwrap();
        let mut frame = |len1: usize, b1: u32, sub1: Option<u32>| {
            let words = crate::wire::packed_words(n, 16) + crate::wire::packed_words(len1, b1);
            let data: Vec<u32> = (0..words).map(|_| g.uniform_below(1 << 32) as u32).collect();
            CtWire { header: CtHeader { bits: Some(16), bits1: Some(b1), sub1, sparse: sub1.is_none(), ..ct.header.clone() }, data }
        };
        let (packed, sparse) = (frame(n / k, 28, Some(k as u32)), frame(n, 22, None));
        c.pack_centred(&packed, 0, 16, 28).unwrap();
        let reps = 50;
        let time = |name: &str, f: &mut dyn FnMut()| {
            let t = std::time::Instant::now();
            (0..reps).for_each(|_| f());
            eprintln!("{name}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
        };
        time("pack_centred 16/28", &mut || drop(c.pack_centred(&packed, 0, 16, 28).unwrap()));
        time("sparse_centred 16/22", &mut || drop(c.sparse_centred(&sparse, 0, 16, 22)));
        time("decrypt packed", &mut || drop(c.decrypt(&packed, Payload::Complex).unwrap()));
        time("decrypt sparse", &mut || drop(c.decrypt(&sparse, Payload::Complex).unwrap()));
    }
}
