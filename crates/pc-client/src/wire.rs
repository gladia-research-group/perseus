//! The versioned wire format (docs/WIRE.md): the key bundle (header + raw payloads, streamed key by key) and the
//! ciphertext frame. Headers are JSON; payloads are little-endian u32 words, each followed by its `sum64` checksum.
use crate::error::{Error, Result};
use crate::spec::{KeyReq, KeyShape};
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom, Write};

pub const VERSION: u32 = 1;
pub const BUNDLE_MAGIC: &[u8; 4] = b"PCKB";
pub const CT_MAGIC: &[u8; 4] = b"PCCT";

/// FNV-1a over little-endian u64 words (a trailing odd u32 word is widened): the payload checksum.
pub fn sum64(words: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut it = words.chunks_exact(2);
    for p in &mut it {
        h ^= p[0] as u64 | (p[1] as u64) << 32;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let [x] = it.remainder() {
        h ^= *x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn write_words<Wr: Write>(w: &mut Wr, words: &[u32]) -> Result<()> {
    let mut buf = Vec::with_capacity(words.len().min(1 << 20) * 4);
    for c in words.chunks(1 << 20) {
        buf.clear();
        c.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes()));
        w.write_all(&buf)?;
    }
    Ok(())
}

pub fn read_words<R: Read>(r: &mut R, n: usize) -> Result<Vec<u32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn write_header<Wr: Write>(w: &mut Wr, magic: &[u8; 4], header: &[u8]) -> Result<()> {
    w.write_all(magic)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(header)?;
    Ok(())
}

fn read_header<R: Read>(r: &mut R, magic: &[u8; 4]) -> Result<Vec<u8>> {
    let mut m = [0u8; 4];
    r.read_exact(&mut m)?;
    if &m != magic {
        return Err(Error::Wire(format!("magic {m:?}, expected {:?}", std::str::from_utf8(magic).unwrap_or("?"))));
    }
    let mut v = [0u8; 4];
    r.read_exact(&mut v)?;
    let ver = u32::from_le_bytes(v);
    if ver != VERSION {
        return Err(Error::Wire(format!("version {ver}; this build reads version {VERSION}")));
    }
    let mut l = [0u8; 8];
    r.read_exact(&mut l)?;
    let len = u64::from_le_bytes(l);
    if len > 1 << 30 {
        return Err(Error::Wire(format!("header of {len} bytes")));
    }
    let mut h = vec![0u8; len as usize];
    r.read_exact(&mut h)?;
    Ok(h)
}

// ── key bundle ──────────────────────────────────────────────────────────────────────────────────────────────────

/// One bundle entry; `words` = u32 words of its payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Entry {
    /// A switching key: `b` packed at the bundle's `bits` (`[digit][slot][row_words]`), `a` = KSKA(seed).
    Ksk {
        #[serde(flatten)]
        key: KeyReq,
        seed: [u32; 8],
        words: usize,
    },
    /// The SPRU bootstrapping-key ciphertexts: `k` ciphertexts `[c][part][rows][N]`, eval form.
    SpruCs { k: usize, rows: usize, level: u32, nf: f64, words: usize },
    /// PCMM keys of shape t (coefficient form, `pc_core::pcmm_keys` layout): fwd, back, pack back to back.
    Pcmm { t: usize, primes: [u64; 3], split: [u64; 4], fwd: usize, back: usize, pack: usize, words: usize },
    /// CCMM keys (coefficient form, `pc_core::ccmm_keys` layout): fwd, back, autos, rlk back to back.
    Ccmm { primes: [u64; 5], fwd: usize, back: usize, autos: usize, rlk: usize, words: usize },
    /// Public key `(-a s + e, a)`: `[part][rows][N]`, eval form over Q ids `0..rows`.
    PublicKey { rows: usize, words: usize },
    /// A pack key (`Client::pack_keygen`): `b`, `[sub][row][N]` u32 words (dense, eval form; rows = the bottom
    /// level's Q ids then its special ids), `a` = `pack_a_row(seed, ..)`.
    PackKsk { sub: usize, rows: usize, seed: [u32; 8], words: usize },
}

impl Entry {
    pub fn words(&self) -> usize {
        match self {
            Entry::Ksk { words, .. }
            | Entry::SpruCs { words, .. }
            | Entry::Pcmm { words, .. }
            | Entry::Ccmm { words, .. }
            | Entry::PublicKey { words, .. }
            | Entry::PackKsk { words, .. } => *words,
        }
    }
}

/// The bundle header: chain identity, the plan contract, the packing, every entry in payload order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BundleHeader {
    pub chain: String,
    pub params_hash: String,
    pub contract: String,
    pub n: usize,
    /// Bits per packed `b` coefficient, and u32 words per packed row (one zero guard word, padded to 4).
    pub bits: u32,
    pub row_words: usize,
    pub entries: Vec<Entry>,
}

/// u32 words of one packed row of `n` values at `bits` (+ one zero guard word, padded to 16 bytes); the device's
/// `packed_row_words`.
pub fn packed_row_words(n: usize, bits: u32) -> usize {
    (n * bits as usize / 32 + 1).div_ceil(4) * 4
}

/// Words of a switching key's packed `b`.
pub fn ksk_words(shape: &KeyShape, row_words: usize) -> usize {
    shape.ndig * shape.rows() * row_words
}

/// Bit-pack one row: value i at bits [i*bits, (i+1)*bits) of a little-endian u32 stream, the rest of `out` zero.
pub fn pack_row(vals: &[u32], bits: u32, out: &mut [u32]) {
    out.iter_mut().for_each(|w| *w = 0);
    let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    for (i, &v) in vals.iter().enumerate() {
        let bit = i as u64 * bits as u64;
        let (w, off) = ((bit >> 5) as usize, (bit & 31) as u32);
        let x = ((v & mask) as u64) << off;
        out[w] |= x as u32;
        if off + bits > 32 {
            out[w + 1] |= (x >> 32) as u32;
        }
    }
}

/// Inverse of `pack_row`.
pub fn unpack_row(packed: &[u32], bits: u32, out: &mut [u32]) {
    let mask = if bits == 32 { u32::MAX as u64 } else { (1u64 << bits) - 1 };
    for (i, o) in out.iter_mut().enumerate() {
        let bit = i as u64 * bits as u64;
        let (w, off) = ((bit >> 5) as usize, bit & 31);
        let lo = packed[w] as u64 | (packed.get(w + 1).copied().unwrap_or(0) as u64) << 32;
        *o = ((lo >> off) & mask) as u32;
    }
}

/// Streams a bundle: the header, then each entry's payload + checksum in header order.
pub struct BundleWriter<Wr: Write> {
    w: Wr,
    header: BundleHeader,
    next: usize,
    pub bytes: u64,
}

impl<Wr: Write> BundleWriter<Wr> {
    pub fn new(mut w: Wr, header: BundleHeader) -> Result<Self> {
        let h = serde_json::to_vec(&header)?;
        write_header(&mut w, BUNDLE_MAGIC, &h)?;
        Ok(BundleWriter { w, header, next: 0, bytes: 16 + h.len() as u64 })
    }

    /// The next entry's payload (must be the header's next entry, of its declared size).
    pub fn put(&mut self, words: &[u32]) -> Result<()> {
        let e = self.header.entries.get(self.next).ok_or_else(|| Error::Wire(format!("entry {} is past the header's {}", self.next, self.header.entries.len())))?;
        if e.words() != words.len() {
            return Err(Error::Wire(format!("entry {} declares {} words, got {}", self.next, e.words(), words.len())));
        }
        write_words(&mut self.w, words)?;
        self.w.write_all(&sum64(words).to_le_bytes())?;
        self.bytes += 4 * words.len() as u64 + 8;
        self.next += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        if self.next != self.header.entries.len() {
            return Err(Error::Wire(format!("bundle closed after {} of {} entries", self.next, self.header.entries.len())));
        }
        self.w.flush()?;
        Ok(self.bytes)
    }
}

/// Reads a bundle file: the header at open, any entry's payload on demand (checksum verified).
pub struct BundleReader<R: Read + Seek> {
    r: R,
    pub header: BundleHeader,
    offsets: Vec<u64>,
}

impl BundleReader<std::io::BufReader<std::fs::File>> {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let f = std::fs::File::open(path).map_err(|e| Error::Wire(format!("{}: {e}", path.display())))?;
        Self::new(std::io::BufReader::with_capacity(1 << 22, f))
    }
}

impl<R: Read + Seek> BundleReader<R> {
    pub fn new(mut r: R) -> Result<Self> {
        let h = read_header(&mut r, BUNDLE_MAGIC)?;
        let header: BundleHeader = serde_json::from_slice(&h)?;
        let mut off = 16 + h.len() as u64;
        let mut offsets = Vec::with_capacity(header.entries.len());
        for e in &header.entries {
            offsets.push(off);
            off += 4 * e.words() as u64 + 8;
        }
        let end = r.seek(SeekFrom::End(0))?;
        if end != off {
            return Err(Error::Wire(format!("bundle is {end} bytes, its header describes {off}")));
        }
        Ok(BundleReader { r, header, offsets })
    }

    /// Payload of entry `i`.
    pub fn read(&mut self, i: usize) -> Result<Vec<u32>> {
        let e = self.header.entries.get(i).ok_or_else(|| Error::Wire(format!("no entry {i}")))?;
        let n = e.words();
        self.r.seek(SeekFrom::Start(self.offsets[i]))?;
        let v = read_words(&mut self.r, n)?;
        let mut c = [0u8; 8];
        self.r.read_exact(&mut c)?;
        if u64::from_le_bytes(c) != sum64(&v) {
            return Err(Error::Wire(format!("entry {i} ({:?}): checksum mismatch", std::mem::discriminant(e))));
        }
        Ok(v)
    }

    /// Total bytes of the bundle.
    pub fn bytes(&self) -> u64 {
        self.offsets.last().map_or(0, |&o| o + 4 * self.header.entries.last().map_or(0, |e| e.words()) as u64 + 8)
    }
}

// ── ciphertext frame ────────────────────────────────────────────────────────────────────────────────────────────

/// Ciphertext metadata as the runtime's `CtMeta` (`level` = primes dropped, `deg` 1 or 2, `nf` the scale, `slots`;
/// `tag` passed through untouched).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireMeta {
    pub level: u32,
    pub deg: u8,
    pub nf: f64,
    pub slots: u32,
    #[serde(default)]
    pub tag: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CtHeader {
    pub params_hash: String,
    pub n: usize,
    /// Ciphertexts in the batch.
    pub k: usize,
    /// Q rows per part (ids `0..rows`; rows = top limb + 1).
    pub rows: usize,
    pub meta: WireMeta,
    /// Compressed frame: each coefficient (coefficient form, the CRT of the `rows` residues) as round(x 2^bits / Q),
    /// `bits` <= 64 per coefficient per part (`pack_coeffs`); None = the eval-form residues of every row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits: Option<u32>,
    /// Seeded frame (a fresh encryption): per ciphertext, c0's eval rows then the `rows` 8-word seeds that c1's rows
    /// are drawn from (`Prng::from_seed(seed, 0).uniform_below(q_r)`, as `Client::encrypt` draws them).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub seeded: bool,
    /// Seeded with the random-access expander instead (`pc_core::keys::cts_expand_row(seed, q_r)`): the server may keep
    /// the seeds and draw c1 where it reads it (`import_ct(..., regen=True)`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub regen: bool,
    /// Compressed frame: c1's bits when they differ from c0's (`bits`); the words are bit-packed (`pack_bits`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits1: Option<u32>,
    /// Encrypted under the client's sparse secret (`SecretKey::s_sparse`: the server switched it before export).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub sparse: bool,
    /// Seeded frame with a short c0 (`Client::encrypt_pre(.., drop)`): c0 in coefficient form, each coefficient the CRT
    /// x in [0, Q) of its `rows` residues without its low `drop` bits, (x + 2^(drop - 1)) >> drop, bit-packed at `bits`
    /// bits (`pack_dropped`); the seeds follow. A rounding of the ciphertext: it adds noise below 2^(drop - 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drop: Option<u32>,
    /// c0's rows are coefficient form (a `drop` frame expanded): `import_ct` runs their forward transform.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub coeff0: bool,
    /// c1's rows (drawn from the seeds of a seeded frame) are coefficient form: a subring client's frame
    /// (`Client::sub`), which the server lifts without a transform of the small ring (`lifted`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub coeff1: bool,
    /// Packed frame (`export_ct_many(..., pack=)`): c1 lies in the subring Z[X^sub1] and only its `n / sub1`
    /// coefficients at the multiples of `sub1` are sent, at `bits1`; under the client's pack secret
    /// (`ClientSecret::pack`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub1: Option<u32>,
}

/// x mod p for any x, m = floor(2^64 / p): the quotient estimate is short by at most one.
#[inline(always)]
fn red(x: u64, p: u64, m: u64) -> u64 {
    let r = x.wrapping_sub(((x as u128 * m as u128) >> 64) as u64 * p);
    if r >= p { r - p } else { r }
}

/// `pack_coeffs` over two primes, Q < 2^63: Garner in 64 bits (x = v0 + q0 ((v1 - v0) q0^-1 mod q1), the product by
/// q0^-1 with its Shoup word, which also reduces the difference) and, up to 32 bits, the rounding through an f64
/// product (x < 2^63: within one unit of the exact rounding, as lossy as the rounding itself); wider words round
/// exactly.
struct Garner2 {
    q: [u64; 2],
    /// floor(2^64 / q_r): residues that arrive unreduced
    m: [u64; 2],
    /// q0^-1 mod q1 and floor(2^64 inv / q1)
    inv: [u64; 2],
    /// the multiple of q1 from q0 up: v1 + lift - v0 >= 0
    lift: u64,
    big: u128,
    scale: f64,
    bits: u32,
    mask: u64,
}

impl Garner2 {
    fn new(q0: u64, q1: u64, bits: u32) -> Self {
        let big = q0 as u128 * q1 as u128;
        let inv = inv_mod(q0 % q1, q1);
        Garner2 { q: [q0, q1], m: [q0, q1].map(|p| ((1u128 << 64) / p as u128) as u64), inv: [inv, (((inv as u128) << 64) / q1 as u128) as u64], lift: q0.div_ceil(q1) * q1, big, scale: (1u128 << bits) as f64 / big as f64, bits, mask: u64::MAX >> (64 - bits) }
    }
    /// The word of residues `a` mod q0, `b` mod q1 (either unreduced).
    #[inline(always)]
    fn word(&self, a: u64, b: u64) -> u64 {
        let [q0, q1] = self.q;
        let v0 = if a < q0 { a } else { red(a, q0, self.m[0]) };
        // q0, q1 < 2^62 (their product is below 2^63): the difference stays in 64 bits
        let d = if b >> 62 == 0 { b } else { red(b, q1, self.m[1]) } + self.lift - v0;
        let t = d.wrapping_mul(self.inv[0]).wrapping_sub((((d as u128 * self.inv[1] as u128) >> 64) as u64).wrapping_mul(q1));
        let x = v0 + q0 * if t >= q1 { t - q1 } else { t };
        if self.bits <= 32 {
            // f64::round (halves away from zero) of a non-negative value below 2^33, without the libm call
            let z = x as i64 as f64 * self.scale;
            // SAFETY: 0 <= z < 2^33
            let f = unsafe { z.to_int_unchecked::<i64>() };
            (f + (z - f as f64 >= 0.5) as i64) as u64 & self.mask
        } else {
            ((((x as u128) << self.bits) + self.big / 2) / self.big) as u64 & self.mask
        }
    }
}

/// Compress coefficient-form residues (`rows[r][i]` mod `q[r]`) to one `bits`-bit word (<= 64) per coefficient:
/// round(x 2^bits / Q) mod 2^bits, x the CRT of the residues in [0, Q). Q must stay below 2^(128 - bits).
pub fn pack_coeffs(q: &[u64], rows: &[&[u64]], bits: u32) -> Result<Vec<u64>> {
    let big: u128 = q.iter().map(|&p| p as u128).product();
    if bits == 0 || bits > 64 || q.iter().map(|&p| 64 - p.leading_zeros()).sum::<u32>() + bits > 127 {
        return Err(Error::Wire(format!("pack_coeffs: {bits} bits over a {}-prime modulus", q.len())));
    }
    let n = rows[0].len();
    if q.len() == 2 && big < 1u128 << 63 {
        use rayon::prelude::*;
        let g = Garner2::new(q[0], q[1], bits);
        let mut out = vec![0u64; n];
        out.par_chunks_mut(4096).enumerate().for_each(|(ci, o)| {
            let (r0, r1) = (&rows[0][ci * 4096..][..o.len()], &rows[1][ci * 4096..][..o.len()]);
            for ((y, &a), &b) in o.iter_mut().zip(r0).zip(r1) {
                *y = g.word(a, b);
            }
        });
        return Ok(out);
    }
    // Garner: x = sum_r v_r * (Q / q_r) * ((Q / q_r)^-1 mod q_r), reduced mod Q
    let parts: Vec<(u128, u64)> = q
        .iter()
        .map(|&p| {
            let m = big / p as u128;
            (m, inv_mod((m % p as u128) as u64, p))
        })
        .collect();
    let mask = u64::MAX as u128 >> (64 - bits);
    Ok((0..n)
        .map(|i| {
            let mut x = 0u128;
            for (r, (&p, &(m, inv))) in q.iter().zip(&parts).enumerate() {
                let v = ((rows[r][i] % p) as u128 * inv as u128 % p as u128) * m % big;
                x = (x + v) % big;
            }
            ((((x << bits) + big / 2) / big) & mask) as u64
        })
        .collect())
}

/// `pack_coeffs`' inverse up to the rounding: the residues mod each `q[r]` of round(y Q / 2^bits).
pub fn unpack_coeffs(q: &[u64], words: &[u64], bits: u32) -> Vec<Vec<u64>> {
    let big: u128 = q.iter().map(|&p| p as u128).product();
    let xs: Vec<u128> = words.iter().map(|&y| ((y as u128 * big) + (1u128 << (bits - 1))) >> bits).collect();
    q.iter().map(|&p| xs.iter().map(|&x| (x % big % p as u128) as u64).collect()).collect()
}

/// Words of `n` values of `bits` bits, bit-packed (one word per value at 32 bits).
pub fn packed_words(n: usize, bits: u32) -> usize {
    (n * bits as usize).div_ceil(32)
}

/// `bits`-bit values (<= 64), LSB first, into `packed_words(len, bits)` words.
pub fn pack_bits(v: &[u64], bits: u32) -> Vec<u32> {
    let mask = u64::MAX >> (64 - bits);
    if bits <= 32 {
        let mut out = Vec::with_capacity(packed_words(v.len(), bits));
        let mut acc = BitAcc::default();
        v.iter().for_each(|&x| acc.push(x & mask, bits, &mut out));
        acc.finish(&mut out);
        return out;
    }
    let mut out = vec![0u32; packed_words(v.len(), bits)];
    for (i, &x) in v.iter().enumerate() {
        let (bit, x) = (i * bits as usize, (x & mask) as u128);
        let (w, o) = (bit / 32, bit % 32);
        let x = x << o;
        for (k, word) in out[w..].iter_mut().take((o + bits as usize).div_ceil(32)).enumerate() {
            *word |= (x >> (32 * k)) as u32;
        }
    }
    out
}

/// `pack_bits`' running word, values of up to 32 bits: under 32 bits pending between pushes.
#[derive(Default)]
struct BitAcc {
    acc: u64,
    nb: u32,
}

impl BitAcc {
    #[inline(always)]
    fn push(&mut self, x: u64, bits: u32, out: &mut Vec<u32>) {
        self.acc |= x << self.nb;
        self.nb += bits;
        if self.nb >= 32 {
            out.push(self.acc as u32);
            self.acc >>= 32;
            self.nb -= 32;
        }
    }
    fn finish(self, out: &mut Vec<u32>) {
        if self.nb > 0 {
            out.push(self.acc as u32);
        }
    }
}

/// `pack_bits(&pack_coeffs(q, rows, bits)?, bits)` appended to `out`, the rows of any word: one pass on the calling
/// thread over two primes (Q < 2^63) up to 32 bits (a frame's parts: several frames pack side by side).
pub fn pack_frame_part<T: Copy + Into<u64>>(q: &[u64], rows: &[&[T]], bits: u32, out: &mut Vec<u32>) -> Result<()> {
    if rows.len() != q.len() || rows.iter().any(|r| r.len() != rows[0].len()) {
        return Err(Error::Wire(format!("pack_frame_part: {} rows over a {}-prime modulus", rows.len(), q.len())));
    }
    if q.len() == 2 && (1..=32).contains(&bits) && (q[0] as u128 * q[1] as u128) < 1u128 << 63 {
        let g = Garner2::new(q[0], q[1], bits);
        out.reserve(packed_words(rows[0].len(), bits));
        // the words a block at a time (independent of each other), then their bits
        let mut acc = BitAcc::default();
        let mut w = [0u64; 256];
        for (r0, r1) in rows[0].chunks(256).zip(rows[1].chunks(256)) {
            for ((y, &a), &b) in w.iter_mut().zip(r0).zip(r1) {
                *y = g.word(a.into(), b.into());
            }
            w[..r0.len()].iter().for_each(|&y| acc.push(y, bits, out));
        }
        acc.finish(out);
        return Ok(());
    }
    let wide: Vec<Vec<u64>> = rows.iter().map(|r| r.iter().map(|&x| x.into()).collect()).collect();
    let refs: Vec<&[u64]> = wide.iter().map(|r| r.as_slice()).collect();
    out.extend(pack_bits(&pack_coeffs(q, &refs, bits)?, bits));
    Ok(())
}

/// Bits of a coefficient without its low `drop` bits (`CtHeader::drop`) over the primes `q`: those of the largest
/// (x + 2^(drop - 1)) >> drop, x < Q. Up to 4 primes below 2^30.
pub fn dropped_bits(q: &[u64], drop: u32) -> Result<u32> {
    let big: u128 = q.iter().map(|&p| p as u128).product();
    if q.is_empty() || q.len() > 4 || q.iter().any(|&p| p >> 30 != 0) || drop >= 128 - big.leading_zeros() {
        return Err(Error::Wire(format!("{drop} bits dropped over a {}-prime modulus", q.len())));
    }
    Ok(128 - ((big - 1 + ((1u128 << drop) >> 1)) >> drop).leading_zeros().min(127))
}

/// Coefficient-form residues (`rows[r][i]` mod `q[r]`) as `CtHeader::drop` words appended to `out`: x by mixed radix
/// (x = t0 + q0 (t1 + q1 (t2 + ..)), t_j < q_j from Shoup products), rounded, `dropped_bits` bits each.
pub fn pack_dropped(q: &[u64], rows: &[&[u32]], drop: u32, out: &mut Vec<u32>) -> Result<u32> {
    use crate::ntt::{mul_shoup, shoup};
    let bits = dropped_bits(q, drop)?;
    let k = q.len();
    if rows.len() != k || rows.iter().any(|r| r.len() != rows[0].len()) {
        return Err(Error::Wire(format!("pack_dropped: {} rows over a {k}-prime modulus", rows.len())));
    }
    // step (i, j), i < j: t_j <- (t_j + lift - t_i) q_i^-1 mod q_j, lift the multiple of q_j from q_i up
    let mut inv = [[(0u32, 0u32, 0u32); 4]; 4];
    for j in 0..k {
        for i in 0..j {
            let w = inv_mod(q[i] % q[j], q[j]) as u32;
            inv[i][j] = (w, shoup(w, q[j] as u32), (q[i].div_ceil(q[j]) * q[j]) as u32);
        }
    }
    let half = (1u128 << drop) >> 1;
    let n = rows[0].len();
    out.reserve(packed_words(n, bits));
    let mut acc = BitAcc::default();
    for i in 0..n {
        let mut t = [0u32; 4];
        for j in 0..k {
            let mut u = rows[j][i];
            for (&ti, a) in t[..j].iter().zip(&inv[..j]) {
                let (w, ws, lift) = a[j];
                u = mul_shoup(u + lift - ti, w, ws, q[j] as u32);
            }
            t[j] = u;
        }
        let mut x = t[k - 1] as u128;
        for j in (0..k - 1).rev() {
            x = x * q[j] as u128 + t[j] as u128;
        }
        let (mut y, mut b) = ((x + half) >> drop, bits);
        while b > 32 {
            acc.push(y as u32 as u64, 32, out);
            y >>= 32;
            b -= 32;
        }
        acc.push(y as u64, b, out);
    }
    acc.finish(out);
    Ok(bits)
}

/// `pack_dropped`'s inverse up to the rounding: the residues of y 2^drop, `rows` = `[row][n]` coefficient form.
pub fn unpack_dropped(q: &[u64], words: &[u32], drop: u32, rows: &mut [u32]) {
    let bits = dropped_bits(q, drop).expect("checked by the caller");
    let n = rows.len() / q.len();
    // per prime: floor(2^64 / p), 2^64 mod p
    let c: Vec<(u64, u64, u64)> = q.iter().map(|&p| (p, ((1u128 << 64) / p as u128) as u64, ((1u128 << 64) % p as u128) as u64)).collect();
    let (mut acc, mut nb, mut k) = (0u64, 0u32, 0usize);
    for i in 0..n {
        let (mut y, mut got) = (0u128, 0u32);
        while got < bits {
            let b = (bits - got).min(32);
            if nb < b {
                acc |= (words[k] as u64) << nb;
                k += 1;
                nb += 32;
            }
            y |= ((acc & (u64::MAX >> (64 - b))) as u128) << got;
            acc >>= b;
            nb -= b;
            got += b;
        }
        let x = y << drop;
        let (lo, hi) = (x as u64, (x >> 64) as u64);
        for (r, &(p, m, top)) in c.iter().enumerate() {
            rows[r * n + i] = if hi == 0 { red(lo, p, m) } else { red(red(hi, p, m) * top + red(lo, p, m), p, m) } as u32;
        }
    }
}

/// `pack_bits`' inverse: `n` values.
pub fn unpack_bits(w: &[u32], n: usize, bits: u32) -> Vec<u64> {
    let mask = u64::MAX >> (64 - bits);
    if bits <= 32 && w.len() >= packed_words(n, bits) {
        // a value's two words as one, shifted: no state carried from value to value
        return (0..n)
            .map(|i| {
                let (k, o) = (i * bits as usize / 32, i * bits as usize % 32);
                let hi = w.get(k + 1).copied().unwrap_or(0);
                ((w[k] as u64 | (hi as u64) << 32) >> o) & mask
            })
            .collect();
    }
    (0..n)
        .map(|i| {
            let bit = i * bits as usize;
            let (k, o) = (bit / 32, bit % 32);
            let x = w[k..].iter().take((o + bits as usize).div_ceil(32)).enumerate().fold(0u128, |a, (j, &y)| a | (y as u128) << (32 * j));
            (x >> o) as u64 & mask
        })
        .collect()
}

fn inv_mod(a: u64, p: u64) -> u64 {
    let (mut r, mut b, mut e) = (1u128, a as u128 % p as u128, p - 2);
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % p as u128;
        }
        b = b * b % p as u128;
        e >>= 1;
    }
    r as u64
}

/// A batch of `k` ciphertexts, eval form, `data` = `[c][part][row][N]` (or compressed: `CtHeader::bits`).
#[derive(Clone, Debug, PartialEq)]
pub struct CtWire {
    pub header: CtHeader,
    pub data: Vec<u32>,
}

impl CtWire {
    pub fn words(h: &CtHeader) -> usize {
        if h.drop.is_some() {
            h.k * (packed_words(h.n, h.bits.unwrap_or(0)) + h.rows * 8)
        } else if let Some(b0) = h.bits {
            let b1 = h.bits1.unwrap_or(b0);
            h.k * (packed_words(h.n, b0) + packed_words(h.n / h.sub1.unwrap_or(1).max(1) as usize, b1))
        } else if h.seeded {
            h.k * h.rows * (h.n + 8)
        } else {
            h.k * 2 * h.rows * h.n
        }
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(4 * self.data.len() + 512);
        self.write(&mut out)?;
        Ok(out)
    }
    pub fn write<Wr: Write>(&self, w: &mut Wr) -> Result<()> {
        if self.data.len() != Self::words(&self.header) {
            return Err(Error::Wire(format!("ciphertext data has {} words, header {:?} needs {}", self.data.len(), self.header, Self::words(&self.header))));
        }
        write_header(w, CT_MAGIC, &serde_json::to_vec(&self.header)?)?;
        write_words(w, &self.data)?;
        w.write_all(&sum64(&self.data).to_le_bytes())?;
        Ok(())
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        let mut r = std::io::Cursor::new(b);
        let w = Self::read(&mut r)?;
        if r.position() as usize != b.len() {
            return Err(Error::Wire(format!("{} trailing bytes after the ciphertext", b.len() - r.position() as usize)));
        }
        Ok(w)
    }
    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let header: CtHeader = serde_json::from_slice(&read_header(r, CT_MAGIC)?)?;
        if header.n == 0 || header.k == 0 || header.rows == 0 || header.n > 1 << 17 || header.k > 64 || header.rows > 128 {
            return Err(Error::Wire(format!("ciphertext header out of range: {header:?}")));
        }
        if header.sub1.is_some_and(|k| k < 2 || !k.is_power_of_two() || header.n % k as usize != 0) {
            return Err(Error::Wire(format!("packed frame header out of range: {header:?}")));
        }
        if header.drop.is_some() && !(header.seeded && !header.regen && header.bits.is_some_and(|b| (1..=126).contains(&b)) && header.rows <= 4) {
            return Err(Error::Wire(format!("short-c0 frame header out of range: {header:?}")));
        }
        let data = read_words(r, Self::words(&header))?;
        let mut c = [0u8; 8];
        r.read_exact(&mut c)?;
        if u64::from_le_bytes(c) != sum64(&data) {
            return Err(Error::Wire("ciphertext checksum mismatch".into()));
        }
        Ok(CtWire { header, data })
    }
    /// A seeded frame as a full one: c1's rows drawn from their seeds (`primes`: the chain's Q primes, row r = q_r).
    pub fn expand_seeded(&self, primes: &[u64]) -> Result<CtWire> {
        use rayon::prelude::*;
        let h = &self.header;
        if !h.seeded {
            return Ok(self.clone());
        }
        let (n, rows) = (h.n, h.rows);
        if primes.len() < rows {
            return Err(Error::Wire(format!("seeded frame of {rows} rows, {} primes", primes.len())));
        }
        let short = match (h.drop, h.bits) {
            (Some(d), Some(b)) if dropped_bits(&primes[..rows], d)? == b => Some((d, packed_words(n, b))),
            (Some(d), b) => return Err(Error::Wire(format!("short-c0 frame of {b:?} bits, {d} dropped over {rows} rows"))),
            _ => None,
        };
        let per = short.map_or(rows * n, |(_, w)| w) + rows * 8;
        let mut data = vec![0u32; h.k * 2 * rows * n];
        for c in 0..h.k {
            let src = &self.data[c * per..(c + 1) * per];
            let (c0, seeds) = src.split_at(per - rows * 8);
            let dst = &mut data[c * 2 * rows * n..(c + 1) * 2 * rows * n];
            let (d0, d1) = dst.split_at_mut(rows * n);
            match short {
                Some((d, _)) => unpack_dropped(&primes[..rows], c0, d, d0),
                None => d0.copy_from_slice(c0),
            }
            d1.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
                let key: [u32; 8] = seeds[r * 8..r * 8 + 8].try_into().expect("8 words");
                if h.regen {
                    pc_core::keys::cts_expand_row(&key, primes[r] as u32, row);
                } else {
                    let mut g = pc_core::keys::Prng::from_seed(key, 0);
                    g.fill_below(primes[r], row);
                }
            });
        }
        Ok(CtWire { header: CtHeader { seeded: false, regen: false, bits: if short.is_some() { None } else { h.bits }, drop: None, coeff0: short.is_some(), ..h.clone() }, data })
    }
    /// A subring frame (`Client::sub`, expanded: both parts in coefficient form over `header.n` | `n` coefficients) in
    /// the ring of `n` coefficients, coefficient i at i n / header.n (X -> X^(n / header.n), a ring map): the same
    /// plaintext's lift under the lifted secret, its slots the small ring's repeated. Every slot live.
    pub fn lifted(&self, n: usize) -> Result<CtWire> {
        let h = &self.header;
        if h.seeded || h.bits.is_some() || !(h.coeff0 && h.coeff1) {
            return Err(Error::Wire("lifted: an expanded frame with both parts in coefficient form (a subring client's)".into()));
        }
        if h.n == 0 || n % h.n != 0 || !(n / h.n).is_power_of_two() {
            return Err(Error::Wire(format!("lifted: a frame of {} coefficients into a ring of {n}", h.n)));
        }
        let k = n / h.n;
        let mut data = vec![0u32; self.data.len() * k];
        for (src, dst) in self.data.chunks(h.n).zip(data.chunks_mut(n)) {
            for (i, &x) in src.iter().enumerate() {
                dst[i * k] = x;
            }
        }
        Ok(CtWire { header: CtHeader { n, meta: WireMeta { slots: (n / 2) as u32, ..h.meta.clone() }, ..h.clone() }, data })
    }
    /// Row `row` of part `part` of ciphertext `c`.
    pub fn row(&self, c: usize, part: usize, row: usize) -> &[u32] {
        let (n, rows) = (self.header.n, self.header.rows);
        let o = ((c * 2 + part) * rows + row) * n;
        &self.data[o..o + n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let n = 1024;
        for bits in [27u32, 28, 31] {
            let vals: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761) & ((1 << bits) - 1)).collect();
            let mut p = vec![0u32; packed_row_words(n, bits)];
            pack_row(&vals, bits, &mut p);
            let mut u = vec![0u32; n];
            unpack_row(&p, bits, &mut u);
            assert_eq!(u, vals, "bits {bits}");
            assert_eq!(*p.last().unwrap(), 0, "guard word");
        }
    }

    #[test]
    fn pack_coeffs_two_primes_wide() {
        // the two-prime path against the exact rounding of the CRT value, above 32 bits
        let q = [268369921u64, 268361729];
        let big = q[0] as u128 * q[1] as u128;
        let n = 4096;
        let x: Vec<u128> = (0..n as u128).map(|i| (i * 0x9e37_79b9_7f4a_7c15 + i * i) % big).collect();
        let rows: Vec<Vec<u64>> = q.iter().map(|&p| x.iter().map(|&v| (v % p as u128) as u64).collect()).collect();
        let refs: Vec<&[u64]> = rows.iter().map(|r| r.as_slice()).collect();
        for bits in [33u32, 41, 48, 56, 64] {
            let got = pack_coeffs(&q, &refs, bits).unwrap();
            let mask = u64::MAX as u128 >> (64 - bits);
            for (g, &v) in got.iter().zip(&x) {
                assert_eq!(*g as u128, (((v << bits) + big / 2) / big) & mask, "bits {bits}");
            }
        }
    }

    /// The packers before the reciprocal reductions and the running word: the reference of `pack_matches_reference`.
    fn pack_coeffs_ref(q: &[u64], rows: &[&[u64]], bits: u32) -> Vec<u64> {
        let (q0, q1) = (q[0], q[1]);
        let big = q0 as u128 * q1 as u128;
        let inv = inv_mod(q0 % q1, q1) as u128;
        let scale = (1u128 << bits) as f64 / big as f64;
        let mask = u64::MAX >> (64 - bits);
        (0..rows[0].len())
            .map(|i| {
                let (v0, v1) = (rows[0][i] % q0, rows[1][i] % q1);
                let d = (v1 + q1 - v0 % q1) % q1;
                let x = v0 + q0 * ((d as u128 * inv % q1 as u128) as u64);
                if bits <= 32 { ((x as f64) * scale).round() as u64 & mask } else { ((((x as u128) << bits) + big / 2) / big) as u64 & mask }
            })
            .collect()
    }
    fn pack_bits_ref(v: &[u64], bits: u32) -> Vec<u32> {
        let mut out = vec![0u32; packed_words(v.len(), bits)];
        for (i, &x) in v.iter().enumerate() {
            let (bit, x) = (i * bits as usize, (x & (u64::MAX >> (64 - bits))) as u128);
            let (w, o) = (bit / 32, bit % 32);
            let x = x << o;
            for (k, word) in out[w..].iter_mut().take((o + bits as usize).div_ceil(32)).enumerate() {
                *word |= (x >> (32 * k)) as u32;
            }
        }
        out
    }
    fn unpack_bits_ref(w: &[u32], n: usize, bits: u32) -> Vec<u64> {
        (0..n)
            .map(|i| {
                let bit = i * bits as usize;
                let (k, o) = (bit / 32, bit % 32);
                let x = w[k..].iter().take((o + bits as usize).div_ceil(32)).enumerate().fold(0u128, |a, (j, &y)| a | (y as u128) << (32 * j));
                (x >> o) as u64 & (u64::MAX >> (64 - bits))
            })
            .collect()
    }

    #[test]
    fn pack_matches_reference() {
        // word for word: random and unreduced residues, and CRT values at the rounding's half points
        for q in [[268042241u64, 265420801], [268369921, 268361729], [4294828033, 2147352577]] {
            let big = q[0] as u128 * q[1] as u128;
            let n = 20000usize;
            for bits in [1u32, 12, 16, 18, 22, 28, 31, 32, 33, 48] {
                let mut x: Vec<u128> = (0..n as u128).map(|i| (i * 0x9e37_79b9_7f4a_7c15 + i * i * 0x2545_f491_4f6c_dd1d) % big).collect();
                for (k, v) in x.iter_mut().enumerate().take(4000) {
                    // around (j + 1/2) Q / 2^bits
                    let j = (k as u128 * 0x9e37_79b9) & ((1u128 << bits) - 1);
                    *v = ((((2 * j + 1) * big) >> (bits + 1)) + big + (k % 5) as u128 - 2) % big;
                }
                let mut rows: Vec<Vec<u64>> = q.iter().map(|&p| x.iter().map(|&v| (v % p as u128) as u64).collect()).collect();
                for i in 4000..5000 {
                    rows[0][i] = u64::MAX - ((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 3);
                    rows[1][i] = (i as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
                }
                let refs: Vec<&[u64]> = rows.iter().map(|r| r.as_slice()).collect();
                let want = pack_coeffs_ref(&q, &refs, bits);
                assert_eq!(pack_coeffs(&q, &refs, bits).unwrap(), want, "pack_coeffs {bits} bits, q {q:?}");
                let packed = pack_bits_ref(&want, bits);
                assert_eq!(pack_bits(&want, bits), packed, "pack_bits {bits} bits");
                assert_eq!(unpack_bits(&packed, n, bits), unpack_bits_ref(&packed, n, bits), "unpack_bits {bits} bits");
                let mut part = vec![7u32];
                pack_frame_part(&q, &refs, bits, &mut part).unwrap();
                assert_eq!(part[1..], packed[..], "pack_frame_part {bits} bits, q {q:?}");
                // a part of 32-bit rows (the session's words)
                let narrow: Vec<Vec<u32>> = rows.iter().map(|r| r.iter().map(|&v| v as u32).collect()).collect();
                let wide: Vec<Vec<u64>> = narrow.iter().map(|r| r.iter().map(|&v| v as u64).collect()).collect();
                let (nrefs, wrefs): (Vec<&[u32]>, Vec<&[u64]>) = (narrow.iter().map(|r| r.as_slice()).collect(), wide.iter().map(|r| r.as_slice()).collect());
                let mut part = Vec::new();
                pack_frame_part(&q, &nrefs, bits, &mut part).unwrap();
                assert_eq!(part, pack_bits_ref(&pack_coeffs_ref(&q, &wrefs, bits), bits), "pack_frame_part of u32 rows, {bits} bits");
            }
        }
        // an odd count's tail word
        for n in [1usize, 3, 31, 33] {
            let v: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15)).collect();
            for bits in [1u32, 7, 22, 32] {
                let packed = pack_bits_ref(&v, bits);
                assert_eq!(pack_bits(&v, bits), packed, "{n} values of {bits} bits");
                assert_eq!(unpack_bits(&packed, n, bits), unpack_bits_ref(&packed, n, bits));
            }
        }
    }

    /// `pack_dropped` / `unpack_dropped`: every residue is that of the rounded coefficient, at the bits `dropped_bits` says.
    #[test]
    fn dropped_roundtrip() {
        let primes = [268042241u64, 265420801, 156499969, 95027201];
        let n = 257usize;
        for k in [1usize, 2, 3, 4] {
            let q = &primes[..k];
            let big: u128 = q.iter().map(|&p| p as u128).product();
            let mut xs: Vec<u128> = (0..n as u128).map(|i| (i + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835) % big).collect();
            xs[0] = 0;
            xs[1] = big - 1;
            let rows: Vec<Vec<u32>> = q.iter().map(|&p| xs.iter().map(|&x| (x % p as u128) as u32).collect()).collect();
            let refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
            for drop in [0u32, 1, 9, 24, 27 * k as u32 - 2] {
                let mut w = Vec::new();
                let bits = pack_dropped(q, &refs, drop, &mut w).unwrap();
                assert_eq!((bits, w.len()), (dropped_bits(q, drop).unwrap(), packed_words(n, bits)), "{k} primes, {drop} dropped");
                let mut back = vec![0u32; k * n];
                unpack_dropped(q, &w, drop, &mut back);
                for (r, &p) in q.iter().enumerate() {
                    for (i, &x) in xs.iter().enumerate() {
                        let y = (x + ((1u128 << drop) >> 1)) >> drop;
                        assert_eq!(back[r * n + i] as u128, (y << drop) % p as u128, "{k} primes, {drop} dropped, row {r}, coefficient {i}");
                    }
                }
            }
        }
    }

    /// Cost of a short-c0 frame's c0 over the four 27..28-bit primes, client side and server side:
    /// `cargo test --release -p pc-client dropped_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dropped_cost() {
        let primes = [268042241u64, 265420801, 156499969, 95027201];
        let n = 1usize << 16;
        for (k, drop) in [(2usize, 10u32), (4, 24)] {
            let q = &primes[..k];
            let rows: Vec<Vec<u32>> = q.iter().map(|&p| (0..n as u64).map(|i| (i.wrapping_mul(0x9e37_79b9_7f4a_7c15) % p) as u32).collect()).collect();
            let refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
            let (mut w, mut back) = (Vec::new(), vec![0u32; k * n]);
            let reps = 50;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                w.clear();
                pack_dropped(q, &refs, drop, &mut w).unwrap();
            }
            let tp = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                unpack_dropped(q, &w, drop, &mut back);
            }
            println!("{k} rows, {drop} dropped ({} bits): pack {tp:.3} ms, unpack {:.3} ms", dropped_bits(q, drop).unwrap(), t.elapsed().as_secs_f64() * 1e3 / reps as f64);
        }
    }

    /// Cost of one compressed frame's packing (16 / 22 bits over two 28-bit primes), alone and eight at once:
    /// `cargo test --release -p pc-client frame_pack_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn frame_pack_cost() {
        let q = [268042241u64, 265420801];
        let n = 1usize << 16;
        let rows: Vec<Vec<u64>> = q.iter().map(|&p| (0..n as u64).map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15) % p).collect()).collect();
        let refs: Vec<&[u64]> = rows.iter().map(|r| r.as_slice()).collect();
        let frame = || {
            for b in [16u32, 22] {
                std::hint::black_box(pack_bits(&pack_coeffs(&q, &refs, b).unwrap(), b));
            }
        };
        let narrow: Vec<Vec<u32>> = rows.iter().map(|r| r.iter().map(|&v| v as u32).collect()).collect();
        let nrefs: Vec<&[u32]> = narrow.iter().map(|r| r.as_slice()).collect();
        let part = || {
            let mut data = Vec::with_capacity(2 * n);
            for b in [16u32, 22] {
                pack_frame_part(&q, &nrefs, b, &mut data).unwrap();
            }
            std::hint::black_box(data);
        };
        let reps = 20;
        let time = |name: &str, f: &(dyn Fn() + Sync)| {
            f();
            let t = std::time::Instant::now();
            for _ in 0..reps {
                f();
            }
            println!("{name}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
        };
        time("pack_coeffs 16", &|| {
            std::hint::black_box(pack_coeffs(&q, &refs, 16).unwrap());
        });
        let v = pack_coeffs(&q, &refs, 22).unwrap();
        time("pack_bits 22", &|| {
            std::hint::black_box(pack_bits(&v, 22));
        });
        time("frame", &frame);
        time("8 frames at once", &|| {
            std::thread::scope(|s| {
                for _ in 0..8 {
                    s.spawn(frame);
                }
            });
        });
        time("frame (pack_frame_part)", &part);
        let buf = std::sync::Mutex::new(Vec::with_capacity(2 * n));
        time("frame (pack_frame_part, buffer kept)", &|| {
            let mut data = buf.lock().unwrap();
            data.clear();
            for b in [16u32, 22] {
                pack_frame_part(&q, &nrefs, b, &mut data).unwrap();
            }
        });
        time("8 frames at once (pack_frame_part)", &|| {
            std::thread::scope(|s| {
                for _ in 0..8 {
                    s.spawn(part);
                }
            });
        });
    }

    #[test]
    fn pack_bits_roundtrip() {
        let n = 1000;
        for bits in [1u32, 20, 26, 32, 33, 46, 57, 64] {
            let vals: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15) & (u64::MAX >> (64 - bits))).collect();
            let p = pack_bits(&vals, bits);
            assert_eq!(p.len(), packed_words(n, bits), "bits {bits}");
            assert_eq!(unpack_bits(&p, n, bits), vals, "bits {bits}");
        }
    }

    #[test]
    fn ct_frame_roundtrip_and_corruption() {
        let h = CtHeader { params_hash: "x".into(), n: 16, k: 2, rows: 3, meta: WireMeta { level: 4, deg: 1, nf: 2f64.powi(54), slots: 8, tag: serde_json::Value::Null }, bits: None, seeded: false, regen: false, bits1: None, sparse: false, drop: None, coeff0: false, coeff1: false, sub1: None };
        let w = CtWire { data: (0..CtWire::words(&h) as u32).collect(), header: h };
        let mut b = w.to_bytes().unwrap();
        assert_eq!(CtWire::from_bytes(&b).unwrap(), w);
        let k = b.len() - 20;
        b[k] ^= 1;
        assert!(CtWire::from_bytes(&b).is_err());
    }
}
