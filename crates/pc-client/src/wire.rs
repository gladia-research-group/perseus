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
    let mut bytes = vec![0u8; n.checked_mul(4).ok_or_else(|| Error::Wire(format!("{n} words")))?];
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
}

impl Entry {
    pub fn words(&self) -> usize {
        match self {
            Entry::Ksk { words, .. } | Entry::SpruCs { words, .. } | Entry::Pcmm { words, .. } | Entry::Ccmm { words, .. } | Entry::PublicKey { words, .. } => *words,
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
            off = (e.words() as u64).checked_mul(4).and_then(|b| off.checked_add(b + 8)).ok_or_else(|| Error::Wire(format!("entry of {} words", e.words())))?;
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
}

/// A batch of `k` ciphertexts, eval form, `data` = `[c][part][row][N]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CtWire {
    pub header: CtHeader,
    pub data: Vec<u32>,
}

impl CtWire {
    pub fn words(h: &CtHeader) -> usize {
        h.k * 2 * h.rows * h.n
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
        let data = read_words(r, Self::words(&header))?;
        let mut c = [0u8; 8];
        r.read_exact(&mut c)?;
        if u64::from_le_bytes(c) != sum64(&data) {
            return Err(Error::Wire("ciphertext checksum mismatch".into()));
        }
        Ok(CtWire { header, data })
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
    fn ct_frame_roundtrip_and_corruption() {
        let h = CtHeader { params_hash: "x".into(), n: 16, k: 2, rows: 3, meta: WireMeta { level: 4, deg: 1, nf: 2f64.powi(54), slots: 8, tag: serde_json::Value::Null } };
        let w = CtWire { data: (0..CtWire::words(&h) as u32).collect(), header: h };
        let mut b = w.to_bytes().unwrap();
        assert_eq!(CtWire::from_bytes(&b).unwrap(), w);
        let k = b.len() - 20;
        b[k] ^= 1;
        assert!(CtWire::from_bytes(&b).is_err());
    }
}
