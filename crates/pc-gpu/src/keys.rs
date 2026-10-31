//! Device key material: the `KskLayout` seam, switching keys (`DevKsk`), the key set (`KeySet`: relin,
//! conj, rotations by signed step, bootstrap encapsulation keys as named fields) and the device secret.
//!
//! Storage is carved from the arena at setup (`KeySetSpec::bytes` + `MemPlan::reserve("keys", ..)`, `KeySet::carve`
//! before `Arena::freeze`), then filled by `keygen` or `import_material` (oracle dumps, saved keys). A key's rows
//! (`KskShape`): digits `0..ndig`, each the Q rows `0..=q_top` then the `kp` special rows; a banded key
//! (`q_top < nq - 1`) drops the rows above `q_top`. Relation stored: OpenFHE `KeySwitchHYBRID::KeySwitchGenInternal`
//! in eval form, `a = KSKA(seed, ...)` (pc_core::keys). Geometry and roles: docs/ARCHITECTURE.md "pc-gpu keys".
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Error, Result};
use crate::ntt::{self, Parts, Span};
use crate::poly::{CtBuf, Layout};
use pc_core::galois;
use pc_core::keys::{KskMaterial, KskSeed, SecretKey};
use pc_core::modmul::mulmod;
use pc_core::params::Chain;
use pc_cuda::{Arena, Compute, DBuf, Stream};
use rayon::prelude::*;
use std::collections::BTreeMap;

pub mod hostref;

/// The key-switching-key storage seam (`KskLayout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KskLayout {
    /// Default: `b` bit-packed at `bits` per coefficient (u32 words only; `bits` >= every prime's width), `a` not
    /// stored: regenerated in-kernel from the key's 256-bit seed (the frozen KSKA expander).
    PackedSeeded { bits: u32 },
    /// `a` and `b` stored dense in words W, `[digit][slot][N]` (imported dense keys, tests).
    Dense,
}

impl KskLayout {
    /// The default packed layout for `chain` at words W: `bits` = the widest prime of Q u P. Errors for W != u32
    /// or a 32-bit prime (nothing to save).
    pub fn packed_for<W: DevWord>(chain: &Chain) -> Result<Self> {
        let bits = chain.prime_bits().into_iter().max().unwrap_or(0);
        if W::BITS != 32 {
            return Err(shape("KskLayout::PackedSeeded", format!("packing is defined for u32 words, the context has {}-bit words (use KskLayout::Dense)", W::BITS)));
        }
        if bits == 0 || bits > 31 {
            return Err(shape("KskLayout::PackedSeeded", format!("widest prime has {bits} bits; packing needs 1..=31")));
        }
        Ok(KskLayout::PackedSeeded { bits })
    }
    pub fn bits(&self) -> u32 {
        match self {
            KskLayout::PackedSeeded { bits } => *bits,
            KskLayout::Dense => 0,
        }
    }
}

/// u32 words of one packed row of N values at `bits` (+ one zero guard word, padded to 16 bytes): readers may load
/// the word after a value's last word (64-bit funnel read) without leaving the row.
pub fn packed_row_words(n: usize, bits: u32) -> usize {
    (n * bits as usize / 32 + 1).div_ceil(4) * 4
}

/// Which rows a key stores (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KskShape {
    /// Highest Q prime id stored (a ciphertext with top limb <= q_top can use the key).
    pub q_top: usize,
    /// Digits stored: `0..ndig` (every digit whose first prime id is <= q_top).
    pub ndig: usize,
    /// Q primes of the chain.
    pub nq: usize,
    /// Special primes.
    pub kp: usize,
}

impl KskShape {
    pub fn full(chain: &Chain) -> Self {
        KskShape { q_top: chain.q.len() - 1, ndig: chain.dnum(), nq: chain.q.len(), kp: chain.p.len() }
    }
    /// Rows for ciphertexts with top limb <= `q_top` only.
    pub fn banded(chain: &Chain, q_top: usize) -> Result<Self> {
        if q_top >= chain.q.len() {
            return Err(shape("rot_key_band", format!("band top {q_top} >= Q primes {}", chain.q.len())));
        }
        let ndig = chain.digits.iter().filter(|d| d.first().is_some_and(|&f| f <= q_top)).count();
        Ok(KskShape { q_top, ndig, nq: chain.q.len(), kp: chain.p.len() })
    }
    /// The bottom composite level only (`level_unit` Q rows, digit 0) over P' (`Chain::p_sparse`): the keys to the
    /// sparse secrets (dense->sparse encapsulation, SPRU atob), whose modulus is q0 x P'.
    pub fn bottom(chain: &Chain) -> Result<Self> {
        Ok(KskShape { kp: chain.p_sparse, ..Self::banded(chain, chain.level_unit - 1)? })
    }
    /// Slots per digit.
    pub fn rows(&self) -> usize {
        self.q_top + 1 + self.kp
    }
    pub fn is_full(&self) -> bool {
        self.q_top + 1 == self.nq
    }
    /// Prime id of every slot.
    pub fn row_ids(&self) -> Vec<usize> {
        (0..=self.q_top).chain(self.nq..self.nq + self.kp).collect()
    }
    /// Slot of prime id `id` (None when not stored).
    pub fn slot_of(&self, id: usize) -> Option<usize> {
        if id <= self.q_top {
            Some(id)
        } else if id >= self.nq && id < self.nq + self.kp {
            Some(self.q_top + 1 + id - self.nq)
        } else {
            None
        }
    }
    /// Can a ciphertext whose top limb is `top` be switched with this key?
    pub fn covers(&self, top: usize) -> bool {
        top <= self.q_top
    }
    /// Words of `b` at `layout` (dense: W words; packed: u32 words).
    pub fn b_words(&self, n: usize, layout: KskLayout) -> usize {
        match layout {
            KskLayout::Dense => self.ndig * self.rows() * n,
            KskLayout::PackedSeeded { bits } => self.ndig * self.rows() * packed_row_words(n, bits),
        }
    }
    /// Words of dense `a` (0 when seeded).
    pub fn a_words(&self, n: usize, layout: KskLayout) -> usize {
        match layout {
            KskLayout::Dense => self.ndig * self.rows() * n,
            KskLayout::PackedSeeded { .. } => 0,
        }
    }
    /// Arena bytes of one key of this shape (carve units, seed included).
    pub fn bytes<W: DevWord>(&self, n: usize, layout: KskLayout) -> usize {
        let seed = Arena::bytes_for::<u32>(8);
        match layout {
            KskLayout::Dense => 2 * Arena::bytes_for::<W>(self.b_words(n, layout)) + seed,
            KskLayout::PackedSeeded { .. } => Arena::bytes_for::<u32>(self.b_words(n, layout)) + seed,
        }
    }
}

/// What a switching key switches (module doc for s_old / s_new).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyRole {
    Relin,
    /// Left rotation by `step` slots (Galois element `galois` = 5^step mod 2N).
    Rot { step: i64, galois: u64 },
    Conj,
    EncDenseToSparse,
    EncSparseToDense,
}

impl KeyRole {
    pub fn rot(step: i64, n: usize) -> Result<Self> {
        Ok(KeyRole::Rot { step, galois: galois::rotation_galois(step, n)? })
    }
    /// The automorphism applied after the switch (None: relin / encapsulation).
    pub fn galois(&self, n: usize) -> Option<u64> {
        match self {
            KeyRole::Rot { galois, .. } => Some(*galois),
            KeyRole::Conj => Some(2 * n as u64 - 1),
            _ => None,
        }
    }
    /// The oracle dump's key-map name (`params.keys`): Galois element, "mult", 2N-2 (dense->sparse), 2N-4.
    pub fn oracle_index(&self, n: usize) -> String {
        match self {
            KeyRole::Relin => "mult".into(),
            KeyRole::Rot { galois, .. } => galois.to_string(),
            KeyRole::Conj => (2 * n - 1).to_string(),
            KeyRole::EncDenseToSparse => (2 * n - 2).to_string(),
            KeyRole::EncSparseToDense => (2 * n - 4).to_string(),
        }
    }
}

/// Device storage of one key.
pub enum KskStore<'a, W: DevWord> {
    /// `b`: `[digit][slot][row_words]` u32 words (`pack_row` format); `a` = KSKA(seed).
    PackedSeeded { b: DBuf<'a, Compute, u32>, bits: u32, row_words: usize },
    /// `[digit][slot][N]` words W.
    Dense { b: DBuf<'a, Compute, W>, a: DBuf<'a, Compute, W> },
}

/// One switching key on the device.
pub struct DevKsk<'a, W: DevWord> {
    pub role: KeyRole,
    pub shape: KskShape,
    /// The KSKA seed (always present: dense keys keep it for checks).
    pub seed: KskSeed,
    /// The seed on the device (8 words), the `seed` argument of the regen kernels.
    pub seed_dev: DBuf<'a, Compute, u32>,
    pub store: KskStore<'a, W>,
}

impl<'a, W: DevWord> DevKsk<'a, W> {
    /// Carve storage for one key (setup only); contents undefined until keygen / import.
    pub fn carve(arena: &'a Arena, role: KeyRole, shape: KskShape, layout: KskLayout, n: usize) -> Result<Self> {
        let store = match layout {
            KskLayout::PackedSeeded { bits } => {
                KskStore::PackedSeeded { b: arena.alloc("keys.b", shape.b_words(n, layout))?, bits, row_words: packed_row_words(n, bits) }
            }
            KskLayout::Dense => KskStore::Dense { b: arena.alloc("keys.b", shape.b_words(n, layout))?, a: arena.alloc("keys.a", shape.a_words(n, layout))? },
        };
        Ok(DevKsk { role, shape, seed: KskSeed([0; 8]), seed_dev: arena.alloc("keys.seed", 8)?, store })
    }
    pub fn layout(&self) -> KskLayout {
        match &self.store {
            KskStore::PackedSeeded { bits, .. } => KskLayout::PackedSeeded { bits: *bits },
            KskStore::Dense { .. } => KskLayout::Dense,
        }
    }
    /// `b` base pointer (packed u32 stream or dense W rows).
    pub fn b_ptr(&self) -> *const std::ffi::c_void {
        match &self.store {
            KskStore::PackedSeeded { b, .. } => b.vptr(),
            KskStore::Dense { b, .. } => b.vptr(),
        }
    }
    /// Dense `a` base pointer (None: regenerate from `seed_dev`).
    pub fn a_ptr(&self) -> Option<*const std::ffi::c_void> {
        match &self.store {
            KskStore::PackedSeeded { .. } => None,
            KskStore::Dense { a, .. } => Some(a.vptr()),
        }
    }
    /// Words between consecutive slots of `b` (dense: N W words; packed: `row_words` u32 words).
    pub fn b_row_stride(&self, n: usize) -> usize {
        match &self.store {
            KskStore::PackedSeeded { row_words, .. } => *row_words,
            KskStore::Dense { .. } => n,
        }
    }
    /// Words between digits of `b` (and of dense `a`): `rows * b_row_stride` (`KsArgs::key_digit_stride` when dense).
    pub fn digit_stride(&self, n: usize) -> usize {
        self.shape.rows() * self.b_row_stride(n)
    }
    pub fn bytes(&self) -> usize {
        let s = match &self.store {
            KskStore::PackedSeeded { b, .. } => Arena::bytes_for::<u32>(b.len()),
            KskStore::Dense { b, a } => Arena::bytes_for::<W>(b.len()) + Arena::bytes_for::<W>(a.len()),
        };
        s + Arena::bytes_for::<u32>(8)
    }

    /// Host copy of the key as dense `[digit][slot][N]` u64 residues `(b, a)`: packed `b` unpacked on the host, `a`
    /// regenerated with the host KSKA twin when not stored (tests and checks; syncs `s`).
    pub fn download_dense(&self, s: &Stream<Compute>, chain: &Chain) -> Result<(Vec<u64>, Vec<u64>)> {
        let n = chain.n;
        let primes = chain.primes();
        let ids = self.shape.row_ids();
        let rows = ids.len();
        let b: Vec<u64> = match &self.store {
            KskStore::PackedSeeded { b, bits, row_words } => {
                let raw = b.download(s)?;
                let (bits, rw) = (*bits, *row_words);
                let mut out = vec![0u64; self.shape.ndig * rows * n];
                out.par_chunks_mut(n).enumerate().for_each(|(r, o)| unpack_row(&raw[r * rw..(r + 1) * rw], bits, o));
                out
            }
            KskStore::Dense { b, .. } => b.download(s)?.into_iter().map(|w| w.to_u64()).collect(),
        };
        let a: Vec<u64> = match &self.store {
            KskStore::Dense { a, .. } => a.download(s)?.into_iter().map(|w| w.to_u64()).collect(),
            KskStore::PackedSeeded { .. } => {
                let mut out = vec![0u64; self.shape.ndig * rows * n];
                let seed = self.seed;
                out.par_chunks_mut(n).enumerate().for_each(|(r, o)| {
                    let mut v = vec![0u32; n];
                    pc_core::keys::kska_expand_row(&seed, (r / rows) as u32, primes[ids[r % rows]] as u32, &mut v);
                    o.iter_mut().zip(v).for_each(|(x, y)| *x = y as u64);
                });
                out
            }
        };
        Ok((b, a))
    }
}

/// Bit-pack one row: value i at bits [i*bits, (i+1)*bits) of a little-endian u32 stream, rest of `out` zero.
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

/// Inverse of `pack_row` into u64 residues.
pub fn unpack_row(packed: &[u32], bits: u32, out: &mut [u64]) {
    let mask = if bits == 32 { u32::MAX as u64 } else { (1u64 << bits) - 1 };
    for (i, o) in out.iter_mut().enumerate() {
        let bit = i as u64 * bits as u64;
        let (w, off) = ((bit >> 5) as usize, bit & 31);
        let lo = packed[w] as u64 | (packed.get(w + 1).copied().unwrap_or(0) as u64) << 32;
        *o = (lo >> off) & mask;
    }
}

/// The bootstrap's encapsulation keys (named fields; FIDESlib's 2N-2 / 2N-4 key-map slots).
pub struct BtsKeys<'a, W: DevWord> {
    /// dense -> sparse, bottom composite level only (OpenFHE patch 0013): shape `KskShape::bottom`.
    pub enc_dense_to_sparse: DevKsk<'a, W>,
    /// sparse -> dense, full rows (used at the raised level).
    pub enc_sparse_to_dense: DevKsk<'a, W>,
}

/// A rotation key request: signed step (normalized mod N/2 when stored) and an optional band top.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotSpec {
    pub step: i64,
    /// Highest ciphertext top limb the key serves (None: full rows).
    pub q_top: Option<usize>,
}

/// What a `KeySet` holds; `bytes` sizes the "keys" MemPlan section.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeySetSpec {
    pub relin: bool,
    pub conj: bool,
    pub rots: Vec<RotSpec>,
    /// The two encapsulation keys (sparse-secret bootstrap routes).
    pub encaps: bool,
}

impl KeySetSpec {
    /// Bootstrap rotation steps at full rows plus model steps limited to ciphertexts with top limb <= `band`
    /// (`cfg.bts.rot_key_band`; None = full rows for every step). A step in both sets is stored once, full; steps
    /// are normalized mod N/2 and step 0 dropped. relin, conj and the encapsulation keys included.
    pub fn with_band(bts_steps: &[i64], model_steps: &[i64], band: Option<usize>, n: usize) -> Self {
        let h = (n / 2) as i64;
        let mut m: BTreeMap<i64, Option<usize>> = BTreeMap::new();
        for &s in model_steps {
            m.insert(s.rem_euclid(h), band);
        }
        for &s in bts_steps {
            m.insert(s.rem_euclid(h), None);
        }
        m.remove(&0);
        KeySetSpec { relin: true, conj: true, rots: m.into_iter().map(|(step, q_top)| RotSpec { step, q_top }).collect(), encaps: true }
    }

    fn rot_shape(chain: &Chain, r: &RotSpec) -> Result<KskShape> {
        match r.q_top {
            Some(t) if t + 1 < chain.q.len() => KskShape::banded(chain, t),
            _ => Ok(KskShape::full(chain)),
        }
    }

    /// Every key (role, shape), carve order.
    pub fn keys(&self, chain: &Chain) -> Result<Vec<(KeyRole, KskShape)>> {
        let full = KskShape::full(chain);
        let mut v = Vec::new();
        if self.relin {
            v.push((KeyRole::Relin, full));
        }
        if self.conj {
            v.push((KeyRole::Conj, full));
        }
        for r in &self.rots {
            v.push((KeyRole::rot(r.step, chain.n)?, Self::rot_shape(chain, r)?));
        }
        if self.encaps {
            v.push((KeyRole::EncDenseToSparse, KskShape::bottom(chain)?));
            v.push((KeyRole::EncSparseToDense, full));
        }
        Ok(v)
    }

    /// Arena bytes of the whole set (reserve as "keys").
    pub fn bytes<W: DevWord>(&self, chain: &Chain, layout: KskLayout) -> Result<usize> {
        Ok(self.keys(chain)?.iter().map(|(_, sh)| sh.bytes::<W>(chain.n, layout)).sum())
    }
}

/// A level relinearization key (PROTOTYPE, bn-evalmod): for a ciphertext at limb top `top`, the special modulus is
/// P' = (Q primes top + 1 ..= q_top, idle at that level) x P and the digits partition Q primes 0..=top into `ndig`
/// balanced runs of at most |P'| limbs (`level_relin_spec`). Relation, every slot t of the shape (Q ids 0..=q_top, P):
///   b_d[t] = -a_d[t] s[t] + e_d[t] + (t <= top && t in digit d ? (P' mod q_t) s^2[t] : 0),  a_d = KSKA(seed, d)
/// i.e. the relinearization key's relation with P' for P and the level's digits: modulus Q P (the chain's), dense
/// ternary s, sigma 3.19 - the RLWE instance of every key of the set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelRelinSpec {
    pub top: usize,
    pub digits: Vec<Vec<usize>>,
    /// P' prime ids: the idle Q ids, then P.
    pub specials: Vec<usize>,
    pub shape: KskShape,
}

/// The level relinearization key for limb top `top` with |P'| <= `cap` (the wide bconv: digits and P' + 2 <= 48), or
/// None when it would not take fewer digits than the chain's partition.
pub fn level_relin_spec(chain: &Chain, top: usize, cap: usize) -> Option<LevelRelinSpec> {
    let (digits, specials, q_top) = chain.level_relin_layout(top, cap)?;
    let shape = KskShape { q_top, ndig: digits.len(), nq: chain.q.len(), kp: chain.p.len() };
    Some(LevelRelinSpec { top, digits, specials, shape })
}

impl LevelRelinSpec {
    /// prime id -> digit (Q ids <= top), -1 otherwise (P' rows), over `rows` ids.
    pub fn digit_of(&self, rows: usize) -> Vec<i32> {
        (0..rows).map(|t| self.digits.iter().position(|g| g.contains(&t)).map_or(-1, |d| d as i32)).collect()
    }
    /// P' mod q_t by prime id (0 on P' rows), over `rows` ids.
    pub fn pmod(&self, primes: &[u64], rows: usize) -> Vec<u64> {
        (0..rows).map(|t| if t <= self.top { self.specials.iter().fold(1u64, |acc, &x| mulmod(acc, primes[x] % primes[t], primes[t])) } else { 0 }).collect()
    }
}

/// The device key set .
pub struct KeySet<'a, W: DevWord> {
    pub layout: KskLayout,
    pub relin: Option<DevKsk<'a, W>>,
    pub conj: Option<DevKsk<'a, W>>,
    /// Rotation keys by step normalized to `[0, N/2)` (`rot` normalizes the lookup).
    pub rots: BTreeMap<i64, DevKsk<'a, W>>,
    pub bts: Option<BtsKeys<'a, W>>,
    /// Level relinearization keys by limb top (`LevelRelinSpec`; prototype, empty unless carved).
    pub relin_levels: BTreeMap<usize, (LevelRelinSpec, DevKsk<'a, W>)>,
    n: usize,
}

impl<'a, W: DevWord> KeySet<'a, W> {
    /// Carve every key of `spec` (setup only, before `Arena::freeze`); fill with keygen / import.
    pub fn carve(arena: &'a Arena, chain: &Chain, spec: &KeySetSpec, layout: KskLayout) -> Result<Self> {
        if let KskLayout::PackedSeeded { bits } = layout {
            let mx = chain.prime_bits().into_iter().max().unwrap_or(0);
            if W::BITS != 32 || bits < mx || bits > 31 {
                return Err(shape("KskLayout::PackedSeeded", format!("bits {bits} at {}-bit words for primes up to {mx} bits", W::BITS)));
            }
        }
        let n = chain.n;
        let mut ks = KeySet { layout, relin: None, conj: None, rots: BTreeMap::new(), bts: None, relin_levels: BTreeMap::new(), n };
        let (mut d2s, mut s2d) = (None, None);
        for (role, sh) in spec.keys(chain)? {
            let k = DevKsk::carve(arena, role, sh, layout, n)?;
            match role {
                KeyRole::Relin => ks.relin = Some(k),
                KeyRole::Conj => ks.conj = Some(k),
                KeyRole::Rot { step, .. } => {
                    ks.rots.insert(step.rem_euclid((n / 2) as i64), k);
                }
                KeyRole::EncDenseToSparse => d2s = Some(k),
                KeyRole::EncSparseToDense => s2d = Some(k),
            }
        }
        if let (Some(a), Some(b)) = (d2s, s2d) {
            ks.bts = Some(BtsKeys { enc_dense_to_sparse: a, enc_sparse_to_dense: b });
        }
        Ok(ks)
    }

    /// The rotation key of `step` (any sign; normalized mod N/2). Missing = error naming the knob.
    pub fn rot(&self, step: i64) -> Result<&DevKsk<'a, W>> {
        let k = step.rem_euclid((self.n / 2) as i64);
        self.rots.get(&k).ok_or_else(|| Error::Mem {
            knob: "KeySetSpec::rots",
            value: k as usize,
            msg: format!("no rotation key for step {step} (normalized {k}); add it to the model/bts rotation set"),
        })
    }
    /// Carve the level relinearization keys of `specs` (setup only, before `Arena::freeze`).
    pub fn carve_relin_levels(&mut self, arena: &'a Arena, specs: &[LevelRelinSpec]) -> Result<()> {
        for sp in specs {
            let k = DevKsk::carve(arena, KeyRole::Relin, sp.shape, self.layout, self.n)?;
            self.relin_levels.insert(sp.top, (sp.clone(), k));
        }
        Ok(())
    }
    /// Arena bytes of `carve_relin_levels`.
    pub fn relin_levels_bytes(specs: &[LevelRelinSpec], layout: KskLayout, n: usize) -> usize {
        specs
            .iter()
            .map(|sp| match layout {
                KskLayout::PackedSeeded { .. } => Arena::bytes_for::<u32>(sp.shape.b_words(n, layout)),
                KskLayout::Dense => Arena::bytes_for::<W>(sp.shape.b_words(n, layout)) + Arena::bytes_for::<W>(sp.shape.a_words(n, layout)),
            } + Arena::bytes_for::<u32>(8))
            .sum()
    }
    pub fn relin(&self) -> Result<&DevKsk<'a, W>> {
        self.relin.as_ref().ok_or_else(|| Error::Mem { knob: "KeySetSpec::relin", value: 0, msg: "no relinearization key".into() })
    }
    pub fn conj(&self) -> Result<&DevKsk<'a, W>> {
        self.conj.as_ref().ok_or_else(|| Error::Mem { knob: "KeySetSpec::conj", value: 0, msg: "no conjugation key".into() })
    }
    pub fn bts(&self) -> Result<&BtsKeys<'a, W>> {
        self.bts.as_ref().ok_or_else(|| Error::Mem { knob: "KeySetSpec::encaps", value: 0, msg: "no encapsulation keys".into() })
    }
    /// Every key, carve order.
    pub fn iter(&self) -> impl Iterator<Item = &DevKsk<'a, W>> {
        self.relin.iter().chain(self.conj.iter()).chain(self.rots.values()).chain(self.bts.iter().flat_map(|b| [&b.enc_dense_to_sparse, &b.enc_sparse_to_dense]))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut DevKsk<'a, W>> {
        self.relin
            .iter_mut()
            .chain(self.conj.iter_mut())
            .chain(self.rots.values_mut())
            .chain(self.bts.iter_mut().flat_map(|b| [&mut b.enc_dense_to_sparse, &mut b.enc_sparse_to_dense]))
    }
    pub fn len(&self) -> usize {
        self.iter().count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Device bytes of every key.
    pub fn bytes(&self) -> usize {
        self.iter().map(|k| k.bytes()).sum()
    }
}

/// Load host key material (full `[digit][limb][N]` rows in prime-id order, `KskMaterial`) into `key`'s layout: rows
/// outside the key's shape are dropped (checked zero first when `require_zero_outside`: the bottom-only
/// encapsulation key), `b` packed on the host, the seed uploaded, dense `a` regenerated from the seed on the host when
/// the layout stores it. Synchronous (setup).
pub fn import_material<W: DevWord>(key: &mut DevKsk<'_, W>, s: &Stream<Compute>, chain: &Chain, m: &KskMaterial, require_zero_outside: bool) -> Result<()> {
    let n = chain.n;
    let primes = chain.primes();
    if m.n != n || m.moduli.len() != primes.len() || m.dnum != chain.dnum() {
        return Err(shape("import_material", format!("material dnum {} x {} limbs x N {} vs chain {} x {} x {}", m.dnum, m.moduli.len(), m.n, chain.dnum(), primes.len(), n)));
    }
    if let Some(i) = m.moduli.iter().zip(&primes).position(|(&a, &b)| a as u64 != b) {
        return Err(shape("import_material", format!("limb {i} modulus {} is not prime id {i} = {}", m.moduli[i], primes[i])));
    }
    let sh = key.shape;
    let ids = sh.row_ids();
    if require_zero_outside {
        for d in 0..m.dnum {
            for l in 0..primes.len() {
                let kept = d < sh.ndig && sh.slot_of(l).is_some();
                if !kept && m.b_row(d, l).iter().any(|&x| x != 0) {
                    return Err(shape("import_material", format!("{:?}: digit {d} limb {l} is outside shape {sh:?} but nonzero", key.role)));
                }
            }
        }
    }
    key.seed = m.seed;
    key.seed_dev.upload(s, &m.seed.0)?;
    let rows = ids.len();
    match &mut key.store {
        KskStore::PackedSeeded { b, bits, row_words } => {
            let (bits, rw) = (*bits, *row_words);
            let mut packed = vec![0u32; sh.ndig * rows * rw];
            packed.par_chunks_mut(rw).enumerate().for_each(|(r, o)| pack_row(m.b_row(r / rows, ids[r % rows]), bits, o));
            b.upload(s, &packed)?;
        }
        KskStore::Dense { b, a } => {
            let mut hb = vec![W::default(); sh.ndig * rows * n];
            hb.par_chunks_mut(n).enumerate().for_each(|(r, o)| {
                o.iter_mut().zip(m.b_row(r / rows, ids[r % rows])).for_each(|(x, &y)| *x = W::from_u64(y as u64));
            });
            b.upload(s, &hb)?;
            // the word's frozen lane (KSKA at u32, KSKB at u64), as the device regenerates it
            hb.par_chunks_mut(n).enumerate().try_for_each(|(r, o)| -> std::result::Result<(), pc_core::error::Error> {
                let mut v = vec![0u64; n];
                pc_core::keys::ksk_expand_row(&m.seed, (r / rows) as u32, primes[ids[r % rows]], W::BITS, &mut v)?;
                o.iter_mut().zip(v).for_each(|(x, y)| *x = W::from_u64(y));
                Ok(())
            })?;
            a.upload(s, &hb)?;
        }
    }
    Ok(())
}

/// Load a key from its own-shape packed `b` (`[digit][slot][row_words]` u32 words at `bits`, the client bundle's
/// format) and its seed: a packed key of the same packing takes the words as they are, any other layout is unpacked
/// on the host (dense `a` regenerated from the seed). Synchronous (setup).
pub fn import_packed<W: DevWord>(key: &mut DevKsk<'_, W>, s: &Stream<Compute>, chain: &Chain, seed: KskSeed, bits: u32, row_words: usize, packed: &[u32]) -> Result<()> {
    let n = chain.n;
    let sh = key.shape;
    let rows = sh.rows();
    if row_words < packed_row_words(n, bits) || packed.len() != sh.ndig * rows * row_words {
        return Err(shape("import_packed", format!("{:?}: {} words at {bits} bits / {row_words} per row, shape {sh:?} needs {}", key.role, packed.len(), sh.ndig * rows * row_words)));
    }
    let primes = chain.primes();
    let ids = sh.row_ids();
    key.seed = seed;
    key.seed_dev.upload(s, &seed.0)?;
    let unpacked = |r: usize, o: &mut [u64]| unpack_row(&packed[r * row_words..(r + 1) * row_words], bits, o);
    match &mut key.store {
        KskStore::PackedSeeded { b, bits: kb, row_words: krw } if *kb == bits && *krw == row_words => b.upload(s, packed)?,
        KskStore::PackedSeeded { b, bits: kb, row_words: krw } => {
            let (kb, krw) = (*kb, *krw);
            let mut out = vec![0u32; sh.ndig * rows * krw];
            out.par_chunks_mut(krw).enumerate().for_each(|(r, o)| {
                let mut v = vec![0u64; n];
                unpacked(r, &mut v);
                pack_row(&v.into_iter().map(|x| x as u32).collect::<Vec<_>>(), kb, o);
            });
            b.upload(s, &out)?;
        }
        KskStore::Dense { b, a } => {
            let mut hb = vec![W::default(); sh.ndig * rows * n];
            hb.par_chunks_mut(n).enumerate().for_each(|(r, o)| {
                let mut v = vec![0u64; n];
                unpacked(r, &mut v);
                o.iter_mut().zip(v).for_each(|(x, y)| *x = W::from_u64(y));
            });
            b.upload(s, &hb)?;
            hb.par_chunks_mut(n).enumerate().try_for_each(|(r, o)| -> std::result::Result<(), pc_core::error::Error> {
                let mut v = vec![0u64; n];
                pc_core::keys::ksk_expand_row(&seed, (r / rows) as u32, primes[ids[r % rows]], W::BITS, &mut v)?;
                o.iter_mut().zip(v).for_each(|(x, y)| *x = W::from_u64(y));
                Ok(())
            })?;
            a.upload(s, &hb)?;
        }
    }
    Ok(())
}

/// The secret on the device: one context-layout ciphertext slot, part 0 = `s`, part 1 = `s_sparse`, both in eval
/// form over every prime id (Q u P).
pub struct DevSecret<'a, W: DevWord> {
    pub buf: CtBuf<'a, Compute, W>,
    /// The sparse secret is loaded (an oracle import may carry only `s`).
    pub has_sparse: bool,
}

impl<'a, W: DevWord> DevSecret<'a, W> {
    pub fn bytes(lay: &Layout) -> usize {
        Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, 1))
    }
    pub fn carve(arena: &'a Arena, lay: Layout) -> Result<Self> {
        Ok(DevSecret { buf: CtBuf::new(arena, "keys.secret", lay, 1)?, has_sparse: false })
    }
    /// Upload `sk` (signed coefficients -> residues per prime id) and NTT it on `s` (synced). An empty `s_sparse`
    /// leaves part 1 = `s` and `has_sparse` false.
    pub fn load<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sk: &SecretKey) -> Result<()> {
        let n = ctx.lay.n;
        if sk.s.len() != n || (!sk.s_sparse.is_empty() && sk.s_sparse.len() != n) {
            return Err(shape("DevSecret::load", format!("secret lengths {} / {} for N = {n}", sk.s.len(), sk.s_sparse.len())));
        }
        let primes = ctx.chain.primes();
        let ids: Vec<usize> = (0..primes.len()).collect();
        for (part, v) in [(0usize, &sk.s), (1, &sk.s_sparse)] {
            let src: &[i8] = if v.is_empty() { &sk.s } else { v };
            let rows = signed_rows::<W>(&src.iter().map(|&x| x as i64).collect::<Vec<_>>(), &primes);
            self.buf.upload_rows(s, 0, part, &ids, &rows)?;
        }
        ntt::forward(ctx, s, &mut self.buf, Span::QP { top: ctx.lay.nq - 1 }, Parts::Both)?;
        s.sync()?;
        self.has_sparse = !sk.s_sparse.is_empty();
        Ok(())
    }
}

/// Signed small coefficients -> residue rows `[prime][N]` in words W.
pub fn signed_rows<W: DevWord>(c: &[i64], primes: &[u64]) -> Vec<W> {
    let n = c.len();
    let mut out = vec![W::default(); primes.len() * n];
    out.par_chunks_mut(n).zip(primes.par_iter()).for_each(|(o, &p)| {
        for (x, &v) in o.iter_mut().zip(c) {
            *x = W::from_u64(v.rem_euclid(p as i64) as u64);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_core::params::ChainSpec;

    #[test]
    fn pack_roundtrip() {
        let n = 1024;
        for bits in [27u32, 28, 31] {
            let vals: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761) & ((1 << bits) - 1)).collect();
            let mut p = vec![0u32; packed_row_words(n, bits)];
            pack_row(&vals, bits, &mut p);
            let mut u = vec![0u64; n];
            unpack_row(&p, bits, &mut u);
            assert!(u.iter().zip(&vals).all(|(a, &b)| *a == b as u64), "bits {bits}");
            assert_eq!(*p.last().unwrap(), 0, "guard word");
        }
    }

    #[test]
    fn shapes_n32() {
        let chain = Chain::generate(&ChainSpec::n32()).unwrap();
        let full = KskShape::full(&chain);
        assert_eq!((full.rows(), full.ndig), (63, 6));
        assert!((0..63).all(|i| full.slot_of(i) == Some(i)));
        let b = KskShape::bottom(&chain).unwrap();
        assert_eq!((b.q_top, b.ndig, b.rows()), (1, 1, 11));
        assert_eq!(b.slot_of(54), Some(2));
        assert_eq!(b.slot_of(2), None);
        let band = KskShape::banded(&chain, 32).unwrap();
        assert_eq!((band.ndig, band.rows()), (4, 42));
        let spec = KeySetSpec::with_band(&[1, -1, 32768], &[1, 2, 5], Some(32), chain.n);
        let keys = spec.keys(&chain).unwrap();
        // 1 full (bts wins), 32767 full, 2 and 5 banded, relin, conj, 2 encaps
        assert_eq!(keys.len(), 8);
        let lay = KskLayout::packed_for::<u32>(&chain).unwrap();
        assert_eq!(lay, KskLayout::PackedSeeded { bits: 28 });
        let full_b = full.bytes::<u32>(chain.n, lay);
        assert_eq!(full_b, (6 * 63 * packed_row_words(chain.n, 28) * 4).div_ceil(256) * 256 + 256);
    }
}
