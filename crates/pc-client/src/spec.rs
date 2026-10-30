//! What a server asks the client to generate (`KeySpec`, sent as JSON before the bundle), and the row geometry of a
//! switching key (`KeyShape`, the same fields and rules as the device's `KskShape`).
//!
//! Relation of every switching key (eval form, slot m = prime id t of digit d, `a` = KSKA(seed, d, p_t)):
//!   b = -a s_new + e + [t in digit d] (P mod q_t) s_old
//! Roles: relin s_old = s^2, s_new = s; rotation by r (Galois g = 5^r mod 2N) and every other automorphism key
//! s_old = s, s_new = sigma_{g^-1}(s) (the evaluator switches, then applies sigma_g); dense->sparse s_old = s,
//! s_new = s_sparse; sparse->dense the reverse; the SPRU block key switch s_old = s, s_new = s_block.
use crate::error::{Error, Result};
use pc_core::galois;
use pc_core::params::Chain;
use serde::{Deserialize, Serialize};

/// Which rows a key stores: digits `0..ndig`, each the Q rows `0..=q_top` then the `kp` special rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KeyShape {
    pub q_top: usize,
    pub ndig: usize,
    pub nq: usize,
    pub kp: usize,
}

impl KeyShape {
    pub fn full(chain: &Chain) -> Self {
        KeyShape { q_top: chain.q.len() - 1, ndig: chain.dnum(), nq: chain.q.len(), kp: chain.p.len() }
    }
    /// Rows for ciphertexts whose top limb is <= `q_top` (every digit starting at or below it).
    pub fn banded(chain: &Chain, q_top: usize) -> Result<Self> {
        if q_top >= chain.q.len() {
            return Err(Error::Spec(format!("band top {q_top} >= Q primes {}", chain.q.len())));
        }
        let ndig = chain.digits.iter().filter(|d| d.first().is_some_and(|&f| f <= q_top)).count();
        Ok(KeyShape { q_top, ndig, nq: chain.q.len(), kp: chain.p.len() })
    }
    /// The bottom composite level only (digit 0, `level_unit` Q rows) over P' (`Chain::p_sparse`): the keys to the
    /// sparse secrets, whose modulus is q0 x P'.
    pub fn bottom(chain: &Chain) -> Result<Self> {
        Ok(KeyShape { kp: chain.p_sparse, ..Self::banded(chain, chain.level_unit - 1)? })
    }
    pub fn rows(&self) -> usize {
        self.q_top + 1 + self.kp
    }
    pub fn row_ids(&self) -> Vec<usize> {
        (0..=self.q_top).chain(self.nq..self.nq + self.kp).collect()
    }
    /// Is this a shape the chain can have (full, banded or bottom with the matching digit count)?
    pub fn check(&self, chain: &Chain) -> Result<()> {
        let bottom = Self::bottom(chain)?;
        let want = if self.q_top == bottom.q_top && self.kp == bottom.kp { bottom } else { Self::banded(chain, self.q_top)? };
        if *self != want {
            return Err(Error::Spec(format!("shape {self:?} does not match the chain's band at q_top {} ({want:?})", self.q_top)));
        }
        Ok(())
    }
}

/// What a switching key switches (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Role {
    Relin,
    /// Left rotation by `step` (Galois element 5^step mod 2N).
    Rot { step: i64, galois: u64 },
    Conj,
    EncDenseToSparse,
    EncSparseToDense,
    /// The SPRU route's dense -> block-binary key switch (bottom shape).
    SpruAtob,
    /// Any other automorphism key, by Galois element (the SPRU route's conj o rot(1), g = 2N - 5).
    Auto { galois: u64 },
}

impl Role {
    pub fn rot(step: i64, n: usize) -> Result<Self> {
        Ok(Role::Rot { step, galois: galois::rotation_galois(step, n)? })
    }
    /// The automorphism of an automorphism key (None for relin and the secret switches).
    pub fn galois(&self, n: usize) -> Option<u64> {
        match self {
            Role::Rot { galois, .. } | Role::Auto { galois } => Some(*galois),
            Role::Conj => Some(2 * n as u64 - 1),
            _ => None,
        }
    }
}

/// One switching key to generate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyReq {
    #[serde(flatten)]
    pub role: Role,
    pub shape: KeyShape,
}

/// The SPRU route's material: the block key switch (a `KeyReq` with `Role::SpruAtob`), the conj o rot(1) key
/// (`Role::Auto`), and the 2n bootstrapping-key ciphertexts Enc(S_u) at `cs_level` with scale `cs_nf`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpruReq {
    /// Blocks of the block-binary key.
    pub h: usize,
    /// Components per slot (2).
    pub n: usize,
    pub cs_level: u32,
    pub cs_nf: f64,
}

impl SpruReq {
    pub fn check(&self, ring_n: usize) -> Result<()> {
        let ok = self.h >= 2 && self.h.is_power_of_two() && self.n == 2 && ring_n % self.h == 0 && (ring_n / self.h) % (2 * self.n) == 0;
        if !ok {
            return Err(Error::Spec(format!("SPRU h = {} must be a power of two dividing N/{}", self.h, 2 * self.n)));
        }
        Ok(())
    }
}

/// Everything a server's plan needs from the client (the server's `Session.key_manifest`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeySpec {
    /// Chain preset name ("n32").
    pub chain: String,
    /// `params_hash` of that chain: both sides must derive the same primes.
    pub params_hash: String,
    /// The plan contract: hash of the load-bearing Config both sides run.
    pub contract: String,
    pub keys: Vec<KeyReq>,
    pub spru: Option<SpruReq>,
    /// PCMM token-lane shapes t (one key set per distinct t).
    pub pcmm_t: Vec<usize>,
    pub ccmm: bool,
    /// A public key, for the server's encryptions of public values.
    pub public_key: bool,
}

impl KeySpec {
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Shapes valid for the chain, roles unique, SPRU keys present iff SPRU is asked for.
    pub fn check(&self, chain: &Chain) -> Result<()> {
        let mut seen = std::collections::BTreeSet::new();
        for k in &self.keys {
            k.shape.check(chain)?;
            let id = serde_json::to_string(&k.role)?;
            if !seen.insert(id.clone()) {
                return Err(Error::Spec(format!("key {id} requested twice")));
            }
            // a sparse secret only at q0 x P' (the modulus its security verdict names)
            if matches!(k.role, Role::EncDenseToSparse | Role::SpruAtob) && k.shape != KeyShape::bottom(chain)? {
                return Err(Error::Spec(format!("{id}: a key to a sparse secret must have the bottom shape {:?}, not {:?}", KeyShape::bottom(chain)?, k.shape)));
            }
            if let Role::Rot { step, galois } = k.role {
                if galois::rotation_galois(step, chain.n)? != galois {
                    return Err(Error::Spec(format!("rotation {step}: Galois element {galois} is not 5^{step} mod 2N")));
                }
            }
        }
        let has = |f: &dyn Fn(&Role) -> bool| self.keys.iter().any(|k| f(&k.role));
        if let Some(s) = &self.spru {
            s.check(chain.n)?;
            if !has(&|r| matches!(r, Role::SpruAtob)) || !has(&|r| matches!(r, Role::Auto { .. })) {
                return Err(Error::Spec("SPRU material asked without its atob / conj o rot(1) keys".into()));
            }
        }
        Ok(())
    }
}

/// FNV-1a 64 (the repo's content hash: EvalMod tables, plan stamps).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Hash of everything key material depends on: ring dimension, primes (Q then P), digit partition, level unit,
/// word size.
pub fn params_hash(chain: &Chain) -> String {
    let mut b = Vec::new();
    b.extend((chain.n as u64).to_le_bytes());
    for &p in chain.q.iter().chain(&chain.p) {
        b.extend(p.to_le_bytes());
    }
    for d in &chain.digits {
        b.extend((d.len() as u64).to_le_bytes());
        for &i in d {
            b.extend((i as u64).to_le_bytes());
        }
    }
    b.extend((chain.level_unit as u64).to_le_bytes());
    b.extend(chain.word_bits.to_le_bytes());
    format!("{:016x}", fnv1a64(&b))
}

/// The chain of a preset name.
pub fn chain_of(name: &str) -> Result<Chain> {
    let spec = pc_core::params::ChainSpec::preset(name)?;
    Ok(Chain::generate(&spec)?)
}
