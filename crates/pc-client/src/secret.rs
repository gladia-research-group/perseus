//! The client's secret: the dense ternary key `s` and the sparse encapsulation key (Hamming weight 32), saved in a
//! small raw file that only the client ever reads.
use crate::error::{Error, Result};
use pc_core::keys::{Prng, SecretKey};

const MAGIC: &[u8; 4] = b"PCSK";

pub struct ClientSecret {
    pub sk: SecretKey,
}

impl ClientSecret {
    pub fn generate(prng: &mut Prng, n: usize) -> Result<Self> {
        Ok(ClientSecret { sk: SecretKey::sample(prng, n)? })
    }

    /// `generate` with a sparse secret of Hamming weight `h`.
    pub fn generate_h(prng: &mut Prng, n: usize, h: usize) -> Result<Self> {
        Ok(ClientSecret { sk: SecretKey::sample_h(prng, n, h)? })
    }

    /// `MAGIC | u32 n | s (n i8) | s_sparse (n i8)`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.sk.s.len();
        let mut b = Vec::with_capacity(8 + 2 * n);
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&(n as u32).to_le_bytes());
        b.extend(self.sk.s.iter().map(|&x| x as u8));
        b.extend(self.sk.s_sparse.iter().map(|&x| x as u8));
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() < 8 || &b[..4] != MAGIC {
            return Err(Error::Wire("not a client secret (magic)".into()));
        }
        let n = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
        if b.len() != 8 + 2 * n {
            return Err(Error::Wire(format!("client secret of {} bytes for n = {n}", b.len())));
        }
        let s: Vec<i8> = b[8..8 + n].iter().map(|&x| x as i8).collect();
        let s_sparse: Vec<i8> = b[8 + n..].iter().map(|&x| x as i8).collect();
        if s.iter().chain(&s_sparse).any(|&x| !(-1..=1).contains(&x)) {
            return Err(Error::Wire("client secret is not ternary".into()));
        }
        Ok(ClientSecret { sk: SecretKey { s, s_sparse } })
    }
}
