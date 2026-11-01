//! FIDESlib's FLEXIBLEAUTO metadata logic (`Ciphertext.cpp`, `Plaintext.cu`), replayed on metadata only, in
//! FIDESlib's own coordinates (top limb index, NoiseLevel, NoiseFactor). This is the ground truth the `Shadow` rules
//! and the EvalMod program builder are both written on; the public API converts to `CtMeta` at its boundary.
//!
//! Every adjustment step is reported as a `Step` so a caller can turn it into a prelude (`Shadow`) or a device op
//! (`evalmod`). Scale arithmetic runs only when scale tables are present (`Fx::st`); without them the levels and
//! degrees are exact and `nf` stays 0.
use crate::error::{rule, Result};
use pc_core::scale::ScaleTables;

/// FIDESlib ciphertext/plaintext metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FMeta {
    /// Top limb index (`getLevel()`).
    pub top: i32,
    /// NoiseLevel (1, or 2 with a pending rescale; 3 transiently inside an adjust).
    pub nl: i32,
    /// NoiseFactor.
    pub nf: f64,
}

/// One metadata-changing step of an adjustment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// `rescale()`: drop `unit` primes, NoiseLevel - 1.
    Rescale,
    /// `dropToLevel(top)` to this top limb index.
    DropTo(i32),
    /// `multScalarNoPrecheck(c)`: the scalar encoded at the current top (`ElemForEvalMult(top, c)`), NoiseLevel + 1.
    Scalar(f64),
}

#[derive(Clone, Copy)]
pub struct Fx<'a> {
    pub d: i32,
    pub st: Option<&'a ScaleTables>,
}

impl<'a> Fx<'a> {
    pub fn sf(&self, top: i32) -> Result<f64> {
        match self.st {
            Some(st) => Ok(st.sf(top)?),
            None => Ok(0.0),
        }
    }
    fn mrp(&self, top: i32) -> f64 {
        self.st.map_or(0.0, |st| st.mod_reduce_product(top))
    }
    fn sf_big(&self, top: i32) -> f64 {
        self.st.map_or(0.0, |st| st.sfr_big.get(top as usize).copied().unwrap_or(0.0))
    }

    pub fn rescale(&self, m: &mut FMeta, steps: &mut Vec<Step>) -> Result<()> {
        if m.nl < 2 {
            return Err(rule("rescale", format!("NoiseLevel {} has no pending rescale (top {})", m.nl, m.top)));
        }
        let g = self.drop_at(m.top);
        if m.top < g {
            return Err(rule("rescale", format!("top limb {} has no {g} primes left to drop", m.top)));
        }
        steps.push(Step::Rescale);
        m.top -= g;
        if self.st.is_some() {
            m.nf /= self.mrp(m.top + g);
        }
        m.nl -= 1;
        Ok(())
    }
    /// Primes a rescale from `top` drops: `d`, but at a lean chain's single-prime level (scale tables only).
    pub fn drop_at(&self, top: i32) -> i32 {
        self.st.and_then(|st| st.drop.get(top as usize).copied()).filter(|&g| g > 0).map_or(self.d, |g| g as i32)
    }
    /// The top of the level below `top` and of the level above it: `top -+ d`, but by the levels' primes on a chain
    /// with an alternating user region (`ScaleTables::singles`; scale tables only).
    pub fn down(&self, top: i32) -> i32 {
        match self.st {
            Some(st) if st.singles != 0 => top - self.drop_at(top),
            _ => top - self.d,
        }
    }
    pub fn up(&self, top: i32) -> i32 {
        match self.st {
            Some(st) if st.singles != 0 => st.up.get(top as usize).copied().filter(|&u| u > 0).map_or(top + self.d, |u| u as i32),
            _ => top + self.d,
        }
    }
    pub fn drop_to(&self, m: &mut FMeta, top: i32, steps: &mut Vec<Step>) {
        if top < m.top {
            steps.push(Step::DropTo(top));
            m.top = top;
        }
    }
    /// `multScalarNoPrecheck(c)`.
    pub fn scalar_noprecheck(&self, m: &mut FMeta, c: f64, steps: &mut Vec<Step>) -> Result<()> {
        let sf = self.sf(m.top)?;
        steps.push(Step::Scalar(c));
        m.nl += 1;
        if self.st.is_some() {
            m.nf *= sf;
        }
        Ok(())
    }
    /// `multScalar(c)`: rescale first if one is pending.
    pub fn scalar(&self, m: &mut FMeta, c: f64, steps: &mut Vec<Step>) -> Result<()> {
        if m.nl == 2 {
            self.rescale(m, steps)?;
        }
        self.scalar_noprecheck(m, c, steps)
    }

    /// `Ciphertext::adjustForAddOrSubBody` (FLEXIBLEAUTO): bring `a` to `b`'s level and degree. false = `a` is the
    /// deeper (or the lower-degree at the same level) operand and nothing was changed; the caller adjusts a copy of `b`.
    pub fn adjust_add(&self, a: &mut FMeta, b: &FMeta, steps: &mut Vec<Step>) -> Result<bool> {
        let (c1lvl, c2lvl, c1d, c2d) = (a.top, b.top, a.nl, b.nl);
        if c1lvl > c2lvl {
            if c1d == 2 {
                if c2d == 2 {
                    let (scf1, scf2) = (a.nf, b.nf);
                    let k = self.k_resc(scf2, scf1, c1lvl)?;
                    self.scalar_noprecheck(a, k, steps)?;
                    self.rescale(a, steps)?;
                    self.drop_to(a, b.top, steps);
                    a.nf = b.nf;
                } else if self.down(c1lvl) == c2lvl {
                    self.rescale(a, steps)?;
                } else {
                    let scf1 = a.nf;
                    let scf2 = self.sf_big(self.up(c2lvl));
                    let k = self.k_resc(scf2, scf1, c1lvl)?;
                    self.scalar_noprecheck(a, k, steps)?;
                    self.rescale(a, steps)?;
                    if a.top > self.up(b.top) {
                        self.drop_to(a, self.up(b.top), steps);
                    }
                    self.rescale(a, steps)?;
                    a.nf = b.nf;
                }
            } else if c2d == 2 {
                let (scf1, scf2) = (a.nf, b.nf);
                let k = self.k_plain(scf2, scf1, c1lvl)?;
                self.scalar_noprecheck(a, k, steps)?;
                self.drop_to(a, c2lvl, steps);
                a.nf = scf2;
            } else {
                let scf1 = a.nf;
                let scf2 = self.sf_big(self.up(c2lvl));
                let k = self.k_plain(scf2, scf1, c1lvl)?;
                self.scalar_noprecheck(a, k, steps)?;
                if c1lvl > self.up(c2lvl) {
                    self.drop_to(a, self.up(c2lvl), steps);
                }
                self.rescale(a, steps)?;
                a.nf = b.nf;
            }
            Ok(true)
        } else if c1lvl < c2lvl {
            Ok(false)
        } else {
            if c1d < c2d {
                self.scalar(a, 1.0, steps)?;
            } else if c2d < c1d {
                return Ok(false);
            }
            Ok(true)
        }
    }

    /// The adjust constants, associated exactly as FIDESlib writes them: `scf2 / scf1 * q1 / scf` when the adjust
    /// rescales, `scf2 / scf1 / scf` when it does not (`scf = sfAtLimb(c1lvl)`, `q1 = modReduceProduct(c1lvl)`).
    /// Levels-only runs carry no scales and record 1.0.
    fn k_resc(&self, scf2: f64, scf1: f64, c1lvl: i32) -> Result<f64> {
        if self.st.is_none() {
            return Ok(1.0);
        }
        Ok(scf2 / scf1 * self.mrp(c1lvl) / self.sf(c1lvl)?)
    }
    fn k_plain(&self, scf2: f64, scf1: f64, c1lvl: i32) -> Result<f64> {
        if self.st.is_none() {
            return Ok(1.0);
        }
        Ok(scf2 / scf1 / self.sf(c1lvl)?)
    }

    /// `Ciphertext::adjustForMultBody`.
    pub fn adjust_mult(&self, a: &mut FMeta, b: &FMeta, steps: &mut Vec<Step>) -> Result<bool> {
        if self.adjust_add(a, b, steps)? {
            if a.nl == 2 {
                self.rescale(a, steps)?;
            }
            Ok(b.nl != 2)
        } else {
            if a.nl == 2 {
                self.rescale(a, steps)?;
            }
            Ok(false)
        }
    }

    /// `Plaintext::adjustPlaintextToCiphertext` (FLEXIBLEAUTO) for the cases FIDESlib can execute: a plaintext at the
    /// ciphertext's top, or a richer degree-1 one. A richer degree-2 plaintext would be rescaled twice there
    /// (`multScalar` prechecks, then `rescale` asserts) and a deeper one fails; both are refused.
    pub fn adjust_pt(&self, p: &mut FMeta, c: &FMeta, steps: &mut Vec<Step>) -> Result<bool> {
        if p.top > c.top {
            if p.nl == 2 {
                return Err(rule(
                    "adjust_pt",
                    format!("a degree-2 plaintext at top {} richer than its ciphertext (top {}) is not executable", p.top, c.top),
                ));
            }
            let scf1 = p.nf;
            if c.nl == 2 {
                let k = self.k_plain(c.nf, scf1, p.top)?;
                self.scalar_noprecheck(p, k, steps)?;
                self.drop_to(p, c.top, steps);
                p.nf = c.nf;
            } else {
                let scf2 = self.sf_big(self.up(c.top));
                let k = self.k_plain(scf2, scf1, p.top)?;
                self.scalar_noprecheck(p, k, steps)?;
                if p.top > self.up(c.top) {
                    self.drop_to(p, self.up(c.top), steps);
                }
                self.rescale(p, steps)?;
                p.nf = c.nf;
            }
            Ok(true)
        } else if p.top < c.top {
            Ok(false)
        } else {
            if p.nl < c.nl {
                self.scalar(p, 1.0, steps)?;
            } else if c.nl < p.nl {
                self.rescale(p, steps)?;
            }
            Ok(true)
        }
    }
}
