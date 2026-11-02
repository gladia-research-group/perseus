//! StC outside a bootstrap (FIDESlib `btsStcHigh`, perseus `bts.stc_high`): the dense route's shipped StC stages
//! applied to an ordinary ciphertext, the kept stages (bit s of `skip_mask` skips stage s: the CCMM's SinC encoding
//! runs stage 0 alone, or stages 1..2) re-levelled to `entry = out + n_kept d` .. `out + d` (on a lean chain the last
//! stage drops one prime of a pair, `StageShape::thin`), the last kept stage scaled by `factor` (times `nf_ref /
//! NoiseFactor` when `nf_ref > 0`, the output then at the reference scale), unfused stage ends, a rescale to `out`.
//!
//! The re-levelled stage sequences are cached per shape (out level, skip mask) in bounded LRU caches
//! (`StageCache`, entries keyed by the factor; FIDESlib kept one process-wide map capped at 24 and synchronized the
//! whole device after every call and before every eviction). Here a fill is ordered on the compute lane and the
//! pinned table buffer's reuse waits on its own event: no device-wide synchronization.
//! Shapes are fixed at setup (`StcHighSetup::shapes`, from the compiled programs): a call with another shape is a
//! setup error naming the knob, not an allocation.
use super::cts::{LtEnd, LtGeom};
use super::stc_first::{run_lt, run_lt_thin, CacheKey, RelevelSrc, StageCache, StageShape};
use super::{common_meta, BtsCx, BtsKeyRing, StcHigh};
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use crate::ks::KskSource;
use crate::poly::Layout;
use pc_cuda::{Arena, Compute, Device, Stream};

/// One stc_high call shape: output level (primes dropped, as `CtMeta::level`) and skip mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StcHighShape {
    pub out_level: u32,
    pub skip_mask: u32,
}

/// The shapes stc_high serves and the cache entries per shape.
#[derive(Clone, Debug, PartialEq)]
pub struct StcHighSetup {
    pub shapes: Vec<StcHighShape>,
    pub slots: usize,
}

/// The re-levelled stages of one shape (execution order). `lone`: a lean chain's single-prime top
/// (`Chain::lone_top`). Each stage drops a level onto the next one's top; an output top that no level lands on (a lean
/// chain's limb 3, inside the pair q3 q4: the CCMM operands) is reached by a last stage that drops one prime.
pub fn shape_stages(sh: StcHighShape, n_stc: usize, n_q: usize, d: usize, lone: Option<usize>) -> Result<Vec<StageShape>> {
    let keep: Vec<usize> = (0..n_stc).filter(|s| sh.skip_mask >> s & 1 == 0).collect();
    if keep.is_empty() {
        return Err(shape("stc_high", "every stage skipped"));
    }
    let out_top = (n_q - 1).checked_sub(sh.out_level as usize).ok_or_else(|| shape("stc_high", format!("output level {} past the chain", sh.out_level)))?;
    // the tops a level's rescale lands on: q0 q1, then every pair above (a lean chain: its single prime, pairs above)
    let lands = |t: usize| match lone {
        None => t % d == d - 1,
        Some(l) => t == d - 1 || t == l || (t > l && (t - l) % d == 0),
    };
    let mut stages = vec![(0, false); keep.len()];
    let mut top = out_top;
    for st in stages.iter_mut().rev() {
        *st = if lands(top) && lands(top + d) { (top + d, false) } else { (top + 1, true) };
        top = st.0;
    }
    if top >= n_q {
        return Err(shape("stc_high", format!("entry limb {top} above the chain top (output level {}, {} stages)", sh.out_level, keep.len())));
    }
    // the factor on the last stage that drops a level: a single prime's plaintexts keep their 27 bits (the products
    // are linear, so where the factor sits does not change the output)
    let scaled = stages.iter().rposition(|s| !s.1).unwrap_or(keep.len() - 1);
    Ok(keep.iter().zip(&stages).enumerate().map(|(ki, (&s, &(top, thin)))| StageShape { src: s, top, end: LtEnd::ModDown, scaled: ki == scaled, thin }).collect())
}

/// stc_high on the device: one cache per shape.
pub struct StcHighPlan<'a, W: DevWord> {
    pub caches: Vec<(StcHighShape, StageCache<'a, W>)>,
}

impl<'a, W: DevWord> StcHighPlan<'a, W> {
    #[allow(clippy::too_many_arguments)]
    pub fn bytes(lay: &Layout, primes: &[u64], h: &StcHighSetup, geoms: &[LtGeom], n_stc: usize, d: usize, lone: Option<usize>, k_max: usize) -> Result<usize> {
        let mut b = 0;
        for &sh in &h.shapes {
            b += StageCache::<W>::bytes(lay, primes, &shape_stages(sh, n_stc, lay.nq, d, lone)?, geoms, h.slots, k_max)?;
        }
        Ok(b)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new<M: DevStrategy<W>>(
        arena: &'a Arena,
        dev: &Device,
        s: &Stream<Compute>,
        ctx: &Context<'_, W, M>,
        h: &StcHighSetup,
        geoms: &[LtGeom],
        n_stc: usize,
        keys: &dyn BtsKeyRing,
        stage_ptrs: Option<(*const std::ffi::c_void, *const std::ffi::c_void)>,
        k_max: usize,
        lt_group: i32,
    ) -> Result<Self> {
        let d = ctx.chain.level_unit;
        let mut caches = Vec::new();
        for &sh in &h.shapes {
            let stages = shape_stages(sh, n_stc, ctx.lay.nq, d, ctx.chain.lone_top())?;
            caches.push((sh, StageCache::new("stc_high", arena, dev, s, ctx, stages, geoms, h.slots, keys, stage_ptrs, k_max, lt_group)?));
        }
        Ok(StcHighPlan { caches })
    }

    /// The cache serving `sh`.
    pub fn cache(&self, sh: StcHighShape) -> Result<&StageCache<'a, W>> {
        self.caches.iter().find(|(s, _)| *s == sh).map(|(_, c)| c).ok_or_else(|| Error::Mem {
            knob: "StcHighSetup::shapes",
            value: sh.out_level as usize,
            msg: format!("stc_high shape {sh:?} not built (built: {:?})", self.caches.iter().map(|c| c.0).collect::<Vec<_>>()),
        })
    }

    /// `btsStcHigh` on every batch of `cts`.
    pub fn run<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, 'a, W, M>, srcs: &[Option<RelevelSrc<'a, W>>], cts: &mut [Ct<'a, W>], spec: &StcHigh, source: KskSource) -> Result<()> {
        let ctx = cx.ctx;
        let n = ctx.lay.n;
        if spec.slots as usize != n / 2 {
            return Err(shape("stc_high", "dense full-slot route only"));
        }
        let sh = StcHighShape { out_level: spec.out_level, skip_mask: spec.skip_mask };
        let cache = self.cache(sh)?;
        let entry_top = cache.shape.first().map(|s| s.top).ok_or_else(|| shape("stc_high", "empty shape"))?;
        let entry_level = ctx.shape.n_q - 1 - entry_top as u32;
        for x in cts.iter_mut() {
            if x.meta.deg == 2 {
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
            if x.meta.level > entry_level {
                return Err(shape("stc_high", format!("input at level {} below entry level {entry_level}", x.meta.level)));
            }
            x.meta.level = entry_level;
            x.meta.slots = spec.slots;
        }
        let nf = common_meta(cts)?.nf;
        let factor = if spec.nf_ref > 0.0 { spec.factor * (spec.nf_ref / nf) } else { spec.factor };
        if !(factor > 0.0) || !factor.is_finite() {
            return Err(shape("stc_high", format!("factor {factor} must be positive")));
        }
        let key = CacheKey([sh.out_level as i64, sh.skip_mask as i64, (factor.log2() * 1e6).round() as i64]);
        let i = cache.get(ctx, &cx.ex.compute, srcs, key, factor)?;
        cache.with(i, |stages| {
            for (st, sh) in stages.iter().zip(&cache.shape) {
                if sh.thin {
                    // the level above leaves its pair pending: dropped first, then this stage's prime
                    for x in cts.iter_mut().filter(|x| x.meta.deg == 2) {
                        cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
                    }
                    run_lt_thin(cx, st, st.pt_nf, cts, source, true)?;
                } else {
                    run_lt(cx, st, cts, source)?;
                }
            }
            Ok(())
        })?;
        for x in cts.iter_mut() {
            if x.meta.deg == 2 {
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
            if x.meta.level != spec.out_level {
                return Err(shape("stc_high", format!("output at level {}, expected {}", x.meta.level, spec.out_level)));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_relevel_to_the_old_entries() {
        // ccmm.py: stc_high(d, S, 3, f, 1, nf_ref): out limb 3 (level 50), stages 1..2 at limbs 7, 5
        let s = shape_stages(StcHighShape { out_level: 50, skip_mask: 1 }, 3, 54, 2, None).unwrap();
        assert_eq!(s.iter().map(|x| (x.src, x.top, x.scaled, x.thin)).collect::<Vec<_>>(), vec![(1, 7, false, false), (2, 5, true, false)]);
        // skip 6: stage 0 alone, one composite level above the output
        let s = shape_stages(StcHighShape { out_level: 30, skip_mask: 6 }, 3, 54, 2, None).unwrap();
        assert_eq!(s.iter().map(|x| (x.src, x.top, x.scaled, x.thin)).collect::<Vec<_>>(), vec![(0, 25, true, false)]);
        assert!(shape_stages(StcHighShape { out_level: 50, skip_mask: 7 }, 3, 54, 2, None).is_err());
        assert!(shape_stages(StcHighShape { out_level: 0, skip_mask: 0 }, 3, 54, 2, None).is_err());
    }

    #[test]
    fn lean_shapes_reach_limb_3() {
        // n32a4 (51 primes, q2 single): limb 3 (level 47) from the pair q5 q6 at limb 6 (with the factor), then q4 alone
        let s = shape_stages(StcHighShape { out_level: 47, skip_mask: 1 }, 3, 51, 2, Some(2)).unwrap();
        assert_eq!(s.iter().map(|x| (x.src, x.top, x.scaled, x.thin)).collect::<Vec<_>>(), vec![(1, 6, true, false), (2, 4, false, true)]);
        // SinCBoot's stage 0 in the paired levels: a pair stage, as on n32 (level 28 = limb 22)
        let s = shape_stages(StcHighShape { out_level: 28, skip_mask: 6 }, 3, 51, 2, Some(2)).unwrap();
        assert_eq!(s.iter().map(|x| (x.src, x.top, x.scaled, x.thin)).collect::<Vec<_>>(), vec![(0, 24, true, false)]);
    }
}
