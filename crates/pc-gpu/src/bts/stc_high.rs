//! StC outside a bootstrap (FIDESlib `btsStcHigh`, perseus `bts.stc_high`): the dense route's shipped StC stages
//! applied to an ordinary ciphertext, the kept stages (bit s of `skip_mask` skips stage s: the CCMM's SinC encoding
//! runs stage 0 alone, or stages 1..2) re-levelled to `entry = out + n_kept d` .. `out + d`, the last kept stage
//! scaled by `factor` (times `nf_ref / NoiseFactor` when `nf_ref > 0`: the output's integers then sit at the
//! reference scale), unfused stage ends (EvalLTStages), a final rescale to the output level.
//!
//! The re-levelled stage sequences are cached per shape (out level, skip mask) in bounded LRU caches
//! (`StageCache`, entries keyed by the factor; FIDESlib kept one process-wide map capped at 24 and synchronized the
//! whole device after every call and before every eviction). Here a fill is ordered on the compute lane and the
//! pinned table buffer's reuse waits on its own event: no device-wide synchronization.
//! Shapes are fixed at setup (`StcHighSetup::shapes`, from the compiled programs): a call with another shape is a
//! setup error naming the knob, not an allocation.
use super::cts::{LtEnd, LtGeom};
use super::stc_first::{run_lt, CacheKey, RelevelSrc, StageCache, StageShape};
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

/// The re-levelled stages of one shape (execution order).
pub fn shape_stages(sh: StcHighShape, n_stc: usize, n_q: usize, d: usize) -> Result<Vec<StageShape>> {
    let keep: Vec<usize> = (0..n_stc).filter(|s| sh.skip_mask >> s & 1 == 0).collect();
    if keep.is_empty() {
        return Err(shape("stc_high", "every stage skipped"));
    }
    let out_top = (n_q - 1).checked_sub(sh.out_level as usize).ok_or_else(|| shape("stc_high", format!("output level {} past the chain", sh.out_level)))?;
    let entry = out_top + keep.len() * d;
    if entry >= n_q {
        return Err(shape("stc_high", format!("entry limb {entry} above the chain top (output level {}, {} stages)", sh.out_level, keep.len())));
    }
    Ok(keep.iter().enumerate().map(|(ki, &s)| StageShape { src: s, top: entry - ki * d, end: LtEnd::ModDown, scaled: ki + 1 == keep.len() }).collect())
}

/// stc_high on the device: one cache per shape.
pub struct StcHighPlan<'a, W: DevWord> {
    pub caches: Vec<(StcHighShape, StageCache<'a, W>)>,
}

impl<'a, W: DevWord> StcHighPlan<'a, W> {
    pub fn bytes(lay: &Layout, primes: &[u64], h: &StcHighSetup, geoms: &[LtGeom], n_stc: usize, d: usize, k_max: usize) -> Result<usize> {
        let mut b = 0;
        for &sh in &h.shapes {
            b += StageCache::<W>::bytes(lay, primes, &shape_stages(sh, n_stc, lay.nq, d)?, geoms, h.slots, k_max)?;
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
            let stages = shape_stages(sh, n_stc, ctx.lay.nq, d)?;
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
        let entry_level = pc_shadow::meta::level_from_top_limb(ctx.shape, entry_top as u32)?;
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
            for st in stages {
                run_lt(cx, st, cts, source)?;
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
        let s = shape_stages(StcHighShape { out_level: 50, skip_mask: 1 }, 3, 54, 2).unwrap();
        assert_eq!(s.iter().map(|x| (x.src, x.top, x.scaled)).collect::<Vec<_>>(), vec![(1, 7, false), (2, 5, true)]);
        // skip 6: stage 0 alone, one composite level above the output
        let s = shape_stages(StcHighShape { out_level: 30, skip_mask: 6 }, 3, 54, 2).unwrap();
        assert_eq!(s.iter().map(|x| (x.src, x.top, x.scaled)).collect::<Vec<_>>(), vec![(0, 25, true)]);
        assert!(shape_stages(StcHighShape { out_level: 50, skip_mask: 7 }, 3, 54, 2).is_err());
        assert!(shape_stages(StcHighShape { out_level: 0, skip_mask: 0 }, 3, 54, 2).is_err());
    }
}
