//! The bootstrap routes one device holds for a planned session (StC-first, dense, SparseB, classic sparse): their
//! arena reservations and their device tables, built the same way on every device that refreshes (the session's own
//! and a remote refresh device, `remote.rs`). SPRU stays with the session: its key material is the session's.
use crate::session::{gerr, M, W};
use pc_core::params::Chain;
use pc_cuda::{Arena, Compute, Device, Stream};
use pc_gpu::bts::dense::{DensePlan, DenseRoute};
use pc_gpu::bts::sparse::{chain_deg, SparseBRoute};
use pc_gpu::bts::sparse_b::{SparseBCfg, SparseBPrecomp};
use pc_gpu::bts::sparse_classic::{ClassicPlan, SparseClassicRoute};
use pc_gpu::bts::stc_first::{StcFirstPlan, StcFirstRoute};
use pc_gpu::bts::{BtsKeyRing, BtsWorkShape};
use pc_gpu::keys::KeySet;
use pc_gpu::ks::KeyGroup;
use pc_gpu::{Context, Layout, MemPlan};
use pc_program::replay::Routes;
use pyo3::PyResult;

/// The route plans of a session (host side, shared by every device).
pub(crate) struct RoutePlans<'p> {
    pub splan: &'p Option<StcFirstPlan>,
    pub dplan: &'p Option<DensePlan>,
    pub sb_cfgs: &'p [SparseBCfg],
    pub cplans: &'p [ClassicPlan],
}

/// The routes built on one device.
pub(crate) struct RouteSet {
    pub dense: Option<StcFirstRoute<'static, W, M>>,
    pub plain: Option<DenseRoute<'static, W, M>>,
    pub sparse: Vec<SparseBRoute<'static, W>>,
    pub classic: Vec<SparseClassicRoute<'static, W, M>>,
}

impl RouteSet {
    pub fn routes(&self) -> Routes<'_, 'static, W, M> {
        let mut r = Routes::new();
        if let Some(d) = &self.dense {
            r.add(d);
        }
        if let Some(d) = &self.plain {
            r.add(d);
        }
        for x in &self.sparse {
            r.add(x);
        }
        for x in &self.classic {
            r.add(x);
        }
        r
    }
}

/// Reserve the route tables of `p` in `mp`; returns the largest work shape they need (None: no route).
pub(crate) fn reserve(mp: &mut MemPlan, chain: &Chain, lay: &Layout, p: &RoutePlans<'_>) -> PyResult<Option<BtsWorkShape>> {
    let e = gerr;
    let mut wshape: Option<BtsWorkShape> = None;
    if let Some(sp) = p.splan {
        let (req, w) = sp.mem::<W>().map_err(e)?;
        req.reserve(mp);
        wshape = Some(w);
    }
    if let Some(dp) = p.dplan {
        let (req, w) = dp.mem::<W>().map_err(e)?;
        req.reserve(mp);
        mp.reserve("bts.route", pc_gpu::bts::raise::EncapsKeys::bytes() + pc_gpu::bts::finish::Pow2::<W>::bytes(lay.rows()) + KeyGroup::bytes(1, true));
        wshape = Some(wshape.map_or(w, |x| x.max(w)));
    }
    let st_tables = pc_core::scale::ScaleTables::precompute(chain).map_err(|x| e(x.into()))?;
    mp.reserve("bts.sparse", SparseBPrecomp::<W>::tails_bytes(chain, &st_tables, lay, p.sb_cfgs).map_err(e)?);
    for (i, c) in p.sb_cfgs.iter().enumerate() {
        let (b, w) = SparseBPrecomp::<W>::bytes(chain, &st_tables, lay, c).map_err(e)?;
        mp.reserve("bts.sparse", b);
        // one route per slot count: its raise variants share the encapsulation keys and the post-factor table
        if p.sb_cfgs[..i].iter().all(|x| x.slots != c.slots) {
            mp.reserve("bts.route", pc_gpu::bts::raise::EncapsKeys::bytes() + pc_gpu::bts::finish::Pow2::<W>::bytes(lay.rows()));
        }
        wshape = Some(wshape.map_or(w, |x| x.max(w)));
    }
    for cp in p.cplans {
        let (req, w) = cp.mem::<W>().map_err(e)?;
        req.reserve(mp);
        wshape = Some(wshape.map_or(w, |x| x.max(w)));
    }
    Ok(wshape)
}

/// Build the route tables of `p` on the device of `a` / `s`, on `keys` (that device's), with `stage` its executor's
/// key staging buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build(a: &'static Arena, device: &Device, s: &Stream<Compute>, ctx: &Context<'static, W, M>, keys: &KeySet<'static, W>, stage: Option<(*const std::ffi::c_void, *const std::ffi::c_void)>, p: &RoutePlans<'_>) -> PyResult<RouteSet> {
    let e = gerr;
    let n = ctx.lay.n;
    let dense = match p.splan {
        Some(sp) => Some(sp.build(a, device, s, ctx, keys, stage).map_err(e)?),
        None => None,
    };
    let plain = match p.dplan {
        Some(dp) => Some(dp.build(a, s, ctx, keys, stage).map_err(e)?),
        None => None,
    };
    let mut sparse: Vec<SparseBRoute<'static, W>> = vec![];
    let sb_tails = SparseBPrecomp::<W>::tails(a, s, ctx, keys, p.sb_cfgs).map_err(e)?;
    for c in p.sb_cfgs {
        let pre = SparseBPrecomp::new(a, s, ctx, c.clone(), keys, &sb_tails, stage).map_err(e)?;
        // the raise variants of one slot count are one route (`SparseBRoute::variant` picks by drop)
        if let Some(r) = sparse.iter_mut().find(|r| r.slots == c.slots) {
            r.variants.push(pre);
            continue;
        }
        let encaps = pc_gpu::bts::raise::EncapsKeys::new(a, s, n, BtsKeyRing::dense_to_sparse(keys).map_err(e)?, BtsKeyRing::sparse_to_dense(keys).map_err(e)?, stage).map_err(e)?;
        let pow2 = pc_gpu::bts::finish::Pow2::new(a, s, ctx).map_err(e)?;
        sparse.push(SparseBRoute { slots: c.slots, ring_n: n, variants: vec![pre], encaps, pow2, cf_default: 6, cf_window: (2, 20), deg: chain_deg(ctx).map_err(e)?, ks_source: c.ks_source });
    }
    let mut classic = vec![];
    for cp in p.cplans {
        let r = cp.build(a, s, ctx, keys, stage).map_err(e)?;
        r.check_mem().map_err(e)?;
        classic.push(r);
    }
    Ok(RouteSet { dense, plain, sparse, classic })
}
