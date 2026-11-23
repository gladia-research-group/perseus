//! A second device for the refresh batches of a planned session, opt-in with `PC_REMOTE_DEVICE=<visible ordinal>`
//! (e.g. `CUDA_VISIBLE_DEVICES=4,5 PC_REMOTE_DEVICE=1`). It holds what a refresh needs and nothing else: the session's
//! keys (copied device to device, whatever made them: device keygen or a client's bundle), the route tables built on
//! them (`routes.rs`), an executor, an op layer, a refresh workspace and a batch buffer. The replay sends the second
//! half of every refresh batch there (`pc_program::replay::Remote`); weights, plaintexts and activations stay on the
//! session's device.
use crate::routes::{RoutePlans, RouteSet};
use crate::session::{gerr, M, W};
use pc_core::params::Chain;
use pc_cuda::{Arena, Compute, Device, Event, Stream};
use pc_gpu::bts::{BtsWork, BtsWorkShape};
use pc_gpu::keys::{ExtraKeySpec, KeySet, KeySetSpec, KskLayout, LevelRelinSpec};
use pc_gpu::ks::{KsPlans, KsWork};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan, PoolClass};
use pc_program::program::GRID_INPUTS;
use pc_gpu::matmul::pcmm::{Pcmm, WeightBuf};
use pc_program::replay::{BTS_BATCHES, BTS_BATCH_CTS, BTS_PAIRS, BTS_SINGLE, REMOTE_ASYNC_CHECKED, REMOTE_ASYNC_PAIRS, REMOTE_ASYNC_SINGLES, REMOTE_CCMM_CHECKED, REMOTE_CCMM_CTS, REMOTE_CCMM_RUNS, REMOTE_CHECKED, REMOTE_CTS, REMOTE_FIN_CHECKED, REMOTE_FIN_CTS, REMOTE_CCMM_OPERANDS, REMOTE_FIN_GROUPS, REMOTE_STC_INPUTS, REMOTE_UNITS};
use pc_shadow::CtMeta;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use std::sync::atomic::Ordering;

/// Ring entries of the asynchronous lane (refreshes in flight between their remote part and their copy back).
pub(crate) const RING: usize = 4;

/// Makes the session's device current again once the remote device's objects are gone (dropped last).
struct Back(Device);

impl Drop for Back {
    fn drop(&mut self) {
        let _ = self.0.make_current();
    }
}

/// The remote device's PCMM engine for joint linears (PC_JOINT_LINEAR): its planes of each joint linear's far outputs,
/// its inputs at the split and its finish-group outputs. Field order = drop order: the engine's GEMM plans borrow the
/// cuBLASLt handle declared last.
pub(crate) struct RemotePcmmDev {
    pub engine: Pcmm<'static, 'static, W>,
    pub planes: std::collections::BTreeMap<String, pc_cuda::DBuf<'static, Compute, u8>>,
    pub ins: Vec<Ct<'static, W>>,
    pub outs: Vec<Ct<'static, W>>,
    #[allow(dead_code)]
    lt: Box<pc_cuda::blas::Lt>,
}

/// What the replica lane (PC_REPLICA, `pc_program::replica`) needs sized: the session's slab (slots, held-chain
/// words) and its programs' key-group bytes.
pub(crate) struct ReplicaSpec {
    pub n_slab: usize,
    pub held_words: usize,
    pub group_bytes: usize,
    /// The plaintext mirror (PC_REPLICA_PT): the session store's slot range (first byte address, bytes), mirrored whole.
    pub pt_span: Option<(usize, usize)>,
}

/// The replica's copy of the session's plaintext store (PC_REPLICA_PT): one buffer as large as the store's slot range,
/// a plaintext's rows at the same offset as there, copied over before a run reads them and again whenever the slot
/// they sit in holds another encoding since.
pub(crate) struct PtMirror {
    pub pool: pc_cuda::DBuf<'static, Compute, W>,
    /// the session store's slot range start (byte address)
    pub base: usize,
    /// byte offset -> the key whose encoding was copied there (`PtStore::content_key`)
    pub owner: std::collections::HashMap<usize, pc_gpu::pt_store::PtKey>,
    pub copies: usize,
    /// keys left unmirrored (not resolved in the store's slots on the session's device), and why (first example each)
    pub left: std::collections::HashSet<pc_gpu::pt_store::PtKey>,
    pub why: std::collections::BTreeMap<&'static str, (usize, String)>,
}

pub(crate) struct ReplicaDev {
    pub bounds: Vec<pc_program::bind::Bound<'static, W>>,
    pub pts: Option<PtMirror>,
    pub slab: pc_program::bind::Slab<'static, W>,
    pub ev: Vec<Event>,
    pub join: Event,
    pub fork: Event,
    /// handoff events (this device): the replica's stream to the compute stream and back (PC_REPLICA_BTS)
    pub hand: Vec<Event>,
    /// PC_REPLICA_BTS: the replica's own refresh workspaces
    pub work: Option<BtsWork<'static, W>>,
    /// PC_REPLICA_DMA: the copy stream here, an event per copy of a block, its join event
    pub dma: Option<(Stream<Compute>, Vec<Event>, Event)>,
    pub ops: Ops<'static, W>,
    pub ex: Executor<'static, W>,
}

/// The remote refresh device. Field order = drop order: everything carved from `arena` before it, `_back` last.
pub(crate) struct RemoteDev {
    /// the replica lane (None: off)
    pub replica: Option<ReplicaDev>,
    /// joint linears (None: no PCMM engine here)
    pub pcmm: Option<RemotePcmmDev>,
    pub routes: RouteSet,
    pub gather: Ct<'static, W>,
    /// the asynchronous lane (PC_REMOTE_ASYNC, `pc_program::replay::RemoteAsync`): `RING` entries of two one-ciphertext
    /// buffers here, an event per entry and per step here (`ring_done`, `lane_in`); the session device's copy-back
    /// stream with an event per step (`cb`, `cb_ev`); the fork (session device), join (here) and copy-back join
    pub ring: Vec<Ct<'static, W>>,
    pub ring_done: Vec<Event>,
    pub lane_in: Vec<Event>,
    pub cb_ev: Vec<Event>,
    /// a stream of the session's device: synchronized and destroyed with that device current (`Drop`)
    pub cb: std::mem::ManuallyDrop<Stream<Compute>>,
    pub lane_fork: Event,
    pub lane_join: Event,
    pub cb_join: Event,
    /// PC_REMOTE_CB: this device's stream for the far work's copies back, and its event
    pub fcb: Stream<Compute>,
    pub fcb_ev: Event,
    pub work: BtsWork<'static, W>,
    pub ops: Ops<'static, W>,
    pub plans: KsPlans<'static>,
    #[allow(dead_code)]
    keys: KeySet<'static, W>,
    pub ctx: Context<'static, W, M>,
    pub ex: Executor<'static, W>,
    /// an event of the session's device: recorded on its compute stream, waited on by this device's
    pub fork: Event,
    /// an event of this device: recorded on its compute stream, waited on by the session's
    pub join: Event,
    pub device: Device,
    #[allow(dead_code)]
    arena: Box<Arena>,
    _back: Back,
}

impl RemoteDev {
    /// Bind `prog` a second time here for the replica lane (after the session bound it: the same order), on this
    /// device's keys, routes and arena (frozen at `complete_setup`). Plaintexts stay unresolved: the replica runs no
    /// plaintext step.
    /// `routes`: the session's (the binding's route checks and refresh batches must be the session's own; the replica
    /// runs no refresh, so it never reads them).
    pub(crate) fn replica_bind(&mut self, prog: &pc_program::Program, dyn_rot: &[i64], streams: Option<&dyn pc_program::bind::StreamSource>, routes: &pc_program::replay::Routes<'_, 'static, W, M>) -> PyResult<()> {
        let e = gerr;
        let Some(rp) = self.replica.as_mut() else { return Ok(()) };
        // SAFETY: the arena outlives the bound blocks (dropped after them, field order)
        let a: &'static Arena = unsafe { &*(self.arena.as_ref() as *const Arena) };
        self.device.make_current().map_err(|x| e(x.into()))?;
        let scalars = pc_program::bind::ScalarTable::new();
        let r = {
            let env = pc_program::bind::BindEnv { arena: a, stream: &rp.ex.compute, lay: self.ctx.lay, keys: &self.keys, stage: rp.ex.ks.stage_ptrs(), scalars: &scalars, streams };
            pc_program::bind::Bound::new_shared(prog, &env, routes, dyn_rot)
        };
        let back = self._back.0.make_current().map_err(|x| e(x.into()));
        let b = r.map_err(|x| pyo3::exceptions::PyRuntimeError::new_err(format!("replica bind: {x}")))?;
        back?;
        rp.bounds.push(b);
        Ok(())
    }

    /// PC_REPLICA_PT: the replica binding `idx`'s plaintexts pointed at their mirrored copies, each copied over from
    /// `main`'s rows (the session's binding of the same block, resolved in `store`) when the mirror holds another
    /// encoding at that offset. A plaintext outside the store's slots stays unresolved there (its steps stay on the
    /// session's device).
    pub(crate) fn replica_pts(&mut self, idx: usize, main: &pc_program::bind::Bound<'static, W>, store: &pc_gpu::pt_store::PtStore<'static, 'static, W>, here: &Device, here_s: &Stream<Compute>) -> PyResult<()> {
        let e = gerr;
        let ce = |x: pc_cuda::Error| gerr(x.into());
        let Some(rp) = self.replica.as_mut() else { return Ok(()) };
        let Some(m) = rp.pts.as_mut() else { return Ok(()) };
        let Some(rb) = rp.bounds.get_mut(idx) else { return Ok(()) };
        let mut copied = false;
        let wb = std::mem::size_of::<W>();
        let pool_bytes = m.pool.len() * wb;
        let n = self.ctx.lay.n;
        self.device.make_current().map_err(ce)?;
        let r = (|| -> PyResult<()> {
            for (k, bp) in rb.pts.iter_mut().enumerate() {
                if bp.stream {
                    continue;
                }
                bp.rows = std::ptr::null();
                bp.meta = None;
                // the same program's plaintext k (the session may have renamed its key to its values' name since)
                let Some(mp) = main.pts.get(k).filter(|x| x.key.level == bp.key.level && x.key.form == bp.key.form && !x.stream && !x.rows.is_null() && x.meta.is_some()) else {
                    if m.left.insert(bp.key.clone()) {
                        let w = m.why.entry("unresolved here").or_insert((0, String::new()));
                        w.0 += 1;
                        if w.1.is_empty() {
                            w.1 = format!("{:?} vs {:?}", bp.key, main.pts.get(k).map(|x| (&x.key, x.rows.is_null(), x.meta.is_some())));
                        }
                    }
                    continue;
                };
                let words = (self.ctx.top_of(bp.key.level).map_err(e)? + 1) * n;
                let a = mp.rows as usize;
                if a < m.base || a - m.base + words * wb > pool_bytes || (a - m.base) % wb != 0 {
                    if m.left.insert(bp.key.clone()) {
                        let w = m.why.entry("outside the store's slots").or_insert((0, String::new()));
                        w.0 += 1;
                        if w.1.is_empty() {
                            w.1 = format!("{:?} at {a:#x}, slots {:#x}..{:#x}, {words} words", bp.key, m.base, m.base + pool_bytes);
                        }
                    }
                    continue;
                }
                let off = a - m.base;
                bp.key = mp.key.clone();
                let Some(ck) = store.content_key(&mp.key) else {
                    if m.left.insert(bp.key.clone()) {
                        let w = m.why.entry("no content key").or_insert((0, String::new()));
                        w.0 += 1;
                        if w.1.is_empty() {
                            w.1 = format!("{:?}", bp.key);
                        }
                    }
                    continue;
                };
                if m.owner.get(&off) != Some(ck) {
                    if !copied {
                        // the session's staging of these rows done before the copy reads them (synchronized from its
                        // own device: a stream is synchronized in its device's context)
                        here.make_current().map_err(ce)?;
                        here_s.sync().map_err(ce)?;
                        self.device.make_current().map_err(ce)?;
                    }
                    // SAFETY: the source rows are a resident PtStore entry of `words` words (its key's level), not
                    // written while the session's device is synchronized above; the destination is the same offset of
                    // this pool (checked inside it)
                    let srcb = unsafe { pc_cuda::DBuf::<Compute, W>::from_raw(mp.rows as *mut W, words) };
                    unsafe { m.pool.copy_from_peer(&self.ex.compute, &self.device, off / wb, &srcb, here, 0, words) }.map_err(ce)?;
                    m.owner.insert(off, ck.clone());
                    m.copies += 1;
                    copied = true;
                }
                bp.rows = m.pool.vptr().wrapping_byte_add(off);
                bp.meta = mp.meta;
            }
            if copied {
                self.ex.compute.sync().map_err(ce)?;
            }
            Ok(())
        })();
        let back = self._back.0.make_current().map_err(|x| e(x.into()));
        r?;
        back
    }

    pub(crate) fn freeze(&self) {
        self.arena.freeze();
    }

    /// A joint linear's far planes (`PcmmRt::joint_far_planes`) uploaded here (synchronous on this device).
    pub(crate) fn pcmm_upload(&mut self, name: &str, planes: &[u8]) -> PyResult<()> {
        let e = gerr;
        let Some(p) = self.pcmm.as_mut() else { return Ok(()) };
        let Some(buf) = p.planes.get_mut(name) else { return Ok(()) };
        if buf.len() != planes.len() {
            return Err(PyValueError::new_err(format!("remote pcmm {name}: {} plane bytes for a buffer of {}", planes.len(), buf.len())));
        }
        self.device.make_current().map_err(|x| e(x.into()))?;
        let r = buf.upload(&self.ex.compute, planes).map_err(|x| e(x.into())).and_then(|_| self.ex.compute.sync().map_err(|x| e(x.into())));
        let back = self._back.0.make_current().map_err(|x| e(x.into()));
        r?;
        back
    }
}

impl Drop for RemoteDev {
    fn drop(&mut self) {
        // this device's objects are released with it current, after nothing is in flight on it
        let _ = self._back.0.make_current();
        let _ = self.cb.sync();
        // SAFETY: dropped once, here, with its device (the session's) current; never used after
        unsafe { std::mem::ManuallyDrop::drop(&mut self.cb) };
        let _ = self.device.make_current();
        let ex = &self.ex;
        let _ = (ex.compute.sync(), ex.aux.sync(), ex.side.sync(), ex.copy.sync(), self.fcb.sync());
        eprintln!(
            "[remote] device {}: {} refresh units split, {} ciphertexts refreshed here, {} units checked word for word",
            self.device.ordinal(),
            REMOTE_UNITS.load(Ordering::Relaxed),
            REMOTE_CTS.load(Ordering::Relaxed),
            REMOTE_CHECKED.load(Ordering::Relaxed)
        );
        eprintln!(
            "[remote] device {}: {} PCMM finish groups ({} ciphertexts) refreshed here, {} checked word for word",
            self.device.ordinal(),
            REMOTE_FIN_GROUPS.load(Ordering::Relaxed),
            REMOTE_FIN_CTS.load(Ordering::Relaxed),
            REMOTE_FIN_CHECKED.load(Ordering::Relaxed)
        );
        eprintln!(
            "[remote] device {}: {} CCMM output halves ({} ciphertexts) refreshed here, {} checked word for word",
            self.device.ordinal(),
            REMOTE_CCMM_RUNS.load(Ordering::Relaxed),
            REMOTE_CCMM_CTS.load(Ordering::Relaxed),
            REMOTE_CCMM_CHECKED.load(Ordering::Relaxed)
        );
        eprintln!(
            "[remote] device {}: {} PCMM inputs' StC-first stage and {} CCMM operand conversions run here",
            self.device.ordinal(),
            REMOTE_STC_INPUTS.load(Ordering::Relaxed),
            REMOTE_CCMM_OPERANDS.load(Ordering::Relaxed)
        );
        eprintln!(
            "[remote] device {}: {} single refreshes and {} refresh pairs on the asynchronous lane, {} checked word for word",
            self.device.ordinal(),
            REMOTE_ASYNC_SINGLES.load(Ordering::Relaxed),
            REMOTE_ASYNC_PAIRS.load(Ordering::Relaxed),
            REMOTE_ASYNC_CHECKED.load(Ordering::Relaxed)
        );
        if let Some(m) = self.replica.as_ref().and_then(|rp| rp.pts.as_ref()) {
            eprintln!("[remote] device {}: plaintext mirror: {} slots held, {} copies over, {} plaintexts left out {:?}", self.device.ordinal(), m.owner.len(), m.copies, m.left.len(), m.why);
        }
        if self.replica.is_some() {
            eprintln!(
                "[remote] device {}: replica lane: {} steps ({} refreshes or pairs, PC_REPLICA_BTS), {} copies onto it, {} copies back, {} blocks checked word for word; ownership (PC_OWN): {} outputs kept there, {} copied over",
                self.device.ordinal(),
                pc_program::replay::REPLICA_STEPS.load(Ordering::Relaxed),
                pc_program::replay::REPLICA_BTS.load(Ordering::Relaxed),
                pc_program::replay::REPLICA_COPIES_IN.load(Ordering::Relaxed),
                pc_program::replay::REPLICA_COPIES_BACK.load(Ordering::Relaxed),
                pc_program::replay::REPLICA_CHECKED.load(Ordering::Relaxed),
                pc_program::replay::OWN_KEPT.load(Ordering::Relaxed),
                pc_program::replay::OWN_COPIED.load(Ordering::Relaxed)
            );
        }
        eprintln!(
            "[remote] refreshes run: {} single bootstrap steps, {} complex pairs, {} local batches of {} ciphertexts (split units not counted)",
            BTS_SINGLE.load(Ordering::Relaxed),
            BTS_PAIRS.load(Ordering::Relaxed),
            BTS_BATCHES.load(Ordering::Relaxed),
            BTS_BATCH_CTS.load(Ordering::Relaxed)
        );
    }
}

/// `PC_REMOTE_DEVICE`: the visible ordinal of the remote refresh device, if set.
pub(crate) fn requested() -> PyResult<Option<i32>> {
    match std::env::var("PC_REMOTE_DEVICE").ok().filter(|v| !v.trim().is_empty()) {
        None => Ok(None),
        Some(v) => v.trim().parse().map(Some).map_err(|_| PyValueError::new_err(format!("PC_REMOTE_DEVICE={v}: a visible device ordinal (e.g. 1)"))),
    }
}

impl RemoteDev {
    /// Open device `ordinal` next to `here` (the session's), carve and fill what a refresh needs there, copy the
    /// session's keys `here_keys` over (after synchronizing `here_s`, the stream that wrote them) and build the routes
    /// of `plans` on them. Leaves `here` current.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        ordinal: i32,
        here: &Device,
        here_keys: &KeySet<'static, W>,
        here_s: &Stream<Compute>,
        chain: &Chain,
        k_max: usize,
        layout: KskLayout,
        spec: &KeySetSpec,
        extra: &ExtraKeySpec,
        levels: &[LevelRelinSpec],
        ocfg: OpsCfg,
        plans: &RoutePlans<'_>,
        n_ev: usize,
        pcmm: Option<crate::pcmm::RemotePcmmSpec>,
        replica: Option<ReplicaSpec>,
    ) -> PyResult<RemoteDev> {
        let e = gerr;
        let ce = |x: pc_cuda::Error| gerr(x.into());
        let t0 = std::time::Instant::now();
        if ordinal == here.ordinal() {
            return Err(PyValueError::new_err(format!("PC_REMOTE_DEVICE={ordinal}: the session's own device")));
        }
        let back = Back(Device::open_ordinal(here.ordinal()).map_err(ce)?);
        let device = Device::open_ordinal(ordinal).map_err(ce)?;
        let peer = (here.enable_peer(&device).map_err(ce)?, device.enable_peer(here).map_err(ce)?);
        device.make_current().map_err(ce)?;
        let mut mp = MemPlan::build::<W, M>(chain, MemCfg { k_max, pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 }).map_err(e)?;
        let lay = mp.lay;
        mp.reserve("ks.plans", KsPlans::bytes::<W>(chain, k_max).map_err(e)?);
        mp.reserve("keys", spec.bytes::<W>(chain, layout).map_err(e)? + extra.bytes::<W>(chain, layout).map_err(e)?);
        if !levels.is_empty() {
            mp.reserve("keys.level", KeySet::<W>::relin_levels_bytes(levels, layout, chain.n));
            mp.reserve("bts.levelks", 512 << 20);
        }
        mp.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, chain.dnum()));
        Ops::<W>::reserve(&mut mp, &ocfg);
        mp.reserve("relin.rr", pc_gpu::relin::RelinRescaler::<W>::bytes(chain, &lay, k_max).map_err(e)?);
        let w = crate::routes::reserve(&mut mp, chain, &lay, plans)?.unwrap_or(BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 });
        mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, w));
        mp.reserve("remote.gather", Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, GRID_INPUTS)));
        mp.reserve("remote.ring", 2 * RING * Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, 1)));
        if let Some(sp) = &pcmm {
            mp.reserve("pcmm", Pcmm::<W>::bytes(&sp.cfg, lay.rows()).map_err(e)?);
            let wb: usize = sp.far.values().map(WeightBuf::bytes).collect::<pc_gpu::Result<Vec<_>>>().map_err(e)?.into_iter().sum();
            mp.reserve("pcmm.weights", wb);
            mp.reserve("pcmm.cts", sp.max_in * Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, 1)) + sp.max_groups * Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, sp.batch)));
        }
        if let Some(rs) = &replica {
            // a second executor's workspaces and op layer (as the session's side lane), the mirrored slab, key groups
            let side = mp.ks_work + mp.bconv_work + mp.scratch + KsWork::<W>::regen_bytes(&lay, chain.dnum());
            mp.reserve("replica.side", side);
            Ops::<W>::reserve(&mut mp, &ocfg);
            mp.reserve("replica.rr", pc_gpu::relin::RelinRescaler::<W>::bytes(chain, &lay, k_max).map_err(e)?);
            mp.reserve("replica.slab", pc_program::bind::Slab::<W>::bytes(&lay, rs.n_slab, rs.held_words));
            mp.reserve("replica.groups", rs.group_bytes);
            if let Some((_, bytes)) = rs.pt_span {
                mp.reserve("replica.pts", bytes);
            }
            if pc_program::replica::bts() {
                // PC_REPLICA_BTS: the replica's own refresh workspaces (its refreshes run on its stream)
                mp.reserve("replica.bts_work", BtsWork::<W>::bytes(&lay, w));
            }
        }
        mp.reserve("slack", 256 << 20);
        let arena = Box::new(mp.allocate(&device).map_err(e)?);
        // SAFETY: the box is owned by the `RemoteDev` built below together with every borrower, and dropped after them
        let a: &'static Arena = unsafe { &*(arena.as_ref() as *const Arena) };
        let mut ex = Executor::<W>::new(&device, a, &mp).map_err(e)?;
        ex.ks.carve_regen(a, &lay).map_err(e)?;
        let ctx = Context::<W, M>::new(a, &ex.copy, chain.clone(), &mp).map_err(e)?;
        let kplans = KsPlans::new(a, &ex.compute, &ctx, k_max).map_err(e)?;
        let mut keys = KeySet::carve(a, chain, spec, layout).map_err(e)?;
        keys.carve_relin_levels(a, levels).map_err(e)?;
        keys.carve_extra(a, chain, extra).map_err(e)?;
        here.make_current().map_err(ce)?;
        here_s.sync().map_err(ce)?;
        device.make_current().map_err(ce)?;
        let tk = std::time::Instant::now();
        // SAFETY: the session's keys are complete (the stream that wrote them synchronized above) and never written again
        let key_bytes = unsafe { keys.copy_from_peer(&ex.compute, &device, here_keys, here) }.map_err(e)?;
        ex.compute.sync().map_err(ce)?;
        let tk = tk.elapsed().as_secs_f64();
        let mut ops = Ops::new(a, &device, &ex.compute, &ctx, ocfg).map_err(e)?;
        ops.rr = Some(pc_gpu::relin::RelinRescaler::new(a, &ex.compute, &ctx, k_max).map_err(e)?);
        let routes = crate::routes::build(a, &device, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs(), plans)?;
        let work = BtsWork::<W>::new(a, lay, w).map_err(e)?;
        let gather = Ct::new(CtMeta::new(0, 1), CtBuf::new(a, "remote.gather", lay, GRID_INPUTS).map_err(e)?);
        let rpcmm = match pcmm {
            Some(sp) => {
                let lt = Box::new(pc_cuda::blas::Lt::new(&device).map_err(ce)?);
                // SAFETY: the box is owned by the RemotePcmmDev built here, declared after the engine whose plans borrow
                // it; its heap address is stable
                let ltr: &'static pc_cuda::blas::lt::Lt = unsafe { &*(&lt.inner as *const pc_cuda::blas::lt::Lt) };
                let engine = Pcmm::<W>::new(a, &device, &ex.compute, &ctx, ltr, sp.cfg, &sp.keys).map_err(e)?;
                let mut planes = std::collections::BTreeMap::new();
                for (name, sub) in &sp.far {
                    planes.insert(name.clone(), a.alloc("pcmm.weights", sub.planes_len().map_err(e)?).map_err(|x| e(x.into()))?);
                }
                let ct = |k: usize| -> PyResult<Ct<'static, W>> { Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(a, "pcmm.cts", lay, k).map_err(e)?)) };
                let ins = (0..sp.max_in).map(|_| ct(1)).collect::<PyResult<Vec<_>>>()?;
                let outs = (0..sp.max_groups).map(|_| ct(sp.batch)).collect::<PyResult<Vec<_>>>()?;
                eprintln!("[remote] device {ordinal}: PCMM engine for {} joint linears", planes.len());
                Some(RemotePcmmDev { engine, planes, ins, outs, lt })
            }
            None => None,
        };
        let ring = (0..2 * RING).map(|_| CtBuf::new(a, "remote.ring", lay, 1).map(|b| Ct::new(CtMeta::new(0, 1), b))).collect::<Result<Vec<_>, _>>().map_err(e)?;
        let join = Event::new(&device).map_err(ce)?;
        let ring_done = (0..RING).map(|_| Event::new(&device)).collect::<Result<Vec<_>, _>>().map_err(ce)?;
        let lane_in = (0..n_ev).map(|_| Event::new(&device)).collect::<Result<Vec<_>, _>>().map_err(ce)?;
        let lane_join = Event::new(&device).map_err(ce)?;
        let fcb = Stream::<Compute>::new(&device).map_err(ce)?;
        let fcb_ev = Event::new(&device).map_err(ce)?;
        let rparts = match &replica {
            Some(rs) => {
                let mut rx = Executor::<W>::new(&device, a, &mp).map_err(e)?;
                rx.ks.carve_regen(a, &lay).map_err(e)?;
                let mut rops = Ops::new(a, &device, &rx.compute, &ctx, ocfg).map_err(e)?;
                rops.rr = Some(pc_gpu::relin::RelinRescaler::new(a, &rx.compute, &ctx, k_max).map_err(e)?);
                let slab = pc_program::bind::Slab::<W>::new(a, lay, rs.n_slab, rs.held_words).map_err(|x| pyo3::exceptions::PyRuntimeError::new_err(x.to_string()))?;
                let rev = (0..n_ev).map(|_| Event::new(&device)).collect::<Result<Vec<_>, _>>().map_err(ce)?;
                let rjoin = Event::new(&device).map_err(ce)?;
                let hand = (0..2).map(|_| Event::new(&device)).collect::<Result<Vec<_>, _>>().map_err(ce)?;
                let rwork = if pc_program::replica::bts() { Some(BtsWork::<W>::new(a, lay, w).map_err(e)?) } else { None };
                let rdma = if pc_program::replica::dma() {
                    let st = Stream::<Compute>::new(&device).map_err(ce)?;
                    let evs = (0..n_ev).map(|_| Event::new(&device)).collect::<Result<Vec<_>, _>>().map_err(ce)?;
                    Some((st, evs, Event::new(&device).map_err(ce)?))
                } else {
                    None
                };
                let pts = match rs.pt_span {
                    Some((base, bytes)) => Some(PtMirror { pool: a.alloc("replica.pts", bytes.div_ceil(std::mem::size_of::<W>())).map_err(ce)?, base, owner: std::collections::HashMap::new(), copies: 0, left: std::collections::HashSet::new(), why: Default::default() }),
                    None => None,
                };
                rx.compute.sync().map_err(ce)?;
                Some((rx, rops, slab, rev, rjoin, hand, rwork, pts, rdma))
            }
            None => None,
        };
        ex.compute.sync().map_err(ce)?;
        if rparts.is_none() {
            a.freeze();
        }
        here.make_current().map_err(ce)?;
        let fork = Event::new(here).map_err(ce)?;
        let lane_fork = Event::new(here).map_err(ce)?;
        let cb_join = Event::new(here).map_err(ce)?;
        let cb = Stream::<Compute>::new(here).map_err(ce)?;
        let cb_ev = (0..n_ev).map(|_| Event::new(here)).collect::<Result<Vec<_>, _>>().map_err(ce)?;
        eprintln!(
            "[remote] device {ordinal}: peer access {}/{}, keys {:.1} GB copied in {:.2} s, arena {:.1} of {:.1} GB, setup {:.1} s",
            peer.0,
            peer.1,
            key_bytes as f64 / 1e9,
            tk,
            a.used_bytes() as f64 / 1e9,
            a.capacity() as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );
        let replica = match rparts {
            Some((rx, rops, slab, rev, rjoin, hand, rwork, pts, rdma)) => {
                let rfork = Event::new(here).map_err(ce)?;
                eprintln!(
                    "[remote] device {ordinal}: replica lane (PC_REPLICA): {} mirrored slots{}",
                    slab.slots.len(),
                    pts.as_ref().map_or(String::new(), |m| format!(", plaintext mirror {:.2} GB (PC_REPLICA_PT)", (m.pool.len() * std::mem::size_of::<W>()) as f64 / 1e9))
                );
                Some(ReplicaDev { bounds: vec![], pts, slab, ev: rev, join: rjoin, fork: rfork, hand, work: rwork, dma: rdma, ops: rops, ex: rx })
            }
            None => None,
        };
        Ok(RemoteDev { replica, pcmm: rpcmm, routes, gather, ring, ring_done, lane_in, cb_ev, cb: std::mem::ManuallyDrop::new(cb), lane_fork, lane_join, cb_join, fcb, fcb_ev, work, ops, plans: kplans, keys, ctx, ex, fork, join, device, arena, _back: back })
    }
}
