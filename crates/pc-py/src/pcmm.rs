//! The PCMM service of a planned session: the engine `pc_gpu::matmul::pcmm` behind the replay's `StepOp::Finish` ops.
//!
//! A captured PCMM linear (pcmm.py `Pcmm.apply`) is one external refresh per output sharing its input set; the
//! compiler merges them into one `Finish` op (descriptor: the v2 `args.external` dict, `{op: "pcmm", ...}`). The
//! driver declares every linear at session setup (`Session(pcmm=[...])`: one resident `WeightBuf` per linear) and
//! uploads the digit planes once (`Session.pcmm_upload`); the keys come from the host secret or the
//! client's bundle (`PcmmKeys::generate`). PC_PCMM_WEIGHTS_PER_PROGRAM=1: the buffers are pooled per shape, as many
//! as the most linears of that shape one program finishes (`PcmmPlan::fit`), and the driver uploads a program's
//! linears before it runs and evicts them after (`Session.pcmm_evict`): a pipeline of different stages holds one
//! stage's weights, not every stage's.
//!
//! A finish = `linear()`: the inputs copied into the service's own ciphertexts, the product, one staged refresh per
//! output, x 2^shift, the outputs copied into their slots at the raw landing the compile pinned. No host sync, no
//! allocation. Details: docs/ARCHITECTURE.md "pc-py PCMM service".
use crate::session::{M, W};
use pc_core::pcmm_keys::PcmmKeys;
use pc_gpu::bts::stc_first::StcFirstRoute;
use pc_gpu::bts::BtsCx;
use pc_gpu::matmul::pcmm::{finish_groups, finish_groups_split, gather_planes, gather_planes_rows, joint_outputs, linear, linear_joint, linear_per_input, linear_rows, LinearHooks, LinearOpts, Pcmm, PcmmCalib, PcmmCfg, PcmmLinear, PcmmShape, WeightBuf};
use pc_gpu::{Ct, CtBuf, MemPlan};
use pc_program::bind::FinishStep;
use pc_program::replay::{copy_ct, FinishExec};
use pc_shadow::CtMeta;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};

/// One linear the session serves: (name, t, n_in, n_out, weight planes, w_scale, shift), as the driver declares it.
pub(crate) type LinSpec = (String, usize, usize, usize, usize, f64, u32);

/// A linear's descriptor as the compiler carries it (`Finish::desc`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Desc {
    pub linear: String,
    pub n_in: usize,
    pub n_out: usize,
    pub t: usize,
    pub shift: u32,
    pub raise_drop: u32,
    /// per-output raise drops (op order) when they differ (the compiler's merged descriptor)
    pub raise_drops: Option<Vec<u32>>,
}

impl Desc {
    /// Every output's raise drop (op order).
    pub fn drops(&self) -> Vec<u32> {
        match &self.raise_drops {
            Some(v) if v.len() == self.n_out => v.clone(),
            _ => vec![self.raise_drop; self.n_out],
        }
    }

    /// `None` for a descriptor that is not a PCMM one (a CCMM finish, a v1 calibration site name).
    pub fn parse(desc: &str) -> Option<Desc> {
        let v: serde_json::Value = serde_json::from_str(desc).ok()?;
        if v.get("op")?.as_str()? != "pcmm" {
            return None;
        }
        let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64);
        Some(Desc {
            linear: v.get("linear")?.as_str()?.to_string(),
            n_in: u("n_in")? as usize,
            n_out: u("n_out")? as usize,
            t: u("t")? as usize,
            shift: u("shift")? as u32,
            raise_drop: u("raise_drop").unwrap_or(0) as u32,
            raise_drops: v.get("raise_drops").and_then(serde_json::Value::as_array).map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect()),
        })
    }
}

/// What the session's MemPlan reserves for the service.
pub(crate) struct PcmmPlan {
    pub cfg: PcmmCfg,
    pub lins: Vec<PcmmLinear>,
    pub max_in: usize,
    pub max_out: usize,
    pub opts: LinearOpts,
    /// Finish group size (PF_PCMM_BATCH, 1..=4; 0/1 = one output at a time) and whether outputs of different raise
    /// drops batch per drop (PF_PCMM_BATCH_DROPS).
    pub batch: usize,
    pub batch_drops: bool,
    /// Most finish groups one finish of the served programs needs (service output ciphertexts of `batch` members).
    pub max_groups: usize,
    /// PC_PCMM_WEIGHTS_PER_PROGRAM=1 (`fit`): weight buffers per `cfg.shapes` entry; None: one per linear
    pub pool: Option<Vec<usize>>,
    /// PC_JOINT_LINEAR (`fit`): per linear, the outputs this device computes and those the remote device computes
    /// (`joint_outputs` of the balanced groups); a linear whose finishes split differently in two programs is absent
    pub joint: BTreeMap<String, (Vec<usize>, Vec<usize>)>,
    /// PC_REDUCE_SPLIT (`fit`): the joint linears split by input blocks instead (`linear_rows`: more input than output
    /// blocks)
    pub rows: std::collections::BTreeSet<String>,
}

/// A row-split linear's inputs on this device (the first ones) out of `n_in`.
pub(crate) fn rows_here(n_in: usize) -> usize {
    n_in - n_in / 2
}

/// PC_JOINT_LINEAR (with a remote refresh device): every PCMM linear split by output blocks across the two devices
/// (`pc_gpu::matmul::pcmm::linear_joint`): each computes its own outputs' product on its own planes and finishes them.
/// With a remote device and the balanced finish split, `PC_JOINT_LINEAR` on by default (faster on every model measured),
/// `0` / `off` = off.
pub(crate) fn joint_linear() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PC_REMOTE_DEVICE").is_ok_and(|v| !v.trim().is_empty()) && remote_finish() && finish_balance() && pc_program::replay::default_on("PC_JOINT_LINEAR"))
}

/// What the remote device needs for joint linears: its engine's shapes (the far sub-shapes), the keys, each linear's
/// far sub-shape, and the service ciphertext counts.
pub(crate) struct RemotePcmmSpec {
    pub cfg: PcmmCfg,
    pub keys: Vec<PcmmKeys>,
    pub far: BTreeMap<String, PcmmShape>,
    pub max_in: usize,
    pub max_groups: usize,
    pub batch: usize,
}

/// The distinct PCMM linears a program's finishes name (op order).
pub(crate) fn program_linears(p: &pc_program::Program) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for op in &p.ops {
        for m in op.members() {
            if let pc_program::Kind::Finish { desc: Some(d), .. } = &m.kind {
                if let Some(d) = Desc::parse(d) {
                    if !out.contains(&d.linear) {
                        out.push(d.linear);
                    }
                }
            }
        }
    }
    out
}

impl PcmmPlan {
    pub fn new(specs: &[LinSpec], cf: u32, batch: usize, batch_drops: bool) -> PyResult<PcmmPlan> {
        if batch > pc_gpu::matmul::pcmm::MAX_BATCH {
            return Err(PyValueError::new_err(format!("pcmm batch (model.prefill.pcmm_batch / PF_PCMM_BATCH) {batch}: at most {}", pc_gpu::matmul::pcmm::MAX_BATCH)));
        }
        let mut lins = vec![];
        let mut shapes: Vec<PcmmShape> = vec![];
        for (name, t, n_in, n_out, nw, w_scale, shift) in specs {
            let shape = PcmmShape { t: *t, n_in: *n_in, n_out: *n_out, nw: *nw };
            shape.check().map_err(crate::session::gerr)?;
            if !(w_scale.is_finite() && *w_scale > 0.0) {
                return Err(PyValueError::new_err(format!("pcmm {name}: w_scale {w_scale}")));
            }
            if lins.iter().any(|l: &PcmmLinear| &l.name == name) {
                return Err(PyValueError::new_err(format!("pcmm {name} declared twice")));
            }
            if !shapes.contains(&shape) {
                shapes.push(shape);
            }
            lins.push(PcmmLinear { name: name.clone(), shape, w_scale: *w_scale, shift: *shift });
        }
        let max_in = lins.iter().map(|l| l.shape.n_in).max().unwrap_or(0);
        let max_out = lins.iter().map(|l| l.shape.n_out).max().unwrap_or(0);
        let batch = batch.max(1);
        Ok(PcmmPlan { cfg: PcmmCfg { shapes, ..PcmmCfg::default() }, lins, max_in, max_out, opts: LinearOpts { cf, ..LinearOpts::default() }, batch, batch_drops, max_groups: max_out, pool: None, joint: BTreeMap::new(), rows: Default::default() })
    }

    /// Size the service outputs for the finishes of `progs`: the most groups any of them splits into.
    pub fn fit(&mut self, progs: &[pc_program::Program]) -> PyResult<()> {
        let mut g = 0;
        for p in progs {
            for op in &p.ops {
                for m in op.members() {
                    if let pc_program::Kind::Finish { desc: Some(d), .. } = &m.kind {
                        if let Some(d) = Desc::parse(d) {
                            g = g.max(finish_groups(&d.drops(), self.batch, self.batch_drops).map_err(crate::session::gerr)?.len());
                            // the balanced split (PC_FINISH_BALANCE) may form a group more per drop class
                            g = g.max(finish_groups_split(&d.drops(), self.batch, self.batch_drops).map_err(crate::session::gerr)?.0.len());
                        }
                    }
                }
            }
        }
        self.max_groups = if g == 0 { self.max_out } else { g };
        // PC_JOINT_LINEAR: each linear's output split (the balanced groups), the same in every program or none
        if joint_linear() && !std::env::var("PC_PCMM_WEIGHTS_PER_PROGRAM").map(|v| !v.is_empty() && v != "0").unwrap_or(false) {
            let mut seen: BTreeMap<String, Option<(Vec<usize>, Vec<usize>)>> = BTreeMap::new();
            for p in progs {
                for op in &p.ops {
                    for m in op.members() {
                        if let pc_program::Kind::Finish { desc: Some(d), .. } = &m.kind {
                            if let Some(d) = Desc::parse(d) {
                                let (gs, n_far) = finish_groups_split(&d.drops(), self.batch, self.batch_drops).map_err(crate::session::gerr)?;
                                let split = (n_far > 0 && n_far < gs.len()).then(|| joint_outputs(&gs, n_far));
                                let e = seen.entry(d.linear.clone()).or_insert_with(|| split.clone());
                                if *e != split {
                                    *e = None;
                                }
                            }
                        }
                    }
                }
            }
            self.joint = seen.into_iter().filter_map(|(k, v)| v.map(|v| (k, v))).collect();
            for l in &self.lins {
                if let Some((h, f)) = self.joint.get(&l.name) {
                    for sub in [PcmmShape { n_out: h.len(), ..l.shape }, PcmmShape { n_out: f.len(), ..l.shape }] {
                        if !self.cfg.shapes.contains(&sub) {
                            self.cfg.shapes.push(sub);
                        }
                    }
                }
            }
            if pc_program::replay::reduce_split() {
                for l in &self.lins {
                    if self.joint.contains_key(&l.name) && l.shape.n_in > l.shape.n_out && l.shape.n_in >= 2 {
                        self.rows.insert(l.name.clone());
                        let sub = PcmmShape { n_in: rows_here(l.shape.n_in), ..l.shape };
                        if !self.cfg.shapes.contains(&sub) {
                            self.cfg.shapes.push(sub);
                        }
                    }
                }
            }
            eprintln!("[pcmm] joint linears (PC_JOINT_LINEAR): {} of {} split, {} of them by input blocks (PC_REDUCE_SPLIT)", self.joint.len(), self.lins.len(), self.rows.len());
        }
        if std::env::var("PC_PCMM_WEIGHTS_PER_PROGRAM").map(|v| !v.is_empty() && v != "0").unwrap_or(false) {
            let shapes = &self.cfg.shapes;
            let shape_of: BTreeMap<&str, usize> =
                self.lins.iter().filter_map(|l| shapes.iter().position(|s| *s == l.shape).map(|i| (l.name.as_str(), i))).collect();
            let mut pool = vec![0; shapes.len()];
            for p in progs {
                let mut per = vec![0; shapes.len()];
                for name in program_linears(p) {
                    if let Some(&i) = shape_of.get(name.as_str()) {
                        per[i] += 1;
                    }
                }
                for (a, b) in pool.iter_mut().zip(per) {
                    *a = (*a).max(b);
                }
            }
            eprintln!("[session] pcmm weights pooled per program: {} buffers for {} linears", pool.iter().sum::<usize>(), self.lins.len());
            self.pool = Some(pool);
        }
        Ok(())
    }

    /// The remote device's share of the joint linears (None: none planned).
    pub fn remote_spec(&self, keys: &[PcmmKeys]) -> Option<RemotePcmmSpec> {
        if self.joint.is_empty() {
            return None;
        }
        let mut far = BTreeMap::new();
        let mut shapes: Vec<PcmmShape> = vec![];
        for l in &self.lins {
            if let Some((_, f)) = self.joint.get(&l.name) {
                // a row-split linear: the remote device's inputs (the last ones), every output
                let sub = if self.rows.contains(&l.name) { PcmmShape { n_in: l.shape.n_in / 2, ..l.shape } } else { PcmmShape { n_out: f.len(), ..l.shape } };
                if !shapes.contains(&sub) {
                    shapes.push(sub);
                }
                far.insert(l.name.clone(), sub);
            }
        }
        let ts: std::collections::BTreeSet<usize> = shapes.iter().map(|s| s.t).collect();
        Some(RemotePcmmSpec {
            cfg: PcmmCfg { shapes, ..PcmmCfg::default() },
            keys: keys.iter().filter(|k| ts.contains(&k.shape.t)).cloned().collect(),
            far,
            max_in: self.max_in,
            max_groups: self.max_groups,
            batch: self.batch,
        })
    }

    /// The distinct token-lane counts (one key set each).
    pub fn ts(&self) -> Vec<usize> {
        let mut ts: Vec<usize> = self.cfg.shapes.iter().map(|s| s.t).collect();
        ts.sort();
        ts.dedup();
        ts
    }

    pub fn reserve(&self, mp: &mut MemPlan, rows: usize) -> PyResult<()> {
        let e = crate::session::gerr;
        mp.reserve("pcmm", Pcmm::<W>::bytes(&self.cfg, rows).map_err(e)?);
        let wb: usize = match &self.pool {
            Some(pool) => self.cfg.shapes.iter().zip(pool).map(|(s, &k)| WeightBuf::bytes(s).map(|b| k * b)).collect::<pc_gpu::Result<Vec<_>>>(),
            None => self
                .lins
                .iter()
                .map(|l| match self.joint.get(&l.name) {
                    // a row-split linear: this device's rows and the whole planes (per-input inputs, the remote check)
                    Some(_) if self.rows.contains(&l.name) => Ok(WeightBuf::bytes(&PcmmShape { n_in: rows_here(l.shape.n_in), ..l.shape })? + WeightBuf::bytes(&l.shape)?),
                    Some((h, f)) => Ok(WeightBuf::bytes(&PcmmShape { n_out: h.len(), ..l.shape })? + WeightBuf::bytes(&PcmmShape { n_out: f.len(), ..l.shape })?),
                    None => WeightBuf::bytes(&l.shape),
                })
                .collect::<pc_gpu::Result<Vec<_>>>(),
        }
        .map_err(e)?
        .into_iter()
        .sum();
        mp.reserve("pcmm.weights", wb);
        let ct = pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&mp.lay, 1));
        let out_ct = pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&mp.lay, self.batch));
        mp.reserve("pcmm.cts", self.max_in * ct + self.max_groups * out_ct);
        Ok(())
    }
}

/// One served linear: its scale / shift / shape and the weight buffer holding its digit planes (`PcmmRt::bufs`;
/// pooled: while uploaded).
pub(crate) struct Served {
    pub lin: PcmmLinear,
    pub buf: Option<usize>,
    pub uploaded: bool,
    /// a joint linear: (this device's outputs, the remote device's, the buffer of this device's planes, of the remote's)
    pub joint: Option<(Vec<usize>, Vec<usize>, usize, usize)>,
    /// a row-split linear: (this device's outputs, the remote device's, the buffer of this device's rows of the planes,
    /// of the whole planes)
    pub rows: Option<(Vec<usize>, Vec<usize>, usize, usize)>,
}

/// The service: the engine, the linears, its own input / output ciphertexts, the call options. Field order = drop
/// order: the engine's GEMM plans borrow the cuBLASLt handle declared last.
pub(crate) struct PcmmRt {
    pub pcmm: Pcmm<'static, 'static, W>,
    pub lins: BTreeMap<String, Served>,
    /// the weight buffers, their shapes and the linear each holds (pooled: per shape, reassigned on upload)
    pub bufs: Vec<WeightBuf<'static>>,
    pub buf_shape: Vec<PcmmShape>,
    pub owner: Vec<Option<String>>,
    pub pooled: bool,
    pub ins: Vec<Ct<'static, W>>,
    pub outs: Vec<Ct<'static, W>>,
    pub opts: LinearOpts,
    pub batch: usize,
    pub batch_drops: bool,
    pub calls: u64,
    /// Finish groups run and outputs refreshed (batching statistics).
    pub groups: u64,
    pub outputs: u64,
    pub keygen_s: f64,
    /// `upload(.., pin)`: the linears' digit planes in pinned host memory, kept for the session (the source of the
    /// asynchronous uploads)
    pub pins: HashMap<String, pc_cuda::PinnedBuf<u8>>,
    /// `PC_FINISH_PROFILE`: timing events around each finish's phases (StC-in, product, output refreshes)
    pub prof: Option<Vec<pc_cuda::Event>>,
    #[allow(dead_code)]
    lt: Box<pc_cuda::blas::Lt>,
}

impl PcmmRt {
    /// Setup (before freeze): the engine (GEMM plans autotuned) on `keys` (one key set per distinct t of
    /// `plan.ts()`, made from the secret by the session or by the client), one weight buffer per linear (pooled: the
    /// plan's per-shape counts), the service ciphertexts.
    pub fn build(plan: PcmmPlan, arena: &'static pc_cuda::Arena, dev: &pc_cuda::Device, s: &pc_cuda::Stream<pc_cuda::Compute>, ctx: &pc_gpu::Context<'static, W, M>, keys: Vec<PcmmKeys>, keygen_s: f64) -> PyResult<PcmmRt> {
        let e = crate::session::gerr;
        let want = plan.ts();
        if keys.iter().map(|k| k.shape.t).collect::<Vec<_>>() != want {
            return Err(PyValueError::new_err(format!("pcmm keys for t = {:?}, the linears need {want:?}", keys.iter().map(|k| k.shape.t).collect::<Vec<_>>())));
        }
        let lt = Box::new(pc_cuda::blas::Lt::new(dev).map_err(|x| e(x.into()))?);
        // SAFETY: the box is owned by the returned PcmmRt and declared after `pcmm`, so it is dropped after the plans
        // that borrow it; its heap address is stable
        let ltr: &'static pc_cuda::blas::lt::Lt = unsafe { &*(&lt.inner as *const pc_cuda::blas::lt::Lt) };
        let shapes = plan.cfg.shapes.clone();
        let pcmm = Pcmm::<W>::new(arena, dev, s, ctx, ltr, plan.cfg, &keys).map_err(e)?;
        let (mut lins, mut bufs, mut buf_shape, mut owner) = (BTreeMap::new(), vec![], vec![], vec![]);
        if let Some(pool) = &plan.pool {
            for (sh, &k) in shapes.iter().zip(pool) {
                for _ in 0..k {
                    bufs.push(WeightBuf::carve(arena, sh).map_err(e)?);
                    buf_shape.push(*sh);
                    owner.push(None);
                }
            }
        }
        for l in plan.lins {
            if let (Some((h, f)), true) = (plan.joint.get(&l.name), plan.rows.contains(&l.name)) {
                // a row-split linear: this device's rows of the planes, and the whole planes
                let mut ix = [0usize; 2];
                for (i, sub) in [PcmmShape { n_in: rows_here(l.shape.n_in), ..l.shape }, l.shape].into_iter().enumerate() {
                    bufs.push(WeightBuf::carve(arena, &sub).map_err(e)?);
                    buf_shape.push(sub);
                    owner.push(Some(l.name.clone()));
                    ix[i] = bufs.len() - 1;
                }
                let rows = Some((h.clone(), f.clone(), ix[0], ix[1]));
                lins.insert(l.name.clone(), Served { lin: l, buf: Some(ix[1]), uploaded: false, joint: None, rows });
                continue;
            }
            if let Some((h, f)) = plan.joint.get(&l.name) {
                // two buffers: this device's outputs' planes, the remote device's (PC_REMOTE_CHECK recomputes those here)
                let mut ix = [0usize; 2];
                for (i, n) in [h.len(), f.len()].into_iter().enumerate() {
                    let sub = PcmmShape { n_out: n, ..l.shape };
                    bufs.push(WeightBuf::carve(arena, &sub).map_err(e)?);
                    buf_shape.push(sub);
                    owner.push(Some(l.name.clone()));
                    ix[i] = bufs.len() - 1;
                }
                let joint = Some((h.clone(), f.clone(), ix[0], ix[1]));
                lins.insert(l.name.clone(), Served { lin: l, buf: Some(ix[0]), uploaded: false, joint, rows: None });
                continue;
            }
            let buf = if plan.pool.is_some() {
                None
            } else {
                bufs.push(WeightBuf::carve(arena, &l.shape).map_err(e)?);
                buf_shape.push(l.shape);
                owner.push(Some(l.name.clone()));
                Some(bufs.len() - 1)
            };
            lins.insert(l.name.clone(), Served { lin: l, buf, uploaded: false, joint: None, rows: None });
        }
        let ct = |k: usize| -> PyResult<Ct<'static, W>> { Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "pcmm.cts", ctx.lay, k).map_err(e)?)) };
        let ins = (0..plan.max_in).map(|_| ct(1)).collect::<PyResult<Vec<_>>>()?;
        let outs = (0..plan.max_groups).map(|_| ct(plan.batch)).collect::<PyResult<Vec<_>>>()?;
        let pooled = plan.pool.is_some();
        let prof = if finish_profile() { Some((0..6).map(|_| pc_cuda::Event::timing(dev)).collect::<Result<Vec<_>, _>>().map_err(|x| e(x.into()))?) } else { None };
        Ok(PcmmRt { pcmm, lins, bufs, buf_shape, owner, pooled, ins, outs, opts: plan.opts, batch: plan.batch, batch_drops: plan.batch_drops, calls: 0, groups: 0, outputs: 0, keygen_s, pins: HashMap::new(), prof, lt })
    }

    /// Upload one linear's digit planes `[nw][rows_out][rows_in]` int8 (synchronous on the compute lane, so after the
    /// finishes issued before it; pooled: into a free buffer of its shape).
    ///
    /// `pin` (the device): the planes are copied once into pinned host memory the service keeps, and the upload is
    /// asynchronous on the compute lane (ordered as before, the host does not wait for the lane to drain); a later
    /// upload of the same linear may pass no bytes.
    pub fn upload(&mut self, s: &pc_cuda::Stream<pc_cuda::Compute>, name: &str, planes: &[u8], pin: Option<&pc_cuda::Device>) -> PyResult<()> {
        let PcmmRt { lins, bufs, buf_shape, owner, pins, .. } = self;
        let sv = lins.get_mut(name).ok_or_else(|| PyKeyError::new_err(format!("pcmm {name}: not declared at Session setup")))?;
        if let Some((_, _, rh, full)) = &sv.rows {
            // a row-split linear: this device's rows of the planes and the whole planes (the remote device's rows go
            // there, `joint_far_planes`)
            let here: Vec<usize> = (0..rows_here(sv.lin.shape.n_in)).collect();
            let (sub, _) = gather_planes_rows(planes, &sv.lin.shape, &here).map_err(crate::session::gerr)?;
            for (b, p) in [(*rh, sub.as_slice()), (*full, planes)] {
                // SAFETY: u8 and i8 have one size and every bit pattern is valid for both
                bufs[b].upload(s, unsafe { std::slice::from_raw_parts(p.as_ptr() as *const i8, p.len()) }).map_err(crate::session::gerr)?;
            }
            sv.uploaded = true;
            return Ok(());
        }
        if let Some((h, f, bh, bf)) = &sv.joint {
            // a joint linear: each device's outputs' planes gathered (synchronous; the remote device's go there too,
            // `joint_far_planes`)
            for (outs, b) in [(h, *bh), (f, *bf)] {
                let (sub, _) = gather_planes(planes, &sv.lin.shape, outs).map_err(crate::session::gerr)?;
                // SAFETY: u8 and i8 have one size and every bit pattern is valid for both
                bufs[b].upload(s, unsafe { std::slice::from_raw_parts(sub.as_ptr() as *const i8, sub.len()) }).map_err(crate::session::gerr)?;
            }
            sv.uploaded = true;
            return Ok(());
        }
        let want = sv.lin.shape.planes_len().map_err(crate::session::gerr)?;
        if planes.len() != want && !(pin.is_some() && planes.is_empty() && pins.contains_key(name)) {
            return Err(PyValueError::new_err(format!("pcmm {name}: {} plane bytes, the shape {:?} needs {want}", planes.len(), sv.lin.shape)));
        }
        let i = match sv.buf {
            Some(i) => i,
            None => {
                let i = (0..bufs.len()).find(|&i| owner[i].is_none() && buf_shape[i] == sv.lin.shape).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("pcmm {name}: every weight buffer of shape {:?} holds another linear (Session.pcmm_evict first)", sv.lin.shape))
                })?;
                owner[i] = Some(name.to_string());
                sv.buf = Some(i);
                i
            }
        };
        if let Some(dev) = pin {
            if !pins.contains_key(name) {
                let mut pb = pc_cuda::PinnedBuf::<u8>::new(dev, want).map_err(|x| crate::session::gerr(x.into()))?;
                pb.as_mut_slice().copy_from_slice(planes);
                pins.insert(name.to_string(), pb);
            }
            // SAFETY: the pinned planes live in `pins` for the service's life and are never written again
            unsafe { bufs[i].buf.upload_pinned_async(s, 0, &pins[name]) }.map_err(|x| crate::session::gerr(x.into()))?;
        } else {
            // SAFETY: u8 and i8 have one size and every bit pattern is valid for both
            let p = unsafe { std::slice::from_raw_parts(planes.as_ptr() as *const i8, planes.len()) };
            bufs[i].upload(s, p).map_err(crate::session::gerr)?;
        }
        sv.uploaded = true;
        Ok(())
    }

    /// Pooled: free the buffers of `names` (a later upload overwrites one after the finishes issued before it, the
    /// compute lane's order). Returns how many were resident.
    pub fn evict(&mut self, names: &[String]) -> PyResult<usize> {
        if !self.pooled {
            return Err(PyRuntimeError::new_err("pcmm_evict: this session keeps every linear resident (PC_PCMM_WEIGHTS_PER_PROGRAM=1 pools them)"));
        }
        let PcmmRt { lins, owner, .. } = self;
        let mut n = 0;
        for name in names {
            let sv = lins.get_mut(name).ok_or_else(|| PyKeyError::new_err(format!("pcmm {name}: not declared at Session setup")))?;
            if let Some(i) = sv.buf.take() {
                owner[i] = None;
                sv.uploaded = false;
                n += 1;
            }
        }
        Ok(n)
    }

    /// A joint linear's remote outputs' planes (None: not a joint linear).
    pub fn joint_far_planes(&self, name: &str, planes: &[u8]) -> PyResult<Option<Vec<u8>>> {
        if let Some(sv) = self.lins.get(name).filter(|sv| sv.rows.is_some()) {
            let far: Vec<usize> = (rows_here(sv.lin.shape.n_in)..sv.lin.shape.n_in).collect();
            return Ok(Some(gather_planes_rows(planes, &sv.lin.shape, &far).map_err(crate::session::gerr)?.0));
        }
        match self.lins.get(name).and_then(|sv| sv.joint.as_ref().map(|j| (sv, j))) {
            Some((sv, (_, f, _, _))) => Ok(Some(gather_planes(planes, &sv.lin.shape, f).map_err(crate::session::gerr)?.0)),
            None => Ok(None),
        }
    }

    /// Linears declared but not uploaded (pooled: not resident now).
    pub fn missing(&self) -> Vec<String> {
        self.lins.values().filter(|v| !v.uploaded).map(|v| v.lin.name.clone()).collect()
    }
}

/// `FinishExec` over the session's service and its StC-first route (`run_block` builds one per run).
pub(crate) struct PcmmExec<'r> {
    pub rt: &'r mut PcmmRt,
    pub route: &'r StcFirstRoute<'static, W, M>,
}

fn ferr(f: &FinishStep, msg: impl Into<String>) -> pc_program::Error {
    pc_program::Error::Replay { block: format!("finish {}", f.desc), msg: msg.into() }
}

/// With a remote refresh device (`PC_REMOTE_DEVICE`), every PCMM finish of two or more groups refreshes its last half
/// of groups there, every CCMM finish the last half of each run of output refreshes (at most `GRID_INPUTS`).
/// `PC_REMOTE_FINISH`: unset, `1` or `on` = both (default: faster or equal on every model measured), `pcmm` = the PCMM
/// finishes only, `0` or `off` = off; anything else is reported and means off.
pub(crate) fn remote_finish() -> bool {
    remote_finish_mode() > 0
}

/// The CCMM finishes too (`PC_REMOTE_FINISH` unset or `1`).
pub(crate) fn remote_finish_ccmm() -> bool {
    remote_finish_mode() > 1
}

fn remote_finish_mode() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("PC_REMOTE_FINISH").as_deref().map(str::trim) {
        Err(_) | Ok("") | Ok("1") | Ok("on") => 2,
        Ok("0") | Ok("off") => 0,
        Ok("pcmm") => 1,
        Ok(v) => {
            eprintln!("[remote] PC_REMOTE_FINISH={v}: not 1 / on, pcmm or 0 / off; the finish split stays off");
            0
        }
    })
}

/// With the finish split, each PCMM finish's groups formed per device, every drop class halved between the two
/// (`finish_groups_split`), instead of the last half of the groups formed for one device. `PC_FINISH_BALANCE`: unset,
/// `1` or `on` = on (default: 3-5 % faster on every model measured), `0` or `off` = the last half of the groups.
pub(crate) fn finish_balance() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("PC_FINISH_BALANCE").as_deref().map(str::trim) {
        Err(_) | Ok("") | Ok("1") | Ok("on") => true,
        Ok("0") | Ok("off") => false,
        Ok(v) => {
            eprintln!("[remote] PC_FINISH_BALANCE={v}: not 1 / on or 0 / off; the balanced finish split stays off");
            false
        }
    })
}

/// `PC_PER_INPUT_FAR` (with the finish split; on by default, `0` / `off` off): a linear whose inputs sit at several
/// scales (`linear_per_input`) sends its last finish groups to the remote device too, as the uniform path does
/// (SANA 2 GPUs: -6.9 % per image, ABBA 11-24).
pub(crate) fn per_input_far() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pc_program::replay::default_on("PC_PER_INPUT_FAR"))
}

/// `PC_FINISH_PROFILE=1` (eager runs only: it waits for each finish): per PCMM / CCMM finish, a `[finprof]` line with
/// the device time of its phases (inputs' StC stages, the product, the output refreshes).
pub(crate) fn finish_profile() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PC_FINISH_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Ownership (PC_OWN): the output slots of finish `f` a joint linear refreshes on the remote device (its balanced
/// split's far groups, in group-member order), which `joint_far` can leave in the replica's slots too; empty for any
/// other linear.
pub(crate) fn owned_far(rt: &PcmmRt, f: &FinishStep) -> Vec<u32> {
    if !(remote_finish() && finish_balance()) {
        return vec![];
    }
    let Some(d) = Desc::parse(&f.desc) else { return vec![] };
    let Some((h, fo, _, _)) = rt.lins.get(&d.linear).and_then(|sv| sv.joint.as_ref()) else { return vec![] };
    let drops = match &d.raise_drops {
        Some(v) if v.len() != d.n_out => return vec![],
        _ => d.drops(),
    };
    let Ok((groups, n)) = finish_groups_split(&drops, rt.batch, rt.batch_drops) else { return vec![] };
    if n == 0 || n >= groups.len() || joint_outputs(&groups, n) != (h.clone(), fo.clone()) {
        return vec![];
    }
    groups[groups.len() - n..].iter().flatten().filter_map(|&m| f.outs.get(m).copied()).collect()
}

impl<'r> FinishExec<'static, W, M> for PcmmExec<'r> {
    fn finish(&mut self, cx: &mut BtsCx<'_, 'static, W, M>, f: &FinishStep, slots: &mut [Ct<'static, W>], remote: Option<&mut dyn pc_program::replay::RemoteLane<'static, W, M>>) -> pc_program::Result<()> {
        let d = Desc::parse(&f.desc).ok_or_else(|| ferr(f, "not a PCMM descriptor (this session serves PCMM finishes only)"))?;
        let PcmmRt { pcmm, lins, bufs, ins, outs, opts, calls, batch, batch_drops, groups: n_groups, outputs: n_outputs, prof, .. } = &mut *self.rt;
        let sv = lins.get(&d.linear).ok_or_else(|| ferr(f, format!("linear {} not declared at Session setup (pcmm=)", d.linear)))?;
        let lin = &sv.lin;
        let w = match (sv.uploaded, sv.buf) {
            (true, Some(i)) => &bufs[i].buf,
            _ => return Err(ferr(f, format!("linear {}: planes not uploaded (Session.pcmm_upload)", d.linear))),
        };
        if (d.n_in, d.n_out, d.t, d.shift) != (lin.shape.n_in, lin.shape.n_out, lin.shape.t, lin.shift) {
            return Err(ferr(f, format!("graph descriptor (n_in {}, n_out {}, t {}, shift {}) vs the declared linear {:?} shift {}", d.n_in, d.n_out, d.t, d.shift, lin.shape, lin.shift)));
        }
        if f.ins.len() != d.n_in || f.outs.len() != d.n_out {
            return Err(ferr(f, format!("{} inputs / {} outputs compiled for n_in {} n_out {}", f.ins.len(), f.outs.len(), d.n_in, d.n_out)));
        }
        for (i, &sl) in f.ins.iter().enumerate() {
            copy_ct(cx.ctx, &cx.ex.compute, &mut ins[i], &slots[sl as usize])?;
        }
        let drops = match &d.raise_drops {
            Some(v) if v.len() != d.n_out => return Err(ferr(f, format!("{} per-output raise drops for {} outputs", v.len(), d.n_out))),
            _ => d.drops(),
        };
        // with the finish split and PC_FINISH_BALANCE: each drop class halved between the devices (`finish_groups_split`)
        let far_on = remote.as_deref().is_some_and(|r| remote_finish() && r.has_finish());
        let (groups, n_far) = if far_on && finish_balance() {
            let (g, n) = finish_groups_split(&drops, *batch, *batch_drops)?;
            (g, Some(n))
        } else {
            (finish_groups(&drops, *batch, *batch_drops)?, None)
        };
        if groups.len() > outs.len() {
            return Err(ferr(f, format!("{} finish groups, the service holds {} output ciphertexts (PcmmPlan::fit)", groups.len(), outs.len())));
        }
        let gd: Vec<u32> = groups.iter().map(|g| drops[g[0]]).collect();
        // each group's ciphertext runs with as many members as the group has (a prefix of its `batch` members)
        for (g, ms) in groups.iter().enumerate() {
            outs[g].buf.k = ms.len();
        }
        let m0 = ins[0].meta;
        let pe = |x: pc_cuda::Error| ferr(f, x.to_string());
        if let Some(ev) = prof.as_ref() {
            ev[0].record(&cx.ex.compute).map_err(pe)?;
        }
        let uniform = ins[..d.n_in].iter().all(|x| (x.meta.level, x.meta.deg, x.meta.nf) == (m0.level, m0.deg, m0.nf));
        let r = if uniform {
            let hooks = prof.as_ref().map(|ev| LinearHooks { after_stc: Some(&ev[1]), product: Some([&ev[1], &ev[2], &ev[3], &ev[4]]), after_finish: Some(&ev[5]) });
            // PC_REMOTE_FINISH: the last half of the finish groups (or the balanced split's last groups) on the remote device
            let mut far = remote.filter(|r| remote_finish() && r.has_finish()).map(|lane| pc_program::replay::FarGroups { lane, n_far });
            match (&sv.rows, far.as_mut()) {
                // PC_REDUCE_SPLIT: each device's inputs' partial product, summed for each device's outputs
                (Some((h, fo, rh, full)), Some(fg)) if n_far.is_some_and(|n| joint_outputs(&groups, n) == (h.clone(), fo.clone())) && pc_gpu::matmul::pcmm::GroupOffload::rows(fg, &lin.name) => {
                    linear_rows(cx, self.route, pcmm, lin, &bufs[*rh].buf, Some(&bufs[*full].buf), &mut ins[..d.n_in], &mut outs[..groups.len()], &groups, &gd, n_far.unwrap_or(0), *opts, fg)
                }
                (Some(_), _) => Err(pc_gpu::Error::Mem { knob: "pcmm rows", value: 0, msg: format!("{}: a row-split linear (PC_REDUCE_SPLIT) without its planned split or the remote engine", lin.name) }),
                (None, far) => match (&sv.joint, far) {
                // PC_JOINT_LINEAR: each device's outputs' product on its own planes (the groups must be the planned split)
                (Some((h, fo, bh, bf)), Some(fg)) if n_far.is_some_and(|n| joint_outputs(&groups, n) == (h.clone(), fo.clone())) && pc_gpu::matmul::pcmm::GroupOffload::joint(fg, &lin.name) => {
                    // ownership: the far groups' output slots, in their order there
                    if !fg.lane.own_wanted().is_empty() {
                        let n_here = groups.len() - n_far.unwrap_or(0);
                        fg.lane.own_next(groups[n_here..].iter().flatten().map(|&m| f.outs[m]).collect(), lin.shift);
                    }
                    linear_joint(cx, self.route, pcmm, lin, &bufs[*bh].buf, Some(&bufs[*bf].buf), &mut ins[..d.n_in], &mut outs[..groups.len()], &groups, &gd, n_far.unwrap_or(0), *opts, fg)
                }
                (Some(_), _) => Err(pc_gpu::Error::Mem { knob: "pcmm joint", value: 0, msg: format!("{}: a joint linear (PC_JOINT_LINEAR) without its planned split or the remote engine", lin.name) }),
                (None, far) => {
                    let off = far.map(|x| x as &mut dyn pc_gpu::matmul::pcmm::GroupOffload<'static, W, M>);
                    linear(cx, self.route, pcmm, lin, w, &mut ins[..d.n_in], &mut outs[..groups.len()], &groups, &gd, *opts, None, hooks, off)
                }
                },
            }
        } else {
            // PC_PER_INPUT_FAR=1: the finish groups' offload (the uniform path's far groups) here too
            let mut far = remote.filter(|r| per_input_far() && remote_finish() && r.has_finish()).map(|lane| pc_program::replay::FarGroups { lane, n_far });
            let off = far.as_mut().map(|x| x as &mut dyn pc_gpu::matmul::pcmm::GroupOffload<'static, W, M>);
            let all: Vec<usize> = (0..lin.shape.n_out).collect();
            match &sv.joint {
                // a joint linear's inputs at several scales: both halves here (this device holds both planes)
                Some((h, fo, bh, bf)) => pc_gpu::matmul::pcmm::linear_per_input_off(cx, self.route, pcmm, lin, &[(&bufs[*bh].buf, h.as_slice()), (&bufs[*bf].buf, fo.as_slice())], &mut ins[..d.n_in], &mut outs[..groups.len()], &groups, &gd, *opts, off),
                None if off.is_some() => pc_gpu::matmul::pcmm::linear_per_input_off(cx, self.route, pcmm, lin, &[(w, all.as_slice())], &mut ins[..d.n_in], &mut outs[..groups.len()], &groups, &gd, *opts, off),
                None => linear_per_input(cx, self.route, pcmm, lin, w, &mut ins[..d.n_in], &mut outs[..groups.len()], &groups, &gd, *opts),
            }
        };
        if let (Some(ev), true) = (prof.as_ref(), r.is_ok()) {
            if !uniform {
                for x in &ev[1..] {
                    x.record(&cx.ex.compute).map_err(pe)?;
                }
            }
            ev[5].sync().map_err(pe)?;
            let ms = |a: usize, b: usize| ev[a].elapsed_ms(&ev[b]).map(f64::from).map_err(pe);
            eprintln!("[finprof] pcmm {} n_out {} groups {} stc_in {:.3} product {:.3} refresh {:.3}{}", d.linear, d.n_out, groups.len(), ms(0, 1)?, ms(1, 4)?, ms(4, 5)?, if uniform { "" } else { " per_input" });
        }
        let mut res: pc_program::Result<()> = r.map_err(Into::into);
        if res.is_ok() {
            'copy: for (g, ms) in groups.iter().enumerate() {
                for (j, &m) in ms.iter().enumerate() {
                    if let Err(x) = copy_member(cx.ctx, &cx.ex.compute, &mut slots[f.outs[m] as usize], &outs[g], j) {
                        res = Err(x);
                        break 'copy;
                    }
                }
            }
        }
        for x in outs.iter_mut() {
            x.buf.k = *batch;
        }
        res?;
        *n_groups += groups.len() as u64;
        *n_outputs += d.n_out as u64;
        *calls += 1;
        Ok(())
    }
}

/// `dst = member j of src` (live Q rows, both parts) and the batch's metadata (one kernel).
fn copy_member(ctx: &pc_gpu::Context<'static, W, M>, s: &pc_cuda::Stream<pc_cuda::Compute>, dst: &mut Ct<'static, W>, src: &Ct<'static, W>, j: usize) -> pc_program::Result<()> {
    if j >= src.k() || dst.k() != 1 {
        return Err(pc_program::Error::Replay { block: "pcmm".into(), msg: format!("copy member {j} of a {}-member batch into a {}-member slot", src.k(), dst.k()) });
    }
    let top = src.top(ctx.shape)?;
    let mut a = src.buf.r();
    // SAFETY: member j < k of a contiguous k-member buffer starts j ciphertext strides in
    a.base = unsafe { (a.base as *mut W).add(j * src.buf.lay.ct_words()) } as *mut std::ffi::c_void;
    let args = pc_kernels::EwArgs { out: dst.buf.w(), a, b: a, rs: ctx.lay.rs_q(top), sc: std::ptr::null(), scs: std::ptr::null(), pr: ctx.pr(), k: 1, nparts: 2, sub: 0, zero_p: 0 };
    // SAFETY: dst is a one-member full-layout ciphertext; rows 0..=top of both parts of member j are read
    pc_cuda::launch_result("ew_copy", unsafe { (ctx.table.ew_copy)(&args, s.raw()) })?;
    dst.meta = src.meta;
    Ok(())
}

/// Calibrated max |y| per linear and the band-shift rule (`pcmm_calib.json`; pcmm.py `band_shift` /
/// `_record_calib`): `PcmmCalib.load(path)`, `.shift(name, margin=1.5, band=0.25)`, `.record(name, ymax)` (a
/// calibration run keeps the running max), `.to_json()`.
#[pyclass(name = "PcmmCalib", module = "perseus._pc")]
#[derive(Clone, Default)]
pub struct PyPcmmCalib(pub PcmmCalib);

#[pymethods]
impl PyPcmmCalib {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        Ok(PyPcmmCalib(PcmmCalib::load(std::path::Path::new(path)).map_err(crate::session::gerr)?))
    }
    #[staticmethod]
    fn parse(text: &str) -> PyResult<Self> {
        Ok(PyPcmmCalib(PcmmCalib::parse(text).map_err(crate::session::gerr)?))
    }
    #[pyo3(signature = (name, margin=1.5, band=0.25))]
    fn shift(&self, name: &str, margin: f64, band: f64) -> PyResult<u32> {
        self.0.shift(name, margin, band).map_err(|e| PyKeyError::new_err(e.to_string()))
    }
    fn record(&mut self, name: &str, ymax: f64) {
        self.0.record(name, ymax)
    }
    #[getter]
    fn ymax(&self) -> BTreeMap<String, f64> {
        self.0.ymax.clone()
    }
    fn to_json(&self) -> String {
        self.0.to_json()
    }
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.0.to_json()).map_err(|e| PyValueError::new_err(format!("{path}: {e}")))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPcmmCalib>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desc() {
        let d = Desc::parse(r#"{"op": "pcmm", "linear": "b0.qkv", "out": 0, "n_in": 1, "n_out": 3, "t": 32, "shift": 5, "raise_drop": 0}"#).unwrap();
        assert_eq!(d, Desc { linear: "b0.qkv".into(), n_in: 1, n_out: 3, t: 32, shift: 5, raise_drop: 0, raise_drops: None });
        assert!(Desc::parse("b0.qkv").is_none());
        assert!(Desc::parse(r#"{"op": "ccmm", "site": "x"}"#).is_none());
    }
}
