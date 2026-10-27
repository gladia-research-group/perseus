//! The PCMM service of a planned session: the engine `pc_gpu::matmul::pcmm` behind the replay's `StepOp::Finish` ops.
//!
//! A captured PCMM linear (pcmm.py `Pcmm.apply`) is one external refresh per output sharing its input set; the
//! compiler merges them into one `Finish` op (descriptor: the v2 `args.external` dict, `{op: "pcmm", ...}`). The
//! driver declares every linear at session setup (`Session(pcmm=[...])`: one resident `WeightBuf` per linear) and
//! uploads the digit planes once (`Session.pcmm_upload`); the keys come from the host secret or the
//! client's bundle (`PcmmKeys::generate`).
//!
//! A finish = `linear()`: the inputs copied into the service's own ciphertexts, the product, one staged refresh per
//! output, x 2^shift, the outputs copied into their slots at the raw landing the compile pinned. No host sync, no
//! allocation. Details: docs/ARCHITECTURE.md "pc-py PCMM service".
use crate::session::{M, W};
use pc_core::pcmm_keys::PcmmKeys;
use pc_gpu::bts::stc_first::StcFirstRoute;
use pc_gpu::bts::BtsCx;
use pc_gpu::matmul::pcmm::{finish_groups, linear, linear_per_input, LinearOpts, Pcmm, PcmmCalib, PcmmCfg, PcmmLinear, PcmmShape, WeightBuf};
use pc_gpu::{Ct, CtBuf, MemPlan};
use pc_program::bind::FinishStep;
use pc_program::replay::{copy_ct, FinishExec};
use pc_shadow::CtMeta;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::collections::BTreeMap;

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
        Ok(PcmmPlan { cfg: PcmmCfg { shapes, ..PcmmCfg::default() }, lins, max_in, max_out, opts: LinearOpts { cf, ..LinearOpts::default() }, batch, batch_drops, max_groups: max_out })
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
                        }
                    }
                }
            }
        }
        self.max_groups = if g == 0 { self.max_out } else { g };
        Ok(())
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
        let wb: usize = self.lins.iter().map(|l| WeightBuf::bytes(&l.shape)).collect::<pc_gpu::Result<Vec<_>>>().map_err(e)?.into_iter().sum();
        mp.reserve("pcmm.weights", wb);
        let ct = pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&mp.lay, 1));
        let out_ct = pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&mp.lay, self.batch));
        mp.reserve("pcmm.cts", self.max_in * ct + self.max_groups * out_ct);
        Ok(())
    }
}

/// One served linear: its scale / shift / shape and its resident digit planes.
pub(crate) struct Served {
    pub lin: PcmmLinear,
    pub w: WeightBuf<'static>,
    pub uploaded: bool,
}

/// The service: the engine, the linears, its own input / output ciphertexts, the call options. Field order = drop
/// order: the engine's GEMM plans borrow the cuBLASLt handle declared last.
pub(crate) struct PcmmRt {
    pub pcmm: Pcmm<'static, 'static, W>,
    pub lins: BTreeMap<String, Served>,
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
    #[allow(dead_code)]
    lt: Box<pc_cuda::blas::Lt>,
}

impl PcmmRt {
    /// Setup (before freeze): the engine (GEMM plans autotuned) on `keys` (one key set per distinct t of
    /// `plan.ts()`, made from the secret by the session or by the client), one weight buffer per linear, the service
    /// ciphertexts.
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
        let pcmm = Pcmm::<W>::new(arena, dev, s, ctx, ltr, plan.cfg, &keys).map_err(e)?;
        let mut lins = BTreeMap::new();
        for l in plan.lins {
            let w = WeightBuf::carve(arena, &l.shape).map_err(e)?;
            lins.insert(l.name.clone(), Served { lin: l, w, uploaded: false });
        }
        let ct = |k: usize| -> PyResult<Ct<'static, W>> { Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "pcmm.cts", ctx.lay, k).map_err(e)?)) };
        let ins = (0..plan.max_in).map(|_| ct(1)).collect::<PyResult<Vec<_>>>()?;
        let outs = (0..plan.max_groups).map(|_| ct(plan.batch)).collect::<PyResult<Vec<_>>>()?;
        Ok(PcmmRt { pcmm, lins, ins, outs, opts: plan.opts, batch: plan.batch, batch_drops: plan.batch_drops, calls: 0, groups: 0, outputs: 0, keygen_s, lt })
    }

    /// Upload one linear's digit planes `[nw][rows_out][rows_in]` int8 (synchronous; setup).
    pub fn upload(&mut self, s: &pc_cuda::Stream<pc_cuda::Compute>, name: &str, planes: &[u8]) -> PyResult<()> {
        let sv = self.lins.get_mut(name).ok_or_else(|| PyKeyError::new_err(format!("pcmm {name}: not declared at Session setup")))?;
        let want = sv.lin.shape.planes_len().map_err(crate::session::gerr)?;
        if planes.len() != want {
            return Err(PyValueError::new_err(format!("pcmm {name}: {} plane bytes, the shape {:?} needs {want}", planes.len(), sv.lin.shape)));
        }
        // SAFETY: u8 and i8 have one size and every bit pattern is valid for both
        let p = unsafe { std::slice::from_raw_parts(planes.as_ptr() as *const i8, planes.len()) };
        sv.w.upload(s, p).map_err(crate::session::gerr)?;
        sv.uploaded = true;
        Ok(())
    }

    /// Linears declared but never uploaded.
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

impl<'r> FinishExec<'static, W, M> for PcmmExec<'r> {
    fn finish(&mut self, cx: &mut BtsCx<'_, 'static, W, M>, f: &FinishStep, slots: &mut [Ct<'static, W>]) -> pc_program::Result<()> {
        let d = Desc::parse(&f.desc).ok_or_else(|| ferr(f, "not a PCMM descriptor (this session serves PCMM finishes only)"))?;
        let PcmmRt { pcmm, lins, ins, outs, opts, calls, batch, batch_drops, groups: n_groups, outputs: n_outputs, .. } = &mut *self.rt;
        let sv = lins.get(&d.linear).ok_or_else(|| ferr(f, format!("linear {} not declared at Session setup (pcmm=)", d.linear)))?;
        let lin = &sv.lin;
        if !sv.uploaded {
            return Err(ferr(f, format!("linear {}: planes never uploaded (Session.pcmm_upload)", d.linear)));
        }
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
        let groups = finish_groups(&drops, *batch, *batch_drops)?;
        if groups.len() > outs.len() {
            return Err(ferr(f, format!("{} finish groups, the service holds {} output ciphertexts (PcmmPlan::fit)", groups.len(), outs.len())));
        }
        let gd: Vec<u32> = groups.iter().map(|g| drops[g[0]]).collect();
        // each group's ciphertext runs with as many members as the group has (a prefix of its `batch` members)
        for (g, ms) in groups.iter().enumerate() {
            outs[g].buf.k = ms.len();
        }
        let m0 = ins[0].meta;
        let r = if ins[..d.n_in].iter().all(|x| (x.meta.level, x.meta.deg, x.meta.nf) == (m0.level, m0.deg, m0.nf)) {
            linear(cx, self.route, pcmm, lin, &sv.w.buf, &mut ins[..d.n_in], &mut outs[..groups.len()], &groups, &gd, *opts, None, None)
        } else {
            linear_per_input(cx, self.route, pcmm, lin, &sv.w.buf, &mut ins[..d.n_in], &mut outs[..groups.len()], &groups, &gd, *opts)
        };
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
