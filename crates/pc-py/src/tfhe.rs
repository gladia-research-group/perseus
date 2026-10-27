//! The TFHE service of a planned session: CKKS -> TFHE -> CKKS lookup tables (pc-gpu `tfhe::TfheSwitch`) behind
//! the replay's `StepOp::Finish` ops.
//!
//! A captured `tfhe_lut` (perseus impl/tfhe.py) is one external refresh of one input; the compiler makes it a
//! `Finish` op (descriptor `{op: "tfhe", lut, slots, ...}`). The session generates the switch keys from the host
//! secret at setup (`Session(tfhe=<values per pass>)`); the driver registers the LUTs and the slot sets they run on
//! by name (`Session.tfhe_lut`, `Session.tfhe_slots`: uploaded once, synchronous). A finish = the input copied into
//! the service's ciphertext, the StC-first input half of a dense refresh at correction factor `CF`, the switch on the
//! set's coefficients (every other slot comes back an exact zero), the finish half, the output copied into its slot
//! at the raw landing the compile pinned (the PCMM finish's: pc-shadow `Finish::Tfhe`). No host sync, no allocation.
//! The finish reads a LUT value v as v / 2^(out_cf - CF) (impl/tfhe.py folds the gain back).
use crate::session::{M, W};
use pc_gpu::bts::staged::{BtsCall, CallOpts, Phase, StagedRef};
use pc_gpu::bts::stc_first::StcFirstRoute;
use pc_gpu::bts::BtsCx;
use pc_gpu::tfhe::TfheSwitch;
use pc_gpu::{Ct, CtBuf, MemPlan};
use pc_program::bind::FinishStep;
use pc_program::replay::{copy_ct, FinishExec};
use pc_shadow::CtMeta;
use pc_tfhe::params::Params;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// The split's correction factor (the forward phase m q0 / 2^CF: |m| < 1 must not wrap; out_cf > CF).
pub(crate) const CF: u32 = 2;
/// LUT tables the session holds (each [2 prime][n_br] residues).
pub(crate) const MAX_LUTS: usize = 16;
/// Coefficient positions the slot sets may hold in total.
pub(crate) const MAX_POS: usize = 1 << 16;
/// The largest LUT value: the warp needs |v| 2 pi < 2^out_cf (8 / 2 pi = 1.27 at out_cf 3).
pub(crate) const MAX_V: f64 = 1.25;

/// A finish descriptor (`Finish::desc`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Desc {
    pub lut: String,
    pub slots: String,
}

impl Desc {
    /// `None` for a descriptor that is not a TFHE one.
    pub fn parse(desc: &str) -> Option<Desc> {
        let v: serde_json::Value = serde_json::from_str(desc).ok()?;
        if v.get("op")?.as_str()? != "tfhe" {
            return None;
        }
        Some(Desc { lut: v.get("lut")?.as_str()?.to_string(), slots: v.get("slots")?.as_str()?.to_string() })
    }
}

/// What the session's MemPlan reserves: the switch (`max` values per pass), the tables, one ciphertext.
pub(crate) struct TfhePlan {
    pub max: usize,
}

impl TfhePlan {
    pub fn reserve(&self, mp: &mut MemPlan, n: usize) -> PyResult<()> {
        let p = Params::new(n).map_err(|x| PyValueError::new_err(x.to_string()))?;
        mp.reserve("tfhe", TfheSwitch::bytes(n, &p, self.max));
        mp.reserve("tfhe.tables", 4 * (MAX_LUTS * 2 * p.n_br + MAX_POS) + 2 * 256);
        mp.reserve("tfhe.ct", pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&mp.lay, 1)));
        Ok(())
    }
}

/// The service. Field order = drop order: the switch's GEMM plans borrow the cuBLASLt handle declared last.
pub(crate) struct TfheRt {
    pub sw: TfheSwitch<'static>,
    pub params: Params,
    rns: pc_tfhe::pbs::Rns,
    luts: BTreeMap<String, usize>,
    lut_buf: pc_cuda::DBuf<'static, pc_cuda::Compute, u32>,
    sets: BTreeMap<String, (usize, usize)>,
    pos_buf: pc_cuda::DBuf<'static, pc_cuda::Compute, u32>,
    pos_used: usize,
    x: Ct<'static, W>,
    n: usize,
    pub calls: u64,
    pub values: u64,
    pub keygen_s: f64,
    lt: Box<pc_cuda::blas::Lt>,
}

impl TfheRt {
    /// Setup (before freeze): the host switch keys from the CKKS secret `s` (ternary, degree N) and the chain's
    /// (p0, p1), then their device image; the tables; the service ciphertext.
    #[allow(clippy::too_many_arguments)]
    pub fn build(plan: TfhePlan, arena: &'static pc_cuda::Arena, dev: &pc_cuda::Device, st: &pc_cuda::Stream<pc_cuda::Compute>, ctx: &pc_gpu::Context<'static, W, M>, s: &[i8], p0: u64, p1: u64, prng: &mut pc_core::keys::Prng) -> PyResult<TfheRt> {
        let e = crate::session::gerr;
        let ve = |x: pc_tfhe::Error| PyValueError::new_err(x.to_string());
        let n = s.len();
        let t0 = std::time::Instant::now();
        let params = Params::new(n).map_err(ve)?;
        let s64: Vec<i64> = s.iter().map(|&v| v as i64).collect();
        let host = pc_tfhe::switch::Switcher::generate(params.clone(), &s64, p0, p1, prng).map_err(ve)?;
        let sw = TfheSwitch::new(arena, st, &host, plan.max).map_err(e)?;
        drop(host);
        let lt = Box::new(pc_cuda::blas::Lt::new(dev).map_err(|x| e(x.into()))?);
        let rns = pc_tfhe::pbs::Rns::new(params.br_primes.0, params.br_primes.1, params.n_br).map_err(ve)?;
        let lut_buf = arena.alloc::<pc_cuda::Compute, u32>("tfhe.tables", MAX_LUTS * 2 * params.n_br).map_err(|x| e(x.into()))?;
        let pos_buf = arena.alloc::<pc_cuda::Compute, u32>("tfhe.tables", MAX_POS).map_err(|x| e(x.into()))?;
        let x = Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "tfhe.ct", ctx.lay, 1).map_err(e)?);
        Ok(TfheRt { sw, params, rns, luts: BTreeMap::new(), lut_buf, sets: BTreeMap::new(), pos_buf, pos_used: 0, x, n, calls: 0, values: 0, keygen_s: t0.elapsed().as_secs_f64(), lt })
    }

    /// Register LUT `name`: `values[j]` = f(m_j) on the grid m_j = 2 j / n_br (j < n_br / 2) and 2 (j - n_br) / n_br
    /// (j >= n_br / 2) of [-1, 1), as read back after the finish (|f| <= `MAX_V`; the arcsine warp and the
    /// negacyclic signs are applied here). Synchronous.
    pub fn lut(&mut self, s: &pc_cuda::Stream<pc_cuda::Compute>, name: &str, values: &[f64]) -> PyResult<()> {
        let nb = self.params.n_br;
        if values.len() != nb {
            return Err(PyValueError::new_err(format!("tfhe lut {name}: {} values, the grid has {nb}", values.len())));
        }
        if let Some(v) = values.iter().find(|v| !(v.is_finite() && v.abs() <= MAX_V)) {
            return Err(PyValueError::new_err(format!("tfhe lut {name}: value {v} outside [-{MAX_V}, {MAX_V}]")));
        }
        let q = self.params.q_br;
        let tv: Vec<u64> = values
            .iter()
            .enumerate()
            .map(|(j, &v)| {
                let x = (q as f64 * pc_tfhe::pbs::warp(v, self.params.out_cf)).round() as i64;
                // the upper half holds -f (the negacyclic rotation flips it back)
                (if j < nb / 2 { x } else { -x }).rem_euclid(q as i64) as u64
            })
            .collect();
        let res: Vec<u32> = self.rns.residues(&tv).iter().flat_map(|r| r.iter().map(|&x| x as u32)).collect();
        let slot = match self.luts.get(name) {
            Some(&k) => k,
            None if self.luts.len() < MAX_LUTS => self.luts.len(),
            None => return Err(PyValueError::new_err(format!("tfhe lut {name}: the session holds {MAX_LUTS} tables"))),
        };
        self.lut_buf.upload_at(s, slot * 2 * nb, &res).map_err(|x| crate::session::gerr(x.into()))?;
        s.sync().map_err(|x| crate::session::gerr(x.into()))?;
        self.luts.insert(name.to_string(), slot);
        Ok(())
    }

    /// Register slot set `name`: the slots (of a real payload, < N/2) a LUT runs on, as split coefficients (slot i at
    /// brev((i - 1) mod N/2): pc-check tfhe). Synchronous; builds the key switch's GEMM plans for the set's chunks.
    pub fn slots(&mut self, s: &pc_cuda::Stream<pc_cuda::Compute>, name: &str, slots: &[usize]) -> PyResult<()> {
        let half = self.n / 2;
        if slots.is_empty() || slots.iter().any(|&i| i >= half) {
            return Err(PyValueError::new_err(format!("tfhe slots {name}: empty or a slot >= {half}")));
        }
        if self.sets.contains_key(name) {
            return Err(PyValueError::new_err(format!("tfhe slots {name} registered twice")));
        }
        if self.pos_used + slots.len() > MAX_POS {
            return Err(PyValueError::new_err(format!("tfhe slots {name}: the session holds {MAX_POS} positions")));
        }
        let bits = half.trailing_zeros();
        let pos: Vec<u32> = slots.iter().map(|&i| (((i + half - 1) % half).reverse_bits() >> (usize::BITS - bits)) as u32).collect();
        self.pos_buf.upload_at(s, self.pos_used, &pos).map_err(|x| crate::session::gerr(x.into()))?;
        s.sync().map_err(|x| crate::session::gerr(x.into()))?;
        // SAFETY: the box is owned by self and outlives the switch's plans (declared after `sw`)
        let ltr: &'static pc_cuda::blas::lt::Lt = unsafe { &*(&self.lt.inner as *const pc_cuda::blas::lt::Lt) };
        let max = self.sw.max();
        self.sw.prepare(ltr, slots.len().min(max)).map_err(crate::session::gerr)?;
        if slots.len() > max && slots.len() % max != 0 {
            self.sw.prepare(ltr, slots.len() % max).map_err(crate::session::gerr)?;
        }
        self.sets.insert(name.to_string(), (self.pos_used, slots.len()));
        self.pos_used += slots.len();
        Ok(())
    }

    pub fn names(&self) -> (Vec<String>, Vec<String>) {
        (self.luts.keys().cloned().collect(), self.sets.keys().cloned().collect())
    }
}

fn ferr(f: &FinishStep, msg: impl Into<String>) -> pc_program::Error {
    pc_program::Error::Replay { block: format!("finish {}", f.desc), msg: msg.into() }
}

/// `FinishExec` over the session's TFHE service and its StC-first route.
pub(crate) struct TfheExec<'r> {
    pub rt: &'r mut TfheRt,
    pub route: &'r StcFirstRoute<'static, W, M>,
}

impl<'r> FinishExec<'static, W, M> for TfheExec<'r> {
    fn finish(&mut self, cx: &mut BtsCx<'_, 'static, W, M>, f: &FinishStep, slots: &mut [Ct<'static, W>]) -> pc_program::Result<()> {
        let d = Desc::parse(&f.desc).ok_or_else(|| ferr(f, "not a TFHE descriptor"))?;
        let rt = &mut *self.rt;
        let &k = rt.luts.get(&d.lut).ok_or_else(|| ferr(f, format!("lut {} never registered (Session.tfhe_lut)", d.lut)))?;
        let &(off, cnt) = rt.sets.get(&d.slots).ok_or_else(|| ferr(f, format!("slot set {} never registered (Session.tfhe_slots)", d.slots)))?;
        if f.ins.len() != 1 || f.outs.len() != 1 {
            return Err(ferr(f, format!("{} inputs / {} outputs (a TFHE finish has one of each)", f.ins.len(), f.outs.len())));
        }
        copy_ct(cx.ctx, &cx.ex.compute, &mut rt.x, &slots[f.ins[0] as usize])?;
        let r = pc_shadow::Route { cf: Some(CF), stc_first: true, real: false, ..pc_shadow::Route::dense() };
        let x = std::slice::from_mut(&mut rt.x);
        let mut call = BtsCall::begin(StagedRef::StcFirst(self.route), cx, x, &r, CallOpts { prescaled: false, own_cf: true })?;
        if call.stc_first_input(cx, x)?.is_none() {
            return Err(ferr(f, "input below the StC-first entry"));
        }
        // SAFETY: the box is owned by the service and outlives the switch's plans
        let ltr: &'static pc_cuda::blas::lt::Lt = unsafe { &*(&rt.lt.inner as *const pc_cuda::blas::lt::Lt) };
        let nb = rt.params.n_br;
        let pos = (rt.pos_buf.vptr() as *const u32).wrapping_add(off) as *const std::ffi::c_void;
        let tv = (rt.lut_buf.vptr() as *const u32).wrapping_add(k * 2 * nb) as *const std::ffi::c_void;
        rt.sw.apply_dev(cx.ctx, &cx.ex.compute, ltr, &mut x[0], pos, cnt, tv, 0, false)?;
        let mut fin = call.finish_call(0)?;
        fin.at_split(cx, x)?;
        fin.run_to(cx, x, Phase::Done)?;
        copy_ct(cx.ctx, &cx.ex.compute, &mut slots[f.outs[0] as usize], &x[0])?;
        rt.calls += 1;
        rt.values += cnt as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desc() {
        let d = Desc::parse(r#"{"op": "tfhe", "lut": "relu_step", "slots": "c0", "gain": 2.0}"#).unwrap();
        assert_eq!((d.lut.as_str(), d.slots.as_str()), ("relu_step", "c0"));
        assert!(Desc::parse(r#"{"op": "pcmm", "linear": "x"}"#).is_none());
    }
}
