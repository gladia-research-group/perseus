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
    /// (j, k): output j of k interleaved LUTs (`TfheRt::lut_multi`); j = 0 runs the switch, the others read its stash
    pub part: Option<(usize, usize)>,
}

impl Desc {
    /// `None` for a descriptor that is not a TFHE one.
    pub fn parse(desc: &str) -> Option<Desc> {
        let v: serde_json::Value = serde_json::from_str(desc).ok()?;
        if v.get("op")?.as_str()? != "tfhe" {
            return None;
        }
        let part = match v.get("part") {
            Some(p) => Some((p.get(0)?.as_u64()? as usize, p.get(1)?.as_u64()? as usize)),
            None => None,
        };
        Some(Desc { lut: v.get("lut")?.as_str()?.to_string(), slots: v.get("slots")?.as_str()?.to_string(), part })
    }
}

/// The switch's parameters: pc-tfhe's defaults and the PC_TFHE_* overrides (PC_TFHE_NU = 2: LUT grids of nu n_br
/// points, impl/tfhe.py GRID).
pub(crate) fn params(n: usize) -> PyResult<Params> {
    let mut p = Params::new(n).map_err(|x| PyValueError::new_err(x.to_string()))?;
    // PC_TFHE_PBS_GADGET=b,l: the bootstrapping key's gadget (two levels run the batch path)
    if let Ok(v) = std::env::var("PC_TFHE_PBS_GADGET") {
        let g = v.split_once(',').and_then(|(b, l)| Some((b.trim().parse::<u32>().ok()?, l.trim().parse::<usize>().ok()?)));
        p.pbs = match g {
            Some(g @ (_, 1 | 2)) if g.0 as usize * g.1 < 62 => g,
            _ => return Err(PyValueError::new_err(format!("PC_TFHE_PBS_GADGET={v}: base log, levels (1 or 2)"))),
        };
    }
    // PC_TFHE_LAT_GROUP=5: the latency key (Params::lat_group)
    if let Ok(v) = std::env::var("PC_TFHE_LAT_GROUP") {
        p.lat_group = v.trim().parse().ok().filter(|&g| g == 0 || (2..=5).contains(&g) && p.n_lwe % g == 0).ok_or_else(|| PyValueError::new_err(format!("PC_TFHE_LAT_GROUP={v}: 2..=5 dividing n_lwe {}", p.n_lwe)))?;
    }
    // PC_TFHE_LWE=n,log2: another LWE key (Params::with_lwe; 990,-21 the precise set)
    if let Ok(v) = std::env::var("PC_TFHE_LWE") {
        let g = v.split_once(',').and_then(|(n, l)| Some((n.trim().parse::<usize>().ok()?, l.trim().parse::<f64>().ok()?)));
        let (n_lwe, l) = g.ok_or_else(|| PyValueError::new_err(format!("PC_TFHE_LWE={v}: dimension, log2 sigma")))?;
        p = p.with_lwe(n_lwe, l).map_err(|x| PyValueError::new_err(x.to_string()))?;
    }
    if let Ok(v) = std::env::var("PC_TFHE_NU") {
        p.nu = match v.trim() {
            "" | "1" => 1,
            "2" => 2,
            _ => return Err(PyValueError::new_err(format!("PC_TFHE_NU={v}: 1 or 2"))),
        };
    }
    Ok(p)
}

/// What the session's MemPlan reserves: the switch (`max` values per pass), the tables, one ciphertext.
pub(crate) struct TfhePlan {
    pub max: usize,
}

impl TfhePlan {
    pub fn reserve(&self, mp: &mut MemPlan, n: usize) -> PyResult<()> {
        let p = params(n)?;
        mp.reserve("tfhe", TfheSwitch::bytes(n, &p, self.max));
        mp.reserve("tfhe.tables", 4 * (MAX_LUTS * 2 * p.tv_len() + MAX_POS) + 2 * 256);
        mp.reserve("tfhe.ct", 2 * pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&mp.lay, 1)));
        Ok(())
    }
}

/// The service. Field order = drop order: the switch's GEMM plans borrow the cuBLASLt handle declared last.
pub(crate) struct TfheRt {
    pub sw: TfheSwitch<'static>,
    pub params: Params,
    rns: pc_tfhe::pbs::Rns,
    /// name -> (table slot, output offset mod q_br)
    luts: BTreeMap<String, (usize, u64)>,
    lut_buf: pc_cuda::DBuf<'static, pc_cuda::Compute, u32>,
    /// name -> (offset, values, copies): positions [copies][values] from `offset`
    sets: BTreeMap<String, (usize, usize, usize)>,
    pos_buf: pc_cuda::DBuf<'static, pc_cuda::Compute, u32>,
    pos_used: usize,
    x: Ct<'static, W>,
    /// a two-LUT finish's second output (finished), for the part-1 op that follows; tagged by its LUT
    x2: Ct<'static, W>,
    stash: Option<String>,
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
        let params = params(n)?;
        let s64: Vec<i64> = s.iter().map(|&v| v as i64).collect();
        let host = pc_tfhe::switch::Switcher::generate(params.clone(), &s64, p0, p1, prng).map_err(ve)?;
        let sw = TfheSwitch::new(arena, dev, st, &host, plan.max).map_err(e)?;
        drop(host);
        let lt = Box::new(pc_cuda::blas::Lt::new(dev).map_err(|x| e(x.into()))?);
        let rns = pc_tfhe::pbs::Rns::new(params.br_primes.0, params.br_primes.1, params.n_br).map_err(ve)?;
        let lut_buf = arena.alloc::<pc_cuda::Compute, u32>("tfhe.tables", MAX_LUTS * 2 * params.tv_len()).map_err(|x| e(x.into()))?;
        let pos_buf = arena.alloc::<pc_cuda::Compute, u32>("tfhe.tables", MAX_POS).map_err(|x| e(x.into()))?;
        let x = Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "tfhe.ct", ctx.lay, 1).map_err(e)?);
        let x2 = Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "tfhe.ct", ctx.lay, 1).map_err(e)?);
        Ok(TfheRt { sw, params, rns, luts: BTreeMap::new(), lut_buf, sets: BTreeMap::new(), pos_buf, pos_used: 0, x, x2, stash: None, n, calls: 0, values: 0, keygen_s: t0.elapsed().as_secs_f64(), lt })
    }

    /// Register LUT `name`: `values[j]` = f(m_j) on the grid m_j = 2 j / g (j < g / 2) and 2 (j - g) / g (j >= g / 2)
    /// of [-1, 1), g = n_br nu, as read back after the finish (|f| <= `MAX_V`). Synchronous.
    pub fn lut(&mut self, s: &pc_cuda::Stream<pc_cuda::Compute>, name: &str, values: &[f64]) -> PyResult<()> {
        let nb = self.params.tv_len();
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
        self.store(s, name, &res, 0)
    }

    /// Register k = `values.len()` LUTs interleaved under `name` (one blind rotation): `values[j]` on the grid of
    /// n_br nu / k points (as `lut`). Synchronous.
    pub fn lut_multi(&mut self, s: &pc_cuda::Stream<pc_cuda::Compute>, name: &str, values: &[Vec<f64>]) -> PyResult<()> {
        let k = values.len();
        let nb = self.params.tv_len();
        if k < 2 || nb % k != 0 || values.iter().any(|v| v.len() != nb / k) {
            return Err(PyValueError::new_err(format!("tfhe lut {name}: {k} tables of the grid of {} points", nb / k.max(1))));
        }
        if let Some(v) = values.iter().flatten().find(|v| !(v.is_finite() && v.abs() <= MAX_V)) {
            return Err(PyValueError::new_err(format!("tfhe lut {name}: value {v} outside [-{MAX_V}, {MAX_V}]")));
        }
        let q = self.params.q_br;
        let tvs: Vec<Vec<u64>> = values
            .iter()
            .map(|vals| {
                let h = vals.len();
                vals.iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        let x = (q as f64 * pc_tfhe::pbs::warp(v, self.params.out_cf)).round() as i64;
                        (if j < h / 2 { x } else { -x }).rem_euclid(q as i64) as u64
                    })
                    .collect()
            })
            .collect();
        let tv = pc_tfhe::pbs::interleave(&tvs);
        let res: Vec<u32> = self.rns.residues(&tv).iter().flat_map(|r| r.iter().map(|&x| x as u32)).collect();
        self.store(s, name, &res, 0)
    }

    /// Register sign LUT `name`: `lo` for m < 0, `hi` for m >= 0 on the whole torus, inputs m in (-2, 2) (a constant
    /// test vector (hi - lo) / 2 and the offset (hi + lo) / 2, warped: the negacyclic sign needs no padding half, so
    /// the input range and the resolution double). |lo|, |hi| <= `MAX_V`. Synchronous.
    pub fn lut_sign(&mut self, s: &pc_cuda::Stream<pc_cuda::Compute>, name: &str, lo: f64, hi: f64) -> PyResult<()> {
        if !(lo.is_finite() && hi.is_finite() && lo.abs() <= MAX_V && hi.abs() <= MAX_V) {
            return Err(PyValueError::new_err(format!("tfhe sign lut {name}: ({lo}, {hi}) outside [-{MAX_V}, {MAX_V}]")));
        }
        let (nb, q) = (self.params.tv_len(), self.params.q_br);
        let w = |v: f64| q as f64 * pc_tfhe::pbs::warp(v, self.params.out_cf);
        let (c, off) = (((w(hi) - w(lo)) / 2.0).round() as i64, ((w(hi) + w(lo)) / 2.0).round() as i64);
        let tv = vec![c.rem_euclid(q as i64) as u64; nb];
        let res: Vec<u32> = self.rns.residues(&tv).iter().flat_map(|r| r.iter().map(|&x| x as u32)).collect();
        self.store(s, name, &res, off.rem_euclid(q as i64) as u64)
    }

    fn store(&mut self, s: &pc_cuda::Stream<pc_cuda::Compute>, name: &str, res: &[u32], offset: u64) -> PyResult<()> {
        let nb = self.params.tv_len();
        let slot = match self.luts.get(name) {
            Some(&(k, _)) => k,
            None if self.luts.len() < MAX_LUTS => self.luts.len(),
            None => return Err(PyValueError::new_err(format!("tfhe lut {name}: the session holds {MAX_LUTS} tables"))),
        };
        self.lut_buf.upload_at(s, slot * 2 * nb, res).map_err(|x| crate::session::gerr(x.into()))?;
        s.sync().map_err(|x| crate::session::gerr(x.into()))?;
        self.luts.insert(name.to_string(), (slot, offset));
        Ok(())
    }

    /// Register slot set `name`: the slots (of a real payload, < N/2) a LUT runs on, as split coefficients (slot i at
    /// brev((i - 1) mod N/2): pc-check tfhe), and `copies`: slot offsets each output is also written at (a layout's
    /// replicas: one bootstrap per value, the repack scatters it again). Synchronous; builds the key switch's GEMM
    /// plans for the set's chunks.
    pub fn slots(&mut self, s: &pc_cuda::Stream<pc_cuda::Compute>, name: &str, slots: &[usize], copies: &[usize]) -> PyResult<()> {
        let half = self.n / 2;
        if slots.is_empty() || slots.iter().any(|&i| i >= half) || copies.iter().any(|&c| c == 0 || c >= half) {
            return Err(PyValueError::new_err(format!("tfhe slots {name}: empty, a slot >= {half} or a copy offset outside 1..{half}")));
        }
        if self.sets.contains_key(name) {
            return Err(PyValueError::new_err(format!("tfhe slots {name} registered twice")));
        }
        let reps = 1 + copies.len();
        if self.pos_used + reps * slots.len() > MAX_POS {
            return Err(PyValueError::new_err(format!("tfhe slots {name}: the session holds {MAX_POS} positions")));
        }
        let bits = half.trailing_zeros();
        let coef = |i: usize| (((i + half - 1) % half).reverse_bits() >> (usize::BITS - bits)) as u32;
        let mut pos: Vec<u32> = slots.iter().map(|&i| coef(i)).collect();
        for &c in copies {
            pos.extend(slots.iter().map(|&i| coef((i + c) % half)));
        }
        self.pos_buf.upload_at(s, self.pos_used, &pos).map_err(|x| crate::session::gerr(x.into()))?;
        s.sync().map_err(|x| crate::session::gerr(x.into()))?;
        // SAFETY: the box is owned by self and outlives the switch's plans (declared after `sw`)
        let ltr: &'static pc_cuda::blas::lt::Lt = unsafe { &*(&self.lt.inner as *const pc_cuda::blas::lt::Lt) };
        let max = self.sw.max();
        self.sw.prepare(ltr, slots.len().min(max)).map_err(crate::session::gerr)?;
        if slots.len() > max && slots.len() % max != 0 {
            self.sw.prepare(ltr, slots.len() % max).map_err(crate::session::gerr)?;
        }
        self.sets.insert(name.to_string(), (self.pos_used, slots.len(), reps));
        self.pos_used += reps * slots.len();
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
        if let Some((j, _)) = d.part.filter(|p| p.0 > 0) {
            // the second output of the many-LUT finish just before (same input: its switch read the same m~)
            if rt.stash.take().as_deref() != Some(d.lut.as_str()) || j != 1 || f.outs.len() != 1 {
                return Err(ferr(f, format!("many-LUT part {j} of {} without its part 0 just before", d.lut)));
            }
            copy_ct(cx.ctx, &cx.ex.compute, &mut slots[f.outs[0] as usize], &rt.x2)?;
            rt.calls += 1;
            return Ok(());
        }
        let &(k, offset) = rt.luts.get(&d.lut).ok_or_else(|| ferr(f, format!("lut {} never registered (Session.tfhe_lut)", d.lut)))?;
        let &(off, cnt, reps) = rt.sets.get(&d.slots).ok_or_else(|| ferr(f, format!("slot set {} never registered (Session.tfhe_slots)", d.slots)))?;
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
        let nb = rt.params.tv_len();
        let pos = (rt.pos_buf.vptr() as *const u32).wrapping_add(off) as *const std::ffi::c_void;
        let tv = (rt.lut_buf.vptr() as *const u32).wrapping_add(k * 2 * nb) as *const std::ffi::c_void;
        match d.part {
            None => {
                rt.sw.apply_dev(cx.ctx, &cx.ex.compute, ltr, &mut x[0], pos, cnt, reps, tv, offset, false)?;
            }
            Some((0, 2)) => {
                rt.sw.apply_dev_k(cx.ctx, &cx.ex.compute, ltr, &mut x[0], &mut [&mut rt.x2], pos, cnt, reps, tv, &[0, 0], false)?;
                let x2 = std::slice::from_mut(&mut rt.x2);
                let mut fin2 = call.finish_call(0)?;
                fin2.at_split(cx, x2)?;
                fin2.run_to(cx, x2, Phase::Done)?;
                rt.stash = Some(d.lut.clone());
            }
            Some(p) => return Err(ferr(f, format!("part {p:?} of a many-LUT finish (two LUTs: part 0 runs the switch)"))),
        }
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
