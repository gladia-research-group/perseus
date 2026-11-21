//! The functional-bootstrap service of a planned session: CKKS functional bootstrapping (FBS) on the StC-first route
//! behind the replay's `StepOp::Finish` ops (pc-gpu `StcFirstRoute::run_fbs`).
//!
//! A captured `fbs` (perseus impl/fbs.py) is one external refresh of one input; the compiler makes it a `Finish` op
//! whose descriptor carries the whole function: `{op: "fbs", fn, cf, amp, post, real}` (`post`: the Chebyshev series
//! on [-1, 1] applied to EvalMod's sine, already scaled by 2^-cf: the StC-first output multiplies by 2^cf). The
//! session collects every distinct function from its programs at setup (no constructor argument), builds one ModRaise
//! table at K = 32 and one EvalMod program per function (`StcFirstPlan::fbs_host`), and runs a finish as: the input
//! copied into the output slot, the StC-first input half at the function's CF, the FBS (ModRaise K32, CtS, EvalMod +
//! series), the StC-first output. It lands at the program's output level (`FbsHost::out`), which impl/fbs.py
//! declares at capture (`Session`-free `fbs_landing`).
use crate::session::{M, W};
use pc_gpu::bts::staged::{BtsCall, CallOpts, StagedRef};
use pc_gpu::bts::stc_first::{FbsHost, FbsPrecomp, FbsSpec, StcFirstPlan, StcFirstRoute};
use pc_gpu::bts::BtsCx;
use pc_gpu::Ct;
use pc_program::bind::FinishStep;
use pc_program::program::{Kind, Program};
use pc_program::replay::{copy_ct, FinishExec};
use std::collections::BTreeMap;

/// A function's descriptor (`Finish::desc`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Desc {
    pub name: String,
    pub cf: u32,
    pub amp: f64,
    pub post: Vec<f64>,
    pub real: bool,
    /// The trigonometric form (`FbsSpec::table_s`); None: the sine form.
    pub trig: Option<Trig>,
    /// The sine form on the paired Paterson-Stockmeyer builder (`FbsSpec::paired`).
    pub paired: bool,
}

/// A trigonometric FBS's descriptor part: Q of P(t) + s Q(t), t's offset, the base pair's degree and double angles
/// (`pc_shadow::evalmod::trig_base`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Trig {
    pub post_s: Vec<f64>,
    pub off: f64,
    pub deg: usize,
    pub da: u32,
}

/// The base pair of a trigonometric FBS by default: degree 27, 5 double angles (4e-12 after the double angles; with Q
/// through the giant powers a degree-n series lands where the paired sine form of degree n does).
pub(crate) const TRIG_DEG: usize = 27;
pub(crate) const TRIG_DA: u32 = 5;

impl Desc {
    /// `None` for a descriptor that is not an FBS one.
    pub fn parse(desc: &str) -> Option<Desc> {
        let v: serde_json::Value = serde_json::from_str(desc).ok()?;
        if v.get("op")?.as_str()? != "fbs" {
            return None;
        }
        let post: Vec<f64> = v.get("post")?.as_array()?.iter().map(|x| x.as_f64()).collect::<Option<_>>()?;
        let trig = match v.get("trig") {
            None => None,
            Some(t) => Some(Trig {
                post_s: t.get("post_s")?.as_array()?.iter().map(|x| x.as_f64()).collect::<Option<_>>()?,
                off: t.get("off").and_then(|x| x.as_f64()).unwrap_or(0.0),
                deg: t.get("deg").and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(TRIG_DEG),
                da: t.get("da").and_then(|x| x.as_u64()).map(|x| x as u32).unwrap_or(TRIG_DA),
            }),
        };
        Some(Desc {
            name: v.get("fn")?.as_str()?.to_string(),
            cf: v.get("cf")?.as_u64()? as u32,
            amp: v.get("amp").and_then(|x| x.as_f64()).unwrap_or(1.0),
            post,
            real: v.get("real").and_then(|x| x.as_bool()).unwrap_or(false),
            trig,
            paired: v.get("paired").and_then(|x| x.as_bool()).unwrap_or(false),
        })
    }

    /// The pc-gpu spec (K = 32 table, raise drop 0); `complex` / `real`: the EvalMod modes built.
    pub fn spec(&self, complex: bool, real: bool) -> FbsSpec {
        let k = pc_core::bts_setup::EVALMOD_K32;
        let (table, da_its, table_s, off, post_s) = match &self.trig {
            None => (pc_core::bts_setup::EVALMOD_K32_CHEB.to_vec(), pc_core::bts_setup::EVALMOD_K32_DA_ITS, vec![], 0.0, vec![]),
            Some(t) => {
                let (c, s) = pc_shadow::evalmod::trig_base(k, t.da, t.deg);
                (c, t.da, s, t.off, t.post_s.clone())
            }
        };
        FbsSpec { table, k, da_its, amp: self.amp, post: self.post.clone(), table_s, off, post_s, paired: self.paired, drop: 0, complex, real }
    }
}

/// Every FBS function the programs' finishes name: name -> (descriptor, needs complex, needs real). Err: one name
/// with two different functions.
pub(crate) fn functions(progs: &[Program]) -> Result<BTreeMap<String, (Desc, bool, bool)>, String> {
    let mut out: BTreeMap<String, (Desc, bool, bool)> = BTreeMap::new();
    for p in progs {
        for op in &p.ops {
            for m in op.members() {
                if let Kind::Finish { desc: Some(d), .. } = &m.kind {
                    if let Some(d) = Desc::parse(d) {
                        let e = out.entry(d.name.clone()).or_insert_with(|| (d.clone(), false, false));
                        if (e.0.cf, e.0.amp.to_bits(), &e.0.post, &e.0.trig, e.0.paired) != (d.cf, d.amp.to_bits(), &d.post, &d.trig, d.paired) {
                            return Err(format!("fbs function {} captured with two different series", d.name));
                        }
                        if d.real {
                            e.2 = true;
                        } else {
                            e.1 = true;
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// The host plans of the functions on the session's StC-first plan.
pub(crate) fn hosts(sp: &StcFirstPlan, fns: &BTreeMap<String, (Desc, bool, bool)>) -> pc_gpu::Result<BTreeMap<String, FbsHost>> {
    fns.iter().map(|(n, (d, c, r))| Ok((n.clone(), sp.fbs_host(&d.spec(*c, *r))?))).collect()
}

/// The device side: one precomputation per function.
pub(crate) struct FbsRt {
    pub pre: BTreeMap<String, FbsPrecomp<'static, W>>,
    pub calls: u64,
}

fn ferr(f: &FinishStep, msg: impl Into<String>) -> pc_program::Error {
    pc_program::Error::Replay { block: format!("finish fbs {}", f.desc.chars().take(80).collect::<String>()), msg: msg.into() }
}

/// `FinishExec` over the session's FBS service and its StC-first route.
pub(crate) struct FbsExec<'r> {
    pub rt: &'r mut FbsRt,
    pub route: &'r StcFirstRoute<'static, W, M>,
}

impl<'r> FinishExec<'static, W, M> for FbsExec<'r> {
    fn finish(&mut self, cx: &mut BtsCx<'_, 'static, W, M>, f: &FinishStep, slots: &mut [Ct<'static, W>]) -> pc_program::Result<()> {
        let d = Desc::parse(&f.desc).ok_or_else(|| ferr(f, "not an FBS descriptor"))?;
        let pre = self.rt.pre.get(&d.name).ok_or_else(|| ferr(f, format!("function {} not built at setup", d.name)))?;
        if f.ins.len() != 1 || f.outs.len() != 1 {
            return Err(ferr(f, format!("{} inputs / {} outputs (an FBS finish has one of each)", f.ins.len(), f.outs.len())));
        }
        let (i, o) = (f.ins[0] as usize, f.outs[0] as usize);
        if i != o {
            let (src, dst) = if i < o {
                let (a, b) = slots.split_at_mut(o);
                (&a[i], &mut b[0])
            } else {
                let (a, b) = slots.split_at_mut(i);
                (&b[0], &mut a[o])
            };
            copy_ct(cx.ctx, &cx.ex.compute, dst, src)?;
        }
        let x = std::slice::from_mut(&mut slots[o]);
        let r = pc_shadow::Route { cf: Some(d.cf), stc_first: true, real: d.real, ..pc_shadow::Route::dense() };
        let mut call = BtsCall::begin(StagedRef::StcFirst(self.route), cx, x, &r, CallOpts { prescaled: false, own_cf: true })?;
        if call.stc_first_input(cx, x)?.is_none() {
            return Err(ferr(f, "input below the StC-first entry"));
        }
        self.route.run_fbs(cx, &call.st, x, pre)?;
        // realized: the output's pending rescale paid here (the capture declares the realized landing, `fbs_landing`)
        if x[0].meta.deg == 2 {
            cx.ops.rescale(cx.ctx, cx.ex, cx.sh, x)?;
        }
        self.rt.calls += 1;
        Ok(())
    }
}

/// The host side of a chain preset an FBS runs on: its chain and the noise model of its StC-first route's drop-0
/// variant (`FbsNoiseModel`, encapsulation weight `h`), built once per (preset, h) (seconds: the route's host plan).
fn model(preset: &str, h: usize) -> Result<std::sync::Arc<(pc_core::params::Chain, pc_gpu::bts::fbs_noise::FbsNoiseModel)>, String> {
    use std::sync::{Arc, Mutex};
    static CACHE: Mutex<BTreeMap<(String, usize), Arc<(pc_core::params::Chain, pc_gpu::bts::fbs_noise::FbsNoiseModel)>>> = Mutex::new(BTreeMap::new());
    let key = (preset.to_string(), h);
    if let Some(m) = CACHE.lock().unwrap().get(&key) {
        return Ok(m.clone());
    }
    let spec = pc_core::params::ChainSpec::preset(preset).map_err(|e| e.to_string())?;
    let chain = pc_core::params::Chain::generate(&spec).map_err(|e| e.to_string())?;
    pc_gpu::bts::stc_first::fbs_chain_check(&chain).map_err(|e| format!("chain preset {preset}: {e}"))?;
    let mut setup = pc_gpu::bts::BtsSetup::n32_shipped();
    setup.raise_drops = vec![0];
    let sp = StcFirstPlan::new(&chain, &setup, &pc_gpu::bts::stc_first::StcFirstSetup::n32_shipped()).map_err(|e| e.to_string())?;
    let m = pc_gpu::bts::fbs_noise::FbsNoiseModel::new(&sp, h).map_err(|e| e.to_string())?;
    let m = Arc::new((chain, m));
    CACHE.lock().unwrap().insert(key, m.clone());
    Ok(m)
}

fn ps_of(amp: f64, post: &[f64], trig: Option<Trig>, paired: bool) -> Result<(pc_shadow::evalmod::PsPlan, FbsSpec), String> {
    let d = Desc { name: String::new(), cf: 0, amp, post: post.to_vec(), real: true, trig, paired };
    let sp = d.spec(false, true);
    Ok((sp.ps_plan().map_err(|e| e.to_string())?, sp))
}

/// The trigonometric part of `fbs_landing` / `fbs_noise`'s arguments (`post_s` None: the sine form).
fn trig_of(post_s: Option<Vec<f64>>, off: f64, tdeg: usize, tda: u32) -> Option<Trig> {
    post_s.map(|post_s| Trig { post_s, off, deg: tdeg, da: tda })
}

/// Where an FBS of `post` (amplitude `amp`, the K = 32 table) on chain preset `preset` lands: (level, 1), its
/// EvalMod output on the StC-first route's drop-0 variant (the session's `FbsHost::out`), realized. Refused on a chain
/// FBS is not validated on (`fbs_chain_check`).
#[pyo3::pyfunction]
#[pyo3(signature = (preset, amp, post, post_s=None, off=0.0, tdeg=TRIG_DEG, tda=TRIG_DA, paired=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fbs_landing(preset: &str, amp: f64, post: Vec<f64>, post_s: Option<Vec<f64>>, off: f64, tdeg: usize, tda: u32, paired: bool) -> pyo3::PyResult<(u32, u8)> {
    use pyo3::exceptions::PyValueError;
    let ve = |e: String| PyValueError::new_err(e);
    let m = model(preset, pc_core::keys::SecretKey::SPARSE_H).map_err(ve)?;
    let (chain, nm) = (&m.0, &m.1);
    let (ps, sp) = ps_of(amp, &post, trig_of(post_s, off, tdeg, tda), paired).map_err(ve)?;
    let st = pc_core::scale::ScaleTables::precompute(chain).map_err(|e| ve(e.to_string()))?;
    let (_, out, _) = pc_gpu::bts::evalmod::EvalModPlan::<u32>::program(&st, nm.em_in, &ps, sp.da_its).map_err(|e| ve(e.to_string()))?;
    let shape = pc_shadow::ChainShape::of(chain);
    // realized (the service rescales the degree-2 output)
    let top = if out.nl == 2 { pc_shadow::fides::Fx { d: st.d as i32, st: Some(&st) }.down(out.top) } else { out.top };
    Ok((pc_shadow::meta::level_from_top_limb(shape, top as u32).map_err(|e| ve(e.to_string()))?, 1))
}

/// The noise model of an FBS (`pc_gpu::bts::fbs_noise`) of `post` (amplitude `amp`, phase x / 2^`cf`, the K = 32
/// table) on chain preset `preset`, the encapsulation secret of Hamming weight `h`: at each input `xs` (input units),
/// the noise-free output's range over the ModRaise integers |I| <= K (`lo`, `hi`) and a bound on its error std
/// (`sigma`), the caller's input noise std `sigma_x` included; the phase std at EvalMod's input, the bootstrap's own
/// (`phase_own`) and total (`phase`), and its terms (`split_x`: input units, `q0_u`: phase units, `cts_y`: EvalMod
/// input units, x K in phase units); `k`: the table's K; `by_i`: per input, (value, std) at each I = -K..=K (NaN off
/// EvalMod's domain).
#[pyo3::pyfunction]
#[pyo3(signature = (preset, cf, amp, post, xs, sigma_x=0.0, h=32, post_s=None, off=0.0, tdeg=TRIG_DEG, tda=TRIG_DA, paired=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fbs_noise(py: pyo3::Python<'_>, preset: &str, cf: u32, amp: f64, post: Vec<f64>, xs: Vec<f64>, sigma_x: f64, h: usize, post_s: Option<Vec<f64>>, off: f64, tdeg: usize, tda: u32, paired: bool) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyDict>> {
    use pyo3::exceptions::PyValueError;
    use pyo3::types::PyDictMethods;
    let ve = |e: String| PyValueError::new_err(e);
    let m = py.allow_threads(|| model(preset, h)).map_err(ve)?;
    let nm = &m.1;
    let (ps, sp) = ps_of(amp, &post, trig_of(post_s, off, tdeg, tda), paired).map_err(ve)?;
    let r = py.allow_threads(|| nm.eval_ps(&ps, sp.da_its, sp.k, cf, &xs, sigma_x)).map_err(|e| ve(e.to_string()))?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("lo", r.points.iter().map(|p| p.lo).collect::<Vec<_>>())?;
    d.set_item("hi", r.points.iter().map(|p| p.hi).collect::<Vec<_>>())?;
    d.set_item("sigma", r.points.iter().map(|p| p.sigma).collect::<Vec<_>>())?;
    d.set_item("phase_own", r.phase_own)?;
    d.set_item("phase", r.phase)?;
    d.set_item("split_x", nm.split_x)?;
    d.set_item("q0_u", nm.q0_u)?;
    d.set_item("cts_y", nm.cts_y)?;
    d.set_item("k", sp.k)?;
    d.set_item("by_i", r.by_i)?;
    Ok(d.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desc() {
        let d = Desc::parse(r#"{"op": "fbs", "fn": "sign", "cf": 3, "amp": 1.0, "post": [0.5, 0.25], "real": true}"#).unwrap();
        assert_eq!((d.name.as_str(), d.cf, d.post.len(), d.real), ("sign", 3, 2, true));
        assert!(Desc::parse(r#"{"op": "tfhe", "lut": "x", "slots": "s"}"#).is_none());
        let t = Desc::parse(r#"{"op": "fbs", "fn": "e", "cf": 3, "amp": 2.0, "post": [0.5, 0.25, 0.1], "real": true, "trig": {"post_s": [0.1, 0.2], "off": -1.0}}"#).unwrap();
        let tr = t.trig.clone().unwrap();
        assert_eq!((tr.post_s.len(), tr.off, tr.deg, tr.da), (2, -1.0, TRIG_DEG, TRIG_DA));
        let sp = t.spec(false, true);
        assert_eq!((sp.table.len(), sp.table_s.len(), sp.da_its, sp.off), (TRIG_DEG + 1, TRIG_DEG + 1, TRIG_DA, -1.0));
    }
}
