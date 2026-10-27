//! `Program` (pc-program) for Python: compile a captured graph against its plan, inspect it, bind it to a session's
//! runtime and run it (the planned mode of `impl/driver.py`).
//!
//! A `Program` is the CUDA-free compile result. `bind(session, store)` (before `Session.complete_setup`) carves its
//! ciphertext slots and key groups in the session's arena and resolves its plaintexts in the session's `PtStore`;
//! the bound block lives in the session (`Planned::bounds`), the Python object keeps its index. `run` executes the
//! whole block in Rust with the GIL released: the dynamic steps (per-position scalars and rotations the driver set
//! with `set_scalar` / `set_rotation`) eagerly, every static segment between them as a captured CUDA graph after the
//! first run (`exec.use_graphs`), then the KV links (cache outputs -> cache inputs for the next token).
use crate::session::{gerr, nogil, Dev, State, W};
use pc_gpu::Ct;
use pc_program::bind::BindEnv;
use pc_program::replay::{copy_ct, Rt, Served};
use pc_program::compile::{compile_dir, CompileOpts};
use pc_program::program::{PtForm, SiteKind};
use pc_program::{Kind, Program};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) fn perr(e: pc_program::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// One compiled block program; `bound` = its index in the session's bound blocks once `bind` ran.
#[pyclass(name = "Program", module = "perseus._pc", unsendable)]
pub struct PyProgram {
    pub(crate) prog: Program,
    /// The plan's runtime Shadow (`compile::runtime_shadow`: the landings the replay's refreshes take).
    pub(crate) sh_cfg: Option<pc_shadow::ShadowConfig>,
    pub(crate) bound: Option<(std::rc::Rc<crate::session::Dev>, usize)>,
}

impl PyProgram {
    pub(crate) fn bound_index(&self) -> PyResult<usize> {
        self.bound.as_ref().map(|b| b.1).ok_or_else(|| PyRuntimeError::new_err(format!("{}: not bound (Program.bind)", self.prog.block)))
    }
    fn dev(&self) -> PyResult<std::rc::Rc<Dev>> {
        self.bound.as_ref().map(|b| b.0.clone()).ok_or_else(|| PyRuntimeError::new_err(format!("{}: not bound (Program.bind)", self.prog.block)))
    }
}

/// One input of `Program.run`.
enum In<'p> {
    Reg(usize),
    Ct(&'p Ct<'static, W>),
    Zero,
}

fn meta_t(m: &pc_shadow::CtMeta) -> (u32, u8, f64) {
    (m.level, m.deg, m.nf)
}

/// The graph name of a var (`v_12~pre` -> `v_12`).
pub(crate) fn base_name(n: &str) -> &str {
    n.split('~').next().unwrap_or(n)
}

#[pymethods]
impl PyProgram {
    /// Compile every block of `plan_dir` against `graph_dir` (`compile_dir`: each block entered at the previous
    /// block's compiled exit). Returns {label: Program} ("block_<b>", "block_<b>_<variant>"). `config`: the session
    /// Config; every load-bearing field a block plan's stamp sets must match it (the contract check; None = skip).
    /// `blocks`: labels to keep (None = all). A block whose compile check fails raises listing the errors, unless
    /// `keep_going`. `live_out` {label: [names]}: final states kept as outputs (K/V states); `input_metas` {label:
    /// {name: (level, deg)}}: the entry state of inputs the driver knows. `schedule`: "program" (the capture's order,
    /// siblings hoisted) or "lockstep" (independent chains advance together: more K-batched refreshes, more live
    /// ciphertexts).
    #[staticmethod]
    /// `routes_built`: the sparse routes the runtime holds while these blocks run (SM_PERIODIC=2: a planned refresh
    /// of a value whose period outgrew its planned slot count takes the smallest built route its period allows).
    /// `dense_fallback`: a refresh whose value outgrew every built sparse route runs dense.
    #[pyo3(signature = (graph_dir, plan_dir, config=None, blocks=None, keep_going=false, no_batch=false, live_out=None, input_metas=None, schedule="program", cf_free=false, routes_built=None, dense_fallback=false))]
    #[allow(clippy::too_many_arguments)]
    fn compile(py: Python<'_>, graph_dir: &str, plan_dir: &str, config: Option<&crate::config::PyConfig>, blocks: Option<Vec<String>>, keep_going: bool, no_batch: bool, live_out: Option<BTreeMap<String, Vec<String>>>, input_metas: Option<BTreeMap<String, BTreeMap<String, (u32, u8)>>>, schedule: &str, cf_free: bool, routes_built: Option<Vec<u32>>, dense_fallback: bool) -> PyResult<Py<PyDict>> {
        let schedule = match schedule {
            "program" => pc_program::compile::batch::Schedule::Program,
            "lockstep" => pc_program::compile::batch::Schedule::Lockstep,
            x => return Err(PyValueError::new_err(format!("Program.compile(schedule={x:?}): program | lockstep"))),
        };
        let opts = CompileOpts { keep_going, no_batch, live_out: live_out.unwrap_or_default(), input_metas: input_metas.unwrap_or_default(), schedule, batch_cf_free: cf_free, routes_built: routes_built.unwrap_or_default(), dense_fallback, ..CompileOpts::default() };
        let (gd, pd) = (graph_dir.to_string(), plan_dir.to_string());
        let dc = nogil(py, || compile_dir(Path::new(&pd), Path::new(&gd), &opts)).map_err(perr)?;
        if let Some(cfg) = config {
            let pdir = pc_program::plan::PlanDir::open(plan_dir).map_err(perr)?;
            let want = cfg.0.load_bearing();
            let mut bad = vec![];
            for (b, v, path) in &pdir.blocks {
                let bp = pc_program::plan::BlockPlan::load(path).map_err(perr)?;
                let stamp = pdir.config(&bp).map_err(perr)?.load_bearing();
                let (a, s) = (serde_json::to_value(&want).map_err(|e| PyValueError::new_err(e.to_string()))?, serde_json::to_value(&stamp).map_err(|e| PyValueError::new_err(e.to_string()))?);
                let diff = json_diff(&a, &s);
                if !diff.is_empty() {
                    bad.push(format!("block_{b}{}: {}", v.as_deref().map(|x| format!("_{x}")).unwrap_or_default(), diff.join(", ")));
                }
            }
            if !bad.is_empty() {
                return Err(crate::ConfigError::new_err(format!("plan {plan_dir} was cut for another Config (session vs stamp): {}", bad.join("; "))));
            }
        }
        let pdir = pc_program::plan::PlanDir::open(plan_dir).map_err(perr)?;
        let plan_cfg = pdir.plan_config().map_err(perr)?;
        let mut shadow_of: BTreeMap<String, pc_shadow::ShadowConfig> = BTreeMap::new();
        for (b, v, path) in &pdir.blocks {
            let label = match v {
                Some(v) => format!("block_{b}_{v}"),
                None => format!("block_{b}"),
            };
            let bp = pc_program::plan::BlockPlan::load(path).map_err(perr)?;
            let cfg = pdir.config(&bp).map_err(perr)?;
            let sh = pc_program::compile::runtime_shadow(&cfg, &plan_cfg, bp.file.rules.bootstrap_level).map_err(perr)?;
            shadow_of.insert(label, sh.cfg.clone());
        }
        let out = PyDict::new(py);
        let mut errs = vec![];
        for (label, r) in dc.programs {
            if let Some(keep) = &blocks {
                if !keep.contains(&label) {
                    continue;
                }
            }
            match r {
                Ok(p) => {
                    if !p.check.errors.is_empty() && !keep_going {
                        errs.push(format!("{label}: {} check error(s): {}", p.check.errors.len(), p.check.errors.iter().take(4).cloned().collect::<Vec<_>>().join(" | ")));
                    }
                    let sh_cfg = shadow_of.get(&label).cloned();
                    out.set_item(label, Py::new(py, PyProgram { prog: p, sh_cfg, bound: None })?)?;
                }
                Err(e) => errs.push(format!("{label}: {e}")),
            }
        }
        if !errs.is_empty() {
            return Err(PyRuntimeError::new_err(format!("compile {graph_dir} x {plan_dir}: {}", errs.join("; "))));
        }
        Ok(out.unbind())
    }

    /// Rename the program (before `bind`): one plan block compiled for several program classes (structures of the
    /// per-position stream) binds once per class under distinct labels.
    fn relabel(&mut self, label: &str) -> PyResult<()> {
        if self.bound.is_some() {
            return Err(PyRuntimeError::new_err(format!("{}: relabel after bind", self.prog.block)));
        }
        self.prog.block = label.to_string();
        Ok(())
    }

    /// A program saved by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        Ok(PyProgram { prog: Program::load(Path::new(path)).map_err(perr)?, sh_cfg: None, bound: None })
    }
    fn save(&self, path: &str) -> PyResult<()> {
        self.prog.save(Path::new(path)).map_err(perr)
    }

    #[getter]
    fn label(&self) -> String {
        self.prog.block.clone()
    }
    /// Ops (launch members) / launches / refresh sites.
    #[getter]
    fn n_ops(&self) -> usize {
        self.prog.ops.iter().map(|o| o.members().len()).sum()
    }
    #[getter]
    fn n_launches(&self) -> usize {
        self.prog.ops.len()
    }
    #[getter]
    fn n_bootstraps(&self) -> usize {
        self.prog.bts_sites.len()
    }
    /// K-batching of the refresh sites: (k_max, refresh launches, {batch size: launches}, {route: (sites, launches)}).
    #[getter]
    fn bts_batches(&self) -> (u32, usize, BTreeMap<usize, usize>, BTreeMap<String, (usize, usize)>) {
        let r = &self.prog.batch_report;
        let mut per: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for op in &self.prog.ops {
            let ms = op.members();
            if let Kind::Bootstrap(s) = ms[0].kind {
                let route = self.prog.bts_sites.get(s as usize).map(|x| format!("{:?}", pc_gpu::bts::RouteKind::of(&x.route, 1 << 16))).unwrap_or_default();
                let e = per.entry(route).or_default();
                e.0 += ms.len();
                e.1 += 1;
            }
        }
        (r.k_max, r.bootstrap_launches, r.bootstrap_hist.clone(), per)
    }
    /// External finishes (PCMM / CCMM ops): (ops, refreshed outputs).
    #[getter]
    fn n_finishes(&self) -> (usize, usize) {
        let mut n = (0, 0);
        for op in &self.prog.ops {
            for m in op.members() {
                if let Kind::Finish { .. } = m.kind {
                    n.0 += 1;
                    n.1 += m.outs.len();
                }
            }
        }
        n
    }
    /// Term counts of the streamed-weight sums (`CoeffSum` ops), op order.
    #[getter]
    fn coeff_sums(&self) -> Vec<usize> {
        let mut v = vec![];
        for op in &self.prog.ops {
            for m in op.members() {
                if let Kind::CoeffSum { pts, .. } = &m.kind {
                    v.push(pts.len());
                }
            }
        }
        v
    }
    /// [(graph name, level, deg, nf)] of the block inputs (first-use order: the driver's write order).
    #[getter]
    fn inputs(&self) -> Vec<(String, u32, u8, f64)> {
        self.prog.inputs.iter().map(|&v| {
            let x = &self.prog.vars[v as usize];
            let (l, d, nf) = meta_t(&x.meta);
            (x.name.clone(), l, d, nf)
        }).collect()
    }
    /// [(graph name, level, deg, nf)] of the outputs (exit var first, then the final states nothing in the block read).
    #[getter]
    fn outputs(&self) -> Vec<(String, u32, u8, f64)> {
        self.prog.outputs.iter().map(|&v| {
            let x = &self.prog.vars[v as usize];
            let (l, d, nf) = meta_t(&x.meta);
            (base_name(&x.name).to_string(), l, d, nf)
        }).collect()
    }
    /// The output position of graph name `name` (aliases included: a `level_hint` / clone names its input's var).
    fn output_index(&self, name: &str) -> PyResult<usize> {
        let v = self.prog.names.get(name).ok_or_else(|| PyValueError::new_err(format!("{}: no var {name}", self.prog.block)))?;
        self.prog.outputs.iter().position(|o| o == v).ok_or_else(|| PyValueError::new_err(format!("{}: {name} is not an output (compile live_out)", self.prog.block)))
    }

    /// [(name, level, deg, form)] of the plaintext operands (one per use).
    #[getter]
    fn pt_refs(&self) -> Vec<(String, u32, u8, &'static str)> {
        self.prog.pt_refs.iter().map(|r| (r.name.clone(), r.level, r.deg, match r.form {
            PtForm::AtUse => "at_use",
            PtForm::Slot => "slot",
            PtForm::Cached => "cached",
            PtForm::CoeffStream => "stream",
        })).collect()
    }
    /// [(var, kind, slots, cf, real, raise_drop, stc_first)] of the refresh sites.
    #[getter]
    fn bts_sites(&self) -> Vec<(String, &'static str, u32, Option<u32>, bool, u32, bool)> {
        self.prog.bts_sites.iter().map(|s| (s.var.clone(), match s.kind {
            SiteKind::Placement => "placement",
            SiteKind::Hint => "hint",
            SiteKind::Deliberate => "deliberate",
        }, s.route.slots, s.route.cf, s.route.real, s.route.raise_drop, s.route.stc_first)).collect()
    }
    /// [(var, input level, input deg, landing level, landing deg)] of the refresh sites, in `bts_sites` order: a dense
    /// StC-first site entering past the entry shows here as a landing at the shipped order's level.
    #[getter]
    fn bts_site_levels(&self) -> Vec<(String, u32, u8, u32, u8)> {
        self.prog.bts_sites.iter().map(|s| (s.var.clone(), s.input.level, s.input.deg, s.landing.level, s.landing.deg)).collect()
    }
    /// {graph name: scalar} of every scalar op (add / mult by a constant; `sub` of c records -c).
    fn scalars(&self) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        for op in &self.prog.ops {
            for o in op.members() {
                if let Kind::AddConst(c) | Kind::MultConst(c) = &o.kind {
                    if let Some(&v) = o.outs.first() {
                        m.insert(base_name(&self.prog.vars[v as usize].name).to_string(), c.value);
                    }
                }
            }
        }
        m
    }
    /// {graph name: k} of every single rotation.
    fn rotations(&self) -> BTreeMap<String, i64> {
        let mut m = BTreeMap::new();
        for op in &self.prog.ops {
            for o in op.members() {
                if let Kind::Rotate(k) = &o.kind {
                    if let Some(&v) = o.outs.first() {
                        m.insert(base_name(&self.prog.vars[v as usize].name).to_string(), *k);
                    }
                }
            }
        }
        m
    }
    /// Every rotation step the block's ops use (normalized as written).
    fn rot_steps(&self) -> Vec<i64> {
        let mut s = std::collections::BTreeSet::new();
        for op in &self.prog.ops {
            for o in op.members() {
                match &o.kind {
                    Kind::Rotate(k) => {
                        s.insert(*k);
                    }
                    Kind::RotateSum(v) => s.extend(v.iter().copied()),
                    _ => {}
                }
            }
        }
        s.into_iter().collect()
    }
    /// The compile check: (pinned vars, errors).
    #[getter]
    fn check(&self) -> (usize, Vec<String>) {
        (self.prog.check.pinned, self.prog.check.errors.clone())
    }
    /// Compiled slab bytes (limb-exact) and slot count.
    #[getter]
    fn mem(&self) -> (u64, usize) {
        (self.prog.mem.bytes, self.prog.mem.slot_bytes.len())
    }
    /// Bind to a planned `session` (before `complete_setup`): key groups (+ one over `dyn_rot`, the session's
    /// by default), the slab's slots, the routes. `store`: the session's `PtStore` (plaintexts are resolved before
    /// every run, so they may be staged after binding).
    #[pyo3(signature = (session, store=None, dyn_rot=None))]
    fn bind(&mut self, session: &crate::session::PySession, store: Option<&crate::store::PyPtStore>, dyn_rot: Option<Vec<i64>>) -> PyResult<()> {
        let _ = store;
        if self.bound.is_some() {
            return Err(PyRuntimeError::new_err(format!("{} is bound already", self.prog.block)));
        }
        let dev = session.dev.clone();
        if dev.frozen.get() {
            return Err(PyRuntimeError::new_err("Program.bind after Session.complete_setup (the arena is frozen)"));
        }
        let sh_cfg = self.sh_cfg.clone().ok_or_else(|| PyValueError::new_err(format!("{}: no runtime Shadow (compile it with Program.compile)", self.prog.block)))?;
        let mut st = dev.st.borrow_mut();
        let State { planned, ex, keys, .. } = &mut *st;
        let pl = planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("Program.bind: the Session was not built with programs=[...]"))?;
        if !pl.spec.programs.iter().any(|p| p.block == self.prog.block && p.mem.slot_bytes.len() >= self.prog.mem.slot_bytes.len()) {
            return Err(PyValueError::new_err(format!("{}: not among the programs the Session was sized for", self.prog.block)));
        }
        let dr = dyn_rot.unwrap_or_else(|| pl.spec.dyn_rot.clone());
        if dr.len() > pl.spec.dyn_rot.len() {
            return Err(PyValueError::new_err(format!("dyn_rot of {} steps, the Session reserved {}", dr.len(), pl.spec.dyn_rot.len())));
        }
        let scalars = pc_program::bind::ScalarTable::new();
        let stage = ex.ks.stage_ptrs();
        let b = {
            let env = BindEnv { arena: dev.arena_static(), stream: &ex.compute, lay: dev.ctx.lay, keys: &*keys, stage, scalars: &scalars, streams: Some(&pl.streams) };
            let routes = pl.routes();
            pc_program::bind::Bound::new_shared(&self.prog, &env, &routes, &dr).map_err(perr)?
        };
        // the weight frame's content id: blocks of different program classes streaming the same linears share it
        let tag = if b.coeff_sums.is_empty() {
            None
        } else {
            Some(match pl.tag_layouts.iter().position(|l| *l == b.streams) {
                Some(t) => t,
                None => {
                    pl.tag_layouts.push(b.streams.clone());
                    pl.tag_layouts.len() - 1
                }
            })
        };
        pl.stream_tag.push(tag);
        pl.bounds.push(b);
        pl.shadows.push(pc_shadow::Shadow::new(sh_cfg, Some(std::sync::Arc::new(dev.ctx.scale.clone()))));
        let i = pl.bounds.len() - 1;
        drop(st);
        self.bound = Some((dev, i));
        Ok(())
    }

    /// The plaintext keys the bound block reads: [(store name, level)] (`<block>/pt_N` for the graph's `pt_N`).
    fn pt_keys(&self) -> PyResult<Vec<(String, u32)>> {
        let dev = self.dev()?;
        let st = dev.st.borrow();
        let pl = st.planned.as_ref().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        Ok(pl.bounds[self.bound_index()?].pt_keys().into_iter().map(|k| (k.name, k.level)).collect())
    }

    /// Re-key the bound block's plaintexts: `alias` {store name: shared name} (e.g. a content hash), so blocks
    /// whose plaintexts carry equal values at one level read one resident slot (the prefill's masks repeat across
    /// blocks: GPT-2 T=64 reads 3465 plaintext keys holding 842 distinct values). Before the first run. Returns the
    /// keys renamed.
    fn alias_pts(&self, alias: std::collections::HashMap<String, String>) -> PyResult<usize> {
        let dev = self.dev()?;
        let mut st = dev.st.borrow_mut();
        let pl = st.planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let b = &mut pl.bounds[self.bound_index()?];
        if b.n_graphs() > 0 {
            return Err(PyRuntimeError::new_err(format!("{}: alias_pts after a graph capture", b.label)));
        }
        let mut n = 0;
        for p in b.pts.iter_mut() {
            if let Some(to) = alias.get(&p.key.name) {
                p.key.name = to.clone();
                p.meta = None;
                p.rows = std::ptr::null();
                n += 1;
            }
        }
        Ok(n)
    }

    /// Point plaintext operands at other store keys: `pairs` [(index into `pt_refs`, store name)] (the level stays
    /// the compiled one). Eager runs only (a captured graph baked the rows): the per-token value aliasing of a
    /// streamed planned run (`StreamedPlannedRun`: equal values across blocks share one slot, every token).
    fn retarget_pts(&self, pairs: Vec<(usize, String)>) -> PyResult<usize> {
        let dev = self.dev()?;
        let mut st = dev.st.borrow_mut();
        let pl = st.planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let b = &mut pl.bounds[self.bound_index()?];
        if b.n_graphs() > 0 {
            return Err(PyRuntimeError::new_err(format!("{}: retarget_pts after a graph capture", b.label)));
        }
        let n = b.pts.len();
        let mut k = 0;
        for (i, name) in pairs {
            let p = b.pts.get_mut(i).ok_or_else(|| PyValueError::new_err(format!("plaintext {i} of {n}")))?;
            if p.stream {
                return Err(PyValueError::new_err(format!("plaintext {i} is a streamed weight row")));
            }
            if p.key.name != name {
                p.key.name = name;
                p.meta = None;
                p.rows = std::ptr::null();
                k += 1;
            }
        }
        Ok(k)
    }

    /// Set the scalar of the constant op producing graph name `name` (dynamic step from now on).
    fn set_scalar(&self, name: &str, value: f64) -> PyResult<()> {
        let dev = self.dev()?;
        let mut st = dev.st.borrow_mut();
        let pl = st.planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        pl.bounds[self.bound_index()?].set_scalar(name, value).map_err(perr)
    }

    /// Set the rotation of the rotate op producing `name` (dynamic step from now on).
    fn set_rotation(&self, name: &str, k: i64) -> PyResult<()> {
        let dev = self.dev()?;
        let mut st = dev.st.borrow_mut();
        let pl = st.planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        pl.bounds[self.bound_index()?].set_rotation(name, k).map_err(perr)
    }

    /// Declare the steps producing `names` dynamic (before the first run).
    fn mark_dynamic(&self, names: Vec<String>) -> PyResult<()> {
        let dev = self.dev()?;
        let mut st = dev.st.borrow_mut();
        let pl = st.planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let b = &mut pl.bounds[self.bound_index()?];
        for n in names {
            b.mark_dynamic(&n).map_err(perr)?;
        }
        Ok(())
    }

    /// Run the bound block. `inputs[i]` (one per `Program.inputs`): a register index, a `Ct`, or None (an
    /// encryption of 0 at the compiled metadata, e.g. a KV state before its first token). `outputs`: [(output
    /// index, register)] copied out after the run. The block runs in Rust with the GIL released: dynamic steps
    /// eagerly, static segments as CUDA graphs once captured. Asynchronous on the compute lane. Returns {served,
    /// host_us, launch_us}.
    ///
    /// Pinned KV (Session(kv=...)): `kv_in` [(input position, unit)] reads a unit's state (joined on compute),
    /// `kv_out` [(output index, unit)] writes one back and enqueues its copy-out, `kv_next` units to reload now
    /// (the next block's, overlapping this one), `kv_done` units whose copy-out is final (the previous block's).
    /// Streamed weights: the block's frame is adopted (staged now if it was not), released after the block, and
    /// the next block of `Session.stream_order` staged behind it.
    #[pyo3(signature = (inputs, outputs=Vec::new(), kv_in=Vec::new(), kv_out=Vec::new(), kv_next=Vec::new(), kv_done=Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn run(&self, py: Python<'_>, inputs: Vec<Bound<'_, PyAny>>, outputs: Vec<(usize, usize)>, kv_in: Vec<(usize, usize)>, kv_out: Vec<(usize, usize)>, kv_next: Vec<usize>, kv_done: Vec<usize>) -> PyResult<Py<PyDict>> {
        let dev = self.dev()?;
        dev.check_frozen()?;
        let idx = self.bound_index()?;
        let mut cts: Vec<Option<PyRef<'_, crate::fhe::PyCt>>> = vec![];
        let mut kinds: Vec<(u8, usize)> = vec![];
        for x in &inputs {
            if x.is_none() {
                kinds.push((0, 0));
                cts.push(None);
            } else if let Ok(r) = x.extract::<usize>() {
                kinds.push((1, r));
                cts.push(None);
            } else {
                let c: PyRef<'_, crate::fhe::PyCt> = x.extract()?;
                kinds.push((2, 0));
                cts.push(Some(c));
            }
        }
        let ins: Vec<In<'_>> = kinds
            .iter()
            .zip(&cts)
            .map(|(&(k, r), c)| match (k, c) {
                (1, _) => In::Reg(r),
                (2, Some(c)) => In::Ct(c.ct_ref()),
                _ => In::Zero,
            })
            .collect();
        let kv = KvIo { inp: &kv_in, out: &kv_out, next: &kv_next, done: &kv_done };
        let info = nogil(py, || run_block(&dev, idx, &ins, &outputs, &kv))?;
        let d = PyDict::new(py);
        d.set_item("served", match info.served {
            Served::Eager => "eager",
            Served::Captured => "captured",
            Served::Replayed => "replayed",
        })?;
        d.set_item("host_us", info.host_us)?;
        d.set_item("launch_us", info.launch_us)?;
        d.set_item("meta_mismatches", info.meta_mismatches)?;
        d.set_item("first_mismatches", info.first_mismatches.clone())?;
        Ok(d.unbind())
    }

    /// A copy of register `r` as an eager `Ct` (to decrypt it).
    #[staticmethod]
    fn reg_ct(session: &crate::session::PySession, r: usize) -> PyResult<crate::fhe::PyCt> {
        let dev = session.dev.clone();
        dev.check_frozen()?;
        let meta = {
            let st = dev.st.borrow();
            let pl = st.planned.as_ref().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
            pl.regs.get(r).map(|c| c.meta).ok_or_else(|| PyValueError::new_err(format!("register {r}")))?
        };
        let mut out = crate::fhe::PyCt::acquire(&dev, meta)?;
        let st = dev.st.borrow();
        let pl = st.planned.as_ref().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        copy_ct(&dev.ctx, &st.ex.compute, out.ct_mut_ref()?, &pl.regs[r]).map_err(perr)?;
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!(
            "Program({}: {} ops, {} launches, {} bootstraps, {} inputs, {} outputs, {} pt refs, {} check errors{})",
            self.prog.block,
            self.n_ops(),
            self.prog.ops.len(),
            self.prog.bts_sites.len(),
            self.prog.inputs.len(),
            self.prog.outputs.len(),
            self.prog.pt_refs.len(),
            self.prog.check.errors.len(),
            if self.bound.is_some() { ", bound" } else { "" }
        )
    }
}

/// The pinned-KV arguments of one `Program.run`.
struct KvIo<'k> {
    inp: &'k [(usize, usize)],
    out: &'k [(usize, usize)],
    next: &'k [usize],
    done: &'k [usize],
}

/// `Program.run` without the GIL: inputs into the slab, the replay, outputs into registers.
fn run_block(dev: &Dev, idx: usize, ins: &[In<'_>], outputs: &[(usize, usize)], kvio: &KvIo<'_>) -> PyResult<pc_program::replay::RunInfo> {
    use pc_program::kv::KvStore;
    let mut st = dev.st.borrow_mut();
    let State { planned, ex, ops, plans, pts, .. } = &mut *st;
    let pl = planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
    let crate::session::Planned { bounds, shadows, replayer, slab, regs, work, dense, plain, sparse, classic, spru, stats, pcmm, ccmm, tfhe, side, coeff, ring, streams, stream_tag, tag_layouts, stream_order, kv, .. } = pl;
    // the block's weight frame: staged while the block before ran (else now), joined into compute
    let spos = match stream_tag.get(idx).copied().flatten() {
        Some(tag) => {
            let r = ring.as_mut().ok_or_else(|| PyRuntimeError::new_err("a streaming block in a session without weight frames"))?;
            let pos = stream_order.iter().position(|&t| t == tag).ok_or_else(|| PyRuntimeError::new_err(format!("{}: its weight frame is not in Session.stream_order", bounds[idx].label)))?;
            let f = pos % r.cfg.frames;
            if !r.staged(f, tag) {
                r.stage(ex, f, tag, &tag_layouts[tag], &*streams).map_err(perr)?;
                r.stats.cold += 1;
            }
            bounds[idx].stream_base = r.adopt(ex, f, tag).map_err(perr)?;
            // the next streaming block's rows now, into its frame (released after the block that read it last):
            // enqueued after this block's K/V copy-outs they would wait for this block's compute
            if !stream_order.is_empty() {
                let np = (pos + 1) % stream_order.len();
                let (ntag, nf) = (stream_order[np], np % r.cfg.frames);
                if nf != f && !r.staged(nf, ntag) {
                    r.stage(ex, nf, ntag, &tag_layouts[ntag], &*streams).map_err(perr)?;
                }
            }
            Some((pos, f))
        }
        None => None,
    };
    // pinned KV: this block's units joined, the next block's reloads enqueued (Copy, overlapping this block)
    if !(kvio.inp.is_empty() && kvio.out.is_empty() && kvio.next.is_empty() && kvio.done.is_empty()) {
        let k = kv.as_mut().ok_or_else(|| PyRuntimeError::new_err("Program.run(kv_*): the Session has no pinned KV (kv=)"))?;
        for &(_, u) in kvio.inp.iter().chain(kvio.out.iter()) {
            k.prefetch(ex, u).map_err(perr)?;
            k.join(ex, u).map_err(perr)?;
        }
        for &u in kvio.next {
            k.prefetch(ex, u).map_err(perr)?;
        }
    }
    let mut routes = pc_program::replay::Routes::new();
    if let Some(d) = dense.as_ref() {
        routes.add(d);
    }
    if let Some(d) = plain.as_ref() {
        routes.add(d);
    }
    for x in sparse.iter() {
        routes.add(x);
    }
    for x in classic.iter() {
        routes.add(x);
    }
    if let Some(x) = spru.as_ref() {
        routes.add(x);
    }
    let locks: Vec<&dyn pc_program::replay::CaptureLock> = dense.iter().flat_map(|d| d.last.iter().map(|l| l as &dyn pc_program::replay::CaptureLock)).collect();
    let ev0: u64 = locks.iter().map(|l| l.evictions()).sum();
    let b = bounds.get_mut(idx).ok_or_else(|| PyRuntimeError::new_err(format!("bound block {idx}")))?;
    if ins.len() != b.inputs.len() {
        return Err(PyValueError::new_err(format!("{}: {} inputs for {} program inputs", b.label, ins.len(), b.inputs.len())));
    }
    b.resolve_pts(pts).map_err(perr)?;
    let ctx = &dev.ctx;
    let sh = &shadows[idx];
    let info = b
        .with_slab(slab, |b| {
            for (i, x) in ins.iter().enumerate() {
                let want = b.inputs[i].2;
                let label = b.label.clone();
                let name = b.inputs[i].0.clone();
                let kv_unit = kvio.inp.iter().find(|(p, _)| *p == i).map(|&(_, u)| u);
                let dst = b.input_mut(i)?;
                let src: Option<&Ct<'static, W>> = match (kv_unit, x) {
                    (Some(u), _) => Some(&*kv.as_mut().ok_or_else(|| pc_program::Error::Replay { block: label.clone(), msg: "no pinned KV".into() })?.page(u)?),
                    (None, In::Reg(r)) => Some(regs.get(*r).ok_or_else(|| pc_program::Error::Replay { block: label.clone(), msg: format!("register {r}") })?),
                    (None, In::Ct(c)) => Some(*c),
                    (None, In::Zero) => None,
                };
                match src {
                    Some(c) => {
                        copy_ct(ctx, &ex.compute, dst, c)?;
                        if (c.meta.level, c.meta.deg) != (want.level, want.deg) {
                            return Err(pc_program::Error::Replay { block: label, msg: format!("input {i} ({name}) at level {} deg {}, compiled level {} deg {}", c.meta.level, c.meta.deg, want.level, want.deg) });
                        }
                    }
                    None => {
                        dst.buf.buf.zero(&ex.compute).map_err(pc_gpu::Error::from)?;
                        dst.meta = want;
                        if want.nf == 0.0 {
                            // the compile leaves block-input scales unknown: an encryption of 0 at the level's own
                            // scaling factor (what a rescaled K/V state carries, so the add it meets adjusts nothing)
                            let top = pc_shadow::meta::top_limb(ctx.shape, want.level).map_err(pc_gpu::Error::from)? as i32;
                            dst.meta.nf = ctx.scale.sf(top).map_err(|x| pc_program::Error::Replay { block: label.clone(), msg: x.to_string() })?.powi(want.deg as i32);
                        }
                    }
                }
            }
            let info = {
                // the PCMM / CCMM / TFHE services run the program's finishes on the StC-first route
                let mut fx = match dense.as_ref() {
                    Some(route) if pcmm.is_some() || ccmm.is_some() || tfhe.is_some() => Some(crate::matmul::FinishSvc { pcmm: pcmm.as_mut(), ccmm: ccmm.as_mut(), tfhe: tfhe.as_mut(), route }),
                    _ => None,
                };
                let finish = fx.as_mut().map(|f| f as &mut dyn pc_program::replay::FinishExec<'static, W, crate::session::M>);
                let mut rt = Rt { ctx, ex: &mut *ex, sh, ks: &*plans, ops: &mut *ops, work: &mut *work, routes: &routes, source: dev.source, locks: &locks, finish, side: side.as_mut(), coeff: coeff.as_mut() };
                replayer.run(b, &mut rt)?
            };
            for &(j, r) in outputs {
                let o = b.output(j)?;
                let label = b.label.clone();
                let dst = regs.get_mut(r).ok_or_else(|| pc_program::Error::Replay { block: label, msg: format!("register {r}") })?;
                copy_ct(ctx, &ex.compute, dst, o)?;
            }
            for &(j, u) in kvio.out {
                let o = b.output(j)?;
                let k = kv.as_mut().ok_or_else(|| pc_program::Error::Replay { block: b.label.clone(), msg: "no pinned KV".into() })?;
                copy_ct(ctx, &ex.compute, k.page(u)?, o)?;
                k.store(ex, u)?;
            }
            Ok(info)
        })
        .map_err(perr)?;
    if let Some(k) = kv.as_mut() {
        for &u in kvio.done {
            k.finalize(ex, u).map_err(perr)?;
        }
    }
    // the frame goes back to Copy behind this block (the next streaming block's rows were enqueued before it ran)
    if let (Some((_pos, f)), Some(r)) = (spos, ring.as_mut()) {
        r.release(ex, f).map_err(perr)?;
    }
    let ev1: u64 = locks.iter().map(|l| l.evictions()).sum();
    stats.runs += 1;
    stats.host_us += info.host_us;
    stats.launch_us += info.launch_us;
    if ev1 > ev0 {
        stats.cache_evictions_seen += ev1 - ev0;
        if replayer.stats.captures > 0 {
            return Err(PyRuntimeError::new_err(format!(
                "StC-first stage cache evicted {} entr(ies) while CUDA graphs that read it exist: raise Session(stc_cache=...)",
                ev1 - ev0
            )));
        }
    }
    let _ = gerr;
    Ok(info)
}

fn json_diff(a: &serde_json::Value, b: &serde_json::Value) -> Vec<String> {
    let mut out = vec![];
    fn walk(p: &str, a: &serde_json::Value, b: &serde_json::Value, out: &mut Vec<String>) {
        match (a, b) {
            (serde_json::Value::Object(x), serde_json::Value::Object(y)) => {
                let keys: std::collections::BTreeSet<&String> = x.keys().chain(y.keys()).collect();
                for k in keys {
                    let n = serde_json::Value::Null;
                    walk(&format!("{p}{}{k}", if p.is_empty() { "" } else { "." }), x.get(k).unwrap_or(&n), y.get(k).unwrap_or(&n), out);
                }
            }
            // the stamp's set fields bind (the plan contract); a field the plan left unset binds nothing
            _ if !b.is_null() && a != b => out.push(format!("{p}: {a} vs {b}")),
            _ => {}
        }
    }
    walk("", a, b, &mut out);
    out
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProgram>()?;
    Ok(())
}
