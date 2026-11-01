//! The staged bootstrap as a call object (`BtsCall`): what PCMM / CCMM work is interleaved into (the stage API of
//! FIDESlib's `BootstrapStages.cuh` as perseus `impl/bootstrap.py` drives it). One call = one `BtsState` + the phase
//! it is at; every transition is checked against the stage order
//!   begin -> [stc_first_input -> Split] -> mod_raise -> fold -> cts_stage(0..n_cts) -> eval_mod ->
//!       ( stc_first_output | stc_enter -> stc_stage(0..n_stc) -> finish ) -> Done
//! on K-batched ciphertexts (`&mut [Ct]`, all with the call's metadata). `run_interleaved` issues several calls
//! stage-major; `BtsRecorder` is the capture hook (the PCMM records each output as a deliberate bootstrap of its
//! inputs). External-op pattern: docs/ARCHITECTURE.md "pc-gpu staged bootstrap".
use super::stc_first::{SplitInfo, StcFirstRoute};
use super::{BtsCx, BtsState, Staged};
use crate::context::{DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use pc_shadow::{CtMeta, Route};

/// Where a call is: the last stage done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Begun,
    /// After `stc_first_input` (or an external op that left its outputs at the split).
    Split,
    Raised,
    Folded,
    /// `n` CtS stages done.
    Cts(usize),
    EvalMod,
    Entered,
    /// `n` StC stages done.
    Stc(usize),
    Done,
}

/// Per-call options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallOpts {
    /// ModRaise's prescaled path (a planned prescale already applied the raise adjust).
    pub prescaled: bool,
    /// StC-first: the mode-2 last stage at the call's own CF (the PCMM's refresh, perseus PF_PCMM_CF).
    pub own_cf: bool,
}

/// The route a call runs on: any `Staged` route, or the StC-first route (whose input stage takes `own_cf`, whose
/// finishes can switch raise variant).
pub enum StagedRef<'r, 'a, W: DevWord, M: DevStrategy<W>> {
    Generic(&'r dyn Staged<'a, W, M>),
    StcFirst(&'r StcFirstRoute<'a, W, M>),
}

impl<'r, 'a, W: DevWord, M: DevStrategy<W>> Clone for StagedRef<'r, 'a, W, M> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'r, 'a, W: DevWord, M: DevStrategy<W>> Copy for StagedRef<'r, 'a, W, M> {}

impl<'r, 'a, W: DevWord, M: DevStrategy<W>> StagedRef<'r, 'a, W, M> {
    fn staged(&self) -> &'r dyn Staged<'a, W, M> {
        match *self {
            StagedRef::Generic(s) => s,
            StagedRef::StcFirst(r) => r,
        }
    }
}

impl<'r, 'a, W: DevWord, M: DevStrategy<W>> From<&'r StcFirstRoute<'a, W, M>> for StagedRef<'r, 'a, W, M> {
    fn from(r: &'r StcFirstRoute<'a, W, M>) -> Self {
        StagedRef::StcFirst(r)
    }
}

/// One staged bootstrap call.
pub struct BtsCall<'r, 'a, W: DevWord, M: DevStrategy<W>> {
    route: StagedRef<'r, 'a, W, M>,
    pub st: BtsState,
    phase: Phase,
    pub opts: CallOpts,
    /// What `stc_first_input` did (StC-first route).
    pub split: Option<SplitInfo>,
}

fn order(what: &str, at: Phase) -> crate::error::Error {
    shape("bts call", format!("{what} out of order (the call is at {at:?})"))
}

impl<'r, 'a, W: DevWord, M: DevStrategy<W>> BtsCall<'r, 'a, W, M> {
    /// `btsBegin` on the ciphertexts' metadata.
    pub fn begin(route: StagedRef<'r, 'a, W, M>, cx: &BtsCx<'_, 'a, W, M>, cts: &[Ct<'a, W>], r: &Route, opts: CallOpts) -> Result<Self> {
        let st = route.staged().begin(cx, cts, r, opts.prescaled)?;
        Ok(BtsCall { route, st, phase: Phase::Begun, opts, split: None })
    }

    /// Resume a call at `phase` with a state (oracle replays; the ciphertexts are in the state after `phase`).
    pub fn resume(route: StagedRef<'r, 'a, W, M>, st: BtsState, phase: Phase, opts: CallOpts) -> Self {
        BtsCall { route, st, phase, opts, split: None }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn stc_first(&self) -> bool {
        self.st.stc_first
    }

    /// The input-side StC (StC-first calls); a shipped-order call stays at `Begun`.
    pub fn stc_first_input(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<Option<SplitInfo>> {
        if self.phase != Phase::Begun {
            return Err(order("stc_first_input", self.phase));
        }
        if !self.st.stc_first {
            return Ok(None);
        }
        let info = match self.route {
            StagedRef::StcFirst(r) => r.stc_first_input_cf(cx, &self.st, cts, self.opts.own_cf)?,
            StagedRef::Generic(s) => {
                if self.opts.own_cf {
                    return Err(shape("bts call", "own_cf needs the StC-first route (StagedRef::StcFirst)"));
                }
                s.stc_first_input(cx, &self.st, cts)?;
                None
            }
        };
        self.split = info;
        self.phase = Phase::Split;
        Ok(info)
    }

    /// The ciphertexts were replaced at the split by an external op (PCMM / CCMM outputs): check they sit where
    /// `stc_first_input` leaves them, then continue from `Split`.
    pub fn at_split(&mut self, cx: &BtsCx<'_, 'a, W, M>, cts: &[Ct<'a, W>]) -> Result<()> {
        if !self.st.stc_first || !matches!(self.phase, Phase::Begun | Phase::Split) {
            return Err(order("at_split", self.phase));
        }
        if let StagedRef::StcFirst(r) = self.route {
            let shp = cx.ctx.shape;
            let top = if self.st.stc_folded { r.split_top() } else { r.split_top() + r.step };
            let want = (pc_shadow::meta::level_from_top_limb(shp, top as u32)?, 2u8);
            if let Some(x) = cts.iter().find(|x| (x.meta.level, x.meta.deg) != want) {
                return Err(shape("bts call", format!("at_split: ciphertext at level {} deg {}, the split is {want:?} (entry limb {})", x.meta.level, x.meta.deg, r.entry_top)));
            }
        }
        self.phase = Phase::Split;
        Ok(())
    }

    /// The same input's call under another raise variant, at the split (the PCMM finish: begun on the input,
    /// finished at the site's raise drop).
    pub fn finish_call(&self, raise_drop: u32) -> Result<BtsCall<'r, 'a, W, M>> {
        let StagedRef::StcFirst(r) = self.route else {
            return Err(shape("bts call", "finish_call needs the StC-first route"));
        };
        if !self.st.stc_first {
            return Err(shape("bts call", "finish_call on a shipped-order call"));
        }
        Ok(BtsCall { route: self.route, st: r.state_for_drop(&self.st, raise_drop)?, phase: Phase::Split, opts: self.opts, split: self.split })
    }

    pub fn mod_raise(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let ok = if self.st.stc_first { self.phase == Phase::Split } else { self.phase == Phase::Begun };
        if !ok {
            return Err(order("mod_raise", self.phase));
        }
        self.route.staged().mod_raise(cx, &mut self.st, cts)?;
        self.phase = Phase::Raised;
        Ok(())
    }

    pub fn fold(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<()> {
        if self.phase != Phase::Raised {
            return Err(order("fold", self.phase));
        }
        self.route.staged().fold(cx, &self.st, cts)?;
        self.phase = Phase::Folded;
        Ok(())
    }

    pub fn cts_stage(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], k: usize) -> Result<()> {
        let done = match self.phase {
            Phase::Folded => 0,
            Phase::Cts(n) => n,
            p => return Err(order("cts_stage", p)),
        };
        if k != done || k >= self.st.n_cts {
            return Err(order(&format!("cts_stage({k})"), self.phase));
        }
        self.route.staged().cts_stage(cx, &self.st, cts, k)?;
        self.phase = Phase::Cts(k + 1);
        Ok(())
    }

    pub fn eval_mod(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<()> {
        if self.phase != Phase::Cts(self.st.n_cts) && !(self.st.n_cts == 0 && self.phase == Phase::Folded) {
            return Err(order("eval_mod", self.phase));
        }
        self.route.staged().eval_mod(cx, &self.st, cts)?;
        self.phase = Phase::EvalMod;
        Ok(())
    }

    pub fn stc_first_output(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<()> {
        if self.phase != Phase::EvalMod || !self.st.stc_first {
            return Err(order("stc_first_output", self.phase));
        }
        self.route.staged().stc_first_output(cx, &self.st, cts)?;
        self.phase = Phase::Done;
        Ok(())
    }

    pub fn stc_enter(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<()> {
        if self.phase != Phase::EvalMod || self.st.stc_first {
            return Err(order("stc_enter", self.phase));
        }
        self.route.staged().stc_enter(cx, &self.st, cts)?;
        self.phase = Phase::Entered;
        Ok(())
    }

    pub fn stc_stage(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], k: usize) -> Result<()> {
        let done = match self.phase {
            Phase::Entered => 0,
            Phase::Stc(n) => n,
            p => return Err(order("stc_stage", p)),
        };
        if k != done || k >= self.st.n_stc {
            return Err(order(&format!("stc_stage({k})"), self.phase));
        }
        self.route.staged().stc_stage(cx, &self.st, cts, k)?;
        self.phase = Phase::Stc(k + 1);
        Ok(())
    }

    pub fn finish(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let ok = self.phase == Phase::Stc(self.st.n_stc) || (self.st.n_stc == 0 && self.phase == Phase::Entered);
        if !ok || self.st.stc_first {
            return Err(order("finish", self.phase));
        }
        self.route.staged().finish(cx, &self.st, cts)?;
        self.phase = Phase::Done;
        Ok(())
    }

    /// The next stage in order; returns the phase reached.
    pub fn step(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<Phase> {
        match self.phase {
            Phase::Begun if self.st.stc_first => {
                self.stc_first_input(cx, cts)?;
            }
            Phase::Begun | Phase::Split => self.mod_raise(cx, cts)?,
            Phase::Raised => self.fold(cx, cts)?,
            Phase::Folded | Phase::Cts(_) => {
                let done = if let Phase::Cts(n) = self.phase { n } else { 0 };
                if done < self.st.n_cts {
                    self.cts_stage(cx, cts, done)?
                } else {
                    self.eval_mod(cx, cts)?
                }
            }
            Phase::EvalMod if self.st.stc_first => self.stc_first_output(cx, cts)?,
            Phase::EvalMod => self.stc_enter(cx, cts)?,
            Phase::Entered | Phase::Stc(_) => {
                let done = if let Phase::Stc(n) = self.phase { n } else { 0 };
                if done < self.st.n_stc {
                    self.stc_stage(cx, cts, done)?
                } else {
                    self.finish(cx, cts)?
                }
            }
            Phase::Done => return Err(order("step", Phase::Done)),
        }
        Ok(self.phase)
    }

    /// Run stages until `until` (inclusive) is reached.
    pub fn run_to(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], until: Phase) -> Result<()> {
        while self.phase != until {
            if self.phase == Phase::Done {
                return Err(shape("bts call", format!("phase {until:?} not on this call's path")));
            }
            self.step(cx, cts)?;
        }
        Ok(())
    }

    /// The capture record of this call's output (perseus `record_external_bootstrap`).
    pub fn external(&self, inputs: &[(&str, u32)], out: &CtMeta, desc: &str, landing: Option<(u32, u8)>) -> ExternalBts {
        ExternalBts {
            inputs: inputs.iter().map(|(n, _)| n.to_string()).collect(),
            input_levels: inputs.iter().map(|(_, l)| *l).collect(),
            out: out.clone(),
            landing,
            route: self.st.route,
            desc: desc.to_string(),
        }
    }
}

/// Run several calls stage-major (each on its own ciphertexts) until every one reaches `until` (perseus
/// `_finish_interleaved_impl`: the finishes of a PCMM's outputs issued side by side).
pub fn run_interleaved<'r, 'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, calls: &mut [BtsCall<'r, 'a, W, M>], cts: &mut [&mut [Ct<'a, W>]], until: Phase) -> Result<()> {
    if calls.len() != cts.len() {
        return Err(shape("run_interleaved", format!("{} calls, {} ciphertext sets", calls.len(), cts.len())));
    }
    loop {
        let mut moved = false;
        for (c, x) in calls.iter_mut().zip(cts.iter_mut()) {
            if c.phase != until {
                if c.phase == Phase::Done {
                    return Err(shape("run_interleaved", format!("phase {until:?} not on a call's path")));
                }
                c.step(cx, x)?;
                moved = true;
            }
        }
        if !moved {
            return Ok(());
        }
    }
}

// ── program capture hooks ────────────────────────────────────────────────────────────────────────────────────────

/// A refresh made outside the recorded ops (the PCMM / CCMM finish), as the capture records it: a deliberate
/// bootstrap of `inputs` whose output has metadata `out` (or the given landing), tag reset.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalBts {
    pub inputs: Vec<String>,
    pub input_levels: Vec<u32>,
    pub out: CtMeta,
    /// A landing the caller declares instead of `out` (perseus `record_external_bootstrap(.., landing=(l, 1))`, CCMM).
    pub landing: Option<(u32, u8)>,
    pub route: Route,
    /// Site descriptor (pcmm_calib site / desc of `Op::Finish`).
    pub desc: String,
}

/// Capture hooks of the staged API (perseus wrapper `level_hint` + `record_external_bootstrap`; the program
/// recorder implements them, pc-gpu only calls them).
pub trait BtsRecorder {
    /// The planner must not let `var` sit deeper than `cap` (primes dropped): StC-first inputs.
    fn level_hint(&mut self, var: &str, level: u32, cap: u32);
    fn external_bootstrap(&mut self, rec: &ExternalBts);
}

/// A recorder that keeps everything (tests, eager runs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BtsLog {
    pub hints: Vec<(String, u32, u32)>,
    pub refreshes: Vec<ExternalBts>,
    pub total_bootstraps: u64,
}

impl BtsRecorder for BtsLog {
    fn level_hint(&mut self, var: &str, level: u32, cap: u32) {
        self.hints.push((var.to_string(), level, cap));
    }
    fn external_bootstrap(&mut self, rec: &ExternalBts) {
        self.total_bootstraps += 1;
        self.refreshes.push(rec.clone());
    }
}

/// The PCMM's input hooks: cap every input at the StC-first entry (after its pending rescale).
pub fn record_stc_first_inputs<W: DevWord, M: DevStrategy<W>>(rec: &mut dyn BtsRecorder, route: &StcFirstRoute<'_, W, M>, cx: &BtsCx<'_, '_, W, M>, inputs: &[(&str, u32)]) -> Result<()> {
    let cap = route.input_cap(cx.ctx.shape)?;
    for (v, l) in inputs {
        rec.level_hint(v, *l, cap);
    }
    Ok(())
}
