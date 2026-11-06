//! The executor: the three lanes, the event pool and every workspace, owned in one object and borrowed `&mut` by
//! the ops that need them (no singletons, no per-poly streams; concurrency inside a block = K ciphertexts per launch).
use crate::context::DevWord;
use crate::error::Result;
use crate::mem::MemPlan;
use crate::poly::{CtBuf, Layout};
use pc_cuda::{lane, Arena, Compute, DBuf, Device, EventPool, PoolEvent, Side, Stream};

pub struct Executor<'a, W: DevWord> {
    /// Every ciphertext kernel.
    pub compute: Stream<Compute>,
    /// H2D / D2H; results are handed to `compute` as `Ready` tokens.
    pub copy: Stream<lane::Copy>,
    /// Compiler-marked independent sub-programs only.
    pub side: Stream<Side>,
    pub events: EventPool,
    /// Key-switching workspace (`MemPlan::ks_work`).
    pub ks: crate::ks::KsWork<'a, W>,
    /// Base-conversion input workspace (`MemPlan::bconv_work`).
    pub bconv_work: DBuf<'a, Compute, W>,
    /// `scratch_cts * k_max` ciphertexts of the context layout.
    pub scratch: CtBuf<'a, Compute, W>,
    /// The d2 rows of one wave of `mult_sum_many` outputs over seeded operands (`carve_seeded_d2`; MemPlan section
    /// "seeded_d2"): `TSEED_OUT` outputs of `rows` rows. None: scratch ciphertexts from the caller.
    pub seeded_d2: Option<(DBuf<'a, Compute, W>, usize)>,
}

impl<'a, W: DevWord> Executor<'a, W> {
    pub fn new(dev: &Device, arena: &'a Arena, plan: &MemPlan) -> Result<Self> {
        let wb = std::mem::size_of::<W>();
        Ok(Executor {
            compute: Stream::new(dev)?,
            copy: Stream::new(dev)?,
            side: Stream::new(dev)?,
            events: EventPool::new(dev, plan.cfg.events)?,
            ks: crate::ks::KsWork::new(arena, plan.lay, plan.cfg.k_max, plan.ks_dnum)?,
            bconv_work: arena.alloc("bconv_work", plan.bconv_work / wb)?,
            scratch: CtBuf::new(arena, "scratch", plan.lay, plan.cfg.scratch_cts * plan.cfg.k_max)?,
            seeded_d2: None,
        })
    }

    /// Bytes of the `seeded_d2` workspace for seeded operands of up to `rows` rows.
    pub fn seeded_d2_bytes(lay: &Layout, rows: usize) -> usize {
        Arena::bytes_for::<W>(pc_kernels::TSEED_OUT * rows * lay.n)
    }
    /// Carve the `seeded_d2` workspace (setup only).
    pub fn carve_seeded_d2(&mut self, arena: &'a Arena, lay: &Layout, rows: usize) -> Result<()> {
        self.seeded_d2 = Some((arena.alloc("seeded_d2", pc_kernels::TSEED_OUT * rows * lay.n)?, rows));
        Ok(())
    }

    /// An event recorded at the current tail of the compute lane (hand it to another lane's `wait`, or `sync` it).
    pub fn fence(&self) -> Result<PoolEvent<'_>> {
        let e = self.events.get()?;
        e.record(&self.compute)?;
        Ok(e)
    }

    /// Host wait for everything issued on the compute lane (host reads of compute buffers).
    pub fn sync_compute(&self) -> Result<()> {
        Ok(self.compute.sync()?)
    }
}
