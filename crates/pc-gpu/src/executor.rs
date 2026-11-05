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
    /// Seeded operands' c1 rows of one `tensor_multi` launch (`carve_cts_expand`; MemPlan section "cts_expand"):
    /// `mult_sum_many` draws them there first (`cts_expand`) instead of inside the tensor. None: in the tensor.
    pub cts_expand: Option<(DBuf<'a, Compute, W>, usize)>,
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
            cts_expand: None,
        })
    }

    /// Bytes of the `cts_expand` workspace for seeded operands of up to `rows` rows.
    pub fn cts_expand_bytes(lay: &Layout, rows: usize) -> usize {
        Arena::bytes_for::<W>(pc_kernels::CTS_EXPAND_MAX * rows * lay.n)
    }
    /// Carve the `cts_expand` workspace (setup only).
    pub fn carve_cts_expand(&mut self, arena: &'a Arena, lay: &Layout, rows: usize) -> Result<()> {
        self.cts_expand = Some((arena.alloc("cts_expand", pc_kernels::CTS_EXPAND_MAX * rows * lay.n)?, rows));
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
