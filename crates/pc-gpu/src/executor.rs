//! The executor: the three lanes, the event pool and every workspace, owned in one object and borrowed `&mut` by
//! the ops that need them (no singletons, no per-poly streams; concurrency inside a block = K ciphertexts per launch).
use crate::context::DevWord;
use crate::error::Result;
use crate::mem::MemPlan;
use crate::poly::CtBuf;
use pc_cuda::{lane, Arena, Compute, DBuf, Device, Event, EventPool, PoolEvent, Side, Stream};

pub struct Executor<'a, W: DevWord> {
    /// Every ciphertext kernel.
    pub compute: Stream<Compute>,
    /// H2D / D2H; results are handed to `compute` as `Ready` tokens.
    pub copy: Stream<lane::Copy>,
    /// Compiler-marked independent sub-programs only.
    pub side: Stream<Side>,
    /// Work forked off the compute lane inside one op (a key switch's Q-row inner products): `aux_fork` recorded on
    /// compute and waited by aux; `aux_join` recorded on aux and waited by compute before the first reader;
    /// `aux_pending`: aux holds work compute has not joined.
    pub aux: Stream<Compute>,
    pub aux_fork: Event,
    pub aux_join: Event,
    pub aux_pending: bool,
    pub events: EventPool,
    /// Key-switching workspace (`MemPlan::ks_work`).
    pub ks: crate::ks::KsWork<'a, W>,
    /// Base-conversion input workspace (`MemPlan::bconv_work`).
    pub bconv_work: DBuf<'a, Compute, W>,
    /// `scratch_cts * k_max` ciphertexts of the context layout.
    pub scratch: CtBuf<'a, Compute, W>,
}

/// Rotations split their inner products by rows: the special rows on compute (the ModDown converts them first), the
/// Q rows on `aux` (read only by the ModDown's last NTT / its consumer). Bit-identical (row-disjoint launches of one
/// kernel). PC_OVL_ROT=0 turns it off (read once).
pub fn split_rotations() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_OVL_ROT").map_or(true, |v| v != "0"))
}

impl<'a, W: DevWord> Executor<'a, W> {
    pub fn new(dev: &Device, arena: &'a Arena, plan: &MemPlan) -> Result<Self> {
        let wb = std::mem::size_of::<W>();
        Ok(Executor {
            compute: Stream::new(dev)?,
            copy: Stream::new(dev)?,
            side: Stream::new(dev)?,
            aux: Stream::new(dev)?,
            aux_fork: Event::new(dev)?,
            aux_join: Event::new(dev)?,
            aux_pending: false,
            events: EventPool::new(dev, plan.cfg.events)?,
            ks: crate::ks::KsWork::new(arena, plan.lay, plan.cfg.k_max, plan.ks_dnum)?,
            bconv_work: arena.alloc("bconv_work", plan.bconv_work / wb)?,
            scratch: CtBuf::new(arena, "scratch", plan.lay, plan.cfg.scratch_cts * plan.cfg.k_max)?,
        })
    }

    /// An event recorded at the current tail of the compute lane (hand it to another lane's `wait`, or `sync` it).
    pub fn fence(&self) -> Result<PoolEvent<'_>> {
        let e = self.events.get()?;
        e.record(&self.compute)?;
        Ok(e)
    }

    /// Fork aux from compute's current tail.
    pub fn fork_aux(&mut self) -> Result<()> {
        self.aux_fork.record(&self.compute)?;
        self.aux.wait(&self.aux_fork)?;
        Ok(())
    }
    /// Mark the end of aux's forked work (compute joins it: `join_aux`, or a consumer handed `aux_join`).
    pub fn end_aux(&mut self) -> Result<()> {
        self.aux_join.record(&self.aux)?;
        self.aux_pending = true;
        Ok(())
    }
    /// Make compute wait for aux's forked work, if any.
    pub fn join_aux(&mut self) -> Result<()> {
        if self.aux_pending {
            self.compute.wait(&self.aux_join)?;
            self.aux_pending = false;
        }
        Ok(())
    }

    /// Host wait for everything issued on the compute lane (host reads of compute buffers).
    pub fn sync_compute(&self) -> Result<()> {
        Ok(self.compute.sync()?)
    }
}
