//! Key switching: seams `KeySwitch` (how a polynomial is decomposed and switched) and `KskSource` (where the key's
//! uniform half `a` comes from).
//!
//! `KeySwitch::Hybrid { dnum }` is OpenFHE/FIDESlib's hybrid key switching, bit-exact with FIDESlib's single-GPU path
//! (`modupMGPU` -> `dotKSKfusedMGPU` / `hoistedRotateDotKSK` -> `moddownMGPU`): ModUp (one INTT, one conversion
//! launch over every live digit, one NTT), the key inner product (`ks_dot` / `ks_dot_regen`), ModDown P -> Q.
//! Plans are built per level at setup for `k_max` ciphertexts (a K < k_max launch runs a prefix); the workspaces
//! (`KsWork`) belong to the `Executor`. Keys are seen through `KsKeyRef` (dense `b` rows `[digit][prime id][N]`,
//! optional expanded `a`, optional 256-bit KSKA seed); `DevKey` serves the oracle import and the benches.
//! Step details: docs/ARCHITECTURE.md "pc-gpu key switching".
use crate::bconv::{BconvPlan, JobSpec};
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Error, Result};
use crate::ntt::{self, Dir, Parts, RowList, Span};
use crate::poly::{CtBuf, Layout};
use pc_core::params::Chain;
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Lane, Stream};
use pc_kernels::{EpiArgs, EwArgs, KsArgs, KsKeyArgs, KsRegenArgs, KskExpandArgs, MdEpiArgs, PolyRef, RowSet};
use std::ffi::c_void;

/// Ciphertexts per `ks_dot` launch (kernel bound).
pub const DOT_KMAX: usize = 8;
/// Ciphertexts per `ks_dot_regen` launch (kernel bound).
pub const REGEN_KMAX: usize = 4;
/// Ciphertexts per `ks_dot_key` launch (kernel bound).
pub const KEY_KMAX: usize = 4;
/// Extended rotations of one ciphertext batch `KsWork::ext` holds at once, for batches of up to `KEY_KMAX`: a radix-3
/// rotate-and-sum group (7 keys) dotted in one go and summed inside the ModDown (`rotate::SumForm::InModDown`).
pub const SUM_KEYS: usize = 7;

/// Seam `KeySwitch`: the decomposition / switching method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeySwitch {
    /// Hybrid key switching with `dnum` digits (OpenFHE numLargeDigits), special primes P.
    Hybrid { dnum: usize },
}

impl KeySwitch {
    pub fn of(chain: &Chain) -> Self {
        KeySwitch::Hybrid { dnum: chain.dnum() }
    }
}

/// Seam `KskSource`: where the inner product reads the key's `a` from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KskSource {
    /// `a` regenerated from the key's 256-bit seed (KSKA ChaCha12 expander) inside the dot kernel: relinearization
    /// through `ks_dot_regen`, every other key switch through `ks_dot_key` (which also reads packed / banded `b` in
    /// place). `KsWork::regen_a` staging remains only for unseeded keys.
    Regen,
    /// `a` expanded once and resident (hot keys; `cfg.mem.expanded_key_cache_mb`).
    Cached,
}

/// What a key switch needs from one device key (`KeySet` entries provide it).
pub trait KsKeyRef {
    /// `b` rows, `[digit][prime id][N]` words.
    fn b_ptr(&self) -> *const c_void;
    /// Expanded `a` rows (same layout), when resident.
    fn a_ptr(&self) -> Option<*const c_void>;
    /// Device copy of the 8-word KSKA seed of `a`, when the key has one.
    fn seed_ptr(&self) -> Option<*const u32>;
    /// OpenFHE automorphism index g of the key (1 = relinearization, 2N - 1 = conjugation, 5^r = rotation by r).
    fn galois(&self) -> u64;
    /// Words between digits of `b` / `a`.
    fn digit_stride(&self) -> usize;
    fn dnum(&self) -> usize;
    /// Highest Q limb id the key holds rows for (band-limited rotation keys sit below the full chain).
    fn q_band(&self) -> usize;
    /// `b` is not dense rows indexed by prime id (28-bit packed rows, or a banded key's slots): the dot reads it after
    /// `stage_key` unpacks it into `KsWork::stage_b`.
    fn staging(&self) -> Option<Staging> {
        None
    }
    /// Resident `a` of the digits from `first` on only (`bts::cts::KeyTails`), as (rows of digit `first`, `first`):
    /// `[digit - first][prime id][N]`, every prime id; the regen kernels read those digits instead of regenerating them.
    fn a_tail(&self) -> Option<(*const c_void, usize)> {
        None
    }
}

/// A key whose `b` must be staged before a dot: storage base and slot geometry (`KskShape` / `KskStore`).
#[derive(Clone, Copy, Debug)]
pub struct Staging {
    pub b: *const c_void,
    /// Packed rows: (bits, u32 words per row); None = dense W rows of N.
    pub packed: Option<(u32, usize)>,
    /// Q slots 0..=q_top, then the kp special slots.
    pub q_top: usize,
    pub ndig: usize,
    /// Slots per digit (q_top + 1 + kp).
    pub srows: usize,
}

impl<W: DevWord> KsKeyRef for crate::keys::DevKsk<'_, W> {
    fn b_ptr(&self) -> *const c_void {
        self.b_ptr()
    }
    fn a_ptr(&self) -> Option<*const c_void> {
        if self.shape.is_full() {
            self.a_ptr()
        } else {
            None
        }
    }
    fn seed_ptr(&self) -> Option<*const u32> {
        Some(self.seed_dev.dptr())
    }
    fn galois(&self) -> u64 {
        self.role.galois(pc_kernels::N).unwrap_or(1)
    }
    fn digit_stride(&self) -> usize {
        // the dot reads dense full-layout rows: the key's own stride when it is that, else the staging buffer's
        (self.shape.nq + self.shape.kp) * pc_kernels::N
    }
    fn dnum(&self) -> usize {
        self.shape.ndig
    }
    fn q_band(&self) -> usize {
        self.shape.q_top
    }
    fn staging(&self) -> Option<Staging> {
        let packed = match &self.store {
            crate::keys::KskStore::PackedSeeded { bits, row_words, .. } => Some((*bits, *row_words)),
            crate::keys::KskStore::Dense { .. } => None,
        };
        if packed.is_none() && self.shape.is_full() {
            return None;
        }
        Some(Staging { b: self.b_ptr(), packed, q_top: self.shape.q_top, ndig: self.shape.ndig, srows: self.shape.rows() })
    }
}

/// How a key's `b` is stored, as the in-place kernels read it (`KeyTab`): packed width (0 = dense W rows), words per
/// row, highest Q prime id held (None = full layout, slot = prime id) and slots per digit (`KskShape`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyForm {
    pub bits: u32,
    pub row_words: usize,
    pub q_top: Option<usize>,
    pub srows: usize,
}

impl KeyForm {
    /// The stored form of `k` (unstaged keys: dense prime-id rows of N words).
    pub fn of(k: &dyn KsKeyRef, n: usize) -> KeyForm {
        match k.staging() {
            None => KeyForm { bits: 0, row_words: n, q_top: None, srows: k.digit_stride() / n },
            Some(st) => {
                let (bits, row_words) = st.packed.unwrap_or((0, n));
                KeyForm { bits, row_words, q_top: Some(st.q_top), srows: st.srows }
            }
        }
    }
}

/// Device tables of an ordered key set read in place by `ks_dot_key` / `cts_fused_regen` (`a` from the seed, `b` from
/// its storage): one `KeyTab` per run of keys sharing a form.
pub struct KeyTabs<'a> {
    kb: ConstBuf<'a, u64>,
    seed: ConstBuf<'a, u64>,
    forms: Vec<KeyForm>,
}

impl<'a> KeyTabs<'a> {
    /// None when some key has no seed.
    pub fn new<S: Lane>(arena: &'a Arena, s: &Stream<S>, n: usize, keys: &[&dyn KsKeyRef]) -> Result<Option<Self>> {
        let Some(seeds) = keys.iter().map(|k| k.seed_ptr().map(|p| p as u64)).collect::<Option<Vec<u64>>>() else { return Ok(None) };
        if keys.is_empty() {
            return Err(shape("KeyTabs", "no keys"));
        }
        let kb: Vec<u64> = keys.iter().map(|k| k.staging().map(|st| st.b).unwrap_or(k.b_ptr()) as u64).collect();
        Ok(Some(KeyTabs {
            kb: ConstBuf::new(arena, "ks.keygroup", s, &kb)?,
            seed: ConstBuf::new(arena, "ks.keygroup", s, &seeds)?,
            forms: keys.iter().map(|k| KeyForm::of(*k, n)).collect(),
        }))
    }
    /// Arena bytes of the tables of `r` keys.
    pub fn bytes(r: usize) -> usize {
        2 * Arena::bytes_for::<u64>(r)
    }
    pub fn form(&self, r: usize) -> Option<KeyForm> {
        self.forms.get(r).copied()
    }
    /// Keys `[r0, ..)` sharing key r0's form, at most `r` of them.
    pub fn run_len(&self, r0: usize, r: usize) -> usize {
        let f = self.forms[r0];
        self.forms[r0..(r0 + r).min(self.forms.len())].iter().take_while(|&&x| x == f).count()
    }
    /// The kernel table of keys from `r0` on (they must share key r0's form; `run_len`).
    pub fn tab(&self, r0: usize) -> Result<pc_kernels::KeyTab> {
        let f = self.form(r0).ok_or_else(|| shape("KeyTabs", format!("key {r0} of {}", self.forms.len())))?;
        Ok(pc_kernels::KeyTab {
            kb: self.kb.dptr_at(r0)? as *const *const c_void,
            seed: self.seed.dptr_at(r0)? as *const *const u32,
            row_words: f.row_words as u64,
            bits: f.bits as i32,
            q_top: f.q_top.map(|q| q as i32).unwrap_or(-1),
            srows: f.srows as i32,
            nregen: i32::MAX,
        })
    }
    /// The smallest Q band of the set (usize::MAX: every key full).
    pub fn q_top(&self) -> usize {
        self.forms.iter().map(|f| f.q_top.unwrap_or(usize::MAX)).min().unwrap_or(0)
    }
}

/// Where a key's `a` comes from at import.
pub enum KeyA<'d, W> {
    /// No dense `a`: Regen only (needs a seed).
    None,
    /// Dense `a` rows from the caller (`[digit][prime id][N]`).
    Dense(&'d [W]),
    /// Expand the seed on the device now (`ksk_expand`) and keep it resident (KskSource::Cached).
    Expand,
}

/// A device key-switching key in the kernels' layout (oracle-imported keys).
pub struct DevKey<'a, W: DevWord> {
    pub b: ConstBuf<'a, W>,
    pub a: Option<DBuf<'a, Compute, W>>,
    pub seed: Option<ConstBuf<'a, u32>>,
    pub galois: u64,
    pub dnum: usize,
    pub rows: usize,
    pub n: usize,
}

impl<'a, W: DevWord> DevKey<'a, W> {
    /// Arena bytes of one key (MemPlan sections).
    pub fn bytes(lay: &Layout, dnum: usize, with_a: bool, with_seed: bool) -> usize {
        let w = dnum * lay.poly_words();
        Arena::bytes_for::<W>(w) * (1 + with_a as usize) + if with_seed { Arena::bytes_for::<u32>(8) } else { 0 }
    }

    /// Upload `b` (`dnum * rows * N` words, prime-id rows) and the seed; `a` per `KeyA`.
    #[allow(clippy::too_many_arguments)]
    pub fn new<M: DevStrategy<W>>(
        arena: &'a Arena,
        s: &Stream<Compute>,
        ctx: &Context<'_, W, M>,
        plans: &KsPlans<'_>,
        galois: u64,
        dnum: usize,
        b: &[W],
        seed: Option<[u32; 8]>,
        a: KeyA<'_, W>,
    ) -> Result<Self> {
        let lay = ctx.lay;
        let words = dnum * lay.poly_words();
        if b.len() != words {
            return Err(shape("DevKey b", format!("{} words, want dnum {dnum} x {} rows x N", b.len(), lay.rows())));
        }
        let bb = ConstBuf::new(arena, "keys.b", s, b)?;
        let seedb = match seed {
            Some(sd) => Some(ConstBuf::new(arena, "keys.seed", s, &sd)?),
            None => None,
        };
        let a = match a {
            KeyA::None => None,
            KeyA::Dense(v) => {
                if v.len() != words {
                    return Err(shape("DevKey a", format!("{} words, want {words}", v.len())));
                }
                let mut d = arena.alloc::<Compute, W>("keys.a", words)?;
                d.upload(s, v)?;
                Some(d)
            }
            KeyA::Expand => {
                let sd = seedb.as_ref().ok_or_else(|| shape("DevKey a", "KeyA::Expand needs a seed"))?;
                let mut d = arena.alloc::<Compute, W>("keys.a", words)?;
                expand_a(ctx, s, plans, sd.dptr(), dnum, &mut d)?;
                Some(d)
            }
        };
        Ok(DevKey { b: bb, a, seed: seedb, galois, dnum, rows: lay.rows(), n: lay.n })
    }
}

impl<W: DevWord> KsKeyRef for DevKey<'_, W> {
    fn b_ptr(&self) -> *const c_void {
        self.b.vptr()
    }
    fn a_ptr(&self) -> Option<*const c_void> {
        self.a.as_ref().map(|a| a.vptr())
    }
    fn seed_ptr(&self) -> Option<*const u32> {
        self.seed.as_ref().map(|s| s.dptr())
    }
    fn galois(&self) -> u64 {
        self.galois
    }
    fn digit_stride(&self) -> usize {
        self.rows * self.n
    }
    fn dnum(&self) -> usize {
        self.dnum
    }
    fn q_band(&self) -> usize {
        usize::MAX
    }
}

/// `a` rows of digits `0..ndig` from a KSKA seed into `out` (`[digit][prime id][N]`, every prime id), on `s`.
pub fn expand_a<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    seed: *const u32,
    ndig: usize,
    out: &mut DBuf<'_, Compute, W>,
) -> Result<()> {
    let rows = ctx.lay.rows();
    if ndig * rows * ctx.lay.n > out.len() {
        return Err(shape("expand_a", format!("{ndig} digits do not fit {} words", out.len())));
    }
    let a = KskExpandArgs { out: out.vptr_mut(), seed, prime_ids: plans.ids.dptr(), pr: ctx.pr(), nmod: rows as i32, dnum: ndig as i32, d0: 0, pad: 0 };
    // SAFETY: table pointers are context constants; out range checked; seed is an 8-word device buffer
    launch_result("ksk_expand", unsafe { (ctx.table.ksk_expand)(&a, s.raw()) })?;
    Ok(())
}

/// Device pointer tables of an ordered set of keys (one hoisted group, a rotate-and-sum radix group, or one key),
/// built at setup: the kernels read key r's `b` / `a` / automorphism exponent through them.
pub struct KeyGroup<'a> {
    kb: ConstBuf<'a, u64>,
    ka: ConstBuf<'a, u64>,
    autok: ConstBuf<'a, u32>,
    /// `a` / `b` pointing at the executor's staging buffers (`KsWork::regen_a` / `stage_b`), one entry per key:
    /// KskSource::Regen rotations and keys whose `b` needs staging.
    ka_regen: Option<ConstBuf<'a, u64>>,
    kb_stage: Option<ConstBuf<'a, u64>>,
    staging: Vec<Option<Staging>>,
    /// The keys read in place (`ks_dot_key`: seeded `a` in-kernel, `b` dense / packed / banded) when all are seeded:
    /// KskSource::Regen and staged keys dot through it with no staging copy.
    tabs: Option<KeyTabs<'a>>,
    /// `ks_dot_key` on keys with a resident dense `a`: digits regenerated (the rest read; `set_regen_digits`).
    regen_digits: usize,
    /// Galois element of each key, in order.
    pub galois: Vec<u64>,
    seeds: Vec<Option<*const u32>>,
    has_a: bool,
    digit_stride: usize,
    dnum: usize,
    q_band: usize,
}

impl<'a> KeyGroup<'a> {
    /// Tables for `keys` (in order). `stage`: the executor's `KsWork::stage_ptrs()` (needed for Regen rotations and
    /// keys whose `b` is packed or banded).
    pub fn new<S: Lane>(arena: &'a Arena, s: &Stream<S>, n: usize, keys: &[&dyn KsKeyRef], stage: Option<(*const c_void, *const c_void)>) -> Result<Self> {
        let first = keys.first().ok_or_else(|| shape("KeyGroup", "no keys"))?;
        let (ds, dnum) = (first.digit_stride(), first.dnum());
        if keys.iter().any(|k| k.digit_stride() != ds || k.dnum() != dnum) {
            return Err(shape("KeyGroup", "keys of one group must share the digit layout"));
        }
        let kb: Vec<u64> = keys.iter().map(|k| k.b_ptr() as u64).collect();
        let has_a = keys.iter().all(|k| k.a_ptr().is_some());
        let ka: Vec<u64> = keys.iter().map(|k| k.a_ptr().map(|p| p as u64).unwrap_or(0)).collect();
        let galois: Vec<u64> = keys.iter().map(|k| k.galois()).collect();
        let autok = galois
            .iter()
            .map(|&g| if g == 1 { Ok(1u32) } else { Ok(pc_core::galois::galois_inverse(g, n)? as u32) })
            .collect::<Result<Vec<u32>>>()?;
        let staging: Vec<Option<Staging>> = keys.iter().map(|k| k.staging()).collect();
        if stage.is_none() && staging.iter().any(|x| x.is_some()) {
            return Err(shape("KeyGroup", "packed / banded keys need the executor's staging buffers (KsWork::carve_regen)"));
        }
        let (ka_regen, kb_stage) = match stage {
            Some((a, b)) => (
                Some(ConstBuf::new(arena, "ks.keygroup", s, &vec![a as u64; keys.len()])?),
                Some(ConstBuf::new(arena, "ks.keygroup", s, &vec![b as u64; keys.len()])?),
            ),
            None => (None, None),
        };
        let tabs = KeyTabs::new(arena, s, n, keys)?;
        Ok(KeyGroup {
            tabs,
            regen_digits: usize::MAX,
            kb: ConstBuf::new(arena, "ks.keygroup", s, &kb)?,
            ka: ConstBuf::new(arena, "ks.keygroup", s, &ka)?,
            autok: ConstBuf::new(arena, "ks.keygroup", s, &autok)?,
            ka_regen,
            kb_stage,
            staging,
            galois,
            seeds: keys.iter().map(|k| k.seed_ptr()).collect(),
            has_a,
            digit_stride: ds,
            dnum,
            q_band: keys.iter().map(|k| k.q_band()).min().unwrap_or(0),
        })
    }
    /// Arena bytes of a group of `r` keys (MemPlan).
    pub fn bytes(r: usize, regen: bool) -> usize {
        Arena::bytes_for::<u64>(r) * (2 + 2 * regen as usize) + Arena::bytes_for::<u32>(r) + KeyTabs::bytes(r)
    }
    /// `ks_dot_key` with only the first `n` digits regenerated when every key has a resident dense `a` (setup).
    pub fn set_regen_digits(&mut self, n: usize) {
        self.regen_digits = n;
    }
    /// The keys can be read in place (`ks_dot_key`): every key seeded.
    pub fn in_place(&self) -> bool {
        self.tabs.is_some()
    }
    /// A dot with `source` reads these keys through `ks_dot_key` (`dot(.., staged = true)`, every key of a range in
    /// one launch): Regen, or keys the Cached kernel cannot read (packed / banded `b`, no resident `a`).
    pub fn reads_in_place(&self, source: KskSource) -> bool {
        self.in_place() && (source == KskSource::Regen || self.needs_stage() || !self.has_a())
    }
    pub fn len(&self) -> usize {
        self.galois.len()
    }
    pub fn is_empty(&self) -> bool {
        self.galois.is_empty()
    }
    /// Every key carries a seed (Regen possible).
    pub fn seeded(&self) -> bool {
        self.seeds.iter().all(|s| s.is_some())
    }
    pub fn has_a(&self) -> bool {
        self.has_a
    }
    pub(crate) fn seed(&self, r: usize) -> Option<*const u32> {
        self.seeds.get(r).copied().flatten()
    }
    /// Some key's `b` must be staged before its dot.
    pub fn needs_stage(&self) -> bool {
        self.staging.iter().any(|x| x.is_some())
    }
    pub(crate) fn staged_b(&self, r: usize) -> bool {
        self.staging.get(r).copied().flatten().is_some()
    }
    /// The in-place table of key `r` when its `b` is not dense prime-id rows (packed / banded) and the group is read
    /// in place: `ks_dot_regen` reads it with no `stage_key` (ABI 7); None = dense rows (or not seeded).
    pub fn regen_tab(&self, r: usize) -> Result<Option<pc_kernels::KeyTab>> {
        match &self.tabs {
            Some(t) if self.staged_b(r) => Ok(Some(t.tab(r)?)),
            _ => Ok(None),
        }
    }
}

/// Per-executor key-switching workspace (`MemPlan::ks_work`).
pub struct KsWork<'a, W: DevWord> {
    /// ModUp output: `[c < k_max][digit < dnum][row][N]`.
    pub digits: DBuf<'a, Compute, W>,
    /// Extended-basis inner-product outputs (`ext_cts(k_max)` ciphertexts).
    pub ext: CtBuf<'a, Compute, W>,
    /// ModDown conversions / extended accumulators (k_max ciphertexts).
    pub conv: CtBuf<'a, Compute, W>,
    /// One expanded key (`dnum` digits) for KskSource::Regen rotations; carved by the owner of that choice
    /// (`KsWork::regen_bytes`, MemPlan section "ks.regen_a") before `freeze`.
    pub regen_a: Option<DBuf<'a, Compute, W>>,
    /// One key's `b` unpacked / re-slotted to dense prime-id rows (packed or banded keys); carved with `regen_a`.
    pub stage_b: Option<DBuf<'a, Compute, W>>,
    pub k_max: usize,
    pub dnum: usize,
}

impl<'a, W: DevWord> KsWork<'a, W> {
    /// Ciphertexts of `ext`: k_max, or `SUM_KEYS` extended rotations of a batch of up to `KEY_KMAX` when more.
    pub fn ext_cts(k_max: usize) -> usize {
        k_max.max(SUM_KEYS * k_max.min(KEY_KMAX))
    }
    /// Element counts (digits, ext, conv).
    pub fn words(lay: &Layout, k_max: usize, dnum: usize) -> [usize; 3] {
        [k_max * dnum * lay.poly_words(), CtBuf::<Compute, W>::words(lay, Self::ext_cts(k_max)), CtBuf::<Compute, W>::words(lay, k_max)]
    }
    /// Arena bytes (MemPlan::ks_work).
    pub fn bytes(lay: &Layout, k_max: usize, dnum: usize) -> usize {
        Self::words(lay, k_max, dnum).iter().map(|&w| Arena::bytes_for::<W>(w)).sum()
    }
    /// Bytes of the staging buffers (`regen_a` + `stage_b`; MemPlan section "ks.regen_a").
    pub fn regen_bytes(lay: &Layout, dnum: usize) -> usize {
        2 * Arena::bytes_for::<W>(dnum * lay.poly_words())
    }
    pub fn new(arena: &'a Arena, lay: Layout, k_max: usize, dnum: usize) -> Result<Self> {
        let [d, _, _] = Self::words(&lay, k_max, dnum);
        Ok(KsWork {
            digits: arena.alloc("ks_work.digits", d)?,
            ext: CtBuf::new(arena, "ks_work.ext", lay, Self::ext_cts(k_max))?,
            conv: CtBuf::new(arena, "ks_work.conv", lay, k_max)?,
            regen_a: None,
            stage_b: None,
            k_max,
            dnum,
        })
    }
    /// Carve the key staging buffers (setup; section "ks.regen_a").
    pub fn carve_regen(&mut self, arena: &'a Arena, lay: &Layout) -> Result<()> {
        self.regen_a = Some(arena.alloc("ks.regen_a", self.dnum * lay.poly_words())?);
        self.stage_b = Some(arena.alloc("ks.regen_a", self.dnum * lay.poly_words())?);
        Ok(())
    }
    /// (a, b) staging pointers for `KeyGroup::new`.
    pub fn stage_ptrs(&self) -> Option<(*const c_void, *const c_void)> {
        Some((self.regen_a.as_ref()?.vptr(), self.stage_b.as_ref()?.vptr()))
    }
}

struct LevelPlan<'a> {
    ndig: usize,
    /// INTT rows of the coefficient workspace: `(c * rows + t, t)`, t <= top, by ciphertext.
    intt: RowList<'a>,
    /// The same rows in a ciphertext buffer, from one part of its first ciphertext: `c * 2 rows + t` (the ModUp's INTT
    /// reads the input there, out of place).
    intt_src: ConstBuf<'a, u32>,
    /// ModUp conversions of every live digit, jobs `[c < k_max][digit]` (digit limbs -> the other live Q limbs and
    /// P): one launch for all digits of the first k ciphertexts (a job prefix).
    modup: BconvPlan<'a>,
    /// NTT rows of the ModUp targets, `[c][digit][target]` (prefix by ciphertext).
    modup_ntt: RowList<'a>,
    /// ModUp NTT rows per ciphertext (prefix unit).
    mu_rows: usize,
    /// Q rows 0..=top then the special rows (dot row table; P-only = suffix, Q-only = prefix).
    ks_rows: ConstBuf<'a, u32>,
    /// ModDown conversion, jobs `[c < k_max][part]`: special rows -> Q rows 0..=top, ciphertext layout.
    moddown: BconvPlan<'a>,
}

/// Host specs of one level (shared by sizing and building).
struct LevelSpecs {
    intt: Vec<(u32, i32)>,
    intt_src: Vec<u32>,
    ndig: usize,
    modup: Vec<JobSpec>,
    modup_ntt: Vec<(u32, i32)>,
    mu_rows: usize,
    ks_rows: Vec<u32>,
    moddown: Vec<JobSpec>,
}

fn level_specs(chain: &Chain, lay: &Layout, top: usize, k_max: usize) -> LevelSpecs {
    let rows = lay.rows();
    let dnum = chain.dnum();
    let specials: Vec<usize> = chain.special_ids();
    // live digits: digit d keeps its limbs <= top (FIDESlib DECOMPmeta sizes cut at the level's limb count)
    let digs: Vec<Vec<usize>> =
        chain.digits.iter().map(|d| d.iter().copied().filter(|&t| t <= top).collect::<Vec<_>>()).filter(|d| !d.is_empty()).collect();
    let (mut intt, mut intt_src) = (Vec::new(), Vec::new());
    for c in 0..k_max {
        for t in 0..=top {
            intt.push(((c * rows + t) as u32, t as i32));
            intt_src.push((c * 2 * rows + t) as u32);
        }
    }
    // [c][digit] order: the first k ciphertexts' jobs (and NTT rows) are a prefix, so one launch serves any k
    let (mut modup, mut modup_ntt) = (Vec::new(), Vec::new());
    let mut mu_rows = 0;
    for c in 0..k_max {
        for (d, dl) in digs.iter().enumerate() {
            let tg: Vec<usize> = (0..=top).filter(|t| !dl.contains(t)).chain(specials.iter().copied()).collect();
            let dst_off = ((c * dnum + d) * rows) as u32;
            for &t in &tg {
                modup_ntt.push((dst_off + t as u32, t as i32));
            }
            if c == 0 {
                mu_rows += tg.len();
            }
            modup.push(JobSpec {
                src_off: (c * rows) as u32,
                dst_off,
                src: dl.iter().map(|&t| (t as u32, t as u32)).collect(),
                tgt: tg.iter().map(|&t| (t as u32, t as u32)).collect(),
                exact: false,
            });
        }
    }
    let ks_rows: Vec<u32> = (0..=top).chain(specials.iter().copied()).map(|t| t as u32).collect();
    let mut moddown = Vec::new();
    for c in 0..k_max {
        for part in 0..2 {
            let base = ((2 * c + part) * rows) as u32;
            moddown.push(JobSpec {
                src_off: base,
                dst_off: base,
                src: specials.iter().map(|&t| (t as u32, t as u32)).collect(),
                tgt: (0..=top).map(|t| (t as u32, t as u32)).collect(),
                exact: false,
            });
        }
    }
    LevelSpecs { intt, intt_src, ndig: digs.len(), modup, modup_ntt, mu_rows, ks_rows, moddown }
}

fn rowlist_bytes(n: usize) -> usize {
    Arena::bytes_for::<u32>(n) + Arena::bytes_for::<i32>(n)
}

/// Key-switching plans of every level for `k_max` ciphertexts (setup; MemPlan section "ks.plans").
pub struct KsPlans<'a> {
    levels: Vec<LevelPlan<'a>>,
    /// prime id -> digit (Q ids), -1 for special ids.
    digit_of: ConstBuf<'a, i32>,
    /// 0..rows (key expansion row list).
    ids: ConstBuf<'a, i32>,
    pub k_max: usize,
    pub dnum: usize,
    pub lay: Layout,
    pub method: KeySwitch,
}

impl<'a> KsPlans<'a> {
    /// Arena bytes of the plans of `chain` at words `W` for `k_max` ciphertexts.
    pub fn bytes<W: DevWord>(chain: &Chain, k_max: usize) -> Result<usize> {
        let lay = Layout::from_chain(chain);
        let primes = chain.primes();
        let mut b = 2 * Arena::bytes_for::<i32>(lay.rows());
        for top in 0..lay.nq {
            let sp = level_specs(chain, &lay, top, k_max);
            b += rowlist_bytes(sp.intt.len()) + Arena::bytes_for::<u32>(sp.intt_src.len()) + Arena::bytes_for::<u32>(sp.ks_rows.len());
            b += BconvPlan::bytes::<W>(&primes, &sp.modup)? + rowlist_bytes(sp.modup_ntt.len());
            b += BconvPlan::bytes::<W>(&primes, &sp.moddown)?;
        }
        Ok(b)
    }

    pub fn new<W: DevWord, M: DevStrategy<W>, S: Lane>(arena: &'a Arena, s: &Stream<S>, ctx: &Context<'_, W, M>, k_max: usize) -> Result<Self> {
        let chain = &ctx.chain;
        let lay = ctx.lay;
        if k_max == 0 || k_max > ctx.k_max {
            return Err(Error::Mem { knob: "MemCfg::k_max", value: k_max, msg: format!("ks plans for {k_max} cts, context row tables cover {}", ctx.k_max) });
        }
        let primes = chain.primes();
        let mut levels = Vec::with_capacity(lay.nq);
        for top in 0..lay.nq {
            let sp = level_specs(chain, &lay, top, k_max);
            levels.push(LevelPlan {
                ndig: sp.ndig,
                intt: RowList::new(arena, s, &sp.intt)?,
                intt_src: ConstBuf::new(arena, "ks.plans", s, &sp.intt_src)?,
                modup: BconvPlan::new::<S, W>(arena, s, &primes, &sp.modup)?,
                modup_ntt: RowList::new(arena, s, &sp.modup_ntt)?,
                mu_rows: sp.mu_rows,
                ks_rows: ConstBuf::new(arena, "ks.plans", s, &sp.ks_rows)?,
                moddown: BconvPlan::new::<S, W>(arena, s, &primes, &sp.moddown)?,
            });
        }
        let digit_of: Vec<i32> = (0..lay.rows()).map(|t| chain.digit_of(t).map(|d| d as i32).unwrap_or(-1)).collect();
        let ids: Vec<i32> = (0..lay.rows() as i32).collect();
        Ok(KsPlans {
            levels,
            digit_of: ConstBuf::new(arena, "ks.plans", s, &digit_of)?,
            ids: ConstBuf::new(arena, "ks.plans", s, &ids)?,
            k_max,
            dnum: chain.dnum(),
            lay,
            method: KeySwitch::of(chain),
        })
    }

    fn level(&self, top: usize) -> Result<&LevelPlan<'a>> {
        self.levels.get(top).ok_or_else(|| shape("ks level", format!("top {top} >= |Q| = {}", self.levels.len())))
    }
    /// Live digits at `top`.
    pub fn ndig(&self, top: usize) -> Result<usize> {
        Ok(self.level(top)?.ndig)
    }
    pub(crate) fn check_k(&self, k: usize) -> Result<()> {
        if k == 0 || k > self.k_max {
            return Err(Error::Mem { knob: "MemCfg::k_max", value: k, msg: format!("key switch of {k} cts, plans cover 1..={}", self.k_max) });
        }
        Ok(())
    }
}

// ── element-wise helpers over ciphertext views ───────────────────────────────────────────────────────────────────

/// A view of ciphertexts (or one polynomial of each) inside some buffer: base pointer + strides (words).
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub base: *mut c_void,
    pub ct_stride: u64,
    pub part_stride: u64,
}

impl View {
    /// Ciphertexts from `first` on of `b` (callers check the range with `check_cts`).
    pub fn of<W: DevWord>(b: &CtBuf<'_, Compute, W>, first: usize) -> View {
        let lay = b.lay;
        View {
            base: (b.buf.dptr() as *mut W).wrapping_add(first * lay.ct_words()) as *mut c_void,
            ct_stride: lay.ct_words() as u64,
            part_stride: lay.poly_words() as u64,
        }
    }
    /// Polynomial `part` of each ciphertext of the view.
    pub fn part<W>(self, part: usize) -> View {
        View { base: (self.base as *mut W).wrapping_add(part * self.part_stride as usize) as *mut c_void, ..self }
    }
    pub fn r(self) -> PolyRef {
        PolyRef { base: self.base, ct_stride: self.ct_stride, part_stride: self.part_stride }
    }
}

pub(crate) fn check_cts<W: DevWord>(b: &CtBuf<'_, Compute, W>, first: usize, k: usize, what: &'static str) -> Result<()> {
    if first + k > b.k {
        return Err(shape(what, format!("cts [{first}, {}) of a {}-ct buffer", first + k, b.k)));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ew {
    Add,
    Copy,
    /// out = (a - b) P^-1 (ModDown epilogue)
    SubMulPinv,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ew<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    kind: Ew,
    out: View,
    a: View,
    b: Option<View>,
    rs: RowSet,
    k: usize,
    nparts: usize,
) -> Result<()> {
    let (f, name, sc, scs) = match kind {
        Ew::Add => (ctx.table.ew_add, "ew_add", std::ptr::null(), std::ptr::null()),
        Ew::Copy => (ctx.table.ew_copy, "ew_copy", std::ptr::null(), std::ptr::null()),
        Ew::SubMulPinv => (ctx.table.ew_sub_mul, "ew_sub_mul", ctx.ks.p_inv_mod_q.v.vptr(), ctx.ks.p_inv_mod_q.s.vptr()),
    };
    let args = EwArgs { out: out.r(), a: a.r(), b: b.unwrap_or(a).r(), rs, sc, scs, pr: ctx.pr(), k: k as i32, nparts: nparts as i32, sub: 0, zero_p: 0 };
    // SAFETY: views are inside live compute-lane buffers of the context's layout (callers check the ciphertext
    // ranges); out aliasing a or b is element-wise safe
    launch_result(name, unsafe { f(&args, s.raw()) })?;
    Ok(())
}

// ── the three steps ──────────────────────────────────────────────────────────────────────────────────────────────

/// ModUp of polynomial `part` of ciphertexts `[first, first + k)` of `x` (live rows 0..=top) into `ks.digits`.
#[allow(clippy::too_many_arguments)]
pub fn modup<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    ks: &mut KsWork<'_, W>,
    coef: &mut DBuf<'_, Compute, W>,
    x: &CtBuf<'_, Compute, W>,
    first: usize,
    k: usize,
    part: usize,
    top: usize,
) -> Result<()> {
    check_cts(x, first, k, "ks modup")?;
    modup_from(ctx, s, plans, ks, coef, View::of(x, first).part::<W>(part), k, top)
}

/// ModUp of the k polynomials of `src` (one part of k ciphertexts of the ciphertext layout, or of one ciphertext:
/// rows 0..=top, evaluation form): the INTT reads them in place of a copy into `coef` (out-of-place first pass).
#[allow(clippy::too_many_arguments)]
pub fn modup_from<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    ks: &mut KsWork<'_, W>,
    coef: &mut DBuf<'_, Compute, W>,
    src: View,
    k: usize,
    top: usize,
) -> Result<()> {
    plans.check_k(k)?;
    let lay = ctx.lay;
    if coef.len() < k * lay.poly_words() {
        return Err(shape("ks modup", format!("coefficient workspace holds {} words, {k} polys needed (MemPlan::bconv_work)", coef.len())));
    }
    if k > 1 && src.ct_stride != lay.ct_words() as u64 {
        return Err(shape("ks modup", "k > 1 sources must have the ciphertext layout"));
    }
    let lv = plans.level(top)?;
    let src = ntt::Src { base: src.base, soff: lv.intt_src.dptr(), n: 1, stride: 0 };
    ntt::transform_rows_prefix_ex(ctx, s, Dir::Inverse, coef, &lv.intt, k * (top + 1), Some(src), None)?;
    modup_tail(ctx, s, lv, ks, coef, k)
}

/// ModUp when rows 0..=top of the k polynomials already sit in `coef` (`[c][row][N]`, evaluation form).
pub fn modup_coef_ready<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    ks: &mut KsWork<'_, W>,
    coef: &mut DBuf<'_, Compute, W>,
    k: usize,
    top: usize,
) -> Result<()> {
    plans.check_k(k)?;
    let lv = plans.level(top)?;
    ntt::transform_rows_prefix(ctx, s, Dir::Inverse, coef, &lv.intt, k * (top + 1))?;
    modup_tail(ctx, s, lv, ks, coef, k)
}

/// The ModUp after its INTT: conversion of every live digit, NTT of the targets.
fn modup_tail<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, lv: &LevelPlan<'_>, ks: &mut KsWork<'_, W>, coef: &DBuf<'_, Compute, W>, k: usize) -> Result<()> {
    lv.modup.run_prefix(ctx, s, coef, 0, &mut ks.digits, 0, k * lv.ndig)?;
    ntt::transform_rows_prefix(ctx, s, Dir::Forward, &mut ks.digits, &lv.modup_ntt, k * lv.mu_rows)
}

/// Which rows a dot computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DotRows {
    /// Q rows 0..=top and P.
    All,
    /// P rows only (relin first half).
    P,
    /// Q rows 0..=top only.
    Q,
}

/// Inputs and outputs of one inner-product launch (raw views; the public ops build them from `Ct`s).
#[derive(Clone, Copy, Debug)]
pub struct DotIo {
    /// The switched polynomial's own evaluation rows (digit d's own limbs are read from here).
    pub c1: *const c_void,
    /// c0 of the input (c0 modes 1/2).
    pub c0: *const c_void,
    /// Words between the input ciphertexts (c1 and c0).
    pub ct_stride: u64,
    /// Output (r, c) part 0 at out + c * out_ct_stride + r * out_rot_stride; part 1 at + part_stride.
    pub out: *mut c_void,
    pub part_stride: u64,
    pub out_ct_stride: u64,
    pub out_rot_stride: u64,
    /// 0 none, 1 P*c0 on Q rows, 2 c0 as stored (extended).
    pub c0_mode: i32,
}

fn dot_rows(lv: &LevelPlan<'_>, lay: &Layout, top: usize, rows: DotRows) -> Result<(*const u32, usize)> {
    Ok(match rows {
        DotRows::All => (lv.ks_rows.dptr(), top + 1 + lay.kp),
        DotRows::Q => (lv.ks_rows.dptr(), top + 1),
        DotRows::P => (lv.ks_rows.dptr_at(top + 1)?, lay.kp),
    })
}

/// `ks_dot` against keys `[r0, r0 + r)` of `group`, K = k ciphertexts of the ModUp in `ks.digits`. `staged`: read
/// key r0 through the staging buffers (`stage_key` ran for it; r must be 1).
#[allow(clippy::too_many_arguments)]
pub fn dot<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    ks: &KsWork<'_, W>,
    group: &KeyGroup<'_>,
    r0: usize,
    r: usize,
    k: usize,
    top: usize,
    rows: DotRows,
    io: DotIo,
    staged: bool,
) -> Result<()> {
    plans.check_k(k)?;
    if staged {
        if let Some(t) = &group.tabs {
            return dot_key(ctx, s, plans, ks, group, t, r0, r, k, top, rows, io);
        }
    }
    if k > DOT_KMAX || r == 0 || r0 + r > group.len() || (staged && r != 1) || (!staged && (r0..r0 + r).any(|j| group.staged_b(j))) {
        return Err(shape("ks dot", format!("k {k} (<= {DOT_KMAX}), keys [{r0}, {}) of {} (regen: one key per dot)", r0 + r, group.len())));
    }
    if top > group.q_band {
        return Err(shape("ks dot", format!("key band ends at limb {}, ciphertext top {top}", group.q_band)));
    }
    let lv = plans.level(top)?;
    if group.dnum < lv.ndig {
        return Err(shape("ks dot", format!("key has {} digits, level needs {}", group.dnum, lv.ndig)));
    }
    let ka = if staged {
        group.ka_regen.as_ref().ok_or_else(|| shape("ks dot", "staged key: group built without KsWork::stage_ptrs"))?.dptr_at(r0)?
    } else {
        if !group.has_a {
            return Err(shape("ks dot", "keys without resident `a` (KskSource::Cached needs KeyA::Expand / Dense)"));
        }
        group.ka.dptr_at(r0)?
    };
    let lay = ctx.lay;
    let (rows_ptr, nrows) = dot_rows(lv, &lay, top, rows)?;
    let a = KsArgs {
        c1: io.c1,
        modup: ks.digits.vptr(),
        c0: io.c0,
        out0: io.out,
        out1: (io.out as *mut W).wrapping_add(io.part_stride as usize) as *mut c_void,
        kb: if staged && group.staged_b(r0) {
            group.kb_stage.as_ref().ok_or_else(|| shape("ks dot", "staged key without stage tables"))?.dptr_at(r0)?
        } else {
            group.kb.dptr_at(r0)?
        } as *const *const c_void,
        ka: ka as *const *const c_void,
        autok: group.autok.dptr_at(r0)?,
        rows: rows_ptr,
        digit_of: plans.digit_of.dptr(),
        pmod: ctx.ks.p_mod_q.v.vptr(),
        pmods: ctx.ks.p_mod_q.s.vptr(),
        pr: ctx.pr(),
        ct_stride: io.ct_stride,
        mu_ct_stride: (plans.dnum * lay.poly_words()) as u64,
        mu_digit_stride: lay.poly_words() as u64,
        out_ct_stride: io.out_ct_stride,
        out_rot_stride: io.out_rot_stride,
        key_digit_stride: group.digit_stride as u64,
        k: k as i32,
        r: r as i32,
        ndig: lv.ndig as i32,
        c0_mode: io.c0_mode,
        nq: lay.nq as i32,
        nrows: nrows as i32,
    };
    // SAFETY: every pointer is a live arena buffer of the context layout (views checked by the callers, key tables
    // built at setup from live keys); outputs never alias the ModUp digits or the keys
    launch_result("ks_dot", unsafe { (ctx.table.ks_dot)(&a, s.raw()) })?;
    Ok(())
}

/// Keys `[r0, r0 + r)` of `group` read in place (`ks_dot_key`: `a` regenerated from the seed, `b` dense / packed /
/// banded as stored): launches of at most `KEY_KMAX` ciphertexts over runs of keys sharing a stored form.
#[allow(clippy::too_many_arguments)]
fn dot_key<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    ks: &KsWork<'_, W>,
    group: &KeyGroup<'_>,
    tabs: &KeyTabs<'_>,
    r0: usize,
    r: usize,
    k: usize,
    top: usize,
    rows: DotRows,
    io: DotIo,
) -> Result<()> {
    if r == 0 || r0 + r > group.len() {
        return Err(shape("ks dot_key", format!("keys [{r0}, {}) of {}", r0 + r, group.len())));
    }
    if top > tabs.q_top() {
        return Err(shape("ks dot_key", format!("key band ends at limb {}, ciphertext top {top}", tabs.q_top())));
    }
    let lv = plans.level(top)?;
    if group.dnum < lv.ndig {
        return Err(shape("ks dot_key", format!("key has {} digits, level needs {}", group.dnum, lv.ndig)));
    }
    let lay = ctx.lay;
    let (rows_ptr, nrows) = dot_rows(lv, &lay, top, rows)?;
    let mu_ct = (plans.dnum * lay.poly_words()) as u64;
    let mut j = r0;
    while j < r0 + r {
        let len = tabs.run_len(j, r0 + r - j);
        let mut key = tabs.tab(j)?;
        let hybrid = group.has_a && !group.needs_stage() && group.regen_digits < lv.ndig;
        if hybrid {
            key.nregen = group.regen_digits as i32;
        }
        let mut c = 0;
        while c < k {
            let kk = (k - c).min(KEY_KMAX);
            let at = |p: *const c_void, stride: u64| (p as *const W).wrapping_add(c * stride as usize) as *const c_void;
            let out = (at(io.out, io.out_ct_stride) as *const W).wrapping_add((j - r0) * io.out_rot_stride as usize) as *mut c_void;
            let a = KsArgs {
                c1: at(io.c1, io.ct_stride),
                modup: at(ks.digits.vptr(), mu_ct),
                c0: if io.c0.is_null() { io.c0 } else { at(io.c0, io.ct_stride) },
                out0: out,
                out1: (out as *mut W).wrapping_add(io.part_stride as usize) as *mut c_void,
                kb: std::ptr::null(),
                ka: if hybrid { group.ka.dptr_at(j)? as *const *const c_void } else { std::ptr::null() },
                autok: group.autok.dptr_at(j)?,
                rows: rows_ptr,
                digit_of: plans.digit_of.dptr(),
                pmod: ctx.ks.p_mod_q.v.vptr(),
                pmods: ctx.ks.p_mod_q.s.vptr(),
                pr: ctx.pr(),
                ct_stride: io.ct_stride,
                mu_ct_stride: mu_ct,
                mu_digit_stride: lay.poly_words() as u64,
                out_ct_stride: io.out_ct_stride,
                out_rot_stride: io.out_rot_stride,
                key_digit_stride: if hybrid { group.digit_stride as u64 } else { 0 },
                k: kk as i32,
                r: len as i32,
                ndig: lv.ndig as i32,
                c0_mode: io.c0_mode,
                nq: lay.nq as i32,
                nrows: nrows as i32,
            };
            let ka = KsKeyArgs { ks: a, key };
            // SAFETY: as `dot`; the key tables (seed pointers, stored `b`) were built at setup from live keys
            launch_result("ks_dot_key", unsafe { (ctx.table.ks_dot_key)(&ka, s.raw()) })?;
            c += kk;
        }
        j += len;
    }
    Ok(())
}

/// Relinearization inner product with the key's `a` regenerated in-kernel (`ks_dot_regen`): key 0 of `group`, no
/// automorphism, no c0 term, K <= 4 per launch (larger K is split). `epi = Some((dst, conv))`: the rows are written
/// through the fused ModDown epilogue `dst = dst + (dot - conv) P^-1` (dst / conv: ciphertext-layout views of the k
/// ciphertexts); None: rows into `io.out`.
#[allow(clippy::too_many_arguments)]
pub fn dot_regen<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    ks: &KsWork<'_, W>,
    group: &KeyGroup<'_>,
    k: usize,
    top: usize,
    rows: DotRows,
    io: DotIo,
    epi: Option<(View, View)>,
) -> Result<()> {
    plans.check_k(k)?;
    let seed = group.seed(0).ok_or_else(|| shape("ks dot_regen", "key without a seed (KskSource::Regen)"))?;
    if group.galois[0] != 1 {
        return Err(shape("ks dot_regen", "in-kernel regeneration is relinearization-only (no automorphism)"));
    }
    let lv = plans.level(top)?;
    let lay = ctx.lay;
    let (rows_ptr, nrows) = dot_rows(lv, &lay, top, rows)?;
    // a packed / banded relin key: read in place through its KeyTab (ABI 7), else the `b` that `stage_key` left in
    // KsWork::stage_b (unseeded tables)
    let tab = group.regen_tab(0)?;
    let kb0 = if group.staged_b(0) && tab.is_none() {
        group.kb_stage.as_ref().ok_or_else(|| shape("ks dot_regen", "staged key without stage tables"))?.dptr()
    } else {
        group.kb.dptr()
    };
    if let Some(t) = &tab {
        if top > t.q_top as usize && t.q_top >= 0 {
            return Err(shape("ks dot_regen", format!("key band ends at limb {}, ciphertext top {top}", t.q_top)));
        }
    }
    let mut c = 0;
    while c < k {
        let kk = (k - c).min(REGEN_KMAX);
        let at = |p: *const c_void, stride: u64| (p as *const W).wrapping_add(c * stride as usize) as *const c_void;
        let out = at(io.out, io.out_ct_stride);
        let ks_args = KsArgs {
            c1: at(io.c1, io.ct_stride),
            modup: at(ks.digits.vptr(), (plans.dnum * lay.poly_words()) as u64),
            c0: std::ptr::null(),
            out0: out as *mut c_void,
            out1: (out as *const W).wrapping_add(io.part_stride as usize) as *mut c_void,
            kb: kb0 as *const *const c_void,
            ka: group.ka.dptr() as *const *const c_void,
            autok: group.autok.dptr(),
            rows: rows_ptr,
            digit_of: plans.digit_of.dptr(),
            pmod: ctx.ks.p_mod_q.v.vptr(),
            pmods: ctx.ks.p_mod_q.s.vptr(),
            pr: ctx.pr(),
            ct_stride: io.ct_stride,
            mu_ct_stride: (plans.dnum * lay.poly_words()) as u64,
            mu_digit_stride: lay.poly_words() as u64,
            out_ct_stride: io.out_ct_stride,
            out_rot_stride: 0,
            key_digit_stride: group.digit_stride as u64,
            k: kk as i32,
            r: 1,
            ndig: lv.ndig as i32,
            c0_mode: 0,
            nq: lay.nq as i32,
            nrows: nrows as i32,
        };
        let (use_epi, e) = match epi {
            Some((dst, conv)) => (
                1,
                EpiArgs {
                    dst: at(dst.base, dst.ct_stride) as *mut c_void,
                    conv: at(conv.base, conv.ct_stride),
                    binv: ctx.ks.p_inv_mod_q.v.vptr(),
                    binvs: ctx.ks.p_inv_mod_q.s.vptr(),
                    k: std::ptr::null(),
                    part_stride: dst.part_stride,
                    dbl: 0,
                    pad: 0,
                },
            ),
            None => (0, EpiArgs::default()),
        };
        let a = KsRegenArgs { ks: ks_args, seed, epi: e, use_epi, pad: 0, key: tab.unwrap_or_default(), add: std::ptr::null() };
        // SAFETY: as `dot`; the epilogue's dst / conv are ciphertext-layout views of the k ciphertexts
        launch_result("ks_dot_regen", unsafe { (ctx.table.ks_dot_regen)(&a, s.raw()) })?;
        c += kk;
    }
    Ok(())
}

/// First half of ModDown on the first `k` ciphertexts of `x` (both parts): INTT of x's special rows (in place),
/// conversion P -> Q rows 0..=top into `conv`, NTT of conv's Q rows. Then `conv` holds the correction.
pub fn moddown_conv<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    x: &mut CtBuf<'_, Compute, W>,
    conv: &mut CtBuf<'_, Compute, W>,
    k: usize,
    top: usize,
) -> Result<()> {
    plans.check_k(k)?;
    check_cts(x, 0, k, "ks moddown")?;
    check_cts(conv, 0, k, "ks moddown")?;
    let lv = plans.level(top)?;
    ntt::transform_k(ctx, s, Dir::Inverse, x, Span::P, Parts::Both, k)?;
    lv.moddown.run_prefix(ctx, s, &x.buf, 0, &mut conv.buf, 0, 2 * k)?;
    ntt::transform_k(ctx, s, Dir::Forward, conv, Span::Q { top }, Parts::Both, k)?;
    Ok(())
}

/// Full ModDown in place on the first `k` ciphertexts of `x`: Q rows 0..=top = (x - conv) P^-1.
pub fn moddown_inplace<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    x: &mut CtBuf<'_, Compute, W>,
    conv: &mut CtBuf<'_, Compute, W>,
    k: usize,
    top: usize,
) -> Result<()> {
    let v = View::of(x, 0);
    moddown_to(ctx, s, plans, x, conv, v, false, k, top)
}

/// ModDown of the first `k` ciphertexts of `x` into `out` (k-ct view of the ciphertext layout; may be x itself):
/// Q rows 0..=top of out = (x - conv) P^-1, or `acc`: out + (x - conv) P^-1. The last step rides the NTT of the
/// conversion (its epilogue), so `conv` holds scratch afterwards; x's special rows are left in coefficient form.
#[allow(clippy::too_many_arguments)]
pub fn moddown_to<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    x: &mut CtBuf<'_, Compute, W>,
    conv: &mut CtBuf<'_, Compute, W>,
    out: View,
    acc: bool,
    k: usize,
    top: usize,
) -> Result<()> {
    plans.check_k(k)?;
    check_cts(x, 0, k, "ks moddown")?;
    check_cts(conv, 0, k, "ks moddown")?;
    let lay = ctx.lay;
    if out.ct_stride != lay.ct_words() as u64 || out.part_stride != lay.poly_words() as u64 {
        return Err(shape("ks moddown_to", "the output view must have the ciphertext layout"));
    }
    let lv = plans.level(top)?;
    ntt::transform_k(ctx, s, Dir::Inverse, x, Span::P, Parts::Both, k)?;
    lv.moddown.run_prefix(ctx, s, &x.buf, 0, &mut conv.buf, 0, 2 * k)?;
    let (o, xv, sc, scs) = (out.base, x.buf.vptr(), ctx.ks.p_inv_mod_q.v.vptr(), ctx.ks.p_inv_mod_q.s.vptr());
    let epi = if acc { ntt::Epi::SubMulAcc { out: o, x: xv, s: sc, ss: scs } } else { ntt::Epi::SubMul { out: o, x: xv, s: sc, ss: scs } };
    ntt::forward_k_epi(ctx, s, conv, Span::Q { top }, Parts::Both, k, epi)
}

/// `out += x_0 + .. + x_(n-1)` ModDowned (the extended ciphertexts x_j of `x`, `stride` ciphertexts apart, k each):
/// the special rows' sum rides the INTT's first pass, the Q rows' sum the conversion NTT's epilogue, so the extended
/// rotations of a hoisted sum are never added row by row. `x`'s special rows of its first k cts are left holding the
/// sum (coefficient form), `conv` scratch.
#[allow(clippy::too_many_arguments)]
pub fn moddown_sum_to<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    x: &mut CtBuf<'_, Compute, W>,
    n: usize,
    stride: usize,
    conv: &mut CtBuf<'_, Compute, W>,
    out: View,
    k: usize,
    top: usize,
) -> Result<()> {
    plans.check_k(k)?;
    check_cts(x, 0, (n - 1) * stride + k, "ks moddown_sum")?;
    check_cts(conv, 0, k, "ks moddown_sum")?;
    let lay = ctx.lay;
    if out.ct_stride != lay.ct_words() as u64 || out.part_stride != lay.poly_words() as u64 {
        return Err(shape("ks moddown_sum_to", "the output view must have the ciphertext layout"));
    }
    let lv = plans.level(top)?;
    let words = stride * lay.ct_words();
    let xb = x.buf.vptr();
    let src = ntt::Src { base: xb, soff: std::ptr::null(), n, stride: words };
    ntt::transform_k_ex(ctx, s, Dir::Inverse, x, Span::P, Parts::Both, k, Some(src), None)?;
    lv.moddown.run_prefix(ctx, s, &x.buf, 0, &mut conv.buf, 0, 2 * k)?;
    let epi = ntt::Epi::SubMulAccSum { out: out.base, x: xb, n, stride: words, s: ctx.ks.p_inv_mod_q.v.vptr(), ss: ctx.ks.p_inv_mod_q.s.vptr() };
    ntt::forward_k_epi(ctx, s, conv, Span::Q { top }, Parts::Both, k, epi)
}

/// `dst += (x - t) P^-1` over Q rows 0..=top, both parts, k ciphertexts (`md_epilogue`).
pub(crate) fn md_epilogue<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dst: View,
    x: View,
    t: View,
    k: usize,
    top: usize,
) -> Result<()> {
    let a = MdEpiArgs {
        dst: dst.r(),
        x: x.r(),
        t: t.r(),
        binv: ctx.ks.p_inv_mod_q.v.vptr(),
        binvs: ctx.ks.p_inv_mod_q.s.vptr(),
        k: std::ptr::null(),
        pr: ctx.pr(),
        dbl: 0,
        n: (top + 1) as i32,
        kk: k as i32,
        pad: 0,
    };
    // SAFETY: k-ct views of the context layout; element-wise (dst may alias neither x nor t, callers)
    launch_result("md_epilogue", unsafe { (ctx.table.md_epilogue)(&a, s.raw()) })?;
    Ok(())
}

/// Stage key `r` of `group` for a dot: `a` of digits `0..ndig` expanded from the seed into `ks.regen_a` (when
/// `with_a`), and a packed / banded `b` unpacked / re-slotted into `ks.stage_b` as dense prime-id rows.
#[allow(clippy::too_many_arguments)]
pub fn stage_key<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    plans: &KsPlans<'_>,
    ks: &mut KsWork<'_, W>,
    group: &KeyGroup<'_>,
    r: usize,
    ndig: usize,
    with_a: bool,
) -> Result<()> {
    if with_a && group.tabs.is_some() {
        // the dot reads this key in place (`ks_dot_key`): nothing to stage
        return Ok(());
    }
    if !with_a && group.regen_tab(r)?.is_some() {
        // the relin dot reads the packed / banded `b` in place (`ks_dot_regen` through KeyTab): nothing to stage
        return Ok(());
    }
    if with_a {
        let seed = group.seed(r).ok_or_else(|| shape("stage_key", "key without a seed"))?;
        let buf = ks.regen_a.as_mut().ok_or_else(|| shape("stage_key", "KsWork::carve_regen (section ks.regen_a) missing"))?;
        expand_a(ctx, s, plans, seed, ndig, buf)?;
    }
    let Some(st) = group.staging.get(r).copied().flatten() else { return Ok(()) };
    if ndig > st.ndig {
        return Err(shape("stage_key", format!("key stores {} digits, the level needs {ndig}", st.ndig)));
    }
    let lay = ctx.lay;
    let (n, nq, kp) = (lay.n, lay.nq, lay.kp);
    let srows = st.q_top + 1 + kp;
    let out = ks.stage_b.as_mut().ok_or_else(|| shape("stage_key", "KsWork::carve_regen (section ks.regen_a) missing"))?;
    let ob = out.vptr_mut() as *mut W;
    for d in 0..ndig {
        // Q slots 0..=q_top -> rows 0..=q_top; special slot q_top+1+j -> row nq+j
        for (slot0, row0, nrows) in [(0, 0, st.q_top + 1), (st.q_top + 1, nq, kp)] {
            let dst = ob.wrapping_add((d * lay.rows() + row0) * n) as *mut c_void;
            match st.packed {
                Some((bits, row_words)) => {
                    let a = pc_kernels::KskUnpackArgs {
                        out: dst,
                        inp: (st.b as *const u32).wrapping_add((d * srows + slot0) * row_words),
                        row_words: row_words as u64,
                        nrows: nrows as i32,
                        bits: bits as i32,
                    };
                    // SAFETY: rows inside the key's packed storage and the staging buffer (dnum full-layout polys)
                    launch_result("ksk_unpack", unsafe { (ctx.table.ksk_unpack)(&a, s.raw()) })?;
                }
                None => {
                    let src = (st.b as *const W).wrapping_add((d * srows + slot0) * n) as *mut c_void;
                    let one = |b| View { base: b, ct_stride: 0, part_stride: 0 };
                    let rs = RowSet { nq: nrows as i32, np: 0, qmax: nq as i32, pad: 0 };
                    ew(ctx, s, Ew::Copy, one(dst), one(src), None, rs, 1, 1)?;
                }
            }
        }
    }
    Ok(())
}
