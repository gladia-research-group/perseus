//! Encoding into device plaintext rows (seam `Encoder`; the payload kind and the slot count are arguments, never a
//! global scope): `HostFft` (`pc_core::encode`, exact i128 rounding, rows uploaded), `DeviceFft` (the `encode`
//! kernel, FIDESlib encodeOnDevice: Stockham FFT in double, llround, 2^61 split), `CoeffRows` (host-scaled
//! coefficients, no FFT), `CoeffRowsDevice` (float32 coefficient rows of a streamed weight expanded inside their
//! consumer, `CoeffRowsDevice::sum`, FIDESlib multCoeffRowsSum).
//!
//! Every path ends with the rows `0..=top` (prime id = row) in a `[row][N]` device buffer, NTT'd unless the caller
//! asks for coefficient form. Host and device FFT differ only where the two FFTs' double errors (<= ~1e-5
//! coefficient units) straddle a .5 boundary (one coefficient off by 1); `pc-check leaf --ops encode` counts them.
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::ntt::{self, Dir, RowList};
use crate::ops::ConstRing;
use crate::poly::Layout;
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::modmul::mulmod;
use pc_cuda::{launch_result, Arena, Compute, DBuf, Device, Event, Lane, PinnedBuf, Stream};
use pc_kernels::{CoeffArgs, EncodeArgs, COEFF_MAXL, COEFF_MAXT};
use pc_shadow::meta::top_limb;
use pc_shadow::{CtMeta, PtMeta, Shadow};
use std::ffi::c_void;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderKind {
    HostFft,
    DeviceFft,
    CoeffRows,
    /// Device expansion of float32 coefficient rows, fused into the weight sum (`CoeffRowsDevice`).
    CoeffRowsDevice,
}

/// Output form of encoded rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PtForm {
    /// NTT domain (what ciphertext products read).
    Eval,
    /// Coefficient domain.
    Coeff,
}

/// What to encode: slot values (`HostFft`, `DeviceFft`) or coefficient values (`CoeffRows`).
#[derive(Clone, Copy, Debug)]
pub enum Payload2<'v> {
    Slots { values: &'v [Complex], slots: usize, payload: Payload },
    Coeffs(&'v [f64]),
}

/// Encoding parameters: rows `0..=top`, `scale` (the level's scaling factor for degree 1), degree (degree 2 rows
/// are multiplied by `round(scale)` mod p, FIDESlib's `deg_factor`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodeAt {
    pub top: usize,
    pub scale: f64,
    pub deg: u8,
    pub form: PtForm,
}

/// Device workspace of the device FFT and the row lists every encoder NTTs through.
pub struct DevEncoder<'a, W: DevWord> {
    lay: Layout,
    /// N/2 complex slots (re, im) as 2 doubles each.
    v: DBuf<'a, Compute, f64>,
    a: DBuf<'a, Compute, f64>,
    b: DBuf<'a, Compute, f64>,
    c: DBuf<'a, Compute, f64>,
    cmax: DBuf<'a, Compute, u64>,
    pin: PinnedBuf<f64>,
    pin_ev: Event,
    pin_used: bool,
    /// host-encoded rows staging (eager HostFft / CoeffRows): one plaintext of words
    pin_w: PinnedBuf<W>,
    pin_w_ev: Event,
    pin_w_used: bool,
    /// `lists[top]`: rows `(l, l)` for `l <= top` of a `[row][N]` buffer
    lists: Vec<RowList<'a>>,
    host: Encoder,
    primes: Vec<u64>,
}

impl<'a, W: DevWord> DevEncoder<'a, W> {
    /// Device bytes (`MemPlan` section "encode").
    pub fn bytes(lay: &Layout) -> usize {
        let n = lay.n;
        let f = |k: usize| Arena::bytes_for::<f64>(k);
        let lists: usize = (0..lay.nq).map(|t| Arena::bytes_for::<u32>(t + 1) + Arena::bytes_for::<i32>(t + 1)).sum();
        f(n) + 3 * f(2 * n) + Arena::bytes_for::<u64>(1) + lists
    }

    pub fn new<S: Lane, M: DevStrategy<W>>(arena: &'a Arena, dev: &Device, s: &Stream<S>, ctx: &Context<'_, W, M>) -> Result<Self> {
        let lay = ctx.lay;
        let n = lay.n;
        let lists = (0..lay.nq)
            .map(|t| RowList::new(arena, s, &(0..=t).map(|l| (l as u32, l as i32)).collect::<Vec<_>>()))
            .collect::<Result<Vec<_>>>()?;
        Ok(DevEncoder {
            lay,
            v: arena.alloc("encode.v", n)?,
            a: arena.alloc("encode.a", 2 * n)?,
            b: arena.alloc("encode.b", 2 * n)?,
            c: arena.alloc("encode.c", 2 * n)?,
            cmax: arena.alloc("encode.cmax", 1)?,
            pin: PinnedBuf::new(dev, n)?,
            pin_ev: Event::new(dev)?,
            pin_used: false,
            pin_w: PinnedBuf::new(dev, lay.nq * n)?,
            pin_w_ev: Event::new(dev)?,
            pin_w_used: false,
            lists,
            host: Encoder::new(&ctx.chain),
            primes: ctx.chain.primes(),
        })
    }

    /// The host encoder (pc-core) this device encoder compares against and stages with.
    pub fn host(&self) -> &Encoder {
        &self.host
    }

    fn check(&self, at: &EncodeAt, out: &DBuf<'_, Compute, W>) -> Result<()> {
        if at.top >= self.lay.nq {
            return Err(shape("encode", format!("top limb {} beyond the chain", at.top)));
        }
        if out.len() < (at.top + 1) * self.lay.n {
            return Err(shape("encode", format!("output holds {} words, rows 0..={} need {}", out.len(), at.top, (at.top + 1) * self.lay.n)));
        }
        if !(1..=2).contains(&at.deg) {
            return Err(shape("encode", format!("degree {} (1 or 2)", at.deg)));
        }
        Ok(())
    }

    /// Encode with `kind` into `out` (rows `0..=at.top`, `[row][N]`), on the compute lane.
    pub fn encode<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        ring: &mut ConstRing<'_, W>,
        kind: EncoderKind,
        what: Payload2<'_>,
        at: EncodeAt,
        out: &mut DBuf<'_, Compute, W>,
    ) -> Result<()> {
        self.check(&at, out)?;
        match (kind, what) {
            (EncoderKind::DeviceFft, Payload2::Slots { values, slots, payload }) => self.device_fft(ctx, s, ring, values, slots, payload, &at, out)?,
            (EncoderKind::HostFft, Payload2::Slots { .. }) | (EncoderKind::CoeffRows, Payload2::Coeffs(_)) => {
                let rows = self.host_rows(what, &at)?;
                self.upload_rows(s, &rows, at.top, out)?;
            }
            (k, _) => return Err(shape("encode", format!("{k:?} does not take this payload"))),
        }
        if at.form == PtForm::Eval {
            ntt::transform_rows(ctx, s, Dir::Forward, out, &self.lists[at.top])?;
        }
        Ok(())
    }

    /// Host rows (coefficient form, words already reduced) of `what` at `at` (the staging worker's encoder).
    pub fn host_rows(&self, what: Payload2<'_>, at: &EncodeAt) -> Result<Vec<Vec<u64>>> {
        host_rows(&self.host, &self.primes, what, at)
    }

    fn upload_rows(&mut self, s: &Stream<Compute>, rows: &[Vec<u64>], top: usize, out: &mut DBuf<'_, Compute, W>) -> Result<()> {
        let n = self.lay.n;
        if self.pin_w_used {
            self.pin_w_ev.sync()?;
        }
        {
            let h = self.pin_w.as_mut_slice();
            for (l, r) in rows.iter().enumerate().take(top + 1) {
                for (o, &x) in h[l * n..(l + 1) * n].iter_mut().zip(r) {
                    *o = W::from_u64(x);
                }
            }
        }
        // SAFETY: the pinned buffer is not rewritten before pin_w_ev (recorded after this copy) completes; the copy
        // covers the whole buffer, rows above `top` land in `out` only when it is large enough
        if out.len() >= self.pin_w.len() {
            unsafe { out.upload_pinned_async(s, 0, &self.pin_w)? };
        } else {
            // a buffer holding exactly rows 0..=top: synchronous copy of the live part
            let words = (top + 1) * n;
            let v: Vec<W> = self.pin_w.as_slice()[..words].to_vec();
            out.upload_at(s, 0, &v)?;
        }
        self.pin_w_ev.record(s)?;
        self.pin_w_used = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn device_fft<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        ring: &mut ConstRing<'_, W>,
        values: &[Complex],
        slots: usize,
        payload: Payload,
        at: &EncodeAt,
        out: &mut DBuf<'_, Compute, W>,
    ) -> Result<()> {
        let nh = self.lay.n / 2;
        if slots == 0 || !slots.is_power_of_two() || slots > nh || values.len() > slots {
            return Err(shape("encode", format!("{} values in {slots} slots (power of two <= {nh})", values.len())));
        }
        if self.pin_used {
            self.pin_ev.sync()?;
        }
        {
            // a sparse vector of `slots` values is the dense encoding of its periodic extension
            let h = self.pin.as_mut_slice();
            for j in 0..nh {
                let x = values.get(j % slots).copied().unwrap_or_default();
                h[2 * j] = x.re;
                h[2 * j + 1] = if payload == Payload::Real { 0.0 } else { x.im };
            }
        }
        // SAFETY: as in upload_rows (pin_ev)
        unsafe { self.v.upload_pinned_async(s, 0, &self.pin)? };
        self.pin_ev.record(s)?;
        self.pin_used = true;
        self.fft_v(ctx, s, ring, at, out)
    }

    /// The slot values of a plaintext as the device FFT takes them: N/2 (re, im) pairs, a shorter vector extended
    /// periodically, the imaginary parts zeroed on a real payload.
    pub fn fill_slots(h: &mut [f64], values: &[Complex], slots: usize, payload: Payload) {
        for j in 0..h.len() / 2 {
            let x = values.get(j % slots).copied().unwrap_or_default();
            h[2 * j] = x.re;
            h[2 * j + 1] = if payload == Payload::Real { 0.0 } else { x.im };
        }
    }

    /// `encode` by the device FFT from slot values the caller keeps in pinned memory (`fill_slots`, N doubles): no
    /// host copy and no wait on the lane.
    ///
    /// # Safety
    /// `src` must stay alive and unwritten until the lane has passed this encode.
    pub unsafe fn encode_pinned<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, ring: &mut ConstRing<'_, W>, src: &PinnedBuf<f64>, at: EncodeAt, out: &mut DBuf<'_, Compute, W>) -> Result<()> {
        self.check(&at, out)?;
        if src.len() != self.lay.n {
            return Err(shape("encode", format!("{} pinned doubles, the device FFT takes {}", src.len(), self.lay.n)));
        }
        self.v.upload_pinned_async(s, 0, src)?;
        self.fft_v(ctx, s, ring, &at, out)?;
        if at.form == PtForm::Eval {
            ntt::transform_rows(ctx, s, Dir::Forward, out, &self.lists[at.top])?;
        }
        Ok(())
    }

    /// The device FFT of the slot values in `v` into `out` (coefficient form).
    fn fft_v<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, ring: &mut ConstRing<'_, W>, at: &EncodeAt, out: &mut DBuf<'_, Compute, W>) -> Result<()> {
        let df: Vec<u64> = self.primes[..=at.top]
            .iter()
            .map(|&p| if at.deg == 2 { (at.scale.round() as u128 % p as u128) as u64 } else { 1 % p })
            .collect();
        let deg_factor = ring.push_plain(s, &df)?;
        let args = EncodeArgs {
            v: self.v.vptr(),
            a: self.a.vptr_mut(),
            b: self.b.vptr_mut(),
            c: self.c.vptr_mut(),
            cmax: self.cmax.dptr_mut(),
            out: out.vptr_mut(),
            deg_factor,
            pr: ctx.pr(),
            scale: at.scale,
            nl: (at.top + 1) as i32,
            pad: 0,
        };
        // SAFETY: workspaces sized by `bytes` (v N doubles, a/b 2N doubles = N double2, c >= N doubles, cmax 1);
        // out holds rows 0..=top (checked)
        launch_result("encode", unsafe { (ctx.table.encode)(&args, s.raw()) })?;
        Ok(())
    }
}

/// Host encoding of `what` at `at`: coefficient-form rows `0..=top` (degree 2 multiplied by round(scale) mod p).
pub fn host_rows(enc: &Encoder, primes: &[u64], what: Payload2<'_>, at: &EncodeAt) -> Result<Vec<Vec<u64>>> {
    let mut rows = match what {
        Payload2::Slots { values, slots, payload } => enc.encode_slots(values, slots, at.top, payload, at.scale)?,
        Payload2::Coeffs(c) => enc.encode_coeff_rows(c, at.top, at.scale)?,
    };
    if at.deg == 2 {
        for (l, r) in rows.iter_mut().enumerate() {
            let p = primes[l];
            let f = (at.scale.round() as u128 % p as u128) as u64;
            r.iter_mut().for_each(|x| *x = mulmod(*x, f, p));
        }
    }
    Ok(rows)
}

/// Raw device pointer of a plaintext buffer (kernel argument form).
pub fn rows_ptr<W: DevWord, S: Lane>(b: &DBuf<'_, S, W>) -> *const c_void {
    b.vptr()
}

/// The device coefficient-row expander (seam `Encoder`, impl `CoeffRowsDevice`), fused into its consumer: `sum`
/// computes `out = sum_r cts[r] * pt_r` where `pt_r[k] = round(rows[r][k] * scale)` at the ciphertexts' level (the
/// scale a plaintext encoded like the ciphertext carries: `PtStore::encode_like`), in chunks of `COEFF_MAXT` terms
/// (two launches each: expansion + NTT column pass into the scratch, NTT row pass + products + sum in registers).
/// Rows are device float32 arrays of N coefficients (a weight frame the copy lane filled). Bookkeeping = the
/// loop's: `mult_pt` of the first term, `add` of the others (all at one level, degree 1: no preludes).
pub struct CoeffRowsDevice<'a, W: DevWord> {
    scratch: DBuf<'a, Compute, W>,
    max_rows: usize,
    overflow: DBuf<'a, Compute, u64>,
    /// Terms per launch pair (<= COEFF_MAXT): small chunks keep the expanded rows (chunk x limbs x N words) in L2
    /// between the two kernels.
    pub chunk: usize,
}

impl<'a, W: DevWord> CoeffRowsDevice<'a, W> {
    /// Device bytes (MemPlan section "encode.coeff"): the scratch of one chunk at `max_rows` limbs + the counter.
    pub fn bytes(lay: &Layout, max_rows: usize) -> usize {
        Arena::bytes_for::<W>(COEFF_MAXT * max_rows.max(1) * lay.n) + Arena::bytes_for::<u64>(1)
    }

    pub fn new(arena: &'a Arena, s: &Stream<Compute>, lay: Layout, max_rows: usize) -> Result<Self> {
        if max_rows == 0 || max_rows > COEFF_MAXL.min(lay.nq) {
            return Err(shape("CoeffRowsDevice", format!("max_rows {max_rows} (1..={})", COEFF_MAXL.min(lay.nq))));
        }
        let mut overflow = arena.alloc("encode.coeff.overflow", 1)?;
        overflow.zero(s)?;
        Ok(CoeffRowsDevice { scratch: arena.alloc("encode.coeff", COEFF_MAXT * max_rows * lay.n)?, max_rows, overflow, chunk: COEFF_MAXT })
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// The output metadata of the sum over `ins` and the expansion scale (the plaintext's), as the loop records it.
    pub fn sum_meta(sh: &Shadow, ins: &[CtMeta]) -> Result<(CtMeta, f64)> {
        let m0 = *ins.first().ok_or_else(|| shape("coeff_sum", "no terms"))?;
        let st = sh.tables.as_ref().ok_or_else(|| shape("coeff_sum", "the Shadow has no scale tables"))?;
        let level = m0.effective_level(sh.unit());
        let top = top_limb(sh.cfg.chain, level)? as i32;
        let scale = st.sf(top)?;
        let pm = PtMeta { level, deg: 1, nf: scale, tag: Default::default() };
        let mut acc: Option<CtMeta> = None;
        for (i, m) in ins.iter().enumerate() {
            if (m.level, m.deg) != (m0.level, 1) || m.nf != m0.nf {
                return Err(shape("coeff_sum", format!("term {i} at level {} deg {} nf 2^{:.3}; term 0 at level {} deg {} nf 2^{:.3} (one level, degree 1)", m.level, m.deg, m.nf.log2(), m0.level, m0.deg, m0.nf.log2())));
            }
            let r = sh.mult_pt(m, &pm)?;
            if !r.pre.is_empty() {
                return Err(shape("coeff_sum", format!("term {i}: the product needs preludes {:?}", r.pre)));
            }
            acc = Some(match acc {
                None => r.out,
                Some(a) => {
                    let s = sh.add(&a, &r.out)?;
                    if !s.pre.is_empty() {
                        return Err(shape("coeff_sum", format!("term {i}: the sum needs preludes {:?}", s.pre)));
                    }
                    s.out
                }
            });
        }
        Ok((acc.ok_or_else(|| shape("coeff_sum", "no terms"))?, scale))
    }

    /// `out = sum_r cts[r] * pt_r` (see the type). `out` must not be one of `cts`. Asynchronous on `s`.
    #[allow(clippy::too_many_arguments)]
    pub fn sum<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sh: &Shadow, out: &mut Ct<'_, W>, cts: &[&Ct<'_, W>], rows: &[*const f32]) -> Result<()> {
        if cts.len() != rows.len() || cts.is_empty() {
            return Err(shape("coeff_sum", format!("{} ciphertexts, {} rows", cts.len(), rows.len())));
        }
        let metas: Vec<CtMeta> = cts.iter().map(|c| c.meta).collect();
        let (meta, scale) = Self::sum_meta(sh, &metas)?;
        let l = top_limb(ctx.shape, metas[0].level)? as usize + 1;
        if l > self.max_rows {
            return Err(shape("coeff_sum", format!("{l} limbs, the scratch holds {} (CoeffRowsDevice::max_rows)", self.max_rows)));
        }
        let ob = out.buf.w().base;
        if cts.iter().any(|c| c.buf.r().base == ob) || rows.iter().any(|r| r.is_null()) {
            return Err(shape("coeff_sum", "the output aliases a term, or a row pointer is null"));
        }
        let ch = self.chunk.clamp(1, COEFF_MAXT);
        for (j, (cc, rc)) in cts.chunks(ch).zip(rows.chunks(ch)).enumerate() {
            let mut a = CoeffArgs {
                scratch: self.scratch.vptr_mut(),
                out: ob,
                fw: ctx.ntt.fw.vptr(),
                fws: ctx.ntt.fws.vptr(),
                pr: ctx.pr(),
                overflow: self.overflow.dptr_mut(),
                part_stride: ctx.lay.poly_words() as u64,
                scale,
                n: cc.len() as i32,
                l: l as i32,
                accumulate: (j > 0) as i32,
                pad: 0,
                rows: [std::ptr::null(); COEFF_MAXT],
                ct: [std::ptr::null(); COEFF_MAXT],
            };
            for (i, (c, r)) in cc.iter().zip(rc).enumerate() {
                a.ct[i] = c.buf.r().base as *const c_void;
                a.rows[i] = *r;
            }
            // SAFETY: scratch holds COEFF_MAXT x max_rows >= n x l rows; every term ciphertext and the output are
            // full-layout single ciphertexts (rows 0..l of both parts); the rows point at N device floats each
            launch_result("coeff_sum", unsafe { (ctx.table.coeff_sum)(&a, s.raw()) })?;
        }
        out.meta = meta;
        Ok(())
    }

    /// Coefficients clamped since setup (nonzero = a row past the int64 range at its scale: a wrong result).
    /// Synchronous on `s`.
    pub fn overflows(&self, s: &Stream<Compute>) -> Result<u64> {
        Ok(self.overflow.download(s)?.first().copied().unwrap_or(0))
    }
}
