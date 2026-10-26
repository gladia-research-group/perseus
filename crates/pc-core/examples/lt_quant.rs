//! B1 gate experiment (CPU): the error a CtS / StC linear transform picks up when its diagonal plaintexts are encoded
//! at a thin scale Delta (one ~30-bit prime per level) instead of the composite 2^54.
//!
//! cargo run --release -p pc-core --example lt_quant -- [log2 Delta ...]
use pc_core::bts_setup::{cts_stc_factors, FftStage, C64};
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::params::{Chain, ChainSpec};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.unif().max(1e-300), self.unif());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn rms(v: &[C64]) -> f64 {
    (v.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>() / v.len() as f64).sqrt()
}

fn diff_rms(a: &[C64], b: &[C64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x.re - y.re).powi(2) + (x.im - y.im).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

/// The diagonal as it decodes after an encoding at `scale` (integer coefficients, then the special FFT).
fn quantize(enc: &Encoder, v: &[C64], scale: f64) -> (Vec<C64>, usize) {
    let c: Vec<Complex> = v.iter().map(|z| Complex::new(z.re, z.im)).collect();
    let co = enc.slot_coeffs(&c, v.len(), Payload::Complex, scale).unwrap();
    let nz = co.iter().filter(|&&x| x != 0).count();
    let h = enc.ring_dim() / 2;
    let mut z: Vec<Complex> = (0..h).map(|i| Complex::new(co[i] as f64 / scale, co[i + h] as f64 / scale)).collect();
    enc.fft_special(&mut z);
    (z.iter().map(|x| C64::new(x.re, x.im)).collect(), nz)
}

fn qstage(enc: &Encoder, st: &FftStage, scale: f64) -> (FftStage, f64, f64, usize) {
    let mut q = st.clone();
    let (mut err2, mut ent2, mut nzmax) = (0.0, 0.0, 0usize);
    for (d, v) in st.diags.iter().enumerate() {
        let (qv, nz) = quantize(enc, v, scale);
        err2 += diff_rms(&qv, v).powi(2);
        ent2 += rms(v).powi(2);
        nzmax = nzmax.max(nz);
        q.diags[d] = qv;
    }
    let nd = st.diags.len() as f64;
    (q, (err2 / nd).sqrt(), (ent2 / nd).sqrt(), nzmax)
}

fn main() {
    let args: Vec<f64> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let logs = if args.is_empty() { vec![29.9, 31.0, 33.0, 36.0, 54.0] } else { args };
    let chain = Chain::generate(&ChainSpec::n32()).unwrap();
    let enc = Encoder::new(&chain);
    let n = chain.n;
    let h = n / 2;
    let f = cts_stc_factors(n, h as u32, [4, 3], [0, 0], 1.0, 1.0).unwrap();

    // CtS input: the slots of a raised polynomial t = I + m (I: overflow integers, rms ~1.2 as measured by L7; m tiny)
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let coeffs: Vec<f64> = (0..n).map(|_| (1.2 * rng.gauss()).round() + 1e-4 * rng.gauss()).collect();
    let mut z: Vec<Complex> = (0..h).map(|i| Complex::new(coeffs[i], coeffs[i + h])).collect();
    enc.fft_special(&mut z);
    let x: Vec<C64> = z.iter().map(|c| C64::new(c.re, c.im)).collect();
    let run_cts = |stages: &[FftStage]| {
        let mut y = x.clone();
        for s in (0..stages.len()).rev() {
            y = stages[s].apply(&y);
        }
        y
    };
    let y0 = run_cts(&f.cts);
    // the unit: CtS output per input coefficient (rms ratio)
    let coef_rms = (coeffs.iter().map(|c| c * c).sum::<f64>() / n as f64).sqrt();
    println!("CtS exact output rms {:.4e} (input coefficient rms {:.3}): output units per q0 unit {:.4e}", rms(&y0), coef_rms, rms(&y0) / (coef_rms * 2f64.sqrt()));

    // StC input: a message vector (slots of the coefficient domain), relative error only
    let xm: Vec<C64> = (0..h).map(|_| C64::new(rng.gauss(), rng.gauss())).collect();
    let run_stc = |stages: &[FftStage]| {
        let mut y = xm.clone();
        for st in stages {
            y = st.apply(&y);
        }
        y
    };
    let w0 = run_stc(&f.stc);

    for &lg in &logs {
        let scale = lg.exp2();
        println!("== Delta = 2^{lg}");
        let mut qcts = Vec::new();
        for st in &f.cts {
            let (q, e, ent, nz) = qstage(&enc, st, scale);
            println!("  CtS s={} stride {} diags {} entry rms {:.3} diag err rms {:.3e} (= 2^{:.2}, x Delta {:.1}) nonzero coeffs <= {}", st.s, st.stride, st.diags.len(), ent, e, e.log2(), e * scale, nz);
            qcts.push(q);
        }
        let y = run_cts(&qcts);
        let rel = diff_rms(&y, &y0) / rms(&y0);
        println!("  CtS output rel err {:.3e} = 2^{:.2}; in q0 units (per coefficient) 2^{:.2}", rel, rel.log2(), (rel * coef_rms).log2());
        // per-stage contributions: quantize one stage at a time
        for k in 0..f.cts.len() {
            let mut one = f.cts.clone();
            one[k] = qcts[k].clone();
            let yk = run_cts(&one);
            let r = diff_rms(&yk, &y0) / rms(&y0);
            println!("    only stage s={}: rel 2^{:.2}", f.cts[k].s, r.log2());
        }
        let mut qstc = Vec::new();
        for st in &f.stc {
            let (q, e, ent, nz) = qstage(&enc, st, scale);
            println!("  StC s={} stride {} diags {} entry rms {:.3} diag err rms 2^{:.2} nonzero coeffs <= {}", st.s, st.stride, st.diags.len(), ent, e.log2(), nz);
            qstc.push(q);
        }
        let w = run_stc(&qstc);
        let relw = diff_rms(&w, &w0) / rms(&w0);
        println!("  StC output rel err 2^{:.2}", relw.log2());
    }
}
