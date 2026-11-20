//! Smooth functional bootstrapping programs: the trigonometric form (f folded into EvalMod's own cosine / sine: Bian et
//! al., ePrint 2026/367; Xiao et al., ePrint 2026/1255; `pc_shadow::evalmod::PsPlan::fbs_trig`, base pairs
//! `pc_shadow::evalmod::trig_base`) and the sine form on the paired Paterson-Stockmeyer builder (`PsPlan::paired`):
//! exact replays through every executor stage and the depth / cost probes.

#[cfg(test)]
mod tests {
    use crate::bts::evalmod::{assign_buffers, rewrite, schedule_relin_rescales, EvalModPlan};
    use pc_shadow::evalmod::{build_evalmod_opts, sim, trig_base, PsPlan};
    use pc_shadow::fides::FMeta;

    fn em_ins() -> (pc_core::scale::ScaleTables, Vec<FMeta>) {
        let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32()).expect("n32");
        let st = pc_core::scale::ScaleTables::precompute(&chain).expect("tables");
        let mut b2 = crate::bts::BtsSetup::n32_shipped();
        b2.raise_drops = vec![0, 1];
        let ems = crate::bts::dense::DensePlan::new(&chain, &b2).expect("dense plan").variants.iter().map(|v| v.em_in).collect();
        (st, ems)
    }

    fn chebval(x: f64, c: &[f64]) -> f64 {
        let (mut b1, mut b2) = (0.0, 0.0);
        for &ck in c.iter().skip(1).rev() {
            let b0 = 2.0 * x * b1 - b2 + ck;
            b2 = b1;
            b1 = b0;
        }
        x * b1 - b2 + c[0] / 2.0
    }

    /// The program on the exact plaintext replay equals P(t) + s Q(t), t = A cos 2 pi u + off, s = A sin 2 pi u, at
    /// every ModRaise integer, through every executor stage.
    #[test]
    fn trig_program_exact() {
        let (st, ems) = em_ins();
        for (bdeg, da, n, amp, off, em) in [(59usize, 3u32, 12usize, 2.0, -1.0, 0usize), (59, 3, 24, 1.0, 0.0, 1), (30, 5, 16, 8.93, -7.93, 0), (22, 6, 8, 1.0, 0.0, 0), (22, 6, 15, 2.0, -1.0, 1)] {
            let (cb, sb) = trig_base(32.0, da, bdeg);
            let g = 2f64.powi(-5);
            let p: Vec<f64> = (0..=n).map(|k| g * 0.8 * 0.7f64.powi(k as i32) * (1.3 + k as f64).cos()).collect();
            let q: Vec<f64> = (0..n).map(|k| g * 0.6 * 0.7f64.powi(k as i32) * (0.4 + k as f64).sin()).collect();
            let ps = PsPlan::fbs_trig(&cb, &sb, da, amp, off, &p, &q).expect("trig plan");
            let (prog, out) = build_evalmod_opts(&st, ems[em], &ps, da as i32, true).expect("program");
            let fused = prog.fused();
            let (rw, rc) = rewrite(fused.clone(), &prog.consts, &st.q);
            let (asg, _) = assign_buffers(schedule_relin_rescales(rw, st.d));
            for (name, ops, consts) in [("raw", &prog.ops, &prog.consts), ("fused", &fused, &prog.consts), ("assigned", &asg, &rc)] {
                let mut emax: f64 = 0.0;
                let cmin = ((-1.0 - off) / amp).max(-1.0);
                let umax = cmin.acos() / (2.0 * std::f64::consts::PI);
                for i in [-31i32, -7, 0, 9, 31] {
                    for j in 0..=10 {
                        let u = umax * (-1.0 + 0.2 * j as f64);
                        let th = 2.0 * std::f64::consts::PI * u;
                        let want = chebval(amp * th.cos() + off, &p) + amp * th.sin() * chebval(amp * th.cos() + off, &q);
                        let y = (u + i as f64) / 32.0;
                        let mut bufs = vec![sim::encode(y * ems[em].nf, &st.q)];
                        sim::run(&st, ops, consts, &mut bufs);
                        let got = sim::decode(&st, &bufs[0], out.top, out.nl, out.nf);
                        emax = emax.max((got - want).abs());
                    }
                }
                println!("trig base {bdeg}/r{da} n {n} {name}: max err {emax:.3e} ({:?} -> {out:?})", ems[em]);
                assert!(emax < 1e-8, "trig base {bdeg} n {n} {name}: {emax:.3e}");
            }
        }
    }

    /// The sine form on the paired Paterson-Stockmeyer builder (`PsPlan::paired`): the exact replay equals the
    /// composition (the K32 table, the double angles at amplitude A, the series) at every executor stage, one
    /// composite level above FIDESlib's program of the same series.
    #[test]
    fn sine_paired_program_exact() {
        let (st, ems) = em_ins();
        let k32 = pc_core::bts_setup::EVALMOD_K32_CHEB.to_vec();
        let da = pc_core::bts_setup::EVALMOD_K32_DA_ITS;
        for (d, amp, c0, em) in [(16usize, 1.0, 0.0, 0usize), (24, 1.41, 0.5, 1), (64, 1.07, 2.0, 0)] {
            let g = 2f64.powi(-6);
            let mut post: Vec<f64> = (0..=d).map(|i| g * 0.3 * 0.97f64.powi(i as i32) * (1.0 + i as f64).sin()).collect();
            post[0] = c0;
            let mut ps = PsPlan::fbs(&k32, da, amp, &post).expect("plan");
            let (_, fides) = build_evalmod_opts(&st, ems[em], &ps, da as i32, true).expect("fides program");
            ps.paired = true;
            let (prog, out) = build_evalmod_opts(&st, ems[em], &ps, da as i32, true).expect("paired program");
            assert_eq!(out.top, fides.top + 2, "degree {d}: paired lands at {}, FIDESlib's at {}", out.top, fides.top);
            let fused = prog.fused();
            let (rw, rc) = rewrite(fused.clone(), &prog.consts, &st.q);
            let (asg, _) = assign_buffers(schedule_relin_rescales(rw, st.d));
            let a = (amp * 2.0 * std::f64::consts::PI).powf(1.0 / (1u64 << da) as f64);
            let bc: Vec<f64> = k32.iter().map(|x| x * a).collect();
            for (name, ops, consts) in [("raw", &prog.ops, &prog.consts), ("assigned", &asg, &rc)] {
                let mut emax: f64 = 0.0;
                for i in [-29i32, 0, 17] {
                    for j in 0..=8 {
                        let u = -0.12 + 0.03 * j as f64;
                        let mut y = chebval((u + i as f64) / 32.0, &bc);
                        for jj in 1..=da as i32 {
                            y = 2.0 * y * y - amp.powf(2f64.powi(jj - da as i32));
                        }
                        let want = chebval(y, &post);
                        let mut bufs = vec![sim::encode((u + i as f64) / 32.0 * ems[em].nf, &st.q)];
                        sim::run(&st, ops, consts, &mut bufs);
                        emax = emax.max((sim::decode(&st, &bufs[0], out.top, out.nl, out.nf) - want).abs());
                    }
                }
                println!("sine paired deg {d} {name}: max err {emax:.3e} (out {out:?}, FIDESlib {fides:?})");
                assert!(emax < 1e-9, "sine paired deg {d} {name}: {emax:.3e}");
            }
        }
    }

    /// Probe (ignored): output levels of the trigonometric and the sine-then-series programs.
    #[test]
    #[ignore]
    fn trig_depth_probe() {
        let (st, ems) = em_ins();
        let em = ems[0];
        let k32 = pc_core::bts_setup::EVALMOD_K32_CHEB.to_vec();
        let lv = |ps: &PsPlan, da: u32| EvalModPlan::<u32>::program(&st, em, ps, da).map(|(p, o, nb)| (o.top, o.nl, p.ops.len(), nb)).map_err(|e| e.to_string());
        println!("em_in {em:?}");
        for d in [8usize, 16, 24, 32, 48, 64, 96, 128, 192, 256] {
            let post: Vec<f64> = (0..=d).map(|i| 0.01 / (1.0 + i as f64)).collect();
            let ps = PsPlan::fbs(&k32, 6, 1.0, &post).expect("plan");
            println!("sine k32/r6 post {d}: {:?}", lv(&ps, 6));
        }
        for (bdeg, da) in [(22usize, 6u32), (23, 5), (27, 5), (29, 5), (31, 5), (39, 4), (59, 3)] {
            let (cb, sb) = trig_base(32.0, da, bdeg);
            let mut emax: f64 = 0.0;
            for i in 0..=4000 {
                let y = -1.0 + i as f64 / 2000.0;
                let (mut c, mut s) = (chebval(y, &cb), chebval(y, &sb));
                for _ in 0..da {
                    (c, s) = (2.0 * c * c - 1.0, 2.0 * s * c);
                }
                let th = 2.0 * std::f64::consts::PI * 32.0 * y;
                emax = emax.max((c - th.cos()).abs()).max((s - th.sin()).abs());
            }
            for n in [0usize, 8, 16, 32] {
                let p: Vec<f64> = (0..=n.max(2)).map(|i| 0.01 / (1.0 + i as f64)).collect();
                let q: Vec<f64> = if n == 0 { vec![] } else { (0..n).map(|i| 0.01 / (2.0 + i as f64)).collect() };
                let ps = PsPlan::fbs_trig(&cb, &sb, da, 1.0, 0.0, &p, &q).expect("plan");
                let c = EvalModPlan::<u32>::program(&st, em, &ps, da).map(|(p, _, _)| pc_shadow::evalmod::op_cost(&p.ops)).unwrap_or(0.0);
                println!("trig base {bdeg}/r{da} (pair err {emax:.1e}) n {n}: {:?} cost {c:.0}", lv(&ps, da));
            }
        }
    }
}

#[cfg(test)]
mod probe {
    use crate::bts::evalmod::EvalModPlan;
    use pc_shadow::evalmod::{trig_base, EmOp, PsPlan};

    /// Probe (ignored): op counts by kind and limbs of the sine and trigonometric programs.
    #[test]
    #[ignore]
    fn trig_op_census() {
        let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32()).expect("n32");
        let st = pc_core::scale::ScaleTables::precompute(&chain).expect("tables");
        let mut b2 = crate::bts::BtsSetup::n32_shipped();
        b2.raise_drops = vec![0];
        let em = crate::bts::dense::DensePlan::new(&chain, &b2).expect("dense plan").variants[0].em_in;
        let census = |name: &str, ps: &PsPlan, da: u32| {
            let (p, o, nb) = EvalModPlan::<u32>::program(&st, em, ps, da).expect("program");
            let mut m: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
            for op in &p.ops {
                let (k, n) = match op {
                    EmOp::Copy { n, .. } => ("copy", *n),
                    EmOp::Add { n, .. } => ("add", *n),
                    EmOp::MulConst { n, .. } => ("mulconst", *n),
                    EmOp::AddConst { n, .. } => ("addconst", *n),
                    EmOp::Rescale { n, .. } => ("rescale", *n),
                    EmOp::Mult { n, .. } => ("mult", *n),
                    EmOp::MultAcc { n, pairs, .. } => ("multacc", *n * pairs.len()),
                    EmOp::WSum { n, srcs, .. } => ("wsum", *n * srcs.len()),
                };
                let e = m.entry(k).or_default();
                e.0 += 1;
                e.1 += n;
            }
            println!("{name}: out top {} bufs {nb} cost {:.0}: {m:?}", o.top, pc_shadow::evalmod::op_cost(&p.ops));
        };
        let k32 = pc_core::bts_setup::EVALMOD_K32_CHEB.to_vec();
        let post: Vec<f64> = (0..=24).map(|i| 0.01 / (1.0 + i as f64)).collect();
        census("sine k32 post24", &PsPlan::fbs(&k32, 6, 1.0, &post).unwrap(), 6);
        census("sine k32 post4", &PsPlan::fbs(&k32, 6, 1.0, &post[..5]).unwrap(), 6);
        for d in [4usize, 8, 16, 24, 32, 48, 64, 96, 128, 256] {
            let pp: Vec<f64> = (0..=d).map(|i| 0.01 / (1.0 + i as f64)).collect();
            let ps = PsPlan::fbs(&k32, 6, 1.0, &pp).unwrap();
            let (_, o, _) = EvalModPlan::<u32>::program(&st, em, &ps, 6).unwrap();
            let (p2, o2) = pc_shadow::evalmod::build_sine_paired(&st, em, &ps, 6).unwrap();
            let (_, o3, _) = EvalModPlan::<u32>::program(&st, em, &PsPlan::fbs_trig(&trig_base(32.0, 6, 22).0, &trig_base(32.0, 6, 22).1, 6, 1.0, 0.0, &pp[..=(d / 2).max(2)], &pp[..(d / 2).max(2)]).unwrap(), 6).unwrap();
            println!("post {d}: sine fides top {} | sine paired top {} cost {:.0} | trig n {} top {}", o.top, o2.top, pc_shadow::evalmod::op_cost(&p2.ops), (d / 2).max(2), o3.top);
        }
        let (cb, sb) = trig_base(32.0, 6, 22);
        let p: Vec<f64> = (0..=12).map(|i| 0.01 / (1.0 + i as f64)).collect();
        let _ = (cb, sb);
        for (bd, r) in [(14usize, 7u32), (18, 7), (22, 6), (26, 6), (30, 5), (40, 4), (59, 3)] {
            let (cb, sb) = trig_base(32.0, r, bd);
            census(&format!("trig {bd}/{r} n12"), &PsPlan::fbs_trig(&cb, &sb, r, 2.0, -1.0, &p, &p[..12]).unwrap(), r);
            census(&format!("trig {bd}/{r} n2"), &PsPlan::fbs_trig(&cb, &sb, r, 2.0, -1.0, &p[..3], &p[..2]).unwrap(), r);
        }
    }
}
