//! The FLEXIBLEAUTO engine against the reference finish dump d3 (FIDESlib's own run):
//!   * EvalMod memo gate: every constant the EvalMod program builder asks for is in FIDESlib's ElemForEvalMult memo
//!     and equal, and the program's output metadata equals the dumped EvalMod output (level, degree, scale bits);
//!   * CtS stage rule: the per-stage encoding levels and scales equal the dumped plaintexts, and the Shadow's
//!     mult_pt + rescale over them reproduces every dumped stage output (level, degree, scale bits).
use pc_core::params::{Chain, ChainSpec};
use pc_core::scale::ScaleTables;
use pc_shadow::evalmod::{build_evalmod, EmOp, PsPlan};
use pc_shadow::fides::FMeta;
use pc_shadow::meta::{level_from_top_limb, ChainShape, CtMeta};
use pc_shadow::route::{cts_stage_levels, stc_stage_levels, BtsConfig};
use pc_shadow::rules::{PtMeta, Shadow, ShadowConfig};
use pc_shadow::Quirks;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

const D3: &str = "/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/d3";

fn load(name: &str) -> Option<Value> {
    let path = format!("{D3}/{name}");
    match std::fs::read(&path) {
        Ok(s) => Some(serde_json::from_slice(&s).expect("dump json")),
        Err(_) => {
            println!("SKIP: {path} not readable");
            None
        }
    }
}

fn tables() -> ScaleTables {
    ScaleTables::precompute(&Chain::generate(&ChainSpec::n32()).unwrap()).unwrap()
}

fn fmeta(j: &Value) -> FMeta {
    FMeta {
        top: j["level"].as_i64().unwrap() as i32,
        nl: j["noise_level"].as_i64().unwrap() as i32,
        nf: j["noise_factor"].as_f64().unwrap(),
    }
}

fn f64v(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn evalmod_memo_gate() {
    let st = tables();
    let (Some(p), Some(memo), Some(cts3), Some(em)) =
        (load("params.json"), load("elem_memo.json"), load("it0.cts3.json"), load("it0.em.json"))
    else {
        return;
    };
    let ps_j = &p["cheb_ps"];
    let ps = PsPlan {
        k: ps_j["k"].as_u64().unwrap() as usize,
        m: ps_j["m"].as_u64().unwrap() as usize,
        divqr_q: f64v(&ps_j["divqr_q"]),
        divcs_q: f64v(&ps_j["divcs_q"]),
        s2: f64v(&ps_j["s2"]),
        coeffs: pc_core::bts_setup::evalmod_coeffs().to_vec(),
        even_shift: None,
    };
    // the computed plan (pc_core::cheb long divisions) must equal the dump's, field for field
    assert_eq!(PsPlan::evalmod(), ps, "pc_core::cheb PS plan != d3 cheb_ps");
    let ps = PsPlan::evalmod();
    let its = p["double_angle_its"].as_i64().unwrap() as i32;
    let memo: HashMap<(i32, i32, u64), Vec<u64>> = memo
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let key = (
                e["level"].as_i64().unwrap() as i32,
                e["level_in"].as_i64().unwrap() as i32,
                e["operand_bits"].as_u64().unwrap(),
            );
            (key, e["res"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect())
        })
        .collect();
    let (prog, out) = build_evalmod(&st, fmeta(&cts3), &ps, its).unwrap();
    let mut missing = 0;
    let mut unequal = 0;
    for &(l, li, c) in &prog.reqs {
        match memo.get(&(l, li, c.to_bits())) {
            None => {
                missing += 1;
                println!("  request NOT in memo: level {l} level_in {li} operand {c:.17e}");
            }
            Some(want) => {
                if &st.elem_for_eval_mult(l, c, li).unwrap() != want {
                    unequal += 1;
                }
            }
        }
    }
    let relins = prog.ops.iter().filter(|o| matches!(o, EmOp::Mult { .. } | EmOp::MultAcc { .. })).count();
    let fused = prog.fused().len();
    println!(
        "[evalmod] {} ops ({fused} fused, {relins} relins), {} constant requests: {missing} not in the memo, {unequal} unequal; out top {} nl {} nf {:e} (dump {} {} {:e})",
        prog.ops.len(),
        prog.reqs.len(),
        out.top,
        out.nl,
        out.nf,
        em["level"],
        em["noise_level"],
        em["noise_factor"].as_f64().unwrap()
    );
    assert!(!prog.reqs.is_empty());
    assert_eq!((missing, unequal), (0, 0));
    let want = fmeta(&em);
    assert_eq!((out.top, out.nl), (want.top, want.nl));
    assert_eq!(out.nf.to_bits(), want.nf.to_bits(), "EvalMod output scale {} vs dump {}", out.nf, want.nf);
}

#[test]
fn cts_stages_match_d3() {
    let st = Arc::new(tables());
    let (Some(p), Some(raise)) = (load("params.json"), load("it0.raise.json")) else { return };
    let boot = &p["boot"];
    let raise_drop = boot["raise_drop"].as_u64().unwrap() as u32;
    let t = boot["cts0_t"].as_u64().unwrap() as u32;
    let cts = boot["CtS"].as_array().unwrap();
    let stages = cts_stage_levels(&st, cts.len() as u32, raise_drop, Some(t)).unwrap();
    let chain = ChainShape::from_tables(&st);
    let shadow = Shadow::new(ShadowConfig { chain, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(st.clone()));
    let r = fmeta(&raise);
    let mut ct = CtMeta { level: level_from_top_limb(chain, r.top as u32).unwrap(), deg: r.nl as u8, nf: r.nf, slots: 32768, tag: Default::default() };
    for (e, (stage, sj)) in stages.iter().zip(cts).enumerate() {
        let pt = &sj["pts"][0];
        let (top, nf) = (pt["level"].as_u64().unwrap() as u32, pt["noise_factor"].as_f64().unwrap());
        println!("[cts] stage {e}: top {} scale {:e} | dump top {top} scale {nf:e}", stage.top, stage.scale);
        assert_eq!(stage.top, top, "stage {e} level");
        assert_eq!(stage.scale.to_bits(), nf.to_bits(), "stage {e} scale");
        let ptm = PtMeta { level: stage.level, deg: 1, nf: stage.scale, tag: Default::default() };
        let prod = shadow.mult_pt(&ct, &ptm).unwrap();
        ct = shadow.rescale(&prod.out).unwrap().out;
        let want = fmeta(&load(&format!("it0.cts{e}.json")).unwrap());
        assert_eq!((chain.top_max() - ct.level, ct.deg as i32), (want.top as u32, want.nl), "stage {e} output");
        assert_eq!(ct.nf.to_bits(), want.nf.to_bits(), "stage {e} output scale {} vs {}", ct.nf, want.nf);
    }
    // StC (shipped order) starts at the effective level of the EvalMod output
    let em = fmeta(&load("it0.em.json").unwrap());
    let stc = stc_stage_levels(&st, chain.top_max() - em.top as u32, em.nl as u8, 3).unwrap();
    assert_eq!(stc[0].top as i32, em.top - st.d as i32);
    println!("[stc] stage tops {:?}", stc.iter().map(|s| s.top).collect::<Vec<_>>());
}
