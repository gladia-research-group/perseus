//! The legacy env bridge: field table sanity, presets, and a round trip (byte-exact; floats by value) of every env value the shipped
//! plans stamp (`placement.json` capture_env, PLAN_CMD.txt `runtime=` / `recipe=` lines).
use pc_config::{presets, Config, PlanConfig};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const PLANS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bootstrap_placements");

/// Equal maps, a value that differs as text compared as f64 (floats round-trip by value: "2e-5" prints "0.00002").
fn same_by_value(a: &BTreeMap<String, String>, b: &BTreeMap<String, String>) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|((ka, va), (kb, vb))| {
            ka == kb
                && (va == vb
                    || matches!((va.parse::<f64>(), vb.parse::<f64>()), (Ok(x), Ok(y)) if x == y))
        })
}

#[test]
fn float_spellings_are_one_value() {
    for s in ["2e-5", "2e-05", "0.00002", "2.0e-5"] {
        let (c, rest) = Config::from_legacy_env([("PF_SILU_SINGLE_ABS", Some(s))]).unwrap();
        assert!(rest.is_empty());
        assert_eq!(c.get("model.prefill.silu_single_abs").unwrap(), serde_json::json!(2e-5), "{s}");
        // the canonical print parses back to the same value
        let back = c.to_legacy_env()["PF_SILU_SINGLE_ABS"].clone();
        let (c2, _) = Config::from_legacy_env([("PF_SILU_SINGLE_ABS", Some(back.as_str()))]).unwrap();
        assert_eq!(c2, c, "{s} -> {back}");
    }
    for bad in ["nan", "inf", "x", "", " 1"] {
        let e = Config::from_legacy_env([("PF_SILU_SINGLE_ABS", Some(bad))]).unwrap_err().to_string();
        assert!(e.contains("PF_SILU_SINGLE_ABS"), "{bad}: {e}");
    }
}

#[test]
fn field_table() {
    let cf = Config::fields();
    let pf = PlanConfig::fields();
    println!("[config] Config {} fields, PlanConfig {} fields", cf.len(), pf.len());
    // inventory sections 1-2, 5a/5b, 6a/6b + exec.use_graphs, exec.trace_alloc, host.n_in_flight, bts.em_fuse_rescale,
    // bts.lt_tails, bts.pair, bts.sparse_b_k, model.prefill.{last_query,qkv_pack,attn_bsgs,sm_hpack}, chain.preset,
    // the GELU output hint, the head-sum cleanse
    assert_eq!(cf.len(), 261);
    // + section 7a: plan.kv_entry_level, plan.kv_entry_deg, plan.keep_placements; 8a: plan.stc_first_reroute, plan.site_cf;
    // plan.fallback_cf, plan.argmax_mincut, plan.ilp_tiebreak, plan.raise_drop_base, plan.raise_drop_search, plan.pack_cf,
    // plan.keep_placements_exact, plan.mag_safety_steps, plan.resume; the lean chain's plan.envelope_strict,
    // plan.env_effective, plan.pcmm_prediv_cap, plan.pcmm_stamps, plan.stc_first_staged_only,
    // plan.step_route_pricing
    assert_eq!(pf.len(), 104);
    let mut envs = BTreeSet::new();
    let mut dotted = BTreeSet::new();
    for f in cf.iter().chain(pf.iter()) {
        assert!(dotted.insert(f.dotted()), "duplicate field {}", f.dotted());
        if !f.env.is_empty() {
            assert!(envs.insert(f.env), "env {} mapped twice", f.env);
            assert!(f.doc.contains(&format!("`{}`", f.env)), "{} not documented with its env name", f.dotted());
        }
    }
    // 337 inventory names - 4 without a field (GPU guard, dropped _client, two upstream-driver inputs) + the three
    // perseus-core runtime switches (PC_EM_FUSE_RESCALE, PC_LT_TAILS, PC_BTS_PAIR) + PF_LAST_QUERY, PF_QKV_PACK, PF_ATTN_BSGS,
    // PF_SM_HPACK, PLAN_FALLBACK_CF, PLAN_ARGMAX_MINCUT, PLAN_ILP_TIEBREAK, PLAN_RAISE_DROP_BASE, PLAN_RAISE_SEARCH,
    // PLAN_PACK_CF, PLAN_KEEP_PLACEMENTS_EXACT, PLAN_MAG_SAFETY_STEPS, PLAN_RESUME, FIDESLIB_BTS_SPARSE_B_K, CHAIN_PRESET,
    // GELU_OUT_HINT, PLAN_ENVELOPE_STRICT, PLAN_ENV_EFFECTIVE, PLAN_PCMM_PREDIV_CAP, PLAN_PCMM_STAMPS,
    // PLAN_STC_FIRST_STAGED_ONLY, PLAN_STEP_ROUTE_PRICING, SM_HRS_CLEANSE
    assert_eq!(envs.len(), 359);
}

#[test]
fn presets_place_every_name() {
    for name in presets::NAMES {
        let t = presets::table(name).unwrap();
        let (c, rest) = Config::from_legacy_env(t.iter().map(|(k, v)| (*k, Some(*v)))).unwrap();
        let (p, rest2) = PlanConfig::from_legacy_env(rest.iter().map(|k| {
            let v = t.iter().find(|(kk, _)| kk == k).unwrap().1;
            (k.as_str(), Some(v))
        }))
        .unwrap();
        assert!(rest2.is_empty(), "{name}: no field for {rest2:?}");
        let mut back = c.to_legacy_env();
        back.extend(p.to_legacy_env());
        let want: BTreeMap<String, String> = t.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        assert_eq!(back, want, "{name}");
        assert_eq!(Config::preset(name).unwrap(), c);
        let j = c.to_json().unwrap();
        assert_eq!(Config::from_json(&j).unwrap(), c, "{name} json");
        println!("[preset] {name}: {} Config + {} PlanConfig values", c.to_legacy_env().len(), p.to_legacy_env().len());
    }
    assert!(Config::preset("nope").is_err());
}

#[test]
fn dotted_access_and_errors() {
    let mut c = Config::preset("n32_gpt2_decode").unwrap();
    assert_eq!(c.get("model.ln_cheb").unwrap(), serde_json::json!(true));
    assert_eq!(c.get("bts.auto_bts_level").unwrap(), serde_json::json!(44));
    assert_eq!(c.get("bts.sparse_bts_slots").unwrap(), serde_json::json!(["512", "1"]));
    assert!(c.get("model.prefill.attn").unwrap().is_null());
    c.set("model.prefill.attn", serde_json::json!("ccmm")).unwrap();
    assert_eq!(c.get_env("PF_ATTN"), Some(Some("ccmm".to_string())));
    assert!(c.get("model.nope").is_err());
    assert!(c.set("bts.auto_bts_level", serde_json::json!("x")).is_err());
    // strict parsers name the knob
    let e = Config::from_legacy_env([("LN_CHEB", Some("yes"))]).unwrap_err().to_string();
    assert!(e.contains("LN_CHEB"), "{e}");
    assert!(Config::from_legacy_env([("AUTO_BTS_LEVEL", Some("044"))]).is_err());
    // unset values (capture_env nulls) stay unset
    let (c2, _) = Config::from_legacy_env([("CHEB_ALIGN", None)]).unwrap();
    assert_eq!(c2, Config::default());
    // load_bearing keeps chain/bts/model/quirks and the contract names elsewhere
    let lb = c.load_bearing();
    assert_eq!(lb.exec.gpt2_cache, Some(true));
    assert_eq!(lb.mem.fhe_pt_coeff_encode, Some(false));
    assert_eq!(lb.host, Default::default());
    assert_eq!(lb.model, c.model);
    // the perseus-core kernel switches bind nothing
    let (k, _) = Config::from_legacy_env([("PC_EM_FUSE_RESCALE", Some("1")), ("PC_LT_TAILS", Some("0"))]).unwrap();
    assert_eq!(k.load_bearing(), Config::default().load_bearing());
}

fn walk(p: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(p) else { return };
    for e in rd.flatten() {
        let q = e.path();
        if q.is_dir() {
            walk(&q, out);
        } else if q.file_name().is_some_and(|n| n == "PLAN_CMD.txt" || n.to_string_lossy().ends_with("placement.json")) {
            out.push(q);
        }
    }
}

#[test]
fn shipped_stamps_round_trip() {
    let mut files = vec![];
    walk(Path::new(PLANS), &mut files);
    if files.is_empty() {
        println!("SKIP: no plan files under {PLANS}");
        return;
    }
    let (mut n_maps, mut n_vals) = (0, 0);
    for f in &files {
        let mut maps: Vec<Vec<(String, Option<String>)>> = vec![];
        let text = std::fs::read_to_string(f).unwrap();
        if f.ends_with("PLAN_CMD.txt") {
            for line in text.lines() {
                for pre in ["runtime=", "recipe="] {
                    if let Some(rest) = line.strip_prefix(pre) {
                        let kv = rest.split_whitespace().filter_map(|kv| kv.split_once('='));
                        maps.push(kv.map(|(k, v)| (k.to_string(), Some(v.to_string()))).collect());
                    }
                }
            }
        } else {
            let j: serde_json::Value = serde_json::from_str(&text).unwrap();
            if let Some(env) = j.pointer("/capture_env/env").and_then(|e| e.as_object()) {
                maps.push(env.iter().map(|(k, v)| (k.clone(), v.as_str().map(str::to_string))).collect());
            }
        }
        for m in maps {
            // a recipe line may set one name twice (`BTS_LEVEL=34 ... BTS_LEVEL=28`): the shell keeps the last
            let last: BTreeMap<&str, Option<&str>> = m.iter().map(|(k, v)| (k.as_str(), v.as_deref())).collect();
            let (c, rest) = Config::from_legacy_env(last.iter().map(|(k, v)| (*k, *v))).unwrap();
            let (p, rest2) = PlanConfig::from_legacy_env(rest.iter().map(|k| (k.as_str(), last[k.as_str()]))).unwrap();
            // IMPL_RESIDENT_BLOCKS: a switch of the C++ runtime the 512-context plan records (perseus.config.RUNTIME_DROPPED)
            let unplaced: Vec<_> = rest2
                .iter()
                .filter(|k| !matches!(k.as_str(), "ML" | "PYTHONHASHSEED" | "IMPL_RESIDENT_BLOCKS"))
                .collect();
            assert!(unplaced.is_empty(), "{}: no field for {unplaced:?}", f.display());
            let mut back = c.to_legacy_env();
            back.extend(p.to_legacy_env());
            let want: BTreeMap<String, String> = last
                .iter()
                .filter(|(k, v)| v.is_some() && !matches!(**k, "ML" | "PYTHONHASHSEED" | "IMPL_RESIDENT_BLOCKS"))
                .map(|(k, v)| (k.to_string(), v.unwrap().to_string()))
                .collect();
            assert!(same_by_value(&back, &want), "{}: {back:?} vs {want:?}", f.display());
            n_maps += 1;
            n_vals += want.len();
        }
    }
    println!(
        "[legacy] {} plan files, {n_maps} env maps, {n_vals} values round-tripped (floats by value)",
        files.len()
    );
    assert!(n_maps > 0);
}
