//! Local tool (not a gate): PLAN_SITES_PLAN=<plan dir> PLAN_SITES_GRAPHS=<base graphs>:<stage-13 graphs> cargo test
//! --release -p pc-program --test plan_sites -- --nocapture: every block's refreshes by (route slots, raise drop,
//! landing) after the compile, and the compile check's errors.
use pc_program::compile::{compile_dir, CompileOpts};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[test]
fn plan_sites() {
    let (Ok(plan), Ok(gr)) = (std::env::var("PLAN_SITES_PLAN"), std::env::var("PLAN_SITES_GRAPHS")) else {
        println!("SKIP: PLAN_SITES_PLAN / PLAN_SITES_GRAPHS unset");
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (base, tail) = gr.split_once(':').unwrap();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("plan_sites_{}", plan.replace('/', "_")));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for b in 0..14 {
        let src = root.join("graphs").join(if b < 13 { base } else { tail });
        std::os::unix::fs::symlink(src.join(format!("block_{b}")), dir.join(format!("block_{b}"))).unwrap();
    }
    for f in ["capture_env.json", "capture_config.json"] {
        let _ = std::fs::copy(root.join("graphs").join(base).join(f), dir.join(f));
    }
    let dc = compile_dir(&root.join("bootstrap_placements").join(&plan), &dir, &CompileOpts { keep_going: true, ..Default::default() }).expect("plan dir");
    let mut hist: BTreeMap<(u32, u32, u32), usize> = BTreeMap::new();
    let mut per_block = vec![];
    let mut errs = 0;
    for (label, r) in dc.programs {
        let p = r.unwrap_or_else(|e| panic!("{label}: {e}"));
        errs += p.check.errors.len();
        for e in p.check.errors.iter().take(3) {
            println!("  {label}: {e}");
        }
        if label.contains("feedback") {
            continue;
        }
        per_block.push((label.clone(), p.bts_sites.len()));
        for s in &p.bts_sites {
            let land = pc_shadow::meta::effective_level(s.landing.level, s.landing.deg, 2);
            *hist.entry((s.route.slots, s.route.raise_drop, land)).or_default() += 1;
        }
    }
    println!("[sites] {plan}: per block {per_block:?}; total {}", per_block.iter().map(|x| x.1).sum::<usize>());
    for ((slots, drop, land), n) in &hist {
        println!("[sites]   slots {slots:4} drop {drop:2} landing {land:3}: {n}");
    }
    println!("[sites] check errors {errs}");
}
