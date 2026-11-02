//! Eager rescales (`compile::rr::early`) on Decode-8's n32sp plans (CPU only): the programs compile with 0 check
//! errors (every planned name keeps its planned state through the fused lists), products rescaled eagerly, and every
//! `~er` var one level unit lower than the planned state it replaces, at degree 1 (its rescale paid). A plan whose
//! graphs are not on disk is skipped (a SKIP line).
use pc_program::compile::{compile_dir, CompileOpts};
use pc_program::program::Kind;
use std::path::{Path, PathBuf};

fn early(plan: &str, base: &str, tail: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let g = root.join("graphs");
    let (base, tail) = (g.join(base), g.join(tail));
    if !base.join("block_0/graph.json").is_file() || !tail.join("block_13/graph.json").is_file() {
        println!("SKIP {plan}: graphs not on disk");
        return;
    }
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("graph_{plan}_early"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for b in 0..14 {
        let src = if b < 13 { &base } else { &tail };
        std::os::unix::fs::symlink(src.join(format!("block_{b}")), dir.join(format!("block_{b}"))).unwrap();
    }
    for f in ["capture_env.json", "capture_config.json"] {
        std::fs::copy(base.join(f), dir.join(f)).unwrap();
    }
    let dc = compile_dir(&root.join("bootstrap_placements").join(plan), &dir, &CompileOpts { keep_going: true, ..Default::default() }).expect("plan dir");
    let (mut products, mut vars) = (0, 0);
    for (label, r) in dc.programs {
        let p = r.unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(p.check.errors.is_empty(), "{label}: {:?}", &p.check.errors[..p.check.errors.len().min(4)]);
        for op in p.ops.iter().flat_map(|o| o.members()) {
            if let Kind::MultRescale { fused, .. } = &op.kind {
                let out = &p.vars[op.outs[0] as usize];
                if !out.name.ends_with("~er") {
                    continue;
                }
                products += 1;
                for &f in fused {
                    let orig = &p.vars[f as usize];
                    let new = p.vars.iter().find(|v| v.name == format!("{}~er", orig.name)).expect("each fused var has its ~er var");
                    vars += 1;
                    if orig.meta.deg == 2 {
                        assert_eq!((new.meta.level, new.meta.deg), (orig.meta.level + 2, 1), "{label} {}", orig.name);
                    }
                }
            }
        }
    }
    println!("[early] {plan}: {products} products rescaled eagerly, {vars} vars moved");
    assert!(products > 0);
}

#[test]
fn n32sp_decode_eager_rescales() {
    early("gpt2_decode_python_n32sp_kc16r_gn", "gpt2_decode_python_n32sp_kc16r_gnm", "gpt2_decode_python_n32sp_kc16r_gn_cm");
    early("gpt2_decode_python_n32sp_kc16r_lh", "gpt2_decode_python_n32sp_kc16r_lhm", "gpt2_decode_python_n32sp_kc16r_lh_cm");
}
