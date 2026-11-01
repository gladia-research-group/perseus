//! Fused relinearization-rescales on the alternating chain n32sp (CPU only): Decode-8's n32sp plans (#19's _lh and the
//! served _gn) compile with products fused on the user region's single-prime levels too (`compile::rr`'s `served`),
//! every one landing where the runtime's tables (`pc_gpu::relin::RelinRescaler`) serve it: a rescale that drops one
//! prime or two from a top limb of at least 3. A plan whose graphs are not on disk is skipped (a SKIP line).
use pc_program::compile::{compile_dir, CompileOpts};
use pc_program::program::Kind;
use pc_shadow::ChainShape;
use std::path::{Path, PathBuf};

#[test]
fn n32sp_decode_fuses_single_prime_rescales() {
    for (plan, base, tail) in [
        ("gpt2_decode_python_n32sp_kc16r_lh", "gpt2_decode_python_n32sp_kc16r_lhm", "gpt2_decode_python_n32sp_kc16r_lh_cm"),
        ("gpt2_decode_python_n32sp_kc16r_gn", "gpt2_decode_python_n32sp_kc16r_gnm", "gpt2_decode_python_n32sp_kc16r_gn_cm"),
    ] {
        fused(plan, base, tail);
    }
}

fn fused(plan: &str, base: &str, tail: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let g = root.join("graphs");
    let (base, tail) = (g.join(base), g.join(tail));
    if !base.join("block_0/graph.json").is_file() || !tail.join("block_13/graph.json").is_file() {
        println!("SKIP {plan}: graphs not on disk");
        return;
    }
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("graph_{plan}_rr"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for b in 0..14 {
        let src = if b < 13 { &base } else { &tail };
        std::os::unix::fs::symlink(src.join(format!("block_{b}")), dir.join(format!("block_{b}"))).unwrap();
    }
    for f in ["capture_env.json", "capture_config.json"] {
        std::fs::copy(base.join(f), dir.join(f)).unwrap();
    }
    let pd = root.join("bootstrap_placements").join(plan);
    let dc = compile_dir(&pd, &dir, &CompileOpts { keep_going: true, ..Default::default() }).expect("plan dir");
    let chain = ChainShape::of(&pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32sp()).unwrap());
    let (mut drops, mut dbl) = ([0usize; 3], [0usize; 3]);
    for (label, r) in dc.programs {
        let p = r.unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(p.check.errors.is_empty(), "{label}: {:?}", &p.check.errors[..p.check.errors.len().min(4)]);
        for op in p.ops.iter().flat_map(|o| o.members()) {
            let Kind::MultRescale { dbl: d, .. } = op.kind else { continue };
            let out = p.vars[op.outs[0] as usize].meta.level;
            let a = chain.primes_dropped(out - chain.unit).expect("the product's level is on the grid");
            let b = chain.primes_dropped(out).expect("the rescaled level is on the grid");
            assert!(b - a == 1 || b - a == 2, "{label}: a fused rescale dropping {} primes", b - a);
            assert!(chain.top_max() - a >= 3, "{label}: a fused product at top {}", chain.top_max() - a);
            drops[(b - a) as usize] += 1;
            dbl[(b - a) as usize] += d as usize;
        }
    }
    println!("[rr] {plan}: fused relinearization-rescales dropping one prime {} ({} doubled), two {} ({} doubled)", drops[1], dbl[1], drops[2], dbl[2]);
    assert!(drops[1] > 0 && drops[2] > 0);
}
