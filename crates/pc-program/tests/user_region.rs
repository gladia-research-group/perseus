//! An alternating user region cannot host EvalMod (`compile::user_region_sites`): the compile refuses a refresh that
//! would run it there instead of computing garbage. The fixture (tests/data/n32sp_entry46) is block 0 of a first n32sp
//! plan whose capture's CPU twin took n32's StC-first entry (46, FAKECAP_STC_ENTRY's default) where n32sp's is 52: its
//! K/V push refresh, an input at 48, was captured at the shipped order's landing (32), which the compile could only reach
//! with a dense raise drop of 3, running EvalMod on the region's 2^36 / 2^44.15 levels.
use pc_program::compile::{compile_dir, CompileOpts};
use std::path::Path;

#[test]
fn n32sp_entry46_k_v_push_refused() {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/n32sp_entry46");
    let dc = compile_dir(&d.join("plan"), &d.join("graph"), &CompileOpts { keep_going: true, ..Default::default() }).expect("plan dir");
    let (label, r) = &dc.programs[0];
    let p = r.as_ref().expect("block 0 compiles (keep_going)");
    let hits: Vec<&String> = p.check.errors.iter().filter(|e| e.starts_with("[user_region]")).collect();
    assert_eq!(hits.len(), 1, "{label}: {hits:?}");
    assert!(hits[0].contains("v_257") && hits[0].contains("raise drop 3"), "{}", hits[0]);
    // without keep_going the compile stops on it
    let strict = compile_dir(&d.join("plan"), &d.join("graph"), &CompileOpts::default()).expect("plan dir");
    assert!(strict.programs[0].1.is_err());
}
