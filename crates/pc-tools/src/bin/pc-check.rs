//! pc-check: cross-checks against the FIDESlib oracle dumps (`perseus-oracle/fx/dumps/<name>`).
//!
//!   pc-check bts|stcf <d3|g0> [--k 1,2,4,8] [--graph]   the d3 finish gates (bit-exact) / StC-first route + staged call
//!   pc-check load|keys \<g0>                             dump inventory and sizes / keys + secret in the device layout
//!   pc-check leaf [--ops ..] \<g0> | leaf-ks --ops .. \<g0> [--source cached|regen|both]   leaf / key-switching op replay
//!   pc-check pcmm|ccmm|spru|sparse-classic ...          matmuls vs host twins, sparse routes (src/check_*.rs, bts_*.rs)
//!   pc-check n64 [--chain n64|n32] [--mm barrett64|barrett32|mont]   the word seam end to end, no oracle
//!
//! Exit status 1 on any mismatch or invalid file. Full usage: docs/ARCHITECTURE.md.
use crate::dump_g0::G0;
use std::process::ExitCode;

#[path = "../dump.rs"]
#[allow(dead_code)]
mod dump;
#[path = "../dump_g0.rs"]
#[allow(dead_code)]
mod dump_g0;
#[path = "pc-check/check_keys.rs"]
mod check_keys;
#[path = "../check_leaf_ks.rs"]
mod check_leaf_ks;
#[path = "../leaf.rs"]
mod leaf;
#[path = "../check_bts.rs"]
mod check_bts;
#[path = "../bts_sparse.rs"]
mod bts_sparse;
#[path = "../bts_classic.rs"]
mod bts_classic;
#[path = "../bts_stc_first.rs"]
#[allow(dead_code)]
mod bts_stc_first;
#[path = "../check_ccmm.rs"]
#[allow(dead_code)]
mod check_ccmm;
#[path = "../check_pcmm.rs"]
#[allow(dead_code)]
mod check_pcmm;
#[path = "../word_seam.rs"]
mod word_seam;
#[path = "../check_tfhe.rs"]
mod check_tfhe;

fn cmd_bts(args: &[String]) -> ExitCode {
    let dir = match args.first() {
        Some(d) => d.clone(),
        None => return usage(),
    };
    let mut ks = vec![1usize];
    let mut graph = false;
    let mut lt_baby = pc_gpu::ks::KskSource::Regen;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lt-baby" => {
                i += 1;
                lt_baby = match args.get(i).map(|s| s.as_str()) {
                    Some("cached") => pc_gpu::ks::KskSource::Cached,
                    Some("regen") => pc_gpu::ks::KskSource::Regen,
                    _ => return usage(),
                };
            }
            "--k" => {
                i += 1;
                ks = args.get(i).map(|v| v.split(',').map(|x| x.parse().expect("--k: integers")).collect()).unwrap_or_default();
            }
            "--graph" => graph = true,
            other => {
                eprintln!("pc-check bts: unknown argument {other}");
                return usage();
            }
        }
        i += 1;
    }
    if std::path::Path::new(&dir).join("bts").is_dir() {
        let mut t = check_bts::Tally::default();
        if let Err(e) = check_bts::g0_bts(&dir, lt_baby, &mut t) {
            println!("[pc-check] bts: ERROR {e}");
            t.bad += 1;
        }
        println!("[pc-check] bts: {} gates, {} mismatches", t.gates, t.bad);
        return if t.bad == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }
    println!("[pc-check] bts {dir}: K-list={ks:?} graph={graph}");
    let mut t = check_bts::Tally::default();
    if let Err(e) = check_bts::d3_finish(&dir, &ks, graph, lt_baby, &mut t) {
        println!("[pc-check] bts: ERROR {e}");
        t.bad += 1;
    }
    println!("[pc-check] bts: {} gates, {} mismatches", t.gates, t.bad);
    if t.bad == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cmd_load(args: &[String]) -> ExitCode {
    let dir = match args.first() {
        Some(d) => d,
        None => return usage(),
    };
    match G0::open(dir) {
        Ok(g) => {
            let inv = g.validate();
            print!("{}", inv.report());
            if inv.errors.is_empty() {
                println!("[pc-check] load {dir}: all {} files valid", inv.files);
                ExitCode::SUCCESS
            } else {
                for e in &inv.errors {
                    println!("[pc-check] INVALID {e}");
                }
                println!("[pc-check] load {dir}: {} invalid of {} files", inv.errors.len(), inv.files);
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            println!("[pc-check] load {dir}: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: pc-check bts <dump> [--k 1,2,4,8] [--graph] [--lt-baby regen|cached (every key source of the bootstrap)]\n       pc-check load <g0 dump>\n       pc-check keys <g0 dump>\n       pc-check leaf [--ops add,sub,...] [--k 1,8] <g0 dump>\n       pc-check leaf-ks --ops <list> <g0 dump> [--k 1,8] [--source cached|regen|both] [--max n]");
    ExitCode::from(2)
}

/// `pc-check n64`: the rig of src/word_seam.rs in check mode.
fn cmd_word_seam(a: &[String]) -> ExitCode {
    let run = || -> word_seam::Res<bool> {
        let mut chain = "n64".to_string();
        let mut mm = "barrett64".to_string();
        let mut levels: Vec<u32> = Vec::new();
        let mut i = 0;
        while i < a.len() {
            let v = a.get(i + 1).cloned().ok_or(format!("{} needs a value", a[i]))?;
            match a[i].as_str() {
                "--chain" => chain = v,
                "--mm" => mm = v,
                "--levels" => levels = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
                o => return Err(format!("unknown argument {o}").into()),
            }
            i += 2;
        }
        let c = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::preset(&chain)?)?;
        if levels.is_empty() {
            let nq = c.q.len() as u32;
            let u = c.level_unit as u32;
            levels = vec![0, nq / 3 / u * u, (2 * nq) / 3 / u * u];
        }
        let o = word_seam::RigOpts { check: true, bench: false, reps: 0, ks: vec![1], levels, seed: 0x64 };
        use pc_core::modmul::{Barrett32, Barrett64, Montgomery};
        let r = match (c.word_bits, mm.as_str()) {
            (64, "barrett64") => word_seam::run::<u64, Barrett64>(c, &o)?,
            (32, "barrett64") => word_seam::run::<u32, Barrett64>(c, &o)?,
            (32, "barrett32") => word_seam::run::<u32, Barrett32>(c, &o)?,
            (32, "mont") => word_seam::run::<u32, Montgomery>(c, &o)?,
            // compile gate only: u16 has no NTT primes at N = 2^16, no chain preset reaches this arm
            (16, "barrett32") => word_seam::run::<u16, Barrett32>(c, &o)?,
            (w, m) => return Err(format!("no instantiation for {w}-bit words x {m}").into()),
        };
        let ok = word_seam::print("pc-check n64", &r);
        let bad = r.checks.iter().filter(|c| !c.4).count();
        println!("[pc-check n64] {} checks, {bad} failed -> {}", r.checks.len(), if ok { "PASS" } else { "FAIL" });
        Ok(ok)
    };
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check n64: {e}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("bts") => cmd_bts(&args[1..]),
        Some("load") => cmd_load(&args[1..]),
        Some("keys") => check_keys::cmd_keys(&args[1..]),
        Some("leaf") => leaf::cmd_leaf(&args[1..]),
        Some("leaf-ks") => check_leaf_ks::cmd_leaf(&args[1..]),
        Some("spru") | Some("sparse") => bts_sparse::cmd(bts_sparse::Mode::Check, &args[1..]),
        Some("fold") => bts_sparse::cmd(bts_sparse::Mode::Fold, &args[1..]),
        Some("sparse-classic") => bts_classic::cmd(bts_classic::Mode::Check, &args[1..]),
        Some("stcf") => bts_stc_first::cmd(bts_stc_first::Mode::Check, &args[1..]),
        Some("pcmm") => check_pcmm::cmd(check_pcmm::Mode::Check, &args[1..]),
        Some("ccmm") => check_ccmm::cmd(check_ccmm::Mode::Check, &args[1..]),
        Some("tfhe") => check_tfhe::cmd(&args[1..]),
        Some("tfhe-pbs") => check_tfhe::cmd_pbs(&args[1..]),
        Some("tfhe-ks") => check_tfhe::cmd_ks(&args[1..]),
        Some("tfhe-bridge") => check_tfhe::cmd_bridge(&args[1..]),
        Some("n64") => cmd_word_seam(&args[1..]),
        _ => usage(),
    }
}
