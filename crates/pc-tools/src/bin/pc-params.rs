//! pc-params: generate a chain preset and its scale/CRT tables; with `--check <params.json>`, compare every chain
//! field of an old-stack dump against them and print the number of differences (exit 1 if any).
use pc_core::params::{check, Chain, ChainSpec};
use pc_core::scale::ScaleTables;
use std::process::ExitCode;

fn usage() -> String {
    "usage: pc-params [--preset n32|n64|n32a4] [--check <params.json>]".to_string()
}

fn run() -> Result<usize, String> {
    let mut preset = "n32".to_string();
    let mut check_path: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--preset" => preset = args.next().ok_or_else(usage)?,
            "--check" => check_path = Some(args.next().ok_or_else(usage)?),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown argument '{other}'\n{}", usage())),
        }
    }
    let spec = ChainSpec::preset(&preset).map_err(|e| e.to_string())?;
    let chain = Chain::generate(&spec).map_err(|e| e.to_string())?;
    let st = ScaleTables::precompute(&chain).map_err(|e| e.to_string())?;
    println!(
        "preset {preset}: N=2^{} |Q|={} |P|={} dnum={} composite={} word={} logQP={} (HE std bound {})",
        spec.logn,
        chain.q.len(),
        chain.p.len(),
        chain.dnum(),
        chain.level_unit,
        chain.word_bits,
        chain.log_qp,
        pc_core::params::max_log_qp(spec.security, chain.n as u32).map_or("none".into(), |b| b.to_string())
    );
    let Some(path) = check_path else {
        println!("q = {:?}", chain.q);
        println!("p = {:?}", chain.p);
        return Ok(0);
    };
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let dump: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    let r = check::compare(&dump, &chain, &st);
    println!("{:<24} {:>8} {:>6}  first difference", "field", "entries", "diff");
    for (f, n, bad, first) in &r.fields {
        println!("{f:<24} {n:>8} {bad:>6}  {}", first.as_deref().unwrap_or("-"));
    }
    for f in &r.missing {
        println!("MISSING from dump: {f}");
    }
    for f in &r.unknown {
        println!("UNCLASSIFIED dump field (counted as a difference): {f}");
    }
    for (f, why) in &r.out_of_scope {
        println!("not compared: {f:<20} ({why})");
    }
    let total = r.differences();
    let leaves: usize = r.fields.iter().map(|f| f.1).sum();
    println!("{} fields, {leaves} values compared: {total} differences", r.fields.len());
    Ok(total)
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(1),
        Err(e) => {
            eprintln!("pc-params: {e}");
            ExitCode::from(2)
        }
    }
}
