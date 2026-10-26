//! Print a chain preset's primes (Q bottom first, then P), digits, level groups and log QP as JSON.
//!   cargo run -p pc-core --example dump_chain -- [preset]
use pc_core::params::{Chain, ChainSpec};
fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "n32".into());
    let c = Chain::generate(&ChainSpec::preset(&name).unwrap()).unwrap();
    let st = pc_core::scale::ScaleTables::precompute(&c).unwrap();
    let sf: Vec<String> = (0..c.q.len()).map(|t| st.sf(t as i32).map(|v| format!("{:.3}", v.log2())).unwrap_or_else(|_| "-".into())).collect();
    println!(
        "{{\"q\": {:?}, \"p\": {:?}, \"digits\": {:?}, \"groups\": {:?}, \"p_sparse\": {}, \"log_qp\": {}, \"log2_sf_by_top\": {:?}}}",
        c.q, c.p, c.digits.iter().map(|d| d.len()).collect::<Vec<_>>(), c.groups, c.p_sparse, c.log_qp, sf
    );
}
