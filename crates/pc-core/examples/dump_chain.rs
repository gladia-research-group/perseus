//! Print the n32 chain's primes (Q bottom first, then P) as JSON.
use pc_core::params::{Chain, ChainSpec};
fn main() {
    let c = Chain::generate(&ChainSpec::n32()).unwrap();
    println!("{{\"q\": {:?}, \"p\": {:?}, \"log_qp\": {}}}", c.q, c.p, c.log_qp);
}
