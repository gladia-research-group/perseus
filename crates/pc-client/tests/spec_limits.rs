//! Limits the client holds a server's key request to, whatever the server asks.
use pc_client::SpruReq;

#[test]
fn spru_block_key_below_h_min_is_refused() {
    let req = |h| SpruReq { h, n: 2, cs_level: 0, cs_nf: 1.0 };
    assert!(req(SpruReq::H_MIN).check(1 << 16).is_ok());
    for h in [2, 4, 8, 16, 32] {
        assert!(req(h).check(1 << 16).is_err(), "h = {h}");
    }
}
