#include "ckks_fixture.h"   // pulls fideslib_wrapper.h (+ inference.h/test_helpers.h deps)

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// 128-bit security BUDGET probe — fast (CPU param-gen, no GPU keygen/bootstrap).
//
// OpenFHE enforces HEStd_128_classic inside GenCryptoContext: given SetRingDim
// (1<<logN) it checks that the ring provides >=128-bit security for the FULL
// keyswitch modulus logQP = logQ + logP (P = HYBRID special primes). An
// over-budget chain THROWS at GenCryptoContext — before any GPU work — and the
// exception text states OpenFHE's own numbers. So the build verdict itself is
// the authoritative 128-bit gate; we scan many candidate chains in seconds.
//
// gen_cc_only() mirrors make_ckks_context()'s param block (fideslib_wrapper.h)
// so PASS/THROW matches the production build path exactly. (The OpenFHE crypto
// params are not reachable through the fideslib CryptoContext wrapper, so we
// report the build verdict + a labeled logQ/logP HEURISTIC, not an introspected
// logQP; the THROW message carries OpenFHE's exact budget when over.)

namespace {

struct Candidate { std::string label; CKKSContextOptions o; };

// Faithful replica of make_ckks_context()'s param construction up to (and
// including) GenCryptoContext, enable_bootstrap=true. Returns the context, or
// throws exactly where the production build would.
CC gen_cc_only(const CKKSContextOptions& o) {
    CCParams<CryptoContextCKKSRNS> params;
    const int actual_scale_bits = o.btp_scale_bits;
    const uint32_t btp_overhead = o.btp_depth_overhead;
    const uint32_t slots = (o.batch_size == 0) ? (1u << (o.logN - 1)) : o.batch_size;

    params.SetMultiplicativeDepth(o.depth + btp_overhead);
    params.SetScalingModSize(actual_scale_bits);
    if (!o.chain_sizes_per_level.empty())
        params.SetScalingModSizePerLevel(o.chain_sizes_per_level);
    params.SetFirstModSize(o.first_mod_bits);
    params.SetScalingTechnique(FLEXIBLEAUTO);
    params.SetBatchSize(slots);
    params.SetSecretKeyDist(o.h_weight > 0 ? fideslib::SPARSE_ENCAPSULATED : UNIFORM_TERNARY);
    params.SetNumLargeDigits(o.num_large_digits);
    params.SetKeySwitchTechnique(HYBRID);
    params.SetSecurityLevel(HEStd_128_classic);  // the gate we are probing
    params.SetRingDim(1 << o.logN);
    return GenCryptoContext(params);             // throws here if over the 128-bit budget
}

// HEURISTIC logQ / logP (bits). logQ = q0 + mult_depth*scale (uniform chain).
// logP ~= (#special primes) * (largest limb): HYBRID puts ceil((towers)/dnum)
// limbs per digit and P must cover one digit; primes are <=60 bits (NATIVEINT=64).
struct ChainBits { double logQ, logP, logQP; int towers, nP; };
ChainBits estimate_bits(const CKKSContextOptions& o) {
    const int mult_depth = o.depth + o.btp_depth_overhead;
    const int towers     = mult_depth + 1;
    const double maxlimb = std::max(o.first_mod_bits, o.btp_scale_bits);
    double logQ = o.first_mod_bits;
    if (o.chain_sizes_per_level.empty()) logQ += static_cast<double>(mult_depth) * o.btp_scale_bits;
    else for (uint32_t s : o.chain_sizes_per_level) logQ += s;
    const int nP   = static_cast<int>(std::ceil(static_cast<double>(towers) / o.num_large_digits));
    const double logP = nP * maxlimb;
    return {logQ, logP, logQ + logP, towers, nP};
}

CKKSContextOptions base_opts() { return CKKSContextOptions{}; }  // THOR uniform-53 defaults

TEST(SecurityBudgetProbe, Scan) {
    constexpr double kBudget16 = 1761.0;  // logN=16 classical budget (logQP), informational

    std::vector<Candidate> cands;
    auto add = [&](const std::string& label, CKKSContextOptions o) { cands.push_back({label, o}); };

    add("BASELINE depth11 scale53 fm60 dnum7 ovh16", base_opts());

    // scale scan (precision/level lever; lower scale widens the UP range wall 2^(fm-scale))
    for (int s : {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58})
        { auto o = base_opts(); o.btp_scale_bits = s; add("scale=" + std::to_string(s), o); }

    // first_mod scan (UP-range wall; 60 is the NATIVEINT=64 single-prime max -> no headroom up)
    for (int f : {58, 59, 60})
        { auto o = base_opts(); o.first_mod_bits = f; add("first_mod=" + std::to_string(f), o); }

    // depth scan (extra usable compute levels)
    for (int d : {10, 11, 12, 13, 14})
        { auto o = base_opts(); o.depth = d; add("depth=" + std::to_string(d), o); }

    // dnum scan (HYBRID digits -> #special primes -> logP)
    for (uint32_t dn : {2u, 4u, 7u, 14u, 28u})
        { auto o = base_opts(); o.num_large_digits = dn; add("dnum=" + std::to_string(dn), o); }

    // btp_overhead scan (bootstrap reserve; bigger level_budget needs more)
    for (uint32_t ovh : {14u, 15u, 16u, 18u, 20u})
        { auto o = base_opts(); o.btp_depth_overhead = ovh; add("btp_overhead=" + std::to_string(ovh), o); }

    // combined "wider range" candidates (lower scale for UP range, spend freed budget on depth)
    { auto o = base_opts(); o.btp_scale_bits = 50;               add("WIDE: scale50 depth11",   o); }
    { auto o = base_opts(); o.btp_scale_bits = 50; o.depth = 13; add("WIDE: scale50 depth13",   o); }
    { auto o = base_opts(); o.btp_scale_bits = 51; o.depth = 12; add("WIDE: scale51 depth12",   o); }

    std::cout << "\n[budget_probe] logN=16 HEStd_128_classic; budget(logQP)~=" << kBudget16
              << " (PASS/THROW from GenCryptoContext is authoritative; logQ/logP/logQP are HEURISTIC)\n"
              << std::string(112, '-') << "\n";
    std::cout << std::left << std::setw(40) << "candidate"
              << std::right << std::setw(6) << "mdep" << std::setw(7) << "towers"
              << std::setw(9) << "~logQ" << std::setw(8) << "~logP" << std::setw(9) << "~logQP"
              << std::setw(9) << "~margin" << "  verdict\n";

    for (const auto& c : cands) {
        ChainBits b = estimate_bits(c.o);
        std::cout << std::left << std::setw(40) << c.label
                  << std::right << std::setw(6) << (c.o.depth + c.o.btp_depth_overhead)
                  << std::setw(7) << b.towers
                  << std::setw(9) << std::fixed << std::setprecision(0) << b.logQ
                  << std::setw(8) << b.logP << std::setw(9) << b.logQP
                  << std::setw(9) << (kBudget16 - b.logQP) << "  ";
        try {
            CC cc = gen_cc_only(c.o);
            (void)cc;
            std::cout << "PASS\n";
        } catch (const std::exception& e) {
            std::cout << "THROW: " << e.what() << "\n";
        }
        std::cout.flush();
    }
    std::cout << std::string(112, '-') << "\n"
              << "[budget_probe] PASS = builds under 128-bit. ~logQP is a heuristic "
                 "(q0+mdep*scale + ceil(towers/dnum)*maxlimb); the THROW text carries "
                 "OpenFHE's exact numbers. first_mod capped at 60 (NATIVEINT=64).\n";
    SUCCEED();
}

}  // namespace
