#include "ckks_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

using SparseEnvelopeTest = CkksFixture;

// Measures the dense/sparse bootstrap envelope.
//
// Sparse-slot routing is worth a large wall saving and extra precision, but it was held
// back by a decode cliff at T=128: dense EvalMod carries a systematic downward bias on
// structured values, the softmax and LayerNorm Goldschmidt chains leaned on that bias for
// band-edge margin, and sparse — being more accurate — removes it.
//
// Measuring that with two processes (a dense control context against a separate sparse one)
// makes every number a cross-run difference over two keysets and two draws, with the arcsine
// reservation present in only one of them. This file instead measures the SAME ciphertext
// through BOTH arms in ONE process. That is legal because dense/sparse is chosen entirely by
// the `slots` argument (Bootstrap.cu: approxModReduction vs approxModReductionSparse) and
// both precomputations coexist when SPARSE_BTS_SLOTS < slots, so SparseBtsScope toggles the
// arm per call and the two results differ only by the bootstrap path.
//
// The tests, in the order they gate adoption:
//   RangeDistributionGrid  — value range x value distribution: where the dense/sparse bias
//                            gap lives, and where each arm's precision fails.
//   PeriodicityTolerance   — sparse assumes the payload is s-periodic and real lanes are only
//                            approximately so, so this fixes how exact a lane must be.
//   ImaginaryLane          — sparse folds ct + conj(ct) and drops the imaginary part, so any
//                            complex-packed lane carrying data in Im is ineligible. Measured
//                            rather than assumed.
//   ReplicateThenSparse    — manufacturing eligibility by replicating a short payload.
//   Vector768Ladder        — the transformer's real shape: 768 live values in 32768 slots.
//   MeanViaSparse          — the sparse bootstrap's own Accumulate as the reduction.
//   MeanRangeWall          — whether that reduction's range wall sits on the elements or
//                            on their mean.
//   CachemirFusedVariance  — the same fusion at the real site, the LayerNorm variance.
//
// Needs the n32 chain environment: source scripts/local_env.sh.

double env_num(const char* k, double dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atof(v) : dflt;
}

// Per-arm statistics against a reference slot vector.
struct Stats {
    double bias = 0.0;      // mean(out - ref): the SIGNED artifact the decode leaned on
    double rel_bias = 0.0;  // bias / A
    double bias_rel_ref = 0.0;  // bias / mean|ref|: scale-free for non-constant structures
    double abs_max = 0.0;
    double rms = 0.0;
    int nonfinite = 0;
};

Stats stats_of(const std::vector<double>& ref, const std::vector<double>& got, double A) {
    Stats s;
    double sum = 0.0, sq = 0.0, aref = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        aref += std::fabs(ref[i]);
        if (!std::isfinite(got[i])) { ++s.nonfinite; continue; }
        const double d = got[i] - ref[i];
        sum += d;
        sq += d * d;
        if (std::fabs(d) > s.abs_max) s.abs_max = std::fabs(d);
        ++n;
    }
    if (n == 0) { s.abs_max = std::numeric_limits<double>::infinity(); return s; }
    s.bias = sum / (double)n;
    s.rms = std::sqrt(sq / (double)n);
    s.rel_bias = A > 0 ? s.bias / A : 0.0;
    aref /= (double)ref.size();
    s.bias_rel_ref = aref > 0 ? s.bias / aref : 0.0;
    return s;
}

// ── the two arms, on a CLONE so the input ciphertext is reused verbatim ──────
Ctx bts_dense(CKKSContext& F, const Ctx& in) {
    Ctx c = in->Clone();
    F.bootstrap(c);
    return c;
}

Ctx bts_sparse(CKKSContext& F, const Ctx& in) {
    Ctx c = in->Clone();
    CKKSContext::SparseBtsScope ss(F);  // -> inner_bootstrap SetSlots(sparse_bts_slots)
    F.bootstrap(c);
    return c;
}

// ── payload structures. Every generator returns a PERIOD-s base vector; the
//    caller tiles it to the full slot count, so the payload is sparse-LEGAL by
//    construction and the two arms see byte-identical plaintext. ─────────────
using Gen = std::function<std::vector<double>(int s, double A, std::mt19937& g)>;

struct Structure {
    const char* name;
    Gen gen;
    bool stochastic;  // draw more reps
    const char* why;  // the model site this mimics
};

std::vector<Structure> structures() {
    return {
        {"const", [](int s, double A, std::mt19937&) { return std::vector<double>(s, A); },
         false, "LayerNorm variance broadcast (period 1) — the maximally coherent case"},

        {"sm16",
         [](int s, double A, std::mt19937&) {
             std::vector<double> v(s);
             for (int i = 0; i < s; ++i) {
                 const int head = (s >= 16) ? (i / std::max(1, s / 16)) : 0;
                 v[i] = A * (0.3 + 0.7 * std::min(15, head) / 15.0);
             }
             return v;
         },
         false, "softmax denominator: 16 distinct levels in blocks"},

        {"rand_u",
         [](int s, double A, std::mt19937& g) {
             std::uniform_real_distribution<double> d(-A, A);
             std::vector<double> v(s);
             for (auto& x : v) x = d(g);
             return v;
         },
         true, "uniform(-A,A): the incoherent control the old probes used"},

        {"rand_pos",
         [](int s, double A, std::mt19937& g) {
             std::uniform_real_distribution<double> d(0.0, A);
             std::vector<double> v(s);
             for (auto& x : v) x = d(g);
             return v;
         },
         true, "post-exp softmax numerators: one-sided, so bias cannot cancel"},

        {"gauss",
         [](int s, double A, std::mt19937& g) {
             std::normal_distribution<double> d(0.0, A / 3.0);
             std::vector<double> v(s);
             for (auto& x : v) x = std::clamp(d(g), -A, A);
             return v;
         },
         true, "activations: mass near 0, tails at +-A"},

        {"expdec",
         [](int s, double A, std::mt19937&) {
             std::vector<double> v(s);
             for (int i = 0; i < s; ++i) v[i] = A * std::exp(-6.0 * i / std::max(1, s - 1));
             return v;
         },
         false, "attention weights after exp: 4 decades of dynamic range in one ct"},

        {"spike",
         [](int s, double A, std::mt19937&) {
             std::vector<double> v(s, A * 1e-3);
             v[0] = A;
             return v;
         },
         false, "one-hot / cutmax argmax output: one big slot, the rest at the floor"},

        {"bimodal",
         [](int s, double A, std::mt19937&) {
             std::vector<double> v(s);
             for (int i = 0; i < s; ++i) v[i] = (i < s / 2) ? A : -A;
             return v;
         },
         false, "signed saturation: both band edges occupied, no interior mass"},

        {"laggard",
         [](int s, double A, std::mt19937&) {
             std::vector<double> v(s, A);
             for (int i = 0; i < s; i += 100) v[i] = A * 1e-3;
             return v;
         },
         false, "cutmax cascade: a coherent bulk plus a 1 % laggard population"},
    };
}

std::vector<double> tile(const std::vector<double>& base, int S) {
    std::vector<double> v(S);
    for (int i = 0; i < S; ++i) v[i] = base[i % base.size()];
    return v;
}

void print_stats(const char* tag, const char* structure, double A, int rep, const char* arm,
                 const Stats& s) {
    std::cout << tag << " struct=" << structure << " A=" << std::scientific
              << std::setprecision(3) << A << " rep=" << rep << " arm=" << arm
              << " bias=" << s.bias << " rel_bias=" << s.rel_bias
              << " bias_rel_ref=" << s.bias_rel_ref << " abs_max=" << s.abs_max
              << " rms=" << s.rms << " bits=" << std::fixed << std::setprecision(2)
              << (s.abs_max > 0 && std::isfinite(s.abs_max) ? -std::log2(s.abs_max) : -999.0)
              << " nonfinite=" << s.nonfinite << std::endl;
}

uint32_t sparse_slots(CKKSContext& F) { return F.sparse_bts_slots; }

// ═════════════════════════════════════════════════════════════════════════════
// Range x distribution, both arms, same ciphertext.
// ═════════════════════════════════════════════════════════════════════════════
TEST_F(SparseEnvelopeTest, RangeDistributionGrid) {
    const int S = slots();
    const uint32_t sp = sparse_slots(fhe());
    ASSERT_GT(sp, 0u) << "build the context with SPARSE_BTS_SLOTS=<s> — this test needs BOTH "
                         "precomputations resident so one process can run both arms";

    // Amplitudes span the floor (1e-3) through the EvalMod wall (|m| ~ 10) and past it,
    // so the grid shows where each arm STOPS working, not just where it is biased.
    std::vector<double> amps = {1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0};
    if (const char* e = std::getenv("SE_AMPS")) {
        amps.clear();
        std::string t;
        for (const char* p = e;; ++p) {
            if (*p && *p != ',') { t += *p; continue; }
            if (!t.empty()) { amps.push_back(std::atof(t.c_str())); t.clear(); }
            if (!*p) break;
        }
    }
    const int reps = (int)env_num("SE_REPS", 3);

    std::cout << "[se_cfg] slots=" << S << " sparse_slots=" << sp << " reps=" << reps
              << " amps=" << amps.size() << std::endl;

    for (const auto& st : structures()) {
        std::cout << "[se_struct] " << st.name << " — " << st.why << std::endl;
        const int R = st.stochastic ? reps : 1;
        for (double A : amps) {
            for (int r = 0; r < R; ++r) {
                std::mt19937 g((uint32_t)0x5E3D ^ (uint32_t)(r * 7919) ^
                               (uint32_t)std::hash<std::string>{}(std::string(st.name)));
                const std::vector<double> base = st.gen((int)sp, A, g);
                const std::vector<double> v = tile(base, S);

                Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());

                for (int arm = 0; arm < 2; ++arm) {
                    const char* name = arm == 0 ? "dense" : "sparse";
                    try {
                        Ctx out = arm == 0 ? bts_dense(fhe(), ct) : bts_sparse(fhe(), ct);
                        auto got = decrypt_slots(fhe(), out);
                        got.resize(v.size());
                        print_stats("[se]", st.name, A, r, name, stats_of(v, got, A));
                    } catch (const std::exception& e) {
                        std::cout << "[se] struct=" << st.name << " A=" << std::scientific
                                  << std::setprecision(3) << A << " rep=" << r << " arm=" << name
                                  << " THREW: " << e.what() << std::endl;
                    }
                }
            }
        }
    }
    SUCCEED();
}

// ═════════════════════════════════════════════════════════════════════════════
// How exactly s-periodic must a lane be to route sparse?
//
// The sparse path folds the N/2 slots down to s (Accumulate with N/2/s copies,
// Bootstrap.cu) and unfolds after StC. If the copies disagree the
// fold AVERAGES them, silently. Nothing throws. So "is this lane eligible" is a
// quantitative question about how much copy-to-copy disagreement the arm
// tolerates before it costs more than the -32 % is worth.
//
// delta = relative perturbation applied to every copy EXCEPT the first.
// Reported against BOTH references: the canonical period (what a folder would
// return) and the true per-slot payload (what the caller actually holds).
// ═════════════════════════════════════════════════════════════════════════════
TEST_F(SparseEnvelopeTest, PeriodicityTolerance) {
    const int S = slots();
    const uint32_t sp = sparse_slots(fhe());
    ASSERT_GT(sp, 0u) << "needs SPARSE_BTS_SLOTS=<s>";

    const std::vector<double> deltas = {0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0};
    const double A = env_num("SE_PERIOD_A", 1.0);

    for (const char* sname : {"const", "sm16", "rand_u"}) {
        Structure st{};
        for (const auto& s : structures())
            if (std::string(s.name) == sname) st = s;

        for (double d : deltas) {
            std::mt19937 g(4242);
            const std::vector<double> base = st.gen((int)sp, A, g);
            std::vector<double> v = tile(base, S);
            std::uniform_real_distribution<double> pert(-1.0, 1.0);
            for (int i = (int)sp; i < S; ++i) v[i] = base[i % sp] * (1.0 + d * pert(g));

            const std::vector<double> canon = tile(base, S);  // what a perfect folder returns

            Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
            Ctx out = bts_sparse(fhe(), ct);
            auto got = decrypt_slots(fhe(), out);
            got.resize(v.size());

            const Stats vs_true = stats_of(v, got, A);
            const Stats vs_canon = stats_of(canon, got, A);
            std::cout << "[se_period] struct=" << sname << " A=" << A << " delta="
                      << std::scientific << std::setprecision(1) << d
                      << " vs_true_absmax=" << std::setprecision(3) << vs_true.abs_max
                      << " vs_true_rms=" << vs_true.rms
                      << " vs_canon_absmax=" << vs_canon.abs_max
                      << " vs_canon_rms=" << vs_canon.rms << std::endl;
        }
    }
    SUCCEED();
}

// ═════════════════════════════════════════════════════════════════════════════
// The imaginary lane.
//
// Dense conjugate-SPLITS (Bootstrap.cu): Re and Im both survive.
// Sparse conjugate-FOLDS (ct += conj(ct)) and runs ONE Chebyshev on
// the real part — so Im should be destroyed. If so, every cachemir/CKKS_COMPLEX
// lane carrying data in Im is structurally ineligible for sparse routing,
// independent of any accuracy argument. Im is drawn INDEPENDENTLY of Re here
// (the old probe used Im = 0.1*Re, which cannot distinguish "dropped" from
// "scaled").
// ═════════════════════════════════════════════════════════════════════════════
TEST_F(SparseEnvelopeTest, ImaginaryLane) {
    const int S = slots();
    const uint32_t sp = sparse_slots(fhe());
    ASSERT_GT(sp, 0u) << "needs SPARSE_BTS_SLOTS=<s>";

    for (double A : {0.5, 1.0, 2.0}) {
        std::mt19937 g(31337);
        std::uniform_real_distribution<double> d(-A, A);
        std::vector<std::complex<double>> base(sp);
        for (auto& z : base) z = {d(g), d(g)};  // Im INDEPENDENT of Re
        std::vector<std::complex<double>> v(S);
        for (int i = 0; i < S; ++i) v[i] = base[i % sp];

        Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());

        // arm -1 = NO bootstrap. Without this control the test silently measures the
        // ENCODING: under CKKS_COMPLEX=0 the wrapper never calls SetCKKSDataTypeComplex
        // (fideslib_wrapper.h), so OpenFHE discards Im at encode and BOTH arms report
        // im_retained=0 for a reason that has nothing to do with the bootstrap. If this row
        // does not show im_retained ~ 1, the other two rows are meaningless — rerun with
        // CKKS_COMPLEX=1.
        for (int arm = -1; arm < 2; ++arm) {
            const char* name = arm == -1 ? "pre" : (arm == 0 ? "dense" : "sparse");
            try {
                Ctx out = arm == -1 ? ct->Clone()
                                    : (arm == 0 ? bts_dense(fhe(), ct) : bts_sparse(fhe(), ct));
                auto pt = decrypt_pt(fhe().cc, out, fhe().sk());
                pt->SetLength(S);
                const auto& cv = pt->GetCKKSPackedValue();
                double re_max = 0.0, im_max = 0.0, im_energy = 0.0, im_ref_energy = 0.0;
                for (int i = 0; i < S && i < (int)cv.size(); ++i) {
                    re_max = std::max(re_max, std::fabs(cv[i].real() - v[i].real()));
                    im_max = std::max(im_max, std::fabs(cv[i].imag() - v[i].imag()));
                    im_energy += cv[i].imag() * cv[i].imag();
                    im_ref_energy += v[i].imag() * v[i].imag();
                }
                std::cout << "[se_imag] A=" << std::fixed << std::setprecision(2) << A
                          << " arm=" << name << std::scientific << std::setprecision(3)
                          << " re_abs_max=" << re_max << " im_abs_max=" << im_max
                          << " im_retained=" << (im_ref_energy > 0
                                                     ? std::sqrt(im_energy / im_ref_energy)
                                                     : 0.0)
                          << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[se_imag] A=" << A << " arm=" << name << " THREW: " << e.what()
                          << std::endl;
            }
        }
    }
    SUCCEED();
}

// ═════════════════════════════════════════════════════════════════════════════
// Manufacturing eligibility: replicate-then-sparse.
//
// The blocker on adoption is not accuracy, it is supply: a transformer's
// periodic objects are only the REDUCTION outputs (LN variance, softmax
// denominator, cutmax scalars — norm.cu "slot-periodic from here"). The
// activations never are.
//
// But eligibility can be BOUGHT. If a ct's live data occupies <= s slots and the
// remaining slots are ZERO — padded / masked lanes routinely are — then
// log2(N/2/s) rotate+adds replicate it into sparse-legal form. The zeros make
// the sum exact, so this costs NO mask and NO level: just the rotations.
//
// This measures whether that trade is profitable: wall and accuracy of
// (replicate + sparse bts) against a plain dense bts on the identical payload,
// compared on the live slots only.
//
// PAY_ROTS=0 skips the replication to confirm the control: a zero-padded ct
// routed sparse WITHOUT replication must come back wrong (the fold averages the
// zeros in, so the live values arrive divided by N/2/s).
// ═════════════════════════════════════════════════════════════════════════════
TEST_F(SparseEnvelopeTest, ReplicateThenSparse) {
    const int S = slots();
    const uint32_t sp = sparse_slots(fhe());
    ASSERT_GT(sp, 0u) << "needs SPARSE_BTS_SLOTS=<s>";
    const int copies = S / (int)sp;
    const int n_rot = (int)std::lround(std::log2((double)copies));
    const int timed = (int)env_num("SE_REP_N", 10);
    const bool pay_rots = env_num("PAY_ROTS", 1) > 0;

    std::cout << "[se_rep] slots=" << S << " sparse=" << sp << " copies=" << copies
              << " rotations=" << (pay_rots ? n_rot : 0) << std::endl;

    for (double A : {0.5, 1.0, 2.0}) {
        std::mt19937 g(20260803);
        std::uniform_real_distribution<double> d(-A, A);
        std::vector<double> live(sp);
        for (auto& x : live) x = d(g);
        std::vector<double> v(S, 0.0);              // live payload, zeros elsewhere
        for (uint32_t i = 0; i < sp; ++i) v[i] = live[i];

        Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());

        // ---- arm 1: plain dense bootstrap on the zero-padded ct
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        Ctx d_out;
        for (int i = 0; i < timed; ++i) d_out = bts_dense(fhe(), ct);
        cudaDeviceSynchronize();
        const double ms_dense =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / timed;

        // ---- arm 2: replicate (rotate+add ladder) then sparse bootstrap
        cudaDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        Ctx s_out;
        for (int i = 0; i < timed; ++i) {
            Ctx r = ct->Clone();
            if (pay_rots)
                for (int k = 0; k < n_rot; ++k)
                    r = fhe().cc->EvalAdd(r, fhe().cc->EvalRotate(r, -(int)sp << k));
            s_out = bts_sparse(fhe(), r);
        }
        cudaDeviceSynchronize();
        const double ms_sparse =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / timed;

        auto gd = decrypt_slots(fhe(), d_out);
        auto gs = decrypt_slots(fhe(), s_out);
        std::vector<double> ref(live), cd(sp), cs(sp);
        for (uint32_t i = 0; i < sp; ++i) { cd[i] = gd[i]; cs[i] = gs[i]; }

        const Stats sd = stats_of(ref, cd, A), ss = stats_of(ref, cs, A);
        std::cout << std::scientific << std::setprecision(3)
                  << "[se_rep] A=" << A
                  << " dense: " << std::fixed << std::setprecision(2) << ms_dense << " ms "
                  << std::scientific << "rms=" << sd.rms << " abs_max=" << sd.abs_max
                  << "  |  replicate+sparse: " << std::fixed << std::setprecision(2)
                  << ms_sparse << " ms " << std::scientific << "rms=" << ss.rms
                  << " abs_max=" << ss.abs_max
                  << "  |  net=" << std::fixed << std::setprecision(2)
                  << (ms_sparse - ms_dense) << " ms ("
                  << (100.0 * (ms_sparse - ms_dense) / ms_dense) << " %)" << std::endl;
    }
    SUCCEED();
}

// ═════════════════════════════════════════════════════════════════════════════
// The transformer's real shape: a 768-dim hidden state (padded to 1024) living in
// a logN=16 slot space, routed sparse at every achievable slot count.
//
// 768 live values in 32768 slots means
// the ciphertext is 97.7 % empty, so the replicate-to-eligible trade above
// applies — and it has a FRONTIER, because s controls three things at once:
//
//   rotations to reach s-periodicity = log2(N/2/s)   (fewer at large s)
//   live slot fraction after replication = live_dim/s (lower at large s
//       => smaller coefficient-domain norm => BETTER EvalMod accuracy)
//   sparse bootstrap wall                             (lower at small s)
//
// Note we replicate ONLY at strides >= s (not from the 1024-block upward):
// that is both fewer rotations AND fewer live slots for the same legality, since
// a window of width s holding the block in [0,1024) and zeros in [1024,s) is
// already s-periodic once the windows agree.
//
// Baseline is a DENSE bootstrap of the same 768-live ct — that is what the
// pipeline does today, and it is a genuinely strong baseline: a 97.7 %-empty ct
// has a small coefficient norm and dense bootstraps it well.
//
// One s per process: SPARSE_BTS_SLOTS is a context build option, so sweep s by
// re-running with a different value.
// ═════════════════════════════════════════════════════════════════════════════
TEST_F(SparseEnvelopeTest, Vector768Ladder) {
    const int S = slots();
    const uint32_t sp = sparse_slots(fhe());
    ASSERT_GT(sp, 0u) << "needs SPARSE_BTS_SLOTS=<s>";
    const int live_dim = (int)env_num("SE_LIVE_DIM", 768);
    ASSERT_LE((uint32_t)live_dim, sp) << "live_dim must fit inside one sparse window";
    const int copies = S / (int)sp;
    const int n_rot = (int)std::lround(std::log2((double)copies));
    const int timed = (int)env_num("SE_REP_N", 10);

    std::cout << "[se768] slots=" << S << " sparse=" << sp << " live_dim=" << live_dim
              << " copies=" << copies << " rotations=" << n_rot << " live_frac="
              << std::fixed << std::setprecision(3) << (double)live_dim / (double)sp << std::endl;

    for (double A : {0.5, 1.0, 2.0, 4.0}) {
        std::mt19937 g(768768);
        std::uniform_real_distribution<double> d(-A, A);
        std::vector<double> v(S, 0.0);
        for (int i = 0; i < live_dim; ++i) v[i] = d(g);
        const std::vector<double> ref(v.begin(), v.begin() + live_dim);

        Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());

        // ---- baseline: dense bootstrap of the 768-live ciphertext, as shipped
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        Ctx d_out;
        for (int i = 0; i < timed; ++i) d_out = bts_dense(fhe(), ct);
        cudaDeviceSynchronize();
        const double ms_dense =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / timed;

        // ---- replicate at strides s, 2s, 4s, ... then sparse bootstrap
        cudaDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        Ctx s_out;
        double ms_rot = 0.0;
        for (int i = 0; i < timed; ++i) {
            Ctx r = ct->Clone();
            // sync FIRST: without it this timer absorbs the tail of the previous
            // iteration's bootstrap and the rot attribution is garbage (it read
            // 7-10 ms for 1-2 rotations, vs ~0.4 ms/rotation from the ladder delta).
            cudaDeviceSynchronize();
            const auto tr = std::chrono::steady_clock::now();
            for (int k = 0; k < n_rot; ++k)
                r = fhe().cc->EvalAdd(r, fhe().cc->EvalRotate(r, -((int)sp << k)));
            cudaDeviceSynchronize();
            ms_rot += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - tr).count();
            s_out = bts_sparse(fhe(), r);
        }
        cudaDeviceSynchronize();
        const double ms_sparse =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / timed;
        ms_rot /= timed;

        // ---- arm 3: NO rotations at all. The bootstrap's own Accumulate folds the
        // copies, and for a zero-padded ct that yields exactly payload/copies (verified
        // by the PAY_ROTS=0 control: rms 0.267 at A=0.5 vs the predicted
        // rms(payload)*15/16 = 0.2706). So a single downstream constant recovers it —
        // free, since any following linear op absorbs a x2^k. The question is whether
        // amplifying the bootstrap's noise floor by `copies` costs more than the
        // replication it saves.
        cudaDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        Ctx n_out;
        for (int i = 0; i < timed; ++i) n_out = bts_sparse(fhe(), ct);
        cudaDeviceSynchronize();
        const double ms_noRot =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / timed;

        auto gd = decrypt_slots(fhe(), d_out);
        auto gs = decrypt_slots(fhe(), s_out);
        auto gn = decrypt_slots(fhe(), n_out);
        std::vector<double> cd(live_dim), cs(live_dim), cn(live_dim);
        for (int i = 0; i < live_dim; ++i) {
            cd[i] = gd[i];
            cs[i] = gs[i];
            cn[i] = gn[i] * (double)copies;   // the free downstream constant
        }
        const Stats sn = stats_of(ref, cn, A);
        std::cout << "[se768scale] s=" << sp << " A=" << std::fixed << std::setprecision(2) << A
                  << " noRot+x" << copies << ": " << std::setprecision(2) << ms_noRot << " ms "
                  << std::scientific << std::setprecision(3) << "rms=" << sn.rms
                  << " abs_max=" << sn.abs_max << std::endl;

        const Stats sd = stats_of(ref, cd, A), ss = stats_of(ref, cs, A);
        auto bits = [](double e) { return (e > 0 && std::isfinite(e)) ? -std::log2(e) : -999.0; };

        std::cout << "[se768] s=" << sp << " A=" << std::fixed << std::setprecision(2) << A
                  << " rot=" << n_rot
                  << " | dense " << std::setprecision(2) << ms_dense << " ms "
                  << std::scientific << std::setprecision(3) << "rms=" << sd.rms
                  << " bits=" << std::fixed << std::setprecision(2) << bits(sd.abs_max)
                  << " | rep+sparse " << std::setprecision(2) << ms_sparse << " ms (rot "
                  << ms_rot << ") " << std::scientific << std::setprecision(3) << "rms=" << ss.rms
                  << " bits=" << std::fixed << std::setprecision(2) << bits(ss.abs_max)
                  << " | net " << std::setprecision(2) << (ms_sparse - ms_dense) << " ms "
                  << (100.0 * (ms_sparse - ms_dense) / ms_dense) << " %"
                  << " dbits " << (bits(ss.abs_max) - bits(sd.abs_max)) << std::endl;
    }
    SUCCEED();
}

// ═════════════════════════════════════════════════════════════════════════════
// Fusing the REDUCTION into the bootstrap.
//
// The sparse arm's Accumulate is an orthogonal projection onto the s-periodic
// subspace with MEAN normalisation. Read that forwards instead of backwards: if
// what you wanted downstream was the mean over those windows, the bootstrap has
// already computed it for you. At s = 1 (32768 windows of width
// 1) that is the mean of the entire vector — a full reduction plus a bootstrap
// in one call.
//
// This is exactly the shape of the pipeline's periodic lanes: LayerNorm
// variance and the softmax denominator are rotate-and-sum reductions that are
// then bootstrapped. Today that costs log2(N/2/s) keyswitches AT THE LANE'S
// LEVEL plus a full dense bootstrap.
//
// But the fold is NOT free, it is RELOCATED: Accumulate runs after ModRaise, at
// the raised full-limb level, the most expensive place to keyswitch. So this
// measures the real question —
//
//   arm 1 (today):  explicit rotate+add ladder, then a DENSE bootstrap
//   arm 2 (fused):  sparse bootstrap straight on the unreduced vector
//
// both compared against the true window-mean of a NON-periodic input.
// ═════════════════════════════════════════════════════════════════════════════
TEST_F(SparseEnvelopeTest, MeanViaSparse) {
    const int S = slots();
    const uint32_t sp = sparse_slots(fhe());
    ASSERT_GT(sp, 0u) << "needs SPARSE_BTS_SLOTS=<s>";
    const int copies = S / (int)sp;
    const int n_rot = (int)std::lround(std::log2((double)copies));
    const int timed = (int)env_num("SE_REP_N", 10);

    std::cout << "[se_mean] slots=" << S << " sparse=" << sp << " windows=" << copies
              << " reduction_rotations=" << n_rot << std::endl;

    for (double A : {0.5, 1.0, 2.0}) {
        std::mt19937 g(515151);
        std::uniform_real_distribution<double> d(-A, A);
        std::vector<double> v(S);
        for (auto& x : v) x = d(g);          // NOT periodic — a general vector

        // ground truth: the window mean, which is what both arms should produce
        std::vector<double> want(sp, 0.0);
        for (int i = 0; i < S; ++i) want[i % sp] += v[i];
        for (auto& x : want) x /= (double)copies;

        Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());

        // ---- arm 1: explicit reduction ladder, then DENSE bootstrap
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        Ctx a1;
        for (int i = 0; i < timed; ++i) {
            Ctx r = ct->Clone();
            for (int k = 0; k < n_rot; ++k)
                r = fhe().cc->EvalAdd(r, fhe().cc->EvalRotate(r, (int)sp << k));
            a1 = bts_dense(fhe(), r);
        }
        cudaDeviceSynchronize();
        const double ms_explicit =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / timed;

        // ---- arm 2: sparse bootstrap on the UNREDUCED vector (fused)
        cudaDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        Ctx a2;
        for (int i = 0; i < timed; ++i) a2 = bts_sparse(fhe(), ct);
        cudaDeviceSynchronize();
        const double ms_fused =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / timed;

        auto g1 = decrypt_slots(fhe(), a1);
        auto g2 = decrypt_slots(fhe(), a2);
        std::vector<double> c1(sp), c2(sp);
        for (uint32_t i = 0; i < sp; ++i) {
            c1[i] = g1[i] / (double)copies;   // the ladder sums; normalise to the mean
            c2[i] = g2[i];                    // the fold already means
        }
        const double scale = A / std::sqrt((double)copies);   // scale of the mean itself
        const Stats s1 = stats_of(want, c1, scale), s2 = stats_of(want, c2, scale);
        std::cout << std::scientific << std::setprecision(3)
                  << "[se_mean] s=" << sp << " A=" << std::fixed << std::setprecision(2) << A
                  << " | explicit(" << n_rot << " rot + dense) " << std::setprecision(2)
                  << ms_explicit << " ms " << std::scientific << "rms=" << s1.rms
                  << " abs_max=" << s1.abs_max
                  << " | fused(sparse) " << std::fixed << std::setprecision(2) << ms_fused
                  << " ms " << std::scientific << "rms=" << s2.rms << " abs_max=" << s2.abs_max
                  << " | net " << std::fixed << std::setprecision(2) << (ms_fused - ms_explicit)
                  << " ms " << (100.0 * (ms_fused - ms_explicit) / ms_explicit) << " %"
                  << std::endl;
    }
    SUCCEED();
}

// ═════════════════════════════════════════════════════════════════════════════
// Where the RANGE wall for the fused reduction sits: on the ELEMENTS or
// on the SUM?
//
// The sum is already ruled out: at s=1, A=2 over 32768 slots the fused call is
// accurate to 1.6e-8 while the sum of that vector is ~±200, twenty times past
// the EvalMod wall. So the fold is not carrying the sum through EvalMod.
//
// Two candidates remain, and they differ enormously for real lanes:
//   (a) the ELEMENTS must satisfy |v_i| <= m  — then the wall is the usual ~10
//       and a reduction over any number of slots is free.
//   (b) the MEAN must satisfy |mean| <= m     — then a zero-mean vector is safe
//       to huge A (mean ~ A/sqrt(n)) but a ONE-SIDED vector is not (mean ~ A).
//
// LayerNorm variance sums SQUARES, so its summands are one-sided and its mean
// does not shrink with n. If (b) holds, that is the binding constraint for the
// exact site this lever targets. So sweep both payload signs.
//
// `sym`  = uniform(-A, A)  : mean ~ A/sqrt(n), elements up to A
// `pos`  = uniform(0, A)   : mean ~ A/2,       elements up to A
// ═════════════════════════════════════════════════════════════════════════════
TEST_F(SparseEnvelopeTest, MeanRangeWall) {
    const int S = slots();
    const uint32_t sp = sparse_slots(fhe());
    ASSERT_GT(sp, 0u) << "needs SPARSE_BTS_SLOTS=<s>";
    const int copies = S / (int)sp;

    std::vector<double> amps = {1, 2, 5, 8, 10, 12, 15, 20, 30, 50};
    if (const char* e = std::getenv("SE_AMPS")) {
        amps.clear();
        std::string t;
        for (const char* p = e;; ++p) {
            if (*p && *p != ',') { t += *p; continue; }
            if (!t.empty()) { amps.push_back(std::atof(t.c_str())); t.clear(); }
            if (!*p) break;
        }
    }

    for (int oneSided = 0; oneSided < 2; ++oneSided) {
        const char* tag = oneSided ? "pos" : "sym";
        for (double A : amps) {
            std::mt19937 g(90909);
            std::uniform_real_distribution<double> d(oneSided ? 0.0 : -A, A);
            std::vector<double> v(S);
            for (auto& x : v) x = d(g);

            std::vector<double> want(sp, 0.0);
            for (int i = 0; i < S; ++i) want[i % sp] += v[i];
            for (auto& x : want) x /= (double)copies;
            double want_absmax = 0.0;
            for (double x : want) want_absmax = std::max(want_absmax, std::fabs(x));

            try {
                Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
                Ctx out = bts_sparse(fhe(), ct);
                auto got = decrypt_slots(fhe(), out);
                std::vector<double> c(sp);
                for (uint32_t i = 0; i < sp; ++i) c[i] = got[i];
                const Stats st = stats_of(want, c, want_absmax);
                std::cout << std::scientific << std::setprecision(3)
                          << "[se_wall] s=" << sp << " dist=" << tag
                          << " elem_max=" << A << " mean_absmax=" << want_absmax
                          << " rms=" << st.rms << " abs_max=" << st.abs_max
                          << " rel=" << (want_absmax > 0 ? st.abs_max / want_absmax : 0.0)
                          << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[se_wall] s=" << sp << " dist=" << tag << " elem_max=" << A
                          << " THREW: " << e.what() << std::endl;
            }
        }
    }
    SUCCEED();
}

// ═════════════════════════════════════════════════════════════════════════════
// The real site: cachemir LayerNorm variance, fused.
//
// Mirrors src/packing/cachemir/cachemir_norm_utils.cu exactly:
//
//   PackedCtx var = inf.fhe->square(centered_x_in);
//   for (int gap = 1; gap < S; gap *= 2)          // 15 rotations at logN 16
//       inf.fhe->inplace_add(var, inf.fhe->rotate(var, gap));
//   inf.fhe->inplace_mult(var, 1.0 / (double)rD); // rD = 768
//
// and the cachemir layout from pack_per_feature_vec in the same file: feature k
// at slot k*t with t = slots/d_pad, everything else ZERO. That full all-reduce
// is only correct because cachemir decode has ONE live token; prefill uses
// diagonal::compute_per_token_var instead. Which is exactly why s=1 fits here.
//
//   arm 1 (today):  square -> 15 rotate+adds -> x1/rD -> DENSE bootstrap
//   arm 2 (fused):  square -> sparse s=1 bootstrap -> x(S/rD), a free constant
//
// Both produce the variance broadcast to every slot, which is what the
// Goldschmidt inv_sqrt chain downstream consumes.
//
// Amplitudes chosen so the variance lands across the Goldschmidt band the
// pipeline actually operates in (~0.006 … 0.4, cf. cutmax sum_lo/sum_hi).
// ═════════════════════════════════════════════════════════════════════════════
TEST_F(SparseEnvelopeTest, CachemirFusedVariance) {
    const int S = slots();
    const uint32_t sp = sparse_slots(fhe());
    ASSERT_EQ(sp, 1u) << "this axis is the s=1 fold — build with SPARSE_BTS_SLOTS=1";
    const int d_pad = (int)env_num("SE_DPAD", 1024);
    const int rD = (int)env_num("SE_RD", 768);
    const int t = S / d_pad;                 // cachemir feature stride
    const int timed = (int)env_num("SE_REP_N", 10);
    ASSERT_LE(rD * t, S);

    std::cout << "[se_cm] slots=" << S << " d_pad=" << d_pad << " rD=" << rD
              << " feature_stride=" << t << " explicit_rotations="
              << (int)std::lround(std::log2((double)S)) << std::endl;

    for (double A : {0.25, 0.5, 1.0, 2.0}) {
        std::mt19937 g(0xCACE);
        std::normal_distribution<double> nd(0.0, A);
        // centered_x in cachemir layout: feature k at slot k*t, zeros elsewhere
        std::vector<double> x(S, 0.0);
        double sumsq = 0.0;
        for (int k = 0; k < rD; ++k) {
            const double val = nd(g);
            x[(size_t)k * t] = val;
            sumsq += val * val;
        }
        const double true_var = sumsq / (double)rD;

        Ctx ct = encrypt(fhe().cc, encode(fhe().cc, x), fhe().pk());

        // ---- arm 1: the unfused path
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        Ctx a1;
        for (int i = 0; i < timed; ++i) {
            Ctx var = fhe().cc->EvalSquare(ct);
            for (int gap = 1; gap < S; gap *= 2)
                var = fhe().cc->EvalAdd(var, fhe().cc->EvalRotate(var, gap));
            var = fhe().cc->EvalMult(var, 1.0 / (double)rD);
            a1 = bts_dense(fhe(), var);
        }
        cudaDeviceSynchronize();
        const double ms_today =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / timed;

        // ---- arm 2: fused. The fold returns sumsq/S; the recovery constant S/rD
        // turns that into sumsq/rD = the variance, and any downstream linear op
        // absorbs it for free.
        cudaDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        Ctx a2;
        for (int i = 0; i < timed; ++i) {
            Ctx sq = fhe().cc->EvalSquare(ct);
            a2 = bts_sparse(fhe(), sq);
        }
        cudaDeviceSynchronize();
        const double ms_fused =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / timed;

        auto g1 = decrypt_slots(fhe(), a1);
        auto g2 = decrypt_slots(fhe(), a2);
        const double recov = (double)S / (double)rD;
        // both arms broadcast the variance to every slot; check the worst slot
        double e1 = 0.0, e2 = 0.0;
        for (int i = 0; i < S; ++i) {
            e1 = std::max(e1, std::fabs(g1[i] - true_var));
            e2 = std::max(e2, std::fabs(g2[i] * recov - true_var));
        }
        std::cout << std::scientific << std::setprecision(3)
                  << "[se_cm] x_sigma=" << std::fixed << std::setprecision(2) << A
                  << " true_var=" << std::scientific << true_var
                  << " folded_mean=" << (true_var * (double)rD / (double)S)
                  << " | today " << std::fixed << std::setprecision(2) << ms_today
                  << " ms err=" << std::scientific << e1 << " rel=" << (e1 / true_var)
                  << " | fused " << std::fixed << std::setprecision(2) << ms_fused
                  << " ms err=" << std::scientific << e2 << " rel=" << (e2 / true_var)
                  << " | net " << std::fixed << std::setprecision(2) << (ms_fused - ms_today)
                  << " ms " << (100.0 * (ms_fused - ms_today) / ms_today) << " %"
                  << " err_ratio " << std::scientific << (e1 > 0 ? e2 / e1 : 0.0) << std::endl;
    }
    SUCCEED();
}

}  // namespace
