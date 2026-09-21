#include "fideslib_wrapper.h"
#include "inference.h"
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace test_helpers;

namespace {

constexpr int kBankStride = 64;   // softmax_v token-lane stride (j*64 rot keys seeded below)

class PackingBench : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {
        CKKSContextOptions o = default_ckks_options();
        const int s = (o.batch_size == 0) ? (1 << (o.logN - 1)) : static_cast<int>(o.batch_size);
        for (int i = 1; i <= s; i *= 2) { o.extra_rot_steps.push_back(i); o.extra_rot_steps.push_back(-i); }
        const char* bv = std::getenv("BANKS");
        const int bmax = (bv && *bv) ? std::atoi(bv) : 8;
        for (int j = 1; j < bmax; ++j)    // bank strides j*64 (non-p2 ones aren't p2-composable)
            o.extra_rot_steps.push_back(j * kBankStride);
        const char* dv = std::getenv("NDIAG");
        const int dmax = (dv && *dv) ? std::atoi(dv) : 8;
        for (int k = 1; k < dmax; ++k)    // diagonal strides k*1024 (= production rot2 = t^2)
            o.extra_rot_steps.push_back(k * 1024);
        ctx_   = make_ckks_context(o);
        slots_ = static_cast<int>(ctx_->cc->GetRingDimension() / 2);
    }
    static void TearDownTestSuite() { ctx_.reset(); }
    static CKKSContext& fhe() { return *ctx_; }
    static int slots() { return slots_; }

    inline static std::shared_ptr<CKKSContext> ctx_;
    inline static int slots_ = 0;
};

double now_ms() { using namespace std::chrono; return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count(); }
int env_int(const char* k, int dflt) { const char* v = std::getenv(k); return (v && *v) ? std::atoi(v) : dflt; }

Ptx i_plaintext(CKKSContext& fhe, int n) { return encode(fhe.cc, std::vector<std::complex<double>>(n, {0.0, 1.0})); }
Ctx pack_ri(CKKSContext& fhe, const Ctx& a, const Ctx& b, Ptx& i_pt) { return fhe.add(a, fhe.mult(b, i_pt)); }
Ctx times_i_mono(CKKSContext& fhe, const Ctx& x) {
    Ctx t = fhe.clone(x);
    fhe.cc->EvalMultMonomialInPlace(t, static_cast<uint32_t>(fhe.cc->GetRingDimension() / 2));
    return t;
}
// (Re, Im) sharing ONE conjugate (mirrors pair_unpack).
std::pair<Ctx, Ctx> unpack_ri(CKKSContext& fhe, const Ctx& P) {
    Ctx cj = fhe.conjugate(P);
    Ctx re = fhe.mult(fhe.add(P, cj), 0.5);
    Ctx im = fhe.mult(times_i_mono(fhe, fhe.sub(cj, P)), 0.5);
    return {re, im};
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b, int n) {
    double m = 0.0; for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i])); return m;
}

struct ArmRun {
    std::vector<Ctx> outs;                   // cold (tallied) outputs — err/level read here
    std::map<std::string, uint64_t> tally;
    double ms = 0.0;
};

uint64_t tget(const std::map<std::string, uint64_t>& t, std::initializer_list<const char*> keys) {
    uint64_t s = 0; for (const char* k : keys) { auto it = t.find(k); if (it != t.end()) s += it->second; }
    return s;
}

void emit_row(const std::string& site, const std::string& arm, int reps, const ArmRun& r, double err) {
    const auto& t = r.tally;
    const uint64_t bts  = tget(t, {"auto_bootstrap", "deliberate_bootstrap"});
    const uint64_t mcc  = tget(t, {"mult_cc"});
    const uint64_t mpt  = tget(t, {"mult_pt", "mult_inplace"});
    const uint64_t msc  = tget(t, {"mult_sc"});
    const uint64_t rot  = tget(t, {"rotate", "rotate_inplace"});
    const uint64_t conj = tget(t, {"conjugate"});
    const uint64_t adds = tget(t, {"add", "add_inplace", "sub", "sub_ct", "sub_inplace", "sub_inplace_ct", "negate", "negate_inplace"});
    const uint64_t sq   = tget(t, {"square", "square_inplace"});
    uint64_t known = bts + mcc + mpt + msc + rot + conj + adds + sq + tget(t, {"mult"});
    uint64_t total = 0; for (const auto& kv : t) total += kv.second;
    const uint64_t other = total - std::min(total, known);
    const int lvl = r.outs.empty() ? -1 : static_cast<int>(level_of(r.outs[0]));

    std::ostringstream os;
    os << site << "," << arm << "," << reps << "," << std::fixed << std::setprecision(2) << r.ms
       << "," << std::scientific << std::setprecision(2) << err << "," << lvl
       << "," << bts << "," << mcc << "," << mpt << "," << msc << "," << rot << "," << conj
       << "," << adds << "," << sq << "," << other;
    std::cout << "[csv] " << os.str() << std::endl;
    if (const char* p = std::getenv("PACKING_TABLE_CSV")) {
        std::ofstream f(p, std::ios::app);
        f << os.str() << "\n";
    }
}

void emit_header() {
    const char* h = "site,arm,reps,ms,err,lvl_out,bts,mult_cc,mult_pt,mult_sc,rot,conj,add_sub,square,other";
    std::cout << "[csv] " << h << std::endl;
    if (const char* p = std::getenv("PACKING_TABLE_CSV")) {
        std::ofstream f(p, std::ios::app);
        f << h << "\n";
    }
}

template <class F>
ArmRun run_arm(CKKSContext& fhe, int reps, F&& f) {
    ArmRun r;
    { auto w = f(); (void)w; cudaDeviceSynchronize(); }
    {
        CKKSContext::OpTallyScope tally(fhe);
        r.outs = f();
        cudaDeviceSynchronize();
        r.tally = fhe.op_tally;
    }
    std::vector<double> ts(reps);
    for (int i = 0; i < reps; ++i) {   // per-rep sync + median: robust to allocator drift
        const double t0 = now_ms();
        auto c = f(); (void)c;
        cudaDeviceSynchronize();
        ts[i] = now_ms() - t0;
    }
    std::sort(ts.begin(), ts.end());
    r.ms = ts[reps / 2];
    return r;
}

double arm_err(CKKSContext& fhe, const ArmRun& cpx, const std::vector<std::vector<double>>& refs, int n) {
    double m = 0.0;
    for (size_t i = 0; i < cpx.outs.size() && i < refs.size(); ++i)
        m = std::max(m, max_abs_diff(refs[i], decrypt_slots(fhe, cpx.outs[i]), n));
    return m;
}

std::vector<std::vector<double>> decrypt_all(CKKSContext& fhe, const ArmRun& r) {
    std::vector<std::vector<double>> v;
    for (const auto& c : r.outs) v.push_back(decrypt_slots(fhe, c));
    return v;
}

}  // namespace

TEST_F(PackingBench, PackingTable) {
    ASSERT_TRUE(std::getenv("CKKS_COMPLEX") && std::getenv("CKKS_COMPLEX")[0] == '1')
        << "run with CKKS_COMPLEX=1 (complex payload)";
    auto& F = fhe();
    const int n  = slots();
    const int R  = env_int("REPS", 5);
    const int ND = env_int("NDIAG", 8);   // diagonals per modeled linear (strides 2^0..2^(ND-1))
    const int B  = env_int("BANKS", 8);   // softmax_v value banks (even)
    const int RO = env_int("RO", 4);      // MLP-up output blocks (even)
    const int K  = env_int("TILES", 8);   // lm_head tiles (even)
    std::mt19937 gen(15);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    auto rnd = [&](double scale) { std::vector<double> v(n); for (auto& x : v) x = d(gen) * scale; return v; };
    auto enc = [&](const std::vector<double>& v) { return encrypt(F.cc, encode(F.cc, v), F.pk()); };
    Ptx i_pt = i_plaintext(F, n);
    auto pack_pt = [&](const std::vector<double>& re, const std::vector<double>& im) {
        std::vector<std::complex<double>> c(n);
        for (int i = 0; i < n; ++i) c[i] = {re[i], im[i]};
        return encode(F.cc, c);
    };

    auto diag_linear = [&](const Ctx& x, std::vector<Ptx>& W) {
        Ctx y = F.mult(x, W[0]);
        for (int k = 1; k < static_cast<int>(W.size()); ++k)
            y = F.add(y, F.mult(F.rotate(x, k * 1024), W[k]));
        return y;
    };
    auto reduce_all = [&](Ctx x) { for (int s = 1; s < n; s *= 2) x = F.add(x, F.rotate(x, s)); return x; };
    emit_header();

    {
        std::vector<std::vector<double>> wk(ND), wv(ND);
        std::vector<Ptx> Wk(ND), Wv(ND), Wkv(ND);
        for (int k = 0; k < ND; ++k) {
            wk[k] = rnd(1.0 / ND); wv[k] = rnd(1.0 / ND);
            Wk[k] = encode(F.cc, wk[k]); Wv[k] = encode(F.cc, wv[k]); Wkv[k] = pack_pt(wk[k], wv[k]);
        }
        Ctx x = enc(rnd(1.0));
        auto real = run_arm(F, R, [&]() {
            Ctx yk = diag_linear(x, Wk); F.bootstrap(yk);
            Ctx yv = diag_linear(x, Wv); F.bootstrap(yv);
            return std::vector<Ctx>{yk, yv};
        });
        auto refs = decrypt_all(F, real);
        auto cpx = run_arm(F, R, [&]() {
            Ctx yp = diag_linear(x, Wkv); F.bootstrap(yp);
            Ctx cj = F.conjugate(yp);
            Ctx k2 = F.add(yp, cj);                       // 2K
            Ctx v2 = times_i_mono(F, F.sub(cj, yp));      // 2V
            return std::vector<Ctx>{k2, v2};
        });
        double e = 0.0;   // compare at half scale (the 2x is mask-absorbed downstream)
        for (size_t i = 0; i < cpx.outs.size(); ++i) {
            auto got = decrypt_slots(F, cpx.outs[i]);
            for (auto& v : got) v *= 0.5;
            e = std::max(e, max_abs_diff(refs[i], got, n));
        }
        emit_row("kv_entry", "real", R, real, 0.0);
        emit_row("kv_entry", "cpx", R, cpx, e);
        EXPECT_LT(e, 1e-1) << "kv_entry lanes diverge (two independent bts noises)";
    }

    {
        std::vector<double> q = rnd(1.0), ka = rnd(1.0 / std::sqrt((double)n)), kb = rnd(1.0 / std::sqrt((double)n));
        Ctx cq = enc(q), cka = enc(ka), ckb = enc(kb);
        Ctx kpack = pack_ri(F, cka, ckb, i_pt);   // built at cache push, amortized
        auto real = run_arm(F, R, [&]() {
            return std::vector<Ctx>{reduce_all(F.mult(cq, cka)), reduce_all(F.mult(cq, ckb))};
        });
        auto refs = decrypt_all(F, real);
        auto cpx = run_arm(F, R, [&]() {
            auto sc = unpack_ri(F, reduce_all(F.mult(cq, kpack)));
            return std::vector<Ctx>{sc.first, sc.second};
        });
        const double e = arm_err(F, cpx, refs, n);
        emit_row("qkt", "real", R, real, 0.0);
        emit_row("qkt", "cpx", R, cpx, e);
        EXPECT_LT(e, 1e-3) << "fused Q.K^T scores diverge";
    }

    {
        std::vector<double> P = rnd(1.0 / B);
        std::vector<Ctx> V(B); for (int j = 0; j < B; ++j) V[j] = enc(rnd(1.0));
        std::vector<Ctx> Vhat(B / 2);
        for (int j = 0; j < B / 2; ++j) Vhat[j] = pack_ri(F, V[2 * j], V[2 * j + 1], i_pt);  // cache-resident
        Ctx cP = enc(P);
        auto real = run_arm(F, R, [&]() {
            Ctx o = F.mult(cP, V[0]);
            for (int j = 1; j < B; ++j) o = F.add(o, F.mult(F.rotate(cP, j * kBankStride), V[j]));
            return std::vector<Ctx>{o};
        });
        auto refs = decrypt_all(F, real);
        auto cpx = run_arm(F, R, [&]() {
            Ctx Pt = F.sub(cP, F.mult(F.rotate(cP, kBankStride), i_pt));   // P − i·rot(P, s)
            Ctx o = F.mult(Pt, Vhat[0]);
            for (int j = 1; j < B / 2; ++j) o = F.add(o, F.mult(F.rotate(Pt, 2 * j * kBankStride), Vhat[j]));
            o = F.mult(F.add(o, F.conjugate(o)), 0.5);                     // 2·Re, ×0.5 rides the head mask
            return std::vector<Ctx>{o};
        });
        const double e = arm_err(F, cpx, refs, n);
        emit_row("softmax_v", "real", R, real, 0.0);
        emit_row("softmax_v", "cpx", R, cpx, e);
        EXPECT_LT(e, 1e-3) << "complex-bank softmax_v context diverges";
    }

    {
        std::vector<std::vector<Ptx>> Wr(RO, std::vector<Ptx>(ND)), Wp(RO / 2, std::vector<Ptx>(ND));
        std::vector<std::vector<std::vector<double>>> wraw(RO, std::vector<std::vector<double>>(ND));
        std::mt19937 gen2(16);
        std::uniform_real_distribution<double> d2(-1.0, 1.0);
        for (int b = 0; b < RO; ++b)
            for (int k = 0; k < ND; ++k) {
                wraw[b][k].resize(n);
                for (auto& x : wraw[b][k]) x = d2(gen2) / ND;
                Wr[b][k] = encode(F.cc, wraw[b][k]);
            }
        for (int p = 0; p < RO / 2; ++p)
            for (int k = 0; k < ND; ++k) Wp[p][k] = pack_pt(wraw[2 * p][k], wraw[2 * p + 1][k]);
        Ctx x = enc(rnd(1.0));
        auto real = run_arm(F, R, [&]() {
            std::vector<Ctx> outs;
            for (int b = 0; b < RO; ++b) outs.push_back(diag_linear(x, Wr[b]));
            return outs;
        });
        auto refs = decrypt_all(F, real);
        auto cpx = run_arm(F, R, [&]() {
            std::vector<Ctx> outs;
            for (int p = 0; p < RO / 2; ++p) {
                auto ri = unpack_ri(F, diag_linear(x, Wp[p]));   // pre-cascade unpack: +1 level
                outs.push_back(ri.first); outs.push_back(ri.second);
            }
            return outs;
        });
        const double e = arm_err(F, cpx, refs, n);
        emit_row("mlp_up", "real", R, real, 0.0);
        emit_row("mlp_up", "cpx", R, cpx, e);
        EXPECT_LT(e, 1e-3) << "output-row-packed MLP up diverges";
    }

    {
        std::vector<std::vector<std::vector<double>>> traw(K, std::vector<std::vector<double>>(ND));
        std::vector<std::vector<Ptx>> Tr(K, std::vector<Ptx>(ND)), Tp(K / 2, std::vector<Ptx>(ND));
        std::mt19937 gen3(17);
        std::uniform_real_distribution<double> d3(-1.0, 1.0);
        for (int m = 0; m < K; ++m)
            for (int k = 0; k < ND; ++k) {
                traw[m][k].resize(n);
                for (auto& x : traw[m][k]) x = d3(gen3) / ND;
                Tr[m][k] = encode(F.cc, traw[m][k]);
            }
        for (int p = 0; p < K / 2; ++p)
            for (int k = 0; k < ND; ++k) Tp[p][k] = pack_pt(traw[2 * p][k], traw[2 * p + 1][k]);
        Ctx h = enc(rnd(1.0));
        auto real = run_arm(F, R, [&]() {
            std::vector<Ctx> outs;
            for (int m = 0; m < K; ++m) outs.push_back(diag_linear(h, Tr[m]));
            return outs;
        });
        auto refs = decrypt_all(F, real);
        auto cpx = run_arm(F, R, [&]() {          // terminal: pairs stay packed (argmax reads them)
            std::vector<Ctx> outs;
            for (int p = 0; p < K / 2; ++p) outs.push_back(diag_linear(h, Tp[p]));
            return outs;
        });

        double e = 0.0;
        for (int p = 0; p < K / 2; ++p) {
            auto ri = unpack_ri(F, cpx.outs[p]);
            e = std::max(e, max_abs_diff(refs[2 * p], decrypt_slots(F, ri.first), n));
            e = std::max(e, max_abs_diff(refs[2 * p + 1], decrypt_slots(F, ri.second), n));
        }
        emit_row("lm_head", "real", R, real, 0.0);
        emit_row("lm_head", "cpx", R, cpx, e);
        EXPECT_LT(e, 1e-3) << "paired lm_head tiles diverge";
    }
}
