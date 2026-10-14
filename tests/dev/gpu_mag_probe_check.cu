// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Alexander Ananyev
// gpu_mag_probe_check — the device magnitude probe (include/gpu_mag_probe.h) against the host path's
// numbers on the same ciphertexts: max|z|, the Re mean and max|z - mean| from the decrypted slots,
// the max coefficient with and without X^0 from FFTSpecialInv of those slots (fideslib_wrapper.h's
// lane_split / slot_stats / coeff_stats, reproduced here). Cases: dense at two levels, a large-value
// vector, a degree-2 product, a 512-slot ciphertext. No gtest: JSON lines; exit 0 iff every relative
// difference is <= 1e-9.
//
//   gpu_mag_probe_check [LEVEL_A=34] [LEVEL_B=44] [SEED=5]      (n32 env; bootstrap off, no rotation keys)
#include "fideslib_wrapper.h"
#include "gpu_mag_probe.h"
#include "test_helpers.h"

#include <CKKS/Ciphertext.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <math/dftransform.h>

#include <algorithm>
#include <any>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

struct HostStats {
    double max_abs = 0, mean = 0, max_dev = 0, max_coeff = 0, max_coeff_ac = 0;
};

// the facade's three measurements on a decoded slot vector
HostStats host_stats(const std::vector<std::complex<double>>& vz, uint32_t ring_dim) {
    HostStats h;
    long double acc = 0.0L;
    for (const auto& z : vz) {
        h.max_abs = std::max(h.max_abs, std::abs(z));
        acc += z.real();
    }
    h.mean = static_cast<double>(acc / static_cast<long double>(vz.size()));
    for (const auto& z : vz) h.max_dev = std::max(h.max_dev, std::abs(z - std::complex<double>(h.mean, 0.0)));
    std::vector<std::complex<double>> inv(vz);
    lbcrypto::DiscreteFourierTransform::FFTSpecialInv(inv, ring_dim * 2);
    for (std::size_t i = 0; i < inv.size(); ++i) {
        const double re = std::abs(inv[i].real()), im = std::abs(inv[i].imag());
        const double m = std::max(re, im);
        h.max_coeff = std::max(h.max_coeff, m);
        h.max_coeff_ac = std::max(h.max_coeff_ac, (i == 0) ? im : m);
    }
    return h;
}

double rel(double a, double b) {
    const double s = std::max(std::fabs(a), std::fabs(b));
    return s > 0 ? std::fabs(a - b) / s : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    const uint32_t LA = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 34u;
    const uint32_t LB = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 44u;
    const unsigned seed = argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 5u;
    CKKSContextOptions o = ckks_options_from_env();
    o.enable_bootstrap = false;
    o.depth = 26;
    o.extra_rot_steps.clear();
    o.h_weight = 0;
    o.ckks_complex_payload = true;
    auto fhe = make_ckks_context(o);
    CC cc = fhe->cc;
    auto& fctx = std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu);
    FIDESlib::CKKS::ContextData& cd = *fctx;
    auto& occ = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
    const uint32_t ring = cc->GetRingDimension();
    const int n = static_cast<int>(ring / 2);

    // the secret key on the device, imported as the facade imports it
    auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(fhe->sk()->pimpl);
    std::vector<double> zeros(static_cast<std::size_t>(n), 0.0);
    lbcrypto::Plaintext spt = occ->MakeCKKSPackedPlaintext(zeros, 1, 0);
    spt->GetElement<lbcrypto::DCRTPoly>() = skImpl->GetPrivateElement();
    auto gs = std::make_shared<FIDESlib::CKKS::Plaintext>(fctx, FIDESlib::CKKS::GetRawPlainText(occ, spt));
    gpu_encode::GpuMagProbe probe(fctx, n, gs);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-0.3, 0.3);
    std::vector<double> x(n), y(n), big(n), x512(512);
    for (auto& v : x) v = U(rng);
    for (auto& v : y) v = U(rng);
    for (int i = 0; i < n; ++i) big[i] = 1.0e3 * x[i];
    for (auto& v : x512) v = U(rng);

    auto enc = [&](const std::vector<double>& v, uint32_t L, uint32_t slots) {
        Ptx pt = slots ? cc->MakeCKKSPackedPlaintext(v, 1, L, nullptr, slots) : cc->MakeCKKSPackedPlaintext(v, 1, L);
        Ctx c = encrypt(cc, pt, fhe->pk());
        return cc->EvalAdd(c, 0.0);   // onto the device, value unchanged
    };

#ifdef GPU_MAG_PROBE_DEBUG
    if (argc > 4 && std::string(argv[4]) == "diag") {
        // Where does the decryption go wrong? Host: the decrypted polynomial's coefficients (OpenFHE
        // Decrypt, element in COEFFICIENT format, towers 0/1 -> integer by CRT). Device: the same from
        // the probe's k_decrypt + INTT, components as stored and swapped. Also the imported key's
        // limb 0 against OpenFHE's tower 0 (EVAL) in natural and bit-reversed order.
        auto crt2 = [](uint64_t r0, uint64_t q0, uint64_t r1, uint64_t q1) {
            auto inv = [](uint64_t a, uint64_t m) { uint64_t e = m - 2, r = 1; a %= m; while (e) { if (e & 1) r = (uint64_t)((__int128)r * a % m); a = (uint64_t)((__int128)a * a % m); e >>= 1; } return r; };
            const uint64_t d = (r1 + q1 - (r0 % q1)) % q1;
            const uint64_t t = (uint64_t)((__int128)d * inv(q0 % q1, q1) % q1);
            __int128 m = (__int128)r0 + (__int128)q0 * t;
            const __int128 M = (__int128)q0 * q1;
            if (m > M / 2) m -= M;
            return (long long)m;
        };
        Ptx xpt = cc->MakeCKKSPackedPlaintext(x, 1, LA);
        Ctx c_cpu = encrypt(cc, xpt, fhe->pk());   // keeps its OpenFHE element (EVAL form)
        Ctx c = cc->EvalAdd(c_cpu, 0.0);             // the device twin, same value
        Plaintext pt;
        Ctx copy = c;
        cc->Decrypt(copy, fhe->sk(), &pt);
        // (the decoded slot values are the host reference; the decrypted polynomial's integers are
        //  not kept in the plaintext's RNS element after Decrypt, so the device's residues are judged
        //  by their per-limb consistency and the host Garner below)
        const double host_scale = std::any_cast<lbcrypto::Plaintext&>(pt->cpu)->GetScalingFactor();
        const double host_mean_times_scale = [&] { const auto vz = pt->GetCKKSPackedValue(); long double a = 0; for (auto& z : vz) a += z.real(); return (double)(a / (long double)vz.size()) * host_scale; }();
        std::printf("{\"diag\":\"host\",\"mean_times_scale\":%.6e,\"scale\":%.6e}\n", host_mean_times_scale, host_scale);
        std::fflush(stdout);
        const uint64_t q0 = cd.param.primes.at(0).p, q1 = cd.param.primes.at(1).p;
        auto gct = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(cc->GetDeviceCiphertext(c->gpu));
        for (int sw = 0; sw < 2; ++sw) {
            auto dev = probe.debug_decrypt(*gct, sw == 1);
            std::printf("{\"diag\":\"device\",\"swap\":%d,\"limbs\":%zu,\"N\":%zu,\"coeffs_crt2\":[", sw, dev.size(), dev.empty() ? 0 : dev[0].size());
            for (int k = 0; k < 6; ++k) std::printf("%s%.6e", k ? "," : "", (double)crt2(dev[0][k], q0, dev[1][k], q1));
            std::printf("],\"res0\":[");
            for (int k = 0; k < 6; ++k) std::printf("%s%lu", k ? "," : "", (unsigned long)dev[0][k]);
            std::printf("]}\n");
            std::fflush(stdout);
        }
        // Are the higher limbs' residues consistent with the small integer the first two limbs give, and
        // does the mixed-radix (Garner) reconstruction on the host agree with it?
        {
            auto dev = probe.debug_decrypt(*gct, false);
            const int nl = static_cast<int>(dev.size());
            std::vector<uint64_t> q(nl);
            for (int l = 0; l < nl; ++l) q[l] = cd.param.primes.at(l).p;
            auto inv = [](uint64_t a, uint64_t m) { uint64_t e = m - 2, r = 1; a %= m; while (e) { if (e & 1) r = (uint64_t)((__int128)r * a % m); a = (uint64_t)((__int128)a * a % m); e >>= 1; } return r; };
            for (int k = 0; k < 4; ++k) {
                const long long m2 = crt2(dev[0][k], q[0], dev[1][k], q[1]);
                std::printf("{\"diag\":\"consistency\",\"k\":%d,\"crt2\":%lld,\"mismatch_limbs\":[", k, m2);
                bool first = true;
                for (int l = 0; l < nl; ++l) {
                    const long long r = ((m2 % (long long)q[l]) + (long long)q[l]) % (long long)q[l];
                    if ((uint64_t)r != dev[l][k]) { std::printf("%s%d", first ? "" : ",", l); first = false; }
                }
                // host Garner, the kernel's algorithm in exact integers
                std::vector<uint64_t> t(nl);
                for (int l = 0; l < nl; ++l) {
                    uint64_t x = dev[l][k];
                    for (int i = 0; i < l; ++i) {
                        uint64_t ti = t[i] % q[l];
                        x = (x >= ti) ? (x - ti) : (x + q[l] - ti);
                        x = (uint64_t)((__int128)x * inv(q[i] % q[l], q[l]) % q[l]);
                    }
                    t[l] = x;
                }
                const bool neg = t[nl - 1] > (q[nl - 1] >> 1);
                long double v = neg ? 1.0L : 0.0L, P = 1.0L;
                int top_nonzero = -1;
                for (int l = 0; l < nl; ++l) {
                    const uint64_t d = neg ? (q[l] - 1 - t[l]) : t[l];
                    if (d) { v += (long double)d * P; top_nonzero = l; }
                    P *= (long double)q[l];
                }
                std::printf("],\"garner_host\":%.6Le,\"neg\":%d,\"top_nonzero_digit\":%d,\"digits\":[", neg ? -v : v, (int)neg, top_nonzero);
                for (int l = 0; l < std::min(nl, 6); ++l) std::printf("%s%lu", l ? "," : "", (unsigned long)t[l]);
                std::printf("]}\n");
            }
            std::fflush(stdout);
        }
        // the key: OpenFHE tower 0 in EVAL form vs the imported limb 0
        const auto& sk_el = skImpl->GetPrivateElement();
        const auto& skt0 = sk_el.GetAllElements()[0].GetValues();
        auto sdev = probe.debug_secret_limbs();
        auto brev16 = [](uint32_t k) { uint32_t r = 0; for (int b = 0; b < 16; ++b) r |= ((k >> b) & 1u) << (15 - b); return r; };
        size_t eq_nat = 0, eq_rev = 0;
        for (uint32_t k = 0; k < ring; ++k) {
            eq_nat += (sdev[0][k] == skt0[k].ConvertToInt());
            eq_rev += (sdev[0][brev16(k)] == skt0[k].ConvertToInt());
        }
        std::printf("{\"diag\":\"secret\",\"format_eval\":%d,\"towers\":%zu,\"dev_limbs\":%zu,\"limb0_eq_natural\":%zu,\"limb0_eq_bitrev\":%zu,\"of\":%u,\"sk0\":[%lu,%lu,%lu],\"dev0\":[%lu,%lu,%lu]}\n",
                    sk_el.GetFormat() == Format::EVALUATION, sk_el.GetAllElements().size(), sdev.size(), eq_nat, eq_rev, ring,
                    (unsigned long)skt0[0].ConvertToInt(), (unsigned long)skt0[1].ConvertToInt(), (unsigned long)skt0[2].ConvertToInt(),
                    (unsigned long)sdev[0][0], (unsigned long)sdev[0][1], (unsigned long)sdev[0][2]);
        // and the ciphertext's own limb 0 against OpenFHE's (EVAL) view of the same ciphertext
        auto& cto = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(c_cpu->cpu);
        std::vector<std::vector<uint64_t>> c0dev;
        gct->c0.store(c0dev);
        cudaDeviceSynchronize();
        const auto& ct0 = cto->GetElements()[0].GetAllElements()[0].GetValues();
        size_t ceq_nat = 0, ceq_rev = 0;
        for (uint32_t k = 0; k < ring; ++k) {
            ceq_nat += (c0dev[0][k] == ct0[k].ConvertToInt());
            ceq_rev += (c0dev[0][brev16(k)] == ct0[k].ConvertToInt());
        }
        std::printf("{\"diag\":\"ciphertext_c0\",\"format_eval\":%d,\"limb0_eq_natural\":%zu,\"limb0_eq_bitrev\":%zu,\"of\":%u,\"cpu_loaded\":%d}\n",
                    cto->GetElements()[0].GetFormat() == Format::EVALUATION, ceq_nat, ceq_rev, ring, (int)c->loaded);
        return 0;
    }
#else
    if (argc > 4 && std::string(argv[4]) == "diag") {
        std::fprintf(stderr, "diag mode needs a build with CACHEMIR_MAG_PROBE_DEBUG=ON (the probe's debug accessors)\n");
        return 2;
    }
#endif
    struct Case {
        std::string name;
        Ctx ct;
    };
    std::vector<Case> cases;
    cases.push_back({"dense_LA", enc(x, LA, 0)});
    cases.push_back({"dense_LB", enc(x, LB, 0)});
    cases.push_back({"big_LA", enc(big, LA, 0)});
    cases.push_back({"deg2_LA", cc->EvalMult(enc(x, LA, 0), enc(y, LA, 0))});
    cases.push_back({"sparse512_LA", enc(x512, LA, 512)});

    double worst = 0.0;
    for (auto& c : cases) {
        if (!c.ct->loaded) {
            std::printf("{\"case\":\"%s\",\"error\":\"ciphertext not on the device\"}\n", c.name.c_str());
            worst = 1.0;
            continue;
        }
        Plaintext pt;
        Ctx copy = c.ct;
        cc->Decrypt(copy, fhe->sk(), &pt);
        const auto vz = pt->GetCKKSPackedValue();
        const HostStats h = host_stats(vz, ring);
        auto gct = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(cc->GetDeviceCiphertext(c.ct->gpu));
        probe.probe(*gct);
        const auto r = probe.fetch().at(0);
        const double d_abs = rel(h.max_abs, r.max_abs), d_dev = rel(h.max_dev, r.max_dev),
                     d_c = rel(h.max_coeff, r.max_coeff), d_cac = rel(h.max_coeff_ac, r.max_coeff_ac),
                     d_mean = h.max_abs > 0 ? std::fabs(h.mean - r.mean) / h.max_abs : 0.0;
        const double w = std::max({d_abs, d_dev, d_c, d_cac, d_mean});
        worst = std::max(worst, w);
        std::printf("{\"case\":\"%s\",\"slots\":%zu,\"noise_deg\":%d,\"gpu_slots\":%d,\"gpu_noise_level\":%d,"
                    "\"host\":{\"max_abs\":%.12e,\"mean\":%.12e,\"max_dev\":%.12e,\"max_coeff\":%.12e,\"max_coeff_ac\":%.12e},"
                    "\"gpu\":{\"max_abs\":%.12e,\"mean\":%.12e,\"max_dev\":%.12e,\"max_coeff\":%.12e,\"max_coeff_ac\":%.12e},"
                    "\"rel\":{\"max_abs\":%.2e,\"mean\":%.2e,\"max_dev\":%.2e,\"max_coeff\":%.2e,\"max_coeff_ac\":%.2e},\"worst\":%.2e}\n",
                    c.name.c_str(), vz.size(), static_cast<int>(c.ct->GetNoiseScaleDeg()), gct->slots, gct->NoiseLevel,
                    h.max_abs, h.mean, h.max_dev, h.max_coeff, h.max_coeff_ac,
                    r.max_abs, r.mean, r.max_dev, r.max_coeff, r.max_coeff_ac,
                    d_abs, d_mean, d_dev, d_c, d_cac, w);
        std::fflush(stdout);
    }
    const auto& st = probe.stats();
    std::printf("{\"probe\":\"stats\",\"items\":%lu,\"enqueue_ms\":%.3f,\"fetch_ms\":%.3f}\n", (unsigned long)st.items,
                st.enqueue_ms, st.fetch_ms);
    const bool ok = worst <= 1e-9;
    std::printf("{\"probe\":\"verdict\",\"worst_rel\":%.3e,\"pass\":%s}\n", worst, ok ? "true" : "false");
    return ok ? 0 : 1;
}
