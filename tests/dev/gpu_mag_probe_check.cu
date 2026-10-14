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
