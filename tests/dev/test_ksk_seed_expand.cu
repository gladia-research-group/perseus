// The ChaCha12 KSK seed expander (KskSeedExpand.cuh) expands bit-identically on CPU and
// GPU. Pure function-level test — no CKKS context, no keys. The OpenFHE patch series
// carries a plain-C++ copy of the same expander, and this gate is what proves the two
// copies agree, by expanding the same (seed, digit, modulus, slot) space on both.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <CKKS/KskSeedExpand.cuh>

using namespace FIDESlib::CKKS::kskexpand;

namespace {

constexpr uint32_t kN = 65536;
constexpr uint32_t kN16 = kN >> 4;

__global__ void expandLimbKernel(uint32_t* out, const uint32_t* key8, uint32_t digit, uint32_t p, uint32_t n16,
 uint32_t n) {
 const uint32_t s = threadIdx.x + blockIdx.x * blockDim.x;
 if (s < n)
 out[s] = expand_coeff(key8, digit, p, s, n16);
}

__global__ void expandLimbKernel64(uint64_t* out, const uint32_t* key8, uint32_t digit, uint64_t p, uint32_t n8,
                                   uint32_t n) {
    const uint32_t s = threadIdx.x + blockIdx.x * blockDim.x;
    if (s < n)
        out[s] = expand_coeff64(key8, digit, p, s, n8);
}

struct LimbCase {
 uint32_t digit;
 uint32_t p;
};

}

TEST(KskSeedExpand, CpuGpuParity) {
 ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

 const uint32_t key[8] = {0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u,
 0xa4093822u, 0x299f31d0u, 0x082efa98u, 0xec4e6c89u};

 const LimbCase cases[] = {
 {0, 134217689u},
 {1, 134217757u},
 {3, 268435399u},
 {5, 268435459u},
 {2, 260301049u},
 };

 uint32_t* d_out = nullptr;
 uint32_t* d_key = nullptr;
 ASSERT_EQ(cudaMalloc(&d_out, kN * sizeof(uint32_t)), cudaSuccess);
 ASSERT_EQ(cudaMalloc(&d_key, 8 * sizeof(uint32_t)), cudaSuccess);
 ASSERT_EQ(cudaMemcpy(d_key, key, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice), cudaSuccess);

 std::vector<uint32_t> cpu(kN), gpu(kN);
 for (const auto& c : cases) {
 uint64_t sum = 0;
 uint32_t escalated = 0;
 const uint32_t m_p = (0xFFFFFFFFu / c.p) * c.p;
 for (uint32_t s = 0; s < kN; ++s) {
 cpu[s] = expand_coeff(key, c.digit, c.p, s, kN16);
 ASSERT_LT(cpu[s], c.p);
 sum += cpu[s];
 uint32_t base[16];
 chacha_block(key, s >> 4, c.digit, c.p, base);
 if (base[s & 15u] >= m_p)
 ++escalated;
 }

 expandLimbKernel<<<(kN + 255) / 256, 256>>>(d_out, d_key, c.digit, c.p, kN16, kN);
 ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
 ASSERT_EQ(cudaMemcpy(gpu.data(), d_out, kN * sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);

 uint32_t mismatches = 0, first_bad = kN;
 for (uint32_t s = 0; s < kN; ++s)
 if (cpu[s] != gpu[s]) {
 if (++mismatches == 1)
 first_bad = s;
 }
 const double mean_frac = (double)sum / kN / c.p;
 std::printf("[ksk_expand] digit=%u p=%u: mismatches=%u/%u escalated=%u mean/p=%.4f\n", c.digit, c.p,
 mismatches, kN, escalated, mean_frac);
 EXPECT_EQ(mismatches, 0u) << "first mismatch at slot " << first_bad;
 if (c.p == 260301049u)
 EXPECT_GT(escalated, 0u);
 EXPECT_NEAR(mean_frac, 0.5, 0.01);
 }

 EXPECT_EQ(expand_coeff(key, 0, 134217689u, 12345u, kN16), [&] {
 for (uint32_t s = 0; s < kN; ++s)
 cpu[s] = expand_coeff(key, 0, 134217689u, s, kN16);
 return cpu[12345];
 }());
 EXPECT_NE(expand_coeff(key, 0, 134217689u, 7u, kN16), expand_coeff(key, 1, 134217689u, 7u, kN16));

 cudaFree(d_out);
 cudaFree(d_key);
}

// ── SPEC v2 (KSKB, 64-bit lane) parity — the NATIVE_SIZE=64 port ─────────────────────
TEST(KskSeedExpand, CpuGpuParity64) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr uint32_t kN8 = kN >> 3;
    const uint32_t key[8] = {0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u,
                             0xa4093822u, 0x299f31d0u, 0x082efa98u, 0xec4e6c89u};
    struct Case64 { uint32_t digit; uint64_t p; };
    // The n64 chain's width classes: 53-bit scale primes, 60-bit q0/special primes, plus a
    // modulus at ~2^64/16.5 for maximal rejection pressure (same construction as the u32 gate).
    const Case64 cases[] = {
        {0, (1ull << 53) - 111ull},   // 53-bit, below the pow2 (prime: 2^53-111)
        {1, (1ull << 53) + 5ull},     // 53-bit, above (2^53+5, prime)
        {3, (1ull << 60) - 93ull},    // 60-bit (2^60-93, prime) — the q0/special class
        {2, 1117984489315730401ull},  // ~2^64/16.5: reject prob ~0.03 -> escalation exercised
    };
    uint64_t* d_out = nullptr;
    uint32_t* d_key = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, kN * sizeof(uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_key, 8 * sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_key, key, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice), cudaSuccess);

    std::vector<uint64_t> cpu(kN), gpu(kN);
    for (const auto& c : cases) {
        long double sum = 0;
        uint32_t escalated = 0;
        const uint64_t m_p = (0xFFFFFFFFFFFFFFFFull / c.p) * c.p;
        for (uint32_t s = 0; s < kN; ++s) {
            cpu[s] = expand_coeff64(key, c.digit, c.p, s, kN8);
            ASSERT_LT(cpu[s], c.p);
            sum += (long double)cpu[s];
            uint32_t base[16];
            const uint32_t w13 = (c.digit & 0xFu) | ((uint32_t)(c.p >> 32) << 4);
            chacha_block_tail(key, s >> 3, w13, (uint32_t)c.p, kDomainSep64, base);
            const uint32_t w = s & 7u;
            const uint64_t v = (uint64_t)base[2u * w] | ((uint64_t)base[2u * w + 1u] << 32);
            if (v >= m_p)
                ++escalated;
        }
        expandLimbKernel64<<<(kN + 255) / 256, 256>>>(d_out, d_key, c.digit, c.p, kN8, kN);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(gpu.data(), d_out, kN * sizeof(uint64_t), cudaMemcpyDeviceToHost), cudaSuccess);
        uint32_t mismatches = 0, first_bad = kN;
        for (uint32_t s = 0; s < kN; ++s)
            if (cpu[s] != gpu[s] && ++mismatches == 1)
                first_bad = s;
        const double mean_frac = (double)(sum / (long double)kN / (long double)c.p);
        std::printf("[ksk_expand64] digit=%u p=%llu: mismatches=%u/%u escalated=%u mean/p=%.4f\n", c.digit,
                    (unsigned long long)c.p, mismatches, kN, escalated, mean_frac);
        EXPECT_EQ(mismatches, 0u) << "first mismatch at slot " << first_bad;
        if (c.p == 1117984489315730401ull)
            EXPECT_GT(escalated, 0u);
        EXPECT_NEAR(mean_frac, 0.5, 0.01);
    }

    // Determinism + lane separation: KSKB re-expansion is stable, and the KSKA lane at the
    // same (digit, p mod 2^32, slot) reads a DIFFERENT stream (domain separator + layout).
    EXPECT_EQ(expand_coeff64(key, 0, (1ull << 53) - 111ull, 12345u, kN8),
              expand_coeff64(key, 0, (1ull << 53) - 111ull, 12345u, kN8));
    EXPECT_NE(expand_coeff64(key, 0, (1ull << 53) - 111ull, 7u, kN8) % 0xFFFFFFFFull,
              (uint64_t)expand_coeff(key, 0, (uint32_t)((1ull << 53) - 111ull), 7u, kN16) % 0xFFFFFFFFull);

    cudaFree(d_out);
    cudaFree(d_key);
}
