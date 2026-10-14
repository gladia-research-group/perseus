// gpu_mag_probe.cu — see include/gpu_mag_probe.h.
#include "gpu_mag_probe.h"

#include "gpu_cuda_raii.h"

#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/Limb.cuh>
#include <CKKS/LimbPartition.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/RNSPoly.cuh>

#include <cuda_runtime.h>
#include <cufft.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace gpu_encode {

namespace {

constexpr int kMaxLimbs = 64;  // FIDESlib's MAXP
constexpr int kThreads = 256;
constexpr const char* kWho = "gpu_mag_probe";

void check(cudaError_t e, const char* what) { cuda_check(e, kWho, what); }
void check_fft(cufftResult r, const char* what) { cufft_check(r, kWho, what); }

__device__ __forceinline__ uint32_t barrett64(uint64_t a, uint64_t q, uint64_t mu) {
    const uint64_t r = a - __umul64hi(a, mu) * q;
    return static_cast<uint32_t>(r >= q ? r - q : r);
}

struct DecParams {
    void* const* c0;  // limb pointer tables (device): c0, c1, secret, destination
    void* const* c1;
    void* const* s;
    void* const* out;
    uint64_t q[kMaxLimbs];
    uint64_t mu[kMaxLimbs];
};

// d_l = c0_l + c1_l * s_l (mod q_l), pointwise in the EVAL layout. grid (N / kThreads, nlimbs).
__global__ void k_decrypt(DecParams p, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (i >= N) return;
    const uint32_t* c0 = static_cast<const uint32_t*>(p.c0[l]);
    const uint32_t* c1 = static_cast<const uint32_t*>(p.c1[l]);
    const uint32_t* s = static_cast<const uint32_t*>(p.s[l]);
    uint32_t* out = static_cast<uint32_t*>(p.out[l]);
    const uint64_t prod = static_cast<uint64_t>(c1[i]) * s[i] + c0[i];  // < 2^57
    out[i] = barrett64(prod, p.q[l], p.mu[l]);
}

// Garner constants, per probe instance (device): cinv[i*kMaxLimbs + j] = q_i^{-1} mod q_j (i < j);
// Pd[i] = prod_{l<i} q_l as a double (inf once past 2^1024: a digit there means a value beyond any
// sane magnitude, reported as inf).
struct GarnerParams {
    void* const* m;  // the scratch poly's limb table after the INTT (natural COEFF order)
    const uint32_t* cinv;
    const double* Pd;
    uint64_t q[kMaxLimbs];
    uint64_t mu[kMaxLimbs];
};

// Every coefficient j of the decrypted polynomial as a double in scale units: mixed-radix digits
// (exact, 32-bit), sign from the top digit, the double from the digits. Also the block maxima of
// |c_j| over the stride-aligned j (the coefficients OpenFHE's sparse decode reads) and the same
// without j = 0. grid N / kThreads.
__global__ void k_garner(GarnerParams p, int nlimbs, int N, int gap, double inv_scale,
                         double* __restrict__ coef, double* __restrict__ pmax, double* __restrict__ pmax_ac) {
    __shared__ double s_max[kThreads], s_max_ac[kThreads];
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (j < N) {
        uint32_t t[kMaxLimbs];
        for (int l = 0; l < nlimbs; ++l) {
            const uint64_t ql = p.q[l];
            uint64_t x = static_cast<const uint32_t*>(p.m[l])[j];
            for (int i = 0; i < l; ++i) {
                // t_i < q_i, and q_0 (28 bits) can exceed twice a 26.5-bit q_l: a full reduction, not
                // one conditional subtract
                const uint64_t ti = barrett64(t[i], ql, p.mu[l]);
                x = (x >= ti) ? (x - ti) : (x + ql - ti);
                x = barrett64(x * __ldg(p.cinv + i * kMaxLimbs + l), ql, p.mu[l]);
            }
            t[l] = static_cast<uint32_t>(x);
        }
        const bool neg = t[nlimbs - 1] > (p.q[nlimbs - 1] >> 1);
        double v = neg ? 1.0 : 0.0;
        for (int l = 0; l < nlimbs; ++l) {
            const uint64_t d = neg ? (p.q[l] - 1 - t[l]) : t[l];
            if (d) v += static_cast<double>(d) * __ldg(p.Pd + l);  // Pd may be inf: a nonzero digit there is inf
        }
        value = (neg ? -v : v) * inv_scale;
        coef[j] = value;
    }
    const bool aligned = (j < N) && ((j % gap) == 0);
    const double a = aligned ? fabs(value) : 0.0;
    s_max[threadIdx.x] = a;
    s_max_ac[threadIdx.x] = (j == 0) ? 0.0 : a;
    __syncthreads();
    for (int st = kThreads / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            s_max[threadIdx.x] = fmax(s_max[threadIdx.x], s_max[threadIdx.x + st]);
            s_max_ac[threadIdx.x] = fmax(s_max_ac[threadIdx.x], s_max_ac[threadIdx.x + st]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pmax[blockIdx.x] = s_max[0];
        pmax_ac[blockIdx.x] = s_max_ac[0];
    }
}

// X_k = (c_{k gap} + i c_{k gap + n}) zeta^{k gap}, k < slots: the input of the inverse DFT of size
// `slots` whose output is the decoded slot vector in permuted order.
__global__ void k_twist_sub(const double* __restrict__ coef, const cufftDoubleComplex* __restrict__ tw, int n,
                            int gap, int slots, cufftDoubleComplex* __restrict__ X) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= slots) return;
    const int j = k * gap;
    const double re = coef[j], im = coef[j + n];
    const cufftDoubleComplex w = tw[j];
    X[k] = make_cuDoubleComplex(re * w.x - im * w.y, re * w.y + im * w.x);
}

// One block: max|Y|, mean Re Y, max|Y - mean| over `slots` values, the coefficient maxima from the
// block partials; writes Result `out`. Launched with kStatsThreads threads.
constexpr int kStatsThreads = 1024;
__global__ void k_stats(const cufftDoubleComplex* __restrict__ Y, int slots, const double* __restrict__ pmax,
                        const double* __restrict__ pmax_ac, int nblocks, GpuMagProbe::Result* out) {
    __shared__ double s_a[kStatsThreads], s_b[kStatsThreads];
    __shared__ double s_mean;
    double mx = 0.0, sum = 0.0;
    for (int k = threadIdx.x; k < slots; k += blockDim.x) {
        const cufftDoubleComplex y = Y[k];
        mx = fmax(mx, hypot(y.x, y.y));
        sum += y.x;
    }
    s_a[threadIdx.x] = mx;
    s_b[threadIdx.x] = sum;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            s_a[threadIdx.x] = fmax(s_a[threadIdx.x], s_a[threadIdx.x + st]);
            s_b[threadIdx.x] += s_b[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out->max_abs = s_a[0];
        s_mean = s_b[0] / slots;
        out->mean = s_mean;
    }
    __syncthreads();
    const double mean = s_mean;
    double dv = 0.0;
    for (int k = threadIdx.x; k < slots; k += blockDim.x) {
        const cufftDoubleComplex y = Y[k];
        dv = fmax(dv, hypot(y.x - mean, y.y));
    }
    double cm = 0.0, cma = 0.0;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        cm = fmax(cm, pmax[b]);
        cma = fmax(cma, pmax_ac[b]);
    }
    s_a[threadIdx.x] = dv;
    s_b[threadIdx.x] = cm;
    __shared__ double s_c[kStatsThreads];
    s_c[threadIdx.x] = cma;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            s_a[threadIdx.x] = fmax(s_a[threadIdx.x], s_a[threadIdx.x + st]);
            s_b[threadIdx.x] = fmax(s_b[threadIdx.x], s_b[threadIdx.x + st]);
            s_c[threadIdx.x] = fmax(s_c[threadIdx.x], s_c[threadIdx.x + st]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out->max_dev = s_a[0];
        out->max_coeff = s_b[0];
        out->max_coeff_ac = s_c[0];
    }
}

// a^{-1} mod m for a prime m < 2^32 (every product of two residues fits in 64 bits).
uint64_t inv_mod(uint64_t a, uint64_t m) {
    uint64_t e = m - 2, r = 1;
    a %= m;
    while (e) {
        if (e & 1) r = (r * a) % m;
        a = (a * a) % m;
        e >>= 1;
    }
    return r;
}

// Both buffers of a limb over their whole allocation: the library's NTT kernels sweep [0, alloc),
// not [0, size), and the aux buffer keeps the transform's intermediate of the last polynomial.
template <class T>
void zero_limb(FIDESlib::CKKS::Limb<T>& lb) noexcept {
    if (lb.v.data) note_teardown(cudaMemset(lb.v.data, 0, sizeof(T) * lb.v.alloc), kWho, "zero limb");
    if (lb.aux.data) note_teardown(cudaMemset(lb.aux.data, 0, sizeof(T) * lb.aux.alloc), kWho, "zero limb aux");
}

}  // namespace

std::string take_pending_cuda_error() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess || benign_pending(e) ? std::string() : std::string(cudaGetErrorString(e));
}

void zero_plaintext_limbs(FIDESlib::CKKS::Plaintext& p) noexcept {
    using FIDESlib::CKKS::Limb;
    for (auto& part : p.c0.GPU) {
        for (auto& var : part.limb) {
            if (auto* lb = std::get_if<Limb<uint32_t>>(&var))
                zero_limb(*lb);
            else if (auto* lb64 = std::get_if<Limb<uint64_t>>(&var))
                zero_limb(*lb64);
        }
    }
    note_teardown(cudaGetLastError(), kWho, "error left pending after zeroing");
}

// A poly whose limbs are zeroed when the owner goes away: the key and the scratch poly (which holds
// the last decrypted polynomial). Runs on the failure paths of the constructor too, since Impl's
// members are destroyed when the constructor throws.
struct WipedPoly {
    std::shared_ptr<FIDESlib::CKKS::Plaintext> p;
    WipedPoly() = default;
    explicit WipedPoly(std::shared_ptr<FIDESlib::CKKS::Plaintext> q) : p(std::move(q)) {}
    ~WipedPoly() {
        if (p) {
            note_teardown(cudaDeviceSynchronize(), kWho, "drain before zeroing a poly");
            zero_plaintext_limbs(*p);
            note_teardown(cudaDeviceSynchronize(), kWho, "drain after zeroing a poly");
        }
    }
    WipedPoly(const WipedPoly&) = delete;
    WipedPoly& operator=(const WipedPoly&) = delete;
    WipedPoly(WipedPoly&&) = default;
    WipedPoly& operator=(WipedPoly&& o) noexcept {
        if (this != &o) {
            if (p) zero_plaintext_limbs(*p);
            p = std::move(o.p);
        }
        return *this;
    }
    FIDESlib::CKKS::Plaintext* operator->() const { return p.get(); }
    explicit operator bool() const { return static_cast<bool>(p); }
};

struct GpuMagProbe::Impl {
    FIDESlib::CKKS::Context& ctx;
    FIDESlib::CKKS::ContextData& cc;
    int N = 0, n = 0;
    WipedPoly secret;   // zeroed when the probe goes away, built or not
    WipedPoly scratch;  // one poly at the top level, level set per probe; zeroed likewise
    void* const* s_table = nullptr;                       // the secret's limb pointer table
    void* const* d_table = nullptr;                       // the scratch's
    std::vector<uint64_t> q, mu;
    bool stream_checked = false;
    // destroyed bottom-up: the plans first, then the buffers
    DevicePtr<uint32_t> cinv;        // kMaxLimbs * kMaxLimbs
    DevicePtr<double> Pd;            // kMaxLimbs + 1
    DevicePtr<double> coef;          // N
    DevicePtr<double> pmax;          // N / kThreads
    DevicePtr<double> pmax_ac;
    DevicePtr<cufftDoubleComplex> tw;  // n: zeta^k
    DevicePtr<cufftDoubleComplex> X;   // n
    DevicePtr<Result> results;         // capacity
    std::size_t capacity = 0, count = 0;
    int nblocks = 0;                   // k_garner's grid: the partial maxima per block
    std::vector<std::pair<int, FftPlan>> plans;
    cudaStream_t s0 = nullptr;  // the legacy default stream

    Impl(FIDESlib::CKKS::Context& c, int slots) : ctx(c), cc(*c), n(slots) {}
    ~Impl() {
        // The decoded coefficients and slots, and their partial maxima, hold the last decrypted
        // value: wiped before their memory returns (the results buffer holds the capture's own
        // statistics). The key and the scratch poly are wiped by their WipedPoly owners, destroyed
        // after this body.
        note_teardown(cudaDeviceSynchronize(), kWho, "drain before the wipe");
        if (coef) note_teardown(cudaMemset(coef.get(), 0, sizeof(double) * N), kWho, "wipe coef");
        if (X) note_teardown(cudaMemset(X.get(), 0, sizeof(cufftDoubleComplex) * n), kWho, "wipe X");
        if (pmax) note_teardown(cudaMemset(pmax.get(), 0, sizeof(double) * nblocks), kWho, "wipe pmax");
        if (pmax_ac) note_teardown(cudaMemset(pmax_ac.get(), 0, sizeof(double) * nblocks), kWho, "wipe pmax_ac");
        note_teardown(cudaDeviceSynchronize(), kWho, "drain after the wipe");
    }

    cufftHandle plan_for(int size) {
        for (auto& [sz, h] : plans)
            if (sz == size) return h.get();
        FftPlan plan = FftPlan::plan1d(size, CUFFT_Z2Z, 1, kWho);
        check_fft(cufftSetStream(plan.get(), s0), "cufftSetStream");
        plans.emplace_back(size, std::move(plan));
        return plans.back().second.get();
    }
};

GpuMagProbe::GpuMagProbe(FIDESlib::CKKS::Context& ctx, int n_slots, std::shared_ptr<FIDESlib::CKKS::Plaintext> secret)
    : impl_(std::make_unique<Impl>(ctx, n_slots)) {
    using FIDESlib::CKKS::Limb;
    Impl& I = *impl_;
    I.N = 1 << I.cc.param.logN;
    if (2 * I.n != I.N) throw std::invalid_argument("gpu_mag_probe: slots must be N/2");
    if (!secret || secret->c0.GPU.size() != 1)
        throw std::invalid_argument("gpu_mag_probe: secret key poly missing or multi-device");
    I.secret = WipedPoly(std::move(secret));
    const int nprimes = I.cc.param.L + 1;
    if (nprimes > kMaxLimbs) throw std::invalid_argument("gpu_mag_probe: too many primes");
    if (static_cast<int>(I.secret->c0.GPU[0].limb.size()) < nprimes)
        throw std::invalid_argument("gpu_mag_probe: the secret key poly must carry every limb of the chain");
    for (int l = 0; l < nprimes; ++l) {
        if (!std::holds_alternative<Limb<uint32_t>>(I.secret->c0.GPU[0].limb[l]))
            throw std::runtime_error("gpu_mag_probe: 32-bit limbs only");
        const auto& lb = std::get<Limb<uint32_t>>(I.secret->c0.GPU[0].limb[l]);
        if (lb.primeid != l) throw std::runtime_error("gpu_mag_probe: secret limb order");
        const uint64_t qv = I.cc.param.primes.at(l).p;
        if (qv >= (1ull << 32)) throw std::runtime_error("gpu_mag_probe: prime wider than 32 bits");
        I.q.push_back(qv);
        I.mu.push_back(~0ull / qv);
    }
    I.s_table = I.secret->c0.GPU[0].limbptr.data;
    // the scratch poly: every limb of the chain, with aux (the INTT needs them)
    I.scratch = WipedPoly(std::make_shared<FIDESlib::CKKS::Plaintext>(I.ctx));
    I.scratch->c0.grow(nprimes - 1, false, false);
    if (static_cast<int>(I.scratch->c0.GPU[0].limb.size()) < nprimes)
        throw std::runtime_error("gpu_mag_probe: scratch poly did not grow");
    I.d_table = I.scratch->c0.GPU[0].limbptr.data;
    // Garner constants
    std::vector<uint32_t> cinv(static_cast<size_t>(kMaxLimbs) * kMaxLimbs, 0);
    for (int i = 0; i < nprimes; ++i)
        for (int j = i + 1; j < nprimes; ++j)
            cinv[static_cast<size_t>(i) * kMaxLimbs + j] = static_cast<uint32_t>(inv_mod(I.q[i] % I.q[j], I.q[j]));
    std::vector<double> Pd(kMaxLimbs + 1, INFINITY);
    long double acc = 1.0L;
    for (int i = 0; i <= nprimes; ++i) {
        Pd[i] = (acc > 1.7e308L) ? INFINITY : static_cast<double>(acc);
        if (i < nprimes) acc *= static_cast<long double>(I.q[i]);
    }
    I.cinv = device_alloc<uint32_t>(cinv.size(), kWho, "cudaMalloc cinv");
    check(cudaMemcpy(I.cinv.get(), cinv.data(), sizeof(uint32_t) * cinv.size(), cudaMemcpyHostToDevice), "cinv upload");
    I.Pd = device_alloc<double>(Pd.size(), kWho, "cudaMalloc Pd");
    check(cudaMemcpy(I.Pd.get(), Pd.data(), sizeof(double) * Pd.size(), cudaMemcpyHostToDevice), "Pd upload");
    // twist zeta^k, k < n, zeta = exp(2 pi i / 4n)
    std::vector<cufftDoubleComplex> tw(I.n);
    for (int k = 0; k < I.n; ++k) {
        const long double ang = 2.0L * 3.141592653589793238462643383279502884L * k / (4.0L * I.n);
        tw[k].x = static_cast<double>(std::cos(ang));
        tw[k].y = static_cast<double>(std::sin(ang));
    }
    I.tw = device_alloc<cufftDoubleComplex>(I.n, kWho, "cudaMalloc tw");
    check(cudaMemcpy(I.tw.get(), tw.data(), sizeof(cufftDoubleComplex) * I.n, cudaMemcpyHostToDevice), "upload tw");
    I.X = device_alloc<cufftDoubleComplex>(I.n, kWho, "cudaMalloc X");
    I.coef = device_alloc<double>(I.N, kWho, "cudaMalloc coef");
    I.nblocks = (I.N + kThreads - 1) / kThreads;
    I.pmax = device_alloc<double>(I.nblocks, kWho, "cudaMalloc pmax");
    I.pmax_ac = device_alloc<double>(I.nblocks, kWho, "cudaMalloc pmax_ac");
    I.capacity = 1 << 16;
    I.results = device_alloc<Result>(I.capacity, kWho, "cudaMalloc results");
}

GpuMagProbe::~GpuMagProbe() {
    // the key, the scratch poly and the decoded buffers are wiped by Impl's teardown
    note_teardown(cudaDeviceSynchronize(), kWho, "drain");
    note_teardown(cudaGetLastError(), kWho, "error left pending at teardown");
}

std::size_t GpuMagProbe::probe(const FIDESlib::CKKS::Ciphertext& ct) {
    Impl& I = *impl_;
    const auto t0 = std::chrono::steady_clock::now();
    if (I.count == I.capacity) throw std::runtime_error("gpu_mag_probe: fetch() before issuing more than 65536 probes");
    const int nlimbs = ct.c0.getLevel() + 1;
    if (nlimbs < 1 || nlimbs > static_cast<int>(I.q.size()))
        throw std::runtime_error("gpu_mag_probe: ciphertext level out of range");
    if (ct.c0.GPU.size() != 1 || ct.c1.GPU.size() != 1)
        throw std::runtime_error("gpu_mag_probe: multi-device ciphertext");
    const int slots = ct.slots;
    if (slots <= 0 || slots > I.n || (I.n % slots) != 0)
        throw std::runtime_error("gpu_mag_probe: slot count " + std::to_string(slots) + " is not a divisor of N/2");
    const int gap = I.n / slots;
    const double scale = ct.NoiseFactor;
    if (!(scale > 0.0)) throw std::runtime_error("gpu_mag_probe: ciphertext without a scale");
    if (!I.stream_checked) {
        if (!stream_is_blocking(ct.c0.GPU[0].getS().ptr(), kWho))
            throw std::runtime_error("gpu_mag_probe: the ciphertext's stream is non-blocking; the legacy-stream "
                                     "ordering this probe relies on does not hold");
        I.stream_checked = true;
    }

    DecParams dp{};
    dp.c0 = ct.c0.GPU[0].limbptr.data;
    dp.c1 = ct.c1.GPU[0].limbptr.data;
    dp.s = I.s_table;
    dp.out = I.d_table;
    GarnerParams gp{};
    gp.m = I.d_table;
    gp.cinv = I.cinv.get();
    gp.Pd = I.Pd.get();
    for (int l = 0; l < nlimbs; ++l) {
        dp.q[l] = gp.q[l] = I.q[l];
        dp.mu[l] = gp.mu[l] = I.mu[l];
    }
    const int nblocks = (I.N + kThreads - 1) / kThreads;
    drain_pending(kWho, "k_decrypt launch");
    k_decrypt<<<dim3(nblocks, nlimbs), kThreads, 0, I.s0>>>(dp, I.N);
    check(cudaGetLastError(), "k_decrypt launch");
    I.scratch->c0.setLevel(nlimbs - 1);
    I.scratch->c0.INTT(I.cc.param.batch, true);
    drain_pending(kWho, "k_garner launch");   // the library's INTT ran in between
    k_garner<<<nblocks, kThreads, 0, I.s0>>>(gp, nlimbs, I.N, gap, 1.0 / scale, I.coef.get(), I.pmax.get(),
                                             I.pmax_ac.get());
    check(cudaGetLastError(), "k_garner launch");
    drain_pending(kWho, "k_twist_sub launch");
    k_twist_sub<<<(slots + kThreads - 1) / kThreads, kThreads, 0, I.s0>>>(I.coef.get(), I.tw.get(), I.n, gap, slots,
                                                                          I.X.get());
    check(cudaGetLastError(), "k_twist_sub launch");
    if (slots > 1) check_fft(cufftExecZ2Z(I.plan_for(slots), I.X.get(), I.X.get(), CUFFT_INVERSE), "cufftExecZ2Z");
    drain_pending(kWho, "k_stats launch");   // cuFFT ran in between
    k_stats<<<1, kStatsThreads, 0, I.s0>>>(I.X.get(), slots, I.pmax.get(), I.pmax_ac.get(), nblocks,
                                           I.results.get() + I.count);
    check(cudaGetLastError(), "k_stats launch");
    stats_.items += 1;
    stats_.enqueue_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return I.count++;
}

std::vector<GpuMagProbe::Result> GpuMagProbe::fetch() {
    Impl& I = *impl_;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Result> out(I.count);
    if (I.count) {
        check(cudaStreamSynchronize(I.s0), "fetch sync");
        check(cudaMemcpy(out.data(), I.results.get(), sizeof(Result) * I.count, cudaMemcpyDeviceToHost), "fetch copy");
    }
    I.count = 0;
    stats_.fetch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace gpu_encode
