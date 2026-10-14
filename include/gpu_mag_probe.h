// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Alexander Ananyev
// gpu_mag_probe.h — the graph capture's magnitude probe on the device.
//
// The capture prices every recorded node by three statistics of its decrypted value: the slot
// magnitude max|z|, the Re-mean and max|z - mean| (the offset arm), and the largest plaintext
// coefficient with and without the X^0 term (what EvalMod's range constraint is on). The host path
// copies each ciphertext down and decrypts it with OpenFHE: at ring 2^16 that is ~35 ms per node and
// ~29k nodes per block, i.e. the capture's wall. Here the same numbers are computed where the
// ciphertext already is: d = c0 + c1*s per limb (the secret key resident on the device in the
// library's own EVAL layout), the library's inverse NTT, an exact mixed-radix reconstruction of every
// coefficient to a double in scale units, one inverse cuFFT for the slots (the capture's statistics
// are invariant under the slot permutation, so the special FFT's output order is not needed), and
// two reductions. Sparse-slot ciphertexts decode as OpenFHE does: the coefficients at stride N/2/slots
// through a transform of size `slots`.
//
// Everything runs on the legacy default stream, which orders after every ciphertext operation that
// produced the value and before every one that may change it in place — provided the library's
// streams are blocking ones, which the first probe checks. The key lives on the device for the
// probe's lifetime. When the probe goes, the key and the scratch polynomial (both buffers of every
// limb, over their whole allocation) and the decoded coefficients and slots of the last probe are
// zeroed before their memory returns to the pool; the statistics buffer is the capture's own
// output. The two debug accessors (the decrypted residues, the key's limbs) exist only in builds
// with GPU_MAG_PROBE_DEBUG (CMake: CACHEMIR_MAG_PROBE_DEBUG=ON, the probe check's diag mode).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace FIDESlib::CKKS {
class Ciphertext;
class Plaintext;
class ContextData;
using Context = std::shared_ptr<ContextData>;
}  // namespace FIDESlib::CKKS

namespace gpu_encode {

// Zero every limb of a poly on the device (both of each limb's buffers, over their whole
// allocation); never throws, a failing memset is reported on stderr. The key's owner calls it
// before the memory returns to the pool, whether the probe was built or its construction failed.
void zero_plaintext_limbs(FIDESlib::CKKS::Plaintext& p) noexcept;

// Consume the thread's pending CUDA error and return its text; "" when there is none, or when it is a
// code the library itself lets through (gpu_cuda_raii.h, benign_pending). The capture asks before each
// probe: an error that earlier work left pending is not the probe's (fideslib_wrapper.h, mag_gpu_probe).
std::string take_pending_cuda_error();

class GpuMagProbe {
  public:
    struct Result {
        double max_abs = 0.0;       // max |z| over the decoded slots
        double mean = 0.0;          // mean of Re z
        double max_dev = 0.0;       // max |z - mean|
        double max_coeff = 0.0;     // max |coefficient| (scale units), all 2*slots coefficients
        double max_coeff_ac = 0.0;  // the same without coefficient X^0
    };
    struct Stats {
        uint64_t items = 0;
        double enqueue_ms = 0;  // host time spent issuing the probes
        double fetch_ms = 0;    // host time spent waiting at fetch (the device work)
    };

    // `secret`: the secret key polynomial imported like any plaintext (Plaintext(ctx, GetRawPlainText))
    // at the top level, EVAL form, all limbs. The probe takes ownership and zeroes it at the end,
    // including when its own construction fails after taking it; a constructor that throws before
    // taking it leaves the zeroing to the caller (zero_plaintext_limbs).
    GpuMagProbe(FIDESlib::CKKS::Context& ctx, int n_slots, std::shared_ptr<FIDESlib::CKKS::Plaintext> secret);
    ~GpuMagProbe();
    GpuMagProbe(const GpuMagProbe&) = delete;
    GpuMagProbe& operator=(const GpuMagProbe&) = delete;

    // Issue the probe of `ct` (EVAL form, any level, any noise degree); returns the index its result
    // will have in the next fetch().
    std::size_t probe(const FIDESlib::CKKS::Ciphertext& ct);
    // Wait for every probe issued since the last fetch and return their results in issue order.
    std::vector<Result> fetch();
    const Stats& stats() const { return stats_; }

#ifdef GPU_MAG_PROBE_DEBUG
    // Diagnostics: the decrypted polynomial's residues per limb after the INTT (natural order, host
    // vectors), with the ciphertext components in the stored order or swapped; and the secret key's
    // limbs as imported.
    std::vector<std::vector<uint64_t>> debug_decrypt(const FIDESlib::CKKS::Ciphertext& ct, bool swap_components);
    std::vector<std::vector<uint64_t>> debug_secret_limbs();
#endif

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Stats stats_;
};

}  // namespace gpu_encode
