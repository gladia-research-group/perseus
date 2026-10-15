// gpu_cuda_raii.h — owners for the CUDA handles the magnitude probe holds: device buffers and
// cuFFT plans. Each is move-only and releases in its destructor, so a constructor that throws halfway
// (typically cudaMalloc out of memory) frees what it had already acquired.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

// The magnitude probe orders its work against the ciphertext operations through the legacy default
// stream (gpu_mag_probe.h); a translation unit compiled for the per-thread default stream would
// silently lose that ordering.
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#error "gpu_mag_probe: built with --default-stream per-thread; the legacy default stream is required"
#endif

namespace gpu_encode {

inline void cuda_check(cudaError_t e, const char* who, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(who) + ": " + what + ": " + cudaGetErrorString(e));
}
inline void cufft_check(cufftResult r, const char* who, const char* what) {
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(who) + ": " + what + ": cufft error " + std::to_string(int(r)));
}
// A failure met while tearing down (a destructor, a noexcept release, a wipe): reported on stderr,
// never thrown. A sticky context error shows up here as the failing synchronize, and is what the
// next real call reports again.
inline void note_teardown(cudaError_t e, const char* who, const char* what) noexcept {
    if (e != cudaSuccess) std::fprintf(stderr, "[%s] teardown: %s: %s\n", who, what, cudaGetErrorString(e));
}
// The codes FIDESlib's own launch check lets through (CudaCheckErrorModNoSync, src/CudaUtils.cuh): a
// graph update that fell back to a new instantiation, peer access already enabled, the runtime unloading
// at exit. The library's normal operation leaves them pending; they are no fault of earlier work.
inline bool benign_pending(cudaError_t e) noexcept {
    return e == cudaErrorGraphExecUpdateFailure || e == cudaErrorPeerAccessAlreadyEnabled ||
           e == cudaErrorCudartUnloading;
}
// The library polls its own launches for errors only under FIDESLIB_ERRCHECK=1, so a failed library
// launch stays pending in the thread's error slot until someone reads it. Drained before each of this
// code's launches and named as earlier work, so that the launch check that follows reports only its
// own launch; the codes the library itself lets through are consumed.
inline void drain_pending(const char* who, const char* before) {
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess && !benign_pending(e))
        throw std::runtime_error(std::string(who) + ": a CUDA error pending from earlier work, found before " +
                                 before + ": " + cudaGetErrorString(e));
}

// A device allocation: unique_ptr<T, DeviceFree>; T is the element type the pointer is used as.
struct DeviceFree {
    void operator()(void* p) const noexcept {
        if (p) cudaFree(p);
    }
};
template <class T>
using DevicePtr = std::unique_ptr<T, DeviceFree>;

template <class T>
DevicePtr<T> device_alloc(std::size_t count, const char* who, const char* what) {
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, sizeof(T) * count), who, what);
    return DevicePtr<T>(static_cast<T*>(p));
}

// A cuFFT plan handle, made by cufftPlan1d.
class FftPlan {
  public:
    FftPlan() = default;
    static FftPlan plan1d(int size, cufftType type, int batch, const char* who) {
        FftPlan p;
        cufft_check(cufftPlan1d(&p.h_, size, type, batch), who, "cufftPlan1d");
        p.live_ = true;
        return p;
    }
    ~FftPlan() {
        if (live_) cufftDestroy(h_);
    }
    FftPlan(const FftPlan&) = delete;
    FftPlan& operator=(const FftPlan&) = delete;
    FftPlan(FftPlan&& o) noexcept : h_(o.h_), live_(std::exchange(o.live_, false)) {}
    FftPlan& operator=(FftPlan&& o) noexcept {
        if (this != &o) {
            if (live_) cufftDestroy(h_);
            h_ = o.h_;
            live_ = std::exchange(o.live_, false);
        }
        return *this;
    }
    cufftHandle get() const { return h_; }

  private:
    cufftHandle h_ = 0;
    bool live_ = false;
};

// True when `s` was created without cudaStreamNonBlocking (the legacy default stream itself counts).
inline bool stream_is_blocking(cudaStream_t s, const char* who) {
    if (s == nullptr) return true;
    unsigned flags = 0;
    cuda_check(cudaStreamGetFlags(s, &flags), who, "cudaStreamGetFlags");
    return (flags & cudaStreamNonBlocking) == 0;
}

}  // namespace gpu_encode
