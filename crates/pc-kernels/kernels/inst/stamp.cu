// PC_TL (diagnostic): a one-thread kernel writing the device's %globaltimer (ns) into buf[idx] (mapped pinned host
// memory, portable: every device writes it directly); capturable like any launch, so it times steps inside graphs.
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void pc_stamp_k(unsigned long long* buf, int idx) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    buf[idx] = t;
}

extern "C" int pc_stamp(unsigned long long* buf, int idx, void* stream) {
    pc_stamp_k<<<1, 1, 0, (cudaStream_t)stream>>>(buf, idx);
    return (int)cudaGetLastError();
}

extern "C" unsigned long long* pc_stamp_alloc(size_t n) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, n * sizeof(unsigned long long), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        return nullptr;
    }
    return (unsigned long long*)p;
}
