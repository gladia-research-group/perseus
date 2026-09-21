#include "residency_pipeline.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

namespace {

class PersistentWorker {
public:
    std::future<void> submit(std::function<void()> job) {
        std::packaged_task<void()> task(std::move(job));
        std::future<void> fut = task.get_future();
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!started_) { th_ = std::thread([this] { loop(); }); started_ = true; }
            task_ = std::move(task);
            has_task_ = true;
        }
        cv_.notify_one();
        return fut;
    }
    ~PersistentWorker() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_one();
        if (th_.joinable()) th_.join();
    }

private:
    void loop() {
        for (;;) {
            std::packaged_task<void()> task;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return has_task_ || stop_; });
                if (stop_ && !has_task_) return;
                task = std::move(task_);
                has_task_ = false;
            }
            task();   // exceptions land in the future, same as std::async
        }
    }
    std::mutex m_;
    std::condition_variable cv_;
    std::packaged_task<void()> task_;
    bool has_task_ = false, stop_ = false, started_ = false;
    std::thread th_;
};

std::future<void> submit_prefetch(std::function<void()> job) {
    static PersistentWorker worker;
    return worker.submit(std::move(job));
}

}  // namespace

void run_residency_pipeline(Inference& inf, std::vector<ResidencyStage> stages,
                            Overlap mode) {
    const int n = static_cast<int>(stages.size());
    if (n <= 0) return;

    auto ACQ = [&](int i, cudaStream_t s) { if (stages[i].acquire) stages[i].acquire(inf, s); };
    auto INS = [&](int i) { if (stages[i].install) stages[i].install(inf); };
    auto CMP = [&](int i) { if (stages[i].compute) stages[i].compute(inf); };
    auto REL = [&](int i) { if (stages[i].release) stages[i].release(inf); };

    bool any_acquire = false;
    for (const auto& st : stages) any_acquire |= static_cast<bool>(st.acquire);
    cudaStream_t stream = nullptr;
    const bool use_stream =
        (mode == Overlap::Stream) || (mode == Overlap::Sync && any_acquire);
    if (use_stream) cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
    const cudaStream_t astream = use_stream ? stream : nullptr;

    bool any_cpu_pf = false;
    for (const auto& st : stages) any_cpu_pf |= static_cast<bool>(st.prefetch_cpu);

    std::future<void> fut;

    if (any_cpu_pf) {

        auto EXTRACT = [&](int k) { if (k < n && stages[k].prefetch_cpu) stages[k].prefetch_cpu(inf); };

        EXTRACT(0);
        ACQ(0, astream);
        if (use_stream) cudaDeviceSynchronize();
        EXTRACT(1);

        for (int i = 0; i < n; ++i) {
            INS(i);
            const bool ext2 = (i + 2 < n) && static_cast<bool>(stages[i + 2].prefetch_cpu);
            if (ext2) fut = submit_prefetch([&, i] { EXTRACT(i + 2); });
            if (i + 1 < n) ACQ(i + 1, astream);   // upload i+1 (stash hit) on astream — overlaps CMP(i)
            CMP(i);
            if (ext2) fut.get();                  // stash[i+2] ready for the next iteration's upload
            REL(i);                               // block_sync drains the astream upload(i+1)
        }
    } else {
        ACQ(0, astream);
        if (use_stream) cudaDeviceSynchronize();
        for (int i = 0; i < n; ++i) {
            INS(i);
            const bool overlap = (mode != Overlap::Sync) && (i + 1 < n) && stages[i].prefetch_next;
            if (overlap) {
                if (mode == Overlap::Threaded)
                    fut = submit_prefetch([&, i] { ACQ(i + 1, astream); });
                else
                    ACQ(i + 1, astream);
            }
            CMP(i);
            if (overlap && mode == Overlap::Stream && use_stream) cudaDeviceSynchronize();
            REL(i);
            if (overlap && mode == Overlap::Threaded) fut.get();   // join before next install
            if (!overlap && i + 1 < n) {                           // Sync boundary
                ACQ(i + 1, astream);
                if (use_stream) cudaDeviceSynchronize();
            }
        }
    }

    if (use_stream) cudaStreamDestroy(stream);
}
