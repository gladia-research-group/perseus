#include <cstdio>
#include "interrupt.h"
#include "residency_pipeline.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <condition_variable>
#include <deque>
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
            q_.push_back(std::move(task));
        }
        cv_.notify_one();
        return fut;
    }
    ~PersistentWorker() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

private:
    void loop() {
        for (;;) {
            std::packaged_task<void()> task;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return !q_.empty() || stop_; });
                if (stop_ && q_.empty()) return;
                task = std::move(q_.front());
                q_.pop_front();
            }
            task();   // exceptions land in the future, same as std::async
        }
    }
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::packaged_task<void()>> q_;
    bool stop_ = false, started_ = false;
    std::thread th_;
};

std::future<void> submit_prefetch(std::function<void()> job) {
    static PersistentWorker worker;
    return worker.submit(std::move(job));
}

// A SECOND worker, for the scoped-mask staging only. It cannot share submit_prefetch's
// worker: that queue is FIFO and the block loop submits EXTRACT(i+2) every iteration and
// joins it in the same iteration, so a long mask job there would stall the extractions.
std::future<void> submit_mask(std::function<void()> job) {
    static PersistentWorker worker;
    return worker.submit(std::move(job));
}

}  // namespace

std::future<void> residency_submit(std::function<void()> job) {
    return submit_prefetch(std::move(job));
}

std::future<void> mask_submit(std::function<void()> job) {
    return submit_mask(std::move(job));
}

void run_residency_pipeline(Inference& inf, std::vector<ResidencyStage> stages,
                            Overlap mode) {
    const int n = static_cast<int>(stages.size());
    if (n <= 0) return;

    const std::thread::id main_tid = std::this_thread::get_id();
    auto on_main = [main_tid] { return std::this_thread::get_id() == main_tid; };

    auto ACQ = [&](int i, cudaStream_t s) {
        if (!stages[i].acquire) return;
        if (on_main()) { WithStep _w(inf, "res_acquire"); stages[i].acquire(inf, s); }
        else            stages[i].acquire(inf, s);
    };
    auto INS = [&](int i) {
        if (!stages[i].install) return;
        if (on_main()) { WithStep _w(inf, "res_install"); stages[i].install(inf); }
        else            stages[i].install(inf);
    };
    auto CMP = [&](int i) {
        perseus_interrupt::poll();   // between stages (blocks): the Python-driven models' Ctrl-C point
        if (stages[i].compute) stages[i].compute(inf);
    };
    auto REL = [&](int i) {
        if (!stages[i].release) return;
        if (on_main()) { WithStep _w(inf, "res_release"); stages[i].release(inf); }
        else            stages[i].release(inf);
    };

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

    // Join the in-flight worker job on every exit path (including an exception unwind):
    // an orphaned worker would otherwise walk this frame's `stages` after it is destroyed.
    struct WorkerJoin {
        std::future<void>& f;
        ~WorkerJoin() {
            if (!f.valid()) return;
            try { f.get(); } catch (...) {}
        }
    } _worker_join{fut};

    if (any_cpu_pf) {

        auto EXTRACT = [&](int k) {
            if (k < n && stages[k].prefetch_cpu) stages[k].prefetch_cpu(inf);
        };

        EXTRACT(0);
        ACQ(0, astream);
        if (use_stream) { WithStep _w(inf, "res_pipeline_sync"); cudaDeviceSynchronize(); }
        EXTRACT(1);

        for (int i = 0; i < n; ++i) {
            INS(i);
            const bool ext2 = (i + 2 < n) && static_cast<bool>(stages[i + 2].prefetch_cpu);
            if (ext2 && use_stream) { WithStep _w(inf, "res_arena_drain"); cudaStreamSynchronize(astream); }
            if (stages[i].stage_owner >= 0) inf.release_stage_block(stages[i].stage_owner);
            if (ext2) fut = submit_prefetch([&, i] { EXTRACT(i + 2); });
            if (i + 1 < n) ACQ(i + 1, astream);   // upload i+1 (stash hit) on astream — overlaps CMP(i)
            CMP(i);
            if (ext2) { WithStep _w(inf, "res_extract_wait"); fut.get(); }
            REL(i);                               // block_sync drains the astream upload(i+1)
        }
    } else {
        ACQ(0, astream);
        if (use_stream) { WithStep _w(inf, "res_pipeline_sync"); cudaDeviceSynchronize(); }
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
            if (overlap && mode == Overlap::Stream && use_stream) { WithStep _w(inf, "res_pipeline_sync"); cudaDeviceSynchronize(); }
            REL(i);
            if (overlap && mode == Overlap::Threaded) { WithStep _w(inf, "res_acquire_wait"); fut.get(); }  // join before next install
            if (!overlap && i + 1 < n) {                           // Sync boundary
                ACQ(i + 1, astream);
                if (use_stream) { WithStep _w(inf, "res_pipeline_sync"); cudaDeviceSynchronize(); }
            }
        }
    }

    if (use_stream) cudaStreamDestroy(stream);
}
