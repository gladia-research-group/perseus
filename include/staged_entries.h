#pragma once

#include "inference.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

class StagedEntries {
public:
    // prefix = the host-staging key namespace. Keys are index-based, so only ONE container per
    // prefix may hold live staged data at a time (producer→consumer pairs must share the prefix
    // AND the size-derived active flag). Concurrent datasets (token-pair A/B halves) use distinct
    // prefixes.
    StagedEntries(Inference& inf, bool active, std::string prefix = "cf.stg.")
        : inf_(inf), active_(active), prefix_(std::move(prefix)) {}

    static bool auto_active(size_t n_entries) { return n_entries > 64; }

    bool   active() const { return active_; }
    size_t size()   const { return v_.size(); }

    void push(PackedCtx&& pc) {
        v_.push_back(std::move(pc));
        if (active_) pending_.push_back(v_.size() - 1);
    }

    void seal() {
        if (!active_ || pending_.empty()) return;
        cudaDeviceSynchronize();
        for (size_t i : pending_)
            if (v_[i].ct) inf_.cc()->KvStoreStaged(v_[i].ct, key_(i), stream_());
        cudaStreamSynchronize(stream_());
        for (size_t i : pending_)
            if (v_[i].ct) inf_.cc()->KvEvict(v_[i].ct);
        pending_.clear();
    }

    void adopt(std::vector<PackedCtx>&& v) { v_ = std::move(v); }
    PackedCtx& load(size_t i) {
        PackedCtx& pc = v_[i];
        if (active_ && pc.ct) {

            cudaDeviceSynchronize();
            inf_.cc()->KvLoadStaged(pc.ct, key_(i), stream_());
            cudaStreamSynchronize(stream_());
        }
        return pc;
    }

    void drop(size_t i) {
        PackedCtx& pc = v_[i];
        if (active_ && pc.ct) inf_.cc()->KvEvict(pc.ct);
    }

    PackedCtx consume(size_t i) {
        load(i);
        return std::move(v_[i]);
    }

    void clear(size_t i) { v_[i] = PackedCtx{}; }

    void set(size_t i, PackedCtx&& pc) {
        v_[i] = std::move(pc);
        if (active_) pending_.push_back(i);
    }

    std::vector<PackedCtx> into_vector() {
        seal();
        return std::move(v_);
    }

private:
    static cudaStream_t stream_() {
        static cudaStream_t s = [] {
            cudaStream_t st = nullptr;
            cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
            return st;
        }();
        return s;
    }

    std::string key_(size_t idx) const { return prefix_ + std::to_string(idx); }

    Inference&             inf_;
    bool                   active_ = false;
    std::string            prefix_;
    std::vector<PackedCtx> v_;
    std::vector<size_t>    pending_;   // pushed/set since the last seal()
};
