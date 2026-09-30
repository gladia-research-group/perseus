#pragma once

#include "fideslib_wrapper.h"
#include "inference.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <memory>

namespace test_helpers {

class CkksFixture : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {
        // default_ckks_options() honours the chain env knobs (BTP_SCALE_BITS,
        // FIRST_MOD_BITS, ...); a bare make_ckks_context() silently pinned every
        // fixture-based probe to the production 58-bit chain.
        CKKSContextOptions o = default_ckks_options();
        // Generic power-of-2 (+ 1024/j, ±5) rotation keys the primitive probes rely on
        // (e.g. the 1..512 reductions). make_ckks_context used
        // to inject these via include_default_rot_keys; now seeded explicitly per the same
        // formula (slots = batch_size?:1<<(logN-1)), since that default was removed.
        const int s = (o.batch_size == 0) ? (1 << (o.logN - 1)) : static_cast<int>(o.batch_size);
        for (int i = 1; i <= s; i *= 2) { o.extra_rot_steps.push_back(i); o.extra_rot_steps.push_back(-i); }
        for (int j = 1; j <= 256; j *= 2) { o.extra_rot_steps.push_back(1024 / j); o.extra_rot_steps.push_back(-(1024 / j)); }
        o.extra_rot_steps.push_back(5); o.extra_rot_steps.push_back(-5);
        // ABORT rather than throw. A throw out of SetUpTestSuite makes gtest SKIP every test in
        // the suite and still print "[  PASSED  ] 0 tests", so a context that cannot be BUILT
        // reads exactly like a suite with nothing in it. Print the chain parameters that decide
        // the throw, since the OpenFHE message names only the bound.
        try {
            ctx_ = make_ckks_context(o);
        } catch (const std::exception& e) {
            std::cerr << "\n[ckks_fixture] FATAL: make_ckks_context failed -- the bit-identity gate "
                         "CANNOT RUN.\n  what(): " << e.what()
                      << "\n  NATIVEINT=" << NATIVEINT
                      << " scale_bits=" << o.scale_bits
                      << " btp_scale_bits=" << o.btp_scale_bits
                      << " first_mod_bits=" << o.first_mod_bits
                      << " composite_degree=" << o.composite_degree
                      << " depth=" << o.depth
                      << "\n  At NATIVEINT=32 OpenFHE requires 15 < scale_bits/composite_degree < 31."
                         "\n  Aborting so this is not mistaken for an empty test suite.\n";
            std::abort();
        }
        slots_ = static_cast<int>(ctx_->cc->GetRingDimension() / 2);
    }
    static void TearDownTestSuite() { ctx_.reset(); }

    static CKKSContext& fhe() { return *ctx_; }
    static int slots() { return slots_; }
    static const std::shared_ptr<CKKSContext>& ctx() { return ctx_; }

    static Ctx enc_const(double v) {
        return ::encrypt_const(fhe().cc, v, slots_, fhe().pk());
    }

    static Inference make_inf(int hidDim = 1024, int dim = 768) {
        Inference inf;
        inf.fhe   = ctx_;
        inf.slots = slots_;
        inf.size.hidDim = hidDim;
        inf.size.dim    = dim;
        return inf;
    }

    inline static std::shared_ptr<CKKSContext> ctx_;
    inline static int slots_ = 0;
};

}  // namespace test_helpers
