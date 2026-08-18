// Sparse-bts env gates. The sparse-routed LN body was unified into
// ln_inv_sqrt_tail (norm.cu) — norm() passes the sparse flag directly.
#include "nonlinear.h"

#include <cstdlib>

bool sparse_ln_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("SPARSE_LN_BTS");
        return v && *v && *v != '0';
    }();
    return on;
}

bool sparse_sm_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("SPARSE_SM_BTS");
        return v && *v && *v != '0';
    }();
    return on;
}
