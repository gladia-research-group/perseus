// Sparse-bootstrap / fold ablation switches (env-gated; defaults are the shipped configuration).
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

bool fused_ln_var_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("FUSED_LN_VAR");
        return v && *v && *v != '0';
    }();
    return on;
}

bool fused_sm_den_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("FUSED_SM_DEN");
        return v && *v && *v != '0';   // the 32-bit preset enables it (scripts/local_env.sh)
    }();
    return on;
}
