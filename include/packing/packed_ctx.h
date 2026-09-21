#pragma once

#include "ckks_types.h"

#include <stdexcept>
#include <string>

// A PackedCtx pairs a raw OpenFHE Ciphertext with a Packing tag describing
// how its slots map to logical tensor coordinates.
//
// Adding a new packing family:
//   1) add an entry to PackingKind (and to_string / parse_packing_kind);
//   2) implement the family-namespaced algorithms (cachemir::, diagonal::, …);
//   3) extend the dispatchers (linear.cu, attention.cu, norm.cu, …) with an
//      `is_<kind>(packing)` branch.

enum class PackingKind {
    Cachemir,         // single-token decoding: lanes carry replicated-token diagonal bundles
    Diagonal,         // multi-token batched: lanes carry independent tokens
    CachemirFilling,  // batched prefill that fills the cachemir KV-cache layout;
                      // shares the token-in-lane (diagonal) linear, diverges at cache/attention
    CachemirComplex,  // same slot layout as Cachemir, but linears carry complex (W_re + i*W_im)

};

inline const char* to_string(PackingKind k) {
    switch (k) {
        case PackingKind::Cachemir:        return "cachemir";
        case PackingKind::Diagonal:        return "diagonal";
        case PackingKind::CachemirFilling: return "cachemir_filling";
        case PackingKind::CachemirComplex: return "cachemir_complex";
    }
    return "unknown";
}

inline PackingKind parse_packing_kind(const std::string& s) {
    if (s == "cachemir")         return PackingKind::Cachemir;
    if (s == "diagonal")         return PackingKind::Diagonal;
    if (s == "cachemir_filling") return PackingKind::CachemirFilling;
    if (s == "cachemir_complex") return PackingKind::CachemirComplex;
    throw std::runtime_error("parse_packing_kind: unknown packing '" + s + "'");
}

struct Packing {
    PackingKind kind = PackingKind::Cachemir;
    int slots    = 0;
    int hidDim   = 0;
    int realDim  = 0;
    int numHeads = 0;
    int t        = 0;

    bool operator==(const Packing& o) const {
        return kind == o.kind && slots == o.slots && hidDim == o.hidDim &&
               realDim == o.realDim && numHeads == o.numHeads && t == o.t;
    }
    bool operator!=(const Packing& o) const { return !(*this == o); }
};

struct PackedCtx {
    Ctx     ct;
    Packing packing;
};

inline void assert_same_packing(const Packing& a, const Packing& b) {
    if (a != b)
        throw std::runtime_error(
            std::string("PackedCtx packing mismatch in binary op: ") + to_string(a.kind) +
            "(t=" + std::to_string(a.t) + ",hid=" + std::to_string(a.hidDim) + ") vs " +
            to_string(b.kind) + "(t=" + std::to_string(b.t) + ",hid=" + std::to_string(b.hidDim) + ")");
}

inline bool is_cachemir(const Packing& p) {
    return p.kind == PackingKind::Cachemir || p.kind == PackingKind::CachemirComplex;
}
inline bool is_diagonal(const Packing& p) { return p.kind == PackingKind::Diagonal; }
inline bool is_cachemir_filling(const Packing& p) { return p.kind == PackingKind::CachemirFilling; }
