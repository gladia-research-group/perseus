#include "slot_layout.h"
#include "fhe_errors.h"

#include "inference.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace slotlayout {

namespace {

// ct-pointer-keyed registry with weak_ptr liveness guard — the same shape as the
// wrapper's ct_tags: an address can be recycled, so a dead weak_ptr (or one that
// resolves to a different object) invalidates the entry instead of mislabeling.
struct Entry {
    std::weak_ptr<Ctx::element_type> wp;
    Kind kind = Kind::Unknown;
};
std::mutex g_mtx;
std::unordered_map<const void*, Entry> g_layouts;

bool g_strict = false;

}  // namespace

const char* name(Kind k) {
    switch (k) {
        case Kind::Unknown:  return "unknown";
        case Kind::Token:    return "token";
        case Kind::Expanded: return "expanded";
        case Kind::Derived:  return "derived";
    }
    return "?";
}

Kind get(const Ctx& ct) {
    if (!ct) return Kind::Unknown;
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_layouts.find((const void*)ct.get());
    if (it == g_layouts.end()) return Kind::Unknown;
    auto live = it->second.wp.lock();
    if (!live || live.get() != ct.get()) {   // original destroyed, address recycled
        g_layouts.erase(it);
        return Kind::Unknown;
    }
    return it->second.kind;
}

void set(const Ctx& ct, Kind k) {
    if (!ct) return;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (k == Kind::Unknown) {
        g_layouts.erase((const void*)ct.get());
        return;
    }
    g_layouts[(const void*)ct.get()] = Entry{ct, k};
}

void propagate(const Ctx& from, const Ctx& to) { set(to, get(from)); }

PackedCtx keep(const Ctx& a, PackedCtx out) {
    propagate(a, out.ct);
    return out;
}

PackedCtx keep(const Ctx& a, const Ctx& b, PackedCtx out) {
    const Kind ka = get(a), kb = get(b);
    // Derived absorbs: junk lanes survive any elementwise combination with a clean operand.
    if (ka == Kind::Derived || kb == Kind::Derived) set(out.ct, Kind::Derived);
    else set(out.ct, ka != Kind::Unknown ? ka : kb);
    return out;
}

bool strict() { return g_strict; }
void set_strict(bool on) { g_strict = on; }

namespace {

void complain(Inference& inf, const std::string& wname, Kind found, Kind wanted) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "[layout_error] linear '%s' encodes the TOKEN basis but its input is in "
                  "the '%s' basis (wanted '%s'). A cachemir linear emits its output "
                  "features REARRANGED, so chaining custom linears computes a different, "
                  "deterministic function — silently. Re-encode the activation between "
                  "chains, keep the chain inside one designed up/down pair, or author the "
                  "second weight in the chain basis. (strict via set_strict_layout)",
                  wname.c_str(), name(found), name(wanted));
    if (strict()) throw fhe::LayoutError(buf);
    static std::mutex warn_mtx;
    static std::unordered_set<std::string> warned;
    std::lock_guard<std::mutex> lk(warn_mtx);
    if (warned.insert(wname).second) {
        std::fprintf(stderr, "%s\n", buf);
        std::fflush(stderr);
    }
    (void)inf;
}

}  // namespace

void check_binary(const Ctx& a, const Ctx& b, const char* op) {
    const Kind ka = get(a), kb = get(b);
    if (ka == Kind::Unknown || kb == Kind::Unknown || ka == kb) return;
    // Token + Derived is legal (lane 0 agrees; the junk lanes are re-based at the next
    // token-basis linear). Only a WIDTH mismatch — Expanded against a hidDim basis — is an
    // operand error.
    if (ka != Kind::Expanded && kb != Kind::Expanded) return;
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "[layout_error] %s: operands sit in different slot bases ('%s' vs '%s') "
                  "-- e.g. a residual add joining a custom linear's DERIVED output to its "
                  "TOKEN-basis skip. Re-encode one side or keep both in one basis. "
                  "(strict via set_strict_layout)", op, name(ka), name(kb));
    if (strict()) throw fhe::LayoutError(buf);
    static std::mutex warn_mtx;
    static std::unordered_set<std::string> warned;
    std::lock_guard<std::mutex> lk(warn_mtx);
    if (warned.insert(op).second) {
        std::fprintf(stderr, "%s\n", buf);
        std::fflush(stderr);
    }
}

PackedCtx rebase_to_token(Inference& inf, const PackedCtx& x) {
    const int d = x.packing.hidDim, t = x.packing.t;
    std::vector<double> lane0(static_cast<size_t>(inf.slots), 0.0);
    for (int k = 0; k < d; ++k) lane0[static_cast<size_t>(k) * t] = 1.0;
    Ptx m = inf.encode_at_cached("layout.lane0", lane0, x);
    PackedCtx y = inf.fhe->mult(x, m);
    set(y.ct, Kind::Token);
    return y;
}

PackedCtx check_linear_input(Inference& inf, const PackedCtx& x, const std::string& wname,
                             int d_in, int d_out) {
    (void)d_out;
    if (!is_cachemir(x.packing)) return x;
    if (!inf.token_basis_weights.count(wname)) return x;   // chain-basis / canonical: no claim
    const Kind found = get(x.ct);
    if (found == Kind::Unknown) return x;                  // no claim, no check
    const Kind wanted = (d_in == x.packing.hidDim) ? Kind::Token : Kind::Expanded;
    if (found == wanted) return x;
    if (found == Kind::Derived && wanted == Kind::Token) {
        if (strict()) complain(inf, wname, found, wanted);   // throws
        static std::mutex note_mtx;
        static std::unordered_set<std::string> noted;
        {
            std::lock_guard<std::mutex> lk(note_mtx);
            if (noted.insert(wname).second) {
                std::fprintf(stderr, "[layout] auto-rebased the input of token-basis linear "
                                     "'%s' (lane-0 mask, one level); strict layout mode "
                                     "throws instead\n", wname.c_str());
                std::fflush(stderr);
            }
        }
        return rebase_to_token(inf, x);
    }
    complain(inf, wname, found, wanted);
    return x;
}

void note_linear_output(Inference& inf, const PackedCtx& y, const std::string& wname,
                        int d_in, int d_out) {
    (void)d_in;
    if (!is_cachemir(y.packing)) return;
    if (!inf.token_basis_weights.count(wname)) return;
    set(y.ct, (d_out != y.packing.hidDim) ? Kind::Expanded : Kind::Derived);
}

}  // namespace slotlayout
