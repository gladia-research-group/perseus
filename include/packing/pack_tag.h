#pragma once

#include <algorithm>
#include <numeric>
#include <string>

namespace packtag {


struct Support {
    // Kind::AP with width w describes a BLOCK support: {offset + k*stride + j : k < count,
    // j < width} — `width` consecutive live slots at the head of each stride cell. width == 1
    // is the classic AP. The Block form represents the KV-cache append (a union of adjacent
    // token lanes with the same stride) without collapsing it to Dense.
    enum class Kind { Empty, AP, Dense } kind = Kind::Dense;
    int offset = 0, stride = 1, count = 0, width = 1;

    static Support empty() { return {Kind::Empty, 0, 1, 0, 1}; }
    static Support dense() { return {Kind::Dense, 0, 1, 0, 1}; }
    static Support ap(int off, int str, int cnt) {
        if (cnt <= 0) return empty();
        if (str <= 1 && off == 0) return dense();   // contiguous from 0 with stride 1
        return {Kind::AP, off, str, cnt, 1};
    }
    static Support block(int off, int str, int w, int cnt) {
        if (cnt <= 0 || w <= 0) return empty();
        if (w == 1) return ap(off, str, cnt);
        if (w >= str) return dense();   // cells full (or invalid): contiguous — stay conservative
        return {Kind::AP, off, str, cnt, w};
    }
    bool is_dense() const { return kind == Kind::Dense; }
    bool is_empty() const { return kind == Kind::Empty; }
};

struct PackTag {
    int slots = 0;
    int period = 0;        // 0 = unknown/aperiodic sentinel filled by top(); else the bound
    Support support{};

    static PackTag top(int slots) { return {slots, slots, Support::dense()}; }
    static PackTag constant(int slots) { return {slots, 1, Support::dense()}; }
    bool known() const { return slots > 0 && period > 0; }

    bool periodic_at(int s) const { return period > 0 && s % period == 0; }
    bool fold_collision_free_at(int s) const {
        if (support.is_empty()) return true;
        if (support.is_dense()) return s >= slots;
        if (support.width > 1) {
            // BLOCK: sound under the aligned case only — s a multiple of stride keeps each
            // cell's residues {k*stride+j mod s} distinct iff count fits s/stride cells and
            // the width stays inside a cell (guaranteed by the representation, checked anyway).
            if (support.stride > 0 && s % support.stride == 0)
                return support.count <= s / support.stride && support.width <= support.stride;
            return false;   // misaligned block: stay conservative
        }
        // live indices offset + k*stride, k < count. Two collide mod s iff
        // (k1-k2)*stride == 0 mod s. The smallest positive multiple of stride divisible by s is
        // s/gcd(stride,s) steps away, so collisions appear once count exceeds that.
        const int g = std::gcd(support.stride, s);
        return support.count <= s / g;
    }

    int min_routable_s() const {
        for (int s = 1; s < slots; s <<= 1)
            if (periodic_at(s) || fold_collision_free_at(s)) return s;
        return slots;
    }

    // True when every live slot already sits below s, so a fold at s moves nothing: the
    // support survives a sparse route in place. Stricter than fold_collision_free_at,
    // which only asks that no two live values land in the same residue class.
    bool fold_transparent_at(int s) const {
        if (support.is_empty()) return true;
        if (support.is_dense()) return s >= slots;
        if (support.count <= 0 || support.offset < 0) return false;
        // Highest live index is offset + (count-1)*stride + (width-1); strictly below s.
        const long long hi = (long long)support.offset +
                             (long long)(support.count - 1) * (long long)support.stride +
                             (long long)(support.width - 1);
        return hi < (long long)s;
    }
};

inline int lcm_capped(int a, int b, int cap) {
    if (a <= 0 || b <= 0) return cap;
    const long long l = (long long)std::lcm(a, b);
    return (l >= cap) ? cap : (int)l;
}

inline Support support_union(const Support& a, const Support& b) {
    if (a.is_empty()) return b;
    if (b.is_empty()) return a;
    if (a.is_dense() || b.is_dense()) return Support::dense();
    if (a.stride == b.stride && a.offset == b.offset && a.width == b.width)
        return Support::block(a.offset, a.stride, a.width, std::max(a.count, b.count));
    // Same stride, offsets within one cell: the union fits a BLOCK — width spans from the
    // lowest live offset to the highest live end. Over-approximating count with max() is a
    // sound superset. This is the KV-append case (adjacent token lanes).
    if (a.stride == b.stride) {
        const int lo = std::min(a.offset, b.offset);
        const int hi = std::max(a.offset + a.width, b.offset + b.width);
        if (hi - lo <= a.stride)
            return Support::block(lo, a.stride, hi - lo, std::max(a.count, b.count));
    }
    return Support::dense();   // conservative: a general union of two APs is not an AP/Block
}

inline Support support_intersect(const Support& a, const Support& b) {
    if (a.is_empty() || b.is_empty()) return Support::empty();
    if (a.is_dense()) return b;
    if (b.is_dense()) return a;
    if (a.stride == b.stride && a.offset == b.offset)
        return Support::block(a.offset, a.stride, std::min(a.width, b.width),
                              std::min(a.count, b.count));
    // A general AP-vs-AP intersection is a CRT problem. Returning EITHER operand is a sound
    // superset of the true intersection; take the sparser-looking one.
    return ((long long)a.count * a.width <= (long long)b.count * b.width) ? a : b;
}

inline PackTag t_mult_scalar(const PackTag& x, double c) {
    if (!x.known()) return PackTag{};
    if (c == 0.0) return {x.slots, 1, Support::empty()};
    return x;
}

inline PackTag t_add_scalar(const PackTag& x, double c) {
    if (!x.known()) return PackTag{};
    if (c == 0.0) return x;
    return {x.slots, x.period, Support::dense()};
}

inline PackTag t_square(const PackTag& x) { return x; }

inline PackTag t_add(const PackTag& a, const PackTag& b) {
    if (!a.known() || !b.known()) return PackTag{};
    return {a.slots, lcm_capped(a.period, b.period, a.slots),
            support_union(a.support, b.support)};
}

inline PackTag t_mult(const PackTag& a, const PackTag& b) {
    if (!a.known() && !b.known()) return PackTag{};
    if (!a.known()) return {b.slots, b.slots, b.support};
    if (!b.known()) return {a.slots, a.slots, a.support};
    return {a.slots, lcm_capped(a.period, b.period, a.slots),
            support_intersect(a.support, b.support)};
}

inline PackTag t_rotate(const PackTag& x, int k) {
    if (!x.known()) return PackTag{};
    PackTag r = x;
    if (x.support.kind == Support::Kind::AP) {
        int off = (x.support.offset - k) % x.slots;
        if (off < 0) off += x.slots;
        r.support = Support::block(off, x.support.stride, x.support.width, x.support.count);
    }
    return r;
}

inline PackTag t_conjugate(const PackTag& x) {
    if (!x.known()) return PackTag{};
    PackTag r = x;
    if (x.support.kind == Support::Kind::AP) r.support = Support::dense();  // mirrored: not an AP
    return r;
}

inline PackTag t_reduce_all(const PackTag& x) { return PackTag::constant(x.slots); }

inline PackTag t_reduce_stride(const PackTag& x, int g) {
    return {x.slots, std::min(g, x.slots), Support::dense()};
}

inline PackTag t_bootstrap(const PackTag& x, int routed_s, int slots) {
    if (routed_s <= 0) return x;
    if (slots <= 0) return PackTag{};
    return {slots, std::min(routed_s, slots), Support::dense()};
}

inline PackTag t_bootstrap(const PackTag& x, int routed_s) {
    if (routed_s <= 0) return x;
    if (x.slots <= 0) return PackTag{};
    return {x.slots, std::min(routed_s, x.slots), Support::dense()};
}

inline PackTag from_signature(int slots, int period, int live, int stride, int window,
                              int offset) {
    PackTag t;
    t.slots = slots;
    t.period = period > 0 ? period : slots;
    if (live <= 0)                 t.support = Support::empty();
    else if (live >= slots)        t.support = Support::dense();
    else if (stride > 1)           t.support = Support::ap(offset, stride, live);
    else                           t.support = Support::ap(offset, 1,
                                                           window > offset ? window - offset
                                                                           : live);
    return t;
}

}  // namespace packtag
