#pragma once

#include "test_helpers.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

namespace test_helpers {

inline std::vector<double> encode_head_token_layout(
        const std::vector<std::vector<double>>& per_head,
        int H_layout, int t, int tH, int nk, int N,
        bool replicate_phantom = false) {
    int H_file = static_cast<int>(per_head.size());
    std::vector<double> encoded(N, 0.0);
    for (int h = 0; h < H_layout; ++h) {
        int src_h = h;
        if (h >= H_file) {
            if (!replicate_phantom || H_file == 0) continue;
            src_h = h % H_file;
        }
        for (int k = 0; k < nk; ++k) {
            double v = 0.0;
            if (k < static_cast<int>(per_head[src_h].size())) {
                v = per_head[src_h][k];
            }
            int idx = k / t * tH + h * t + k % t;
            encoded[idx] = v;
        }
    }
    return encoded;
}

inline std::vector<int> build_active_indices(int H, int nk, int t, int tH) {
    std::vector<int> idx;
    idx.reserve(H * nk);
    for (int h = 0; h < H; ++h)
        for (int k = 0; k < nk; ++k)
            idx.push_back(k / t * tH + h * t + k % t);
    return idx;
}

struct KLStats {
    double mean_kl_fwd = 0.0;
    double max_kl_fwd  = 0.0;
    double mean_kl_rev = 0.0;
    double max_kl_rev  = 0.0;
    double mean_js     = 0.0;
    double max_js      = 0.0;
    int    n_heads     = 0;
};

inline double kl_pq(const std::vector<double>& p,
                    const std::vector<double>& q,
                    double eps = 1e-12) {
    double kl = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] > eps) {
            double q_safe = std::max(q[i], eps);
            kl += p[i] * std::log(p[i] / q_safe);
        }
    }
    return kl;
}

inline KLStats compute_kl_per_head(const std::vector<double>& got,
                                   const std::vector<double>& ref,
                                   int H_file, int nk, int t, int tH) {
    KLStats s;
    double sum_fwd = 0.0, sum_rev = 0.0, sum_js = 0.0;
    for (int h = 0; h < H_file; ++h) {
        std::vector<double> p(nk), q(nk);
        double sp = 0.0, sq = 0.0;
        for (int k = 0; k < nk; ++k) {
            int idx = k / t * tH + h * t + k % t;
            p[k] = std::max(ref[idx], 0.0);
            q[k] = std::max(got[idx], 0.0);
            sp += p[k];
            sq += q[k];
        }
        if (sp <= 0.0 || sq <= 0.0) continue;
        for (int k = 0; k < nk; ++k) { p[k] /= sp; q[k] /= sq; }

        const double kl_fwd = kl_pq(p, q);
        const double kl_rev = kl_pq(q, p);
        std::vector<double> m(nk);
        for (int k = 0; k < nk; ++k) m[k] = 0.5 * (p[k] + q[k]);
        const double js = 0.5 * kl_pq(p, m) + 0.5 * kl_pq(q, m);

        sum_fwd += kl_fwd; sum_rev += kl_rev; sum_js += js;
        if (kl_fwd > s.max_kl_fwd) s.max_kl_fwd = kl_fwd;
        if (kl_rev > s.max_kl_rev) s.max_kl_rev = kl_rev;
        if (js     > s.max_js)     s.max_js     = js;
        s.n_heads++;
    }
    if (s.n_heads > 0) {
        s.mean_kl_fwd = sum_fwd / s.n_heads;
        s.mean_kl_rev = sum_rev / s.n_heads;
        s.mean_js     = sum_js  / s.n_heads;
    }
    return s;
}

struct TopKStats {
    double top1_acc = 0.0;
    double top5_acc = 0.0;
    int    n_heads  = 0;
};

inline std::vector<int> topk_indices(const std::vector<double>& v, int k) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < v.size(); ++i) idx[i] = static_cast<int>(i);
    int kk = std::min<int>(k, v.size());
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(kk);
    return idx;
}

inline TopKStats compute_topk_per_head(const std::vector<double>& got,
                                       const std::vector<double>& ref,
                                       int H_file, int nk, int t, int tH) {
    TopKStats s;
    int sum_top1 = 0, sum_top5 = 0;
    for (int h = 0; h < H_file; ++h) {
        std::vector<double> p(nk), q(nk);
        for (int k = 0; k < nk; ++k) {
            int idx = k / t * tH + h * t + k % t;
            p[k] = ref[idx];
            q[k] = got[idx];
        }
        auto p_top1 = topk_indices(p, 1);
        auto q_top1 = topk_indices(q, 1);
        if (!p_top1.empty() && !q_top1.empty() && p_top1[0] == q_top1[0]) sum_top1++;

        auto p_top5 = topk_indices(p, 5);
        auto q_top5 = topk_indices(q, 5);
        int hits = 0;
        for (int pi : p_top5)
            for (int qi : q_top5)
                if (pi == qi) { ++hits; break; }
        sum_top5 += hits;
        s.n_heads++;
    }
    if (s.n_heads > 0) {
        s.top1_acc = static_cast<double>(sum_top1) / s.n_heads;
        s.top5_acc = static_cast<double>(sum_top5) / (s.n_heads * 5);
    }
    return s;
}

inline void report_slot(const std::string& label, const SlotStats& s) {
    std::cout << std::scientific << std::setprecision(3)
              << "[slot] " << std::left << std::setw(40) << label
              << " n=" << s.n
              << " max_abs=" << s.max_abs
              << " mean_abs=" << s.mean_abs
              << " rmse=" << s.rmse
              << " max_rel=" << s.max_rel
              << " mean_rel=" << s.mean_rel
              << " |ref|_max=" << s.max_ref
              << " |got|_max=" << s.max_got
              << "\n";
}

inline void report_kl(const std::string& label, const KLStats& s) {
    constexpr double NATS_TO_BITS = 1.4426950408889634;  // 1 / ln(2)
    std::cout << std::scientific << std::setprecision(3)
              << "[KL]   " << label << "\n"
              << "         mean_kl_fwd=" << s.mean_kl_fwd
              << " (" << std::fixed << std::setprecision(3)
              << (s.mean_kl_fwd * NATS_TO_BITS) << " bits)"
              << std::scientific << std::setprecision(3)
              << "  max_kl_fwd=" << s.max_kl_fwd << "\n"
              << "         mean_kl_rev=" << s.mean_kl_rev
              << "  max_kl_rev=" << s.max_kl_rev << "\n"
              << "         mean_js="    << s.mean_js
              << "  max_js="    << s.max_js
              << "  (heads=" << s.n_heads << ")\n";
    std::cout.unsetf(std::ios::fixed);
}

inline void report_topk(const std::string& label, const TopKStats& s) {
    std::cout << std::fixed << std::setprecision(4)
              << "[top]  " << label
              << " top1_acc=" << s.top1_acc
              << " top5_acc=" << s.top5_acc
              << " (heads=" << s.n_heads << ")\n";
    std::cout.unsetf(std::ios::fixed);
}

}  // namespace test_helpers
