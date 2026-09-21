#pragma once

#include "test_helpers.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

namespace test_helpers {

// Per-token sweep accumulator — tallies pass/fail at a threshold and tracks
// max/mean of per-token max_rel across the sweep.
struct SweepSummary {
    int n = 0;
    int n_fail = 0;
    double max_rel = 0.0;
    double sum_rel = 0.0;
    void add(const AccStats& s, double fail_thresh) {
        ++n;
        if (s.max_rel >= fail_thresh) ++n_fail;
        max_rel = std::max(max_rel, s.max_rel);
        sum_rel += s.max_rel;
    }
    double mean_rel() const { return n ? sum_rel / n : 0.0; }
};

inline void print_token_header() {
    std::cout << "   tok |   max_abs |   max_rel |  mean_rel | worst[idx] got           ref\n"
              << "   ----+-----------+-----------+-----------+-----------------------------------\n";
}

inline void print_token_row(int tok_idx, const AccStats& s) {
    std::cout << std::scientific << std::setprecision(2)
              << "   " << std::setw(3) << tok_idx
              << " |  " << s.max_abs
              << " |  " << s.max_rel
              << " |  " << s.mean_rel
              << " | [" << std::setw(3) << s.worst_idx << "]   "
              << std::showpos << std::setprecision(3)
              << s.worst_got << " / " << s.worst_ref << std::noshowpos << "\n";
}

inline void print_sweep_summary(int T_val, const SweepSummary& sum, double fail_thresh) {
    std::cout << std::scientific << std::setprecision(2)
              << "  [T=" << T_val << " summary]"
              << "  n=" << sum.n
              << "  max_rel max=" << sum.max_rel
              << "  mean=" << sum.mean_rel()
              << "  failed=" << sum.n_fail << "/" << sum.n
              << "  (thr=" << fail_thresh << ")\n";
}

}
