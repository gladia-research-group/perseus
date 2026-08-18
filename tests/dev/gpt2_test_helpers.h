#pragma once

#include <algorithm>
#include <iostream>
#include <string>

namespace test_helpers {

// Progress-printing token loop. Pure test scaffolding — every production-side
// concern (encode, LN, K/V projection, push) is the caller's responsibility.
template <typename PushFn>
inline void warmup_kv_cache(int total, int T_val, PushFn&& push) {
    const int step = std::max(1, total / 16);
    for (int i = 0; i < total; ++i) {
        push(i);
        if (i % step == 0 || i == total - 1) {
            std::cout << "  [T=" << T_val << "] cache warmup "
                      << (i + 1) << "/" << total << std::endl;
        }
    }
}

}  // namespace test_helpers
