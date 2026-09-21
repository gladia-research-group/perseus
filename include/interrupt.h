#pragma once

namespace perseus_interrupt {

using Fn = void (*)();

inline Fn& hook() {
    static Fn f = nullptr;
    return f;
}

inline void poll() {
    if (Fn f = hook()) f();
}

}  // namespace perseus_interrupt
