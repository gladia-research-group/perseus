#pragma once

#include "fideslib_wrapper.h"

#ifndef PERSEUS_VERSION
#define PERSEUS_VERSION "0.1.0"
#endif

namespace perseus_stamp {
#if defined(NATIVEINT)
static constexpr int kNativeIntBits = NATIVEINT;
#else
static constexpr int kNativeIntBits = 0;
#endif
static constexpr const char* kChain =
    kNativeIntBits == 32 ? "n32" : (kNativeIntBits == 64 ? "n64" : "unknown");
}  // namespace perseus_stamp
