#pragma once

#include "inference.h"
#include <string>

namespace diagonal {

PackedCtx linear(Inference& inf, const PackedCtx& x,
                 const std::string& wname, int d_in, int d_out, bool stream_pt = false);

}  // namespace diagonal
