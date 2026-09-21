#pragma once

#include <stdexcept>
#include <string>

namespace fhe {

struct FHEError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct PlanError : FHEError {
    using FHEError::FHEError;
};

struct MaskError : FHEError {
    using FHEError::FHEError;
};

struct LayoutError : FHEError {
    using FHEError::FHEError;
};

}  // namespace fhe
