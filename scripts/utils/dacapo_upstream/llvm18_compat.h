// Force-included when building hecate against LLVM 18: its sources use the cast helpers
// unqualified, which older LLVM re-exported into the mlir namespace.
#if __has_include("llvm/Support/Casting.h")
#include "llvm/Support/Casting.h"
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;
#endif
