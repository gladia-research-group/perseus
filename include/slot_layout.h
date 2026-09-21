#pragma once

#include "packing/packed_ctx.h"

#include <string>

struct Inference;

namespace slotlayout {

enum class Kind {
    Unknown,    
    Token,      
    Expanded,   
    Derived,    
};

const char* name(Kind k);

Kind get(const Ctx& ct);
void set(const Ctx& ct, Kind k);
void propagate(const Ctx& from, const Ctx& to);

bool strict();
void set_strict(bool on);

PackedCtx keep(const Ctx& a, PackedCtx out);
PackedCtx keep(const Ctx& a, const Ctx& b, PackedCtx out);
void check_binary(const Ctx& a, const Ctx& b, const char* op);

PackedCtx rebase_to_token(Inference& inf, const PackedCtx& x);

PackedCtx check_linear_input(Inference& inf, const PackedCtx& x, const std::string& wname,
                             int d_in, int d_out);
void note_linear_output(Inference& inf, const PackedCtx& y, const std::string& wname,
                        int d_in, int d_out);

}  // namespace slotlayout
