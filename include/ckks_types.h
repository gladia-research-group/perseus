#pragma once

#include <fideslib.hpp>

using namespace fideslib;

using CC  = CryptoContext<DCRTPoly>;
using Ctx = Ciphertext<DCRTPoly>;
using Ptx = Plaintext;
using KP  = KeyPair<DCRTPoly>;

inline uint32_t level_of(const Ctx& ct) {
    return (uint32_t)ct->GetLevel();
}
