#include "client_context.h"
#include "fhe_errors.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <set>

namespace perseus_client {

namespace {

// BootstrapPrecomputation.cuh and RawCiphertext.cu.
constexpr bool AFFINE_LT = true;
constexpr bool MAKE_CTS_LT_FRIENDLY = true;
constexpr bool MAKE_STC_LT_FRIENDLY = true;

// The index-bearing part of FIDESlib::CKKS::BootstrapPrecomputation::LTstep.
struct LTstep {
    int slots = -1;
    int bStep = -1;
    int gStep = -1;
    std::vector<int> rotIn;
    std::vector<int> rotOut;
};

using CKKSRNSParams = lbcrypto::CryptoParametersCKKSRNS;
using RelinKey = std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>;
using SKImpl = std::shared_ptr<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>;

std::shared_ptr<lbcrypto::FHECKKSRNS> fhe_of(const LbCC& cc) {
    auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
    if (!fhe) throw fhe::FHEError("bootstrap keys: the context has no FHE scheme (Enable(FHE) first)");
    return fhe;
}

// ParameterSwitch.cu
LbCC create_switchable_context(LbCC& cc, int limbs, int digits, int hamming_weight) {
    std::shared_ptr<CKKSRNSParams> init_param =
        std::dynamic_pointer_cast<CKKSRNSParams>(cc->GetCryptoParameters());
    auto& init_encode_param = init_param->GetEncodingParams();
    auto& init_elem_param = init_param->GetElementParams();
    LbCC cc_res;

    CKKSRNSParams param{*init_param};
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetBatchSize(init_encode_param->GetBatchSize());
    parameters.SetDecryptionNoiseMode(param.GetDecryptionNoiseMode());
    parameters.SetDigitSize(digits);
    parameters.SetExecutionMode(param.GetExecutionMode());
    parameters.SetFirstModSize(init_elem_param->GetParams().at(0)->GetModulus().GetMSB());
    parameters.SetInteractiveBootCompressionLevel(param.GetMPIntBootCiphertextCompressionLevel());
    parameters.SetKeySwitchTechnique(init_param->GetKeySwitchTechnique());
    parameters.SetMaxRelinSkDeg(init_param->GetMaxRelinSkDeg());
    parameters.SetMultiplicativeDepth(limbs - 1);

    parameters.SetNumAdversarialQueries(param.GetNumAdversarialQueries());
    parameters.SetNumLargeDigits(param.GetNumPartQ());
    parameters.SetPREMode(param.GetPREMode());
    parameters.SetRingDim(param.GetElementParams()->GetRingDimension());
    const int srcCompositeDegree = static_cast<int>(init_param->GetCompositeDegree());
    int scale = init_elem_param->GetParams().at(srcCompositeDegree)->GetModulus().GetMSB();
    if (srcCompositeDegree > 1 && scale >= static_cast<int>(MAX_MODULUS_SIZE))
        scale = static_cast<int>(MAX_MODULUS_SIZE) - 1;
    parameters.SetScalingModSize(scale);
    // A keygen-only, single-prime helper context: COMPOSITESCALING* becomes FLEXIBLEAUTO.
    {
        auto st = init_param->GetScalingTechnique();
        if (st == lbcrypto::COMPOSITESCALINGAUTO || st == lbcrypto::COMPOSITESCALINGMANUAL)
            st = lbcrypto::FLEXIBLEAUTO;
        parameters.SetScalingTechnique(st);
    }
    parameters.SetSecretKeyDist(hamming_weight == static_cast<int>(cc->GetRingDimension() / 2)
                                    ? lbcrypto::UNIFORM_TERNARY
                                    : lbcrypto::SPARSE_TERNARY);
    parameters.SetSecurityLevel(param.GetStdLevel());
    parameters.SetStatisticalSecurity(param.GetStatisticalSecurity());

    cc_res = lbcrypto::GenCryptoContext(parameters);
    cc_res->Enable(lbcrypto::PKE | lbcrypto::KEYSWITCH | lbcrypto::LEVELEDSHE);

    return cc_res;
}

// ParameterSwitch.cu
std::pair<std::pair<RelinKey, RelinKey>, SKImpl>
create_context_switching_keys(LbCC& cca, LbCC& ccb, const LbSK& a, int hamming_weight_b) {
    lbcrypto::DCRTPoly::TugType tug;
    lbcrypto::DCRTPoly sNew(tug, cca->GetElementParams(), Format::EVALUATION, hamming_weight_b);
    // sparse key used for the modraising step
    auto skNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(ccb);
    skNew->SetPrivateElement(std::move(sNew));
    skNew->SetKeyTag(a->GetKeyTag());
    auto scaling =
        std::dynamic_pointer_cast<CKKSRNSParams>(cca->GetCryptoParameters())->GetScalingTechnique();

    RelinKey atob;
    const int srcCompositeDegree = static_cast<int>(
        std::dynamic_pointer_cast<CKKSRNSParams>(cca->GetCryptoParameters())->GetCompositeDegree());
    if (srcCompositeDegree > 1) {
        auto skNewMain = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(cca);
        skNewMain->SetPrivateElement(skNew->GetPrivateElement());
        skNewMain->SetKeyTag(a->GetKeyTag());
        atob = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
            cca->GetScheme()->KeySwitchGen(a, skNewMain));
    } else if (scaling != lbcrypto::FLEXIBLEAUTOEXT) {
        atob = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
            ccb->GetScheme()->KeySwitchGen(a, skNew));
    } else {
        lbcrypto::DCRTPoly saNew = a->GetPrivateElement();
        saNew.SetElementAtIndex(ccb->GetElementParams()->GetParams().size() - 1,
                                saNew.GetAllElements().back());
        saNew.DropLastElements(saNew.GetAllElements().size() -
                               ccb->GetElementParams()->GetParams().size());
        auto skaNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(cca);
        skaNew->SetPrivateElement(std::move(saNew));
        skaNew->SetKeyTag(a->GetKeyTag());

        lbcrypto::DCRTPoly sbNew = skNew->GetPrivateElement();
        sbNew.SetElementAtIndex(ccb->GetElementParams()->GetParams().size() - 1,
                                sbNew.GetAllElements().back());
        sbNew.DropLastElements(sbNew.GetAllElements().size() -
                               ccb->GetElementParams()->GetParams().size());
        auto skbNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(ccb);
        skbNew->SetPrivateElement(std::move(sbNew));
        skbNew->SetKeyTag(a->GetKeyTag());

        atob = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
            ccb->GetScheme()->KeySwitchGen(skaNew, skbNew));
    }
    RelinKey btoa = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
        cca->GetScheme()->KeySwitchGen(skNew, a));

    return {{atob, btoa}, skNew};
}

}  // namespace

// AccumulateBroadcast.cu
std::vector<int> accumulate_rotation_indices(const int bStep, const int stride, const int size) {
    std::vector<int> indices;
    int logbStep = std::bit_width(static_cast<uint32_t>(bStep)) - 1;
    for (int s = stride; s < stride * size; s <<= logbStep) {
        for (int idx = s; idx < s * bStep && idx < stride * size; idx += s) {
            indices.push_back(idx);
        }
    }
    return indices;
}

// RawCiphertext.cu
std::shared_ptr<LbEvalKeyMap> gen_rotation_keys(const LbSK& keys, const std::vector<int>& indexes) {
    LbCC cc = keys->GetCryptoContext();
    std::set<int> indexes2(indexes.begin(), indexes.end());
    std::vector<int> indexes3;
    for (int i : indexes2) {
        if (i) {
            indexes3.emplace_back(i);
        }
    }
    auto evalKeys = cc->GetScheme()->EvalAtIndexKeyGen(nullptr, keys, indexes3);
    lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::InsertEvalAutomorphismKey(evalKeys, keys->GetKeyTag());
    return evalKeys;
}

// RawCiphertext.cu (index vector only; the GPU precomputation struct is replaced
// by the local LTstep list).
std::vector<int> bootstrap_indexes(const LbCC& cc, int slots) {
    std::vector<int> indexes;
    auto fhe = fhe_of(cc);
    auto it = fhe->m_bootPrecomMap.find(static_cast<uint32_t>(slots));
    if (it == fhe->m_bootPrecomMap.end())
        throw fhe::FHEError("bootstrap_indexes: no EvalBootstrapSetup for " + std::to_string(slots) +
                            " slots");
    auto precom = it->second;
    using namespace lbcrypto;
    std::vector<LTstep> CtS, StC;

    if (precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1 &&
        precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) {
        const int bStep = (precom->m_dim1 == 0) ? static_cast<int>(std::ceil(std::sqrt(slots)))
                                                : static_cast<int>(precom->m_dim1);
        for (int i = 1; i < bStep; ++i) {
            indexes.push_back(i);
        }
        if constexpr (AFFINE_LT) {
            indexes.push_back(bStep);
        } else {
            for (int i = bStep; i < slots; i += bStep) indexes.push_back(i);
        }
    } else {
        {  // CoeffToSlots metadata
            uint32_t M = cc->GetCyclotomicOrder();
            int32_t levelBudget = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
            int32_t layersCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_COLL];
            int32_t remCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_REM];
            int32_t numRotations = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
            int32_t b = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP];
            int32_t g = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP];
            int32_t numRotationsRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
            int32_t bRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP_REM];
            int32_t gRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

            int32_t stop = -1;
            int32_t flagRem = 0;

            if (remCollapse != 0) {
                stop = 0;
                flagRem = 1;
            }

            // precompute the inner and outer rotations
            {
                CtS.resize(levelBudget);
                for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
                    if (flagRem == 1 && i == 0) {
                        // remainder corresponds to index 0 in encoding and to last index in decoding
                        CtS[i].bStep = gRem;
                        CtS[i].gStep = bRem;
                        CtS[i].slots = numRotationsRem;
                        CtS[i].rotIn.resize(gRem);
                        CtS[i].rotOut.resize(bRem);
                    } else {
                        CtS[i].bStep = g;
                        CtS[i].gStep = b;
                        CtS[i].slots = numRotations;
                        CtS[i].rotIn.resize(g);
                        CtS[i].rotOut.resize(b);
                    }
                }

                for (int32_t s = levelBudget - 1; s > stop; s--) {
                    for (int32_t j = 0; j < g; j++) {
                        CtS[s].rotIn[j] =
                            ReduceRotation((j - int32_t((numRotations + 1) / 2) + 1) *
                                               (1 << ((s - flagRem) * layersCollapse + remCollapse)),
                                           slots);
                    }

                    for (int32_t i = 0; i < b; i++) {
                        CtS[s].rotOut[i] =
                            ReduceRotation((g * i) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), M / 4);
                    }
                }

                if (flagRem) {
                    for (int32_t j = 0; j < gRem; j++) {
                        CtS[stop].rotIn[j] = ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1), slots);
                    }

                    for (int32_t i = 0; i < bRem; i++) {
                        CtS[stop].rotOut[i] = ReduceRotation((gRem * i), M / 4);
                    }
                }

                if constexpr (AFFINE_LT && MAKE_CTS_LT_FRIENDLY) {
                    for (int32_t s = 0; s < levelBudget; s++) {
                        int offset = CtS.at(s).rotIn[0];
                        for (auto& i : CtS.at(s).rotIn) {
                            i = (i - offset);
                        }
                        for (auto& i : CtS.at(s).rotOut) {
                            i = (i + offset);
                        }
                    }
                }
            }
        }

        {  // SlotToCoeff metadata
            uint32_t M = cc->GetCyclotomicOrder();

            int32_t levelBudget = precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
            int32_t layersCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_COLL];
            int32_t remCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_REM];
            int32_t numRotations = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
            int32_t b = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP];
            int32_t g = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP];
            int32_t numRotationsRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
            int32_t bRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP_REM];
            int32_t gRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

            int32_t flagRem = 0;

            if (remCollapse != 0) {
                flagRem = 1;
            }

            // precompute the inner and outer rotations
            {
                StC.resize(levelBudget);
                for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
                    if (flagRem == 1 && i == uint32_t(levelBudget - 1)) {
                        // remainder corresponds to index 0 in encoding and to last index in decoding
                        StC[i].bStep = gRem;
                        StC[i].gStep = bRem;
                        StC[i].slots = numRotationsRem;
                        StC[i].rotIn.resize(gRem);
                        StC.at(i).rotOut.resize(bRem);
                    } else {
                        StC[i].bStep = g;
                        StC[i].gStep = b;
                        StC[i].slots = numRotations;
                        StC[i].rotIn.resize(g);
                        StC.at(i).rotOut.resize(b);
                    }
                }

                for (int32_t s = 0; s < levelBudget - flagRem; s++) {
                    for (int32_t j = 0; j < g; j++) {
                        StC.at(s).rotIn.at(j) = ReduceRotation(
                            (j - int32_t((numRotations + 1) / 2) + 1) * (1 << (s * layersCollapse)), M / 4);
                    }

                    for (int32_t i = 0; i < b; i++) {
                        StC.at(s).rotOut.at(i) = ReduceRotation((g * i) * (1 << (s * layersCollapse)), M / 4);
                    }
                }

                if (flagRem) {
                    int32_t s = levelBudget - flagRem;
                    for (int32_t j = 0; j < gRem; j++) {
                        StC.at(s).rotIn.at(j) = ReduceRotation(
                            (j - int32_t((numRotationsRem + 1) / 2) + 1) * (1 << (s * layersCollapse)), M / 4);
                    }

                    for (int32_t i = 0; i < bRem; i++) {
                        StC.at(s).rotOut.at(i) = ReduceRotation((gRem * i) * (1 << (s * layersCollapse)), M / 4);
                    }
                }

                if constexpr (AFFINE_LT && MAKE_STC_LT_FRIENDLY) {
                    for (int32_t s = 0; s < levelBudget; s++) {
                        int offset = StC.at(s).rotIn[0];
                        for (auto& i : StC.at(s).rotIn) {
                            i = (i - offset);
                        }
                        for (auto& i : StC.at(s).rotOut) {
                            i = (i + offset);
                        }
                    }
                }
            }
        }

        std::reverse(CtS.begin(), CtS.end());

        int acc_offset = 0;
        if constexpr (AFFINE_LT && MAKE_CTS_LT_FRIENDLY) {
            for (size_t s = 0; s < CtS.size(); s++) {
                int offset = CtS.at(s).rotOut[0];
                acc_offset += CtS.at(s).rotOut[0];
                for (int i = 1; i < CtS.at(s).gStep; ++i) {
                    CtS.at(s).rotOut[i] -= offset;
                    CtS.at(s).rotOut[i] %= std::min(2 * slots, static_cast<int>(cc->GetRingDimension()) / 2);
                }

                for (int i = 0; i < CtS.at(s).gStep; ++i) {
                    for (int j = 0; j < CtS.at(s).bStep; ++j) {
                        if (i * CtS.at(s).bStep + j < CtS.at(s).slots) {
                            if (j > 0) {
                                if (CtS.at(s).rotIn[j] - CtS.at(s).rotIn[j - 1] !=
                                    CtS.at(s).rotIn[1] - CtS.at(s).rotIn[0]) {
                                    int new_in = CtS.at(s).rotIn[j - 1] + CtS.at(s).rotIn[1] -
                                                 CtS.at(s).rotIn[0];
                                    CtS.at(s).rotIn[j] = new_in;
                                }
                            }
                        }
                    }
                }
            }
        }

        if constexpr (AFFINE_LT && MAKE_STC_LT_FRIENDLY) {
            for (size_t s = 0; s < StC.size(); s++) {
                int offset = StC.at(s).rotOut[0];
                acc_offset += StC.at(s).rotOut[0];
                for (int i = 1; i < StC.at(s).gStep; ++i) {
                    StC.at(s).rotOut[i] -= offset;
                    StC.at(s).rotOut[i] %= std::min(2 * slots, static_cast<int>(cc->GetRingDimension()) / 2);
                }

                for (int i = 0; i < StC.at(s).gStep; ++i) {
                    for (int j = 0; j < StC.at(s).bStep; ++j) {
                        if (i * StC.at(s).bStep + j < StC.at(s).slots) {
                            if (j > 0) {
                                if (StC.at(s).rotIn[j] - StC.at(s).rotIn[j - 1] !=
                                    StC.at(s).rotIn[1] - StC.at(s).rotIn[0]) {
                                    int new_in = StC.at(s).rotIn[j - 1] + StC.at(s).rotIn[1] -
                                                 StC.at(s).rotIn[0];
                                    StC.at(s).rotIn[j] = new_in;
                                }
                            }
                        }
                    }
                }
            }
        }

        indexes.emplace_back(acc_offset);

        for (auto* v : {&CtS, &StC}) {
            for (auto& i : *v) {
                for (auto& j : i.rotIn) {
                    indexes.push_back(j);
                }
                if constexpr (AFFINE_LT) {
                    // rotOut[0] is not included (it is later set to 0; the last is set to acc_offset)
                    indexes.push_back(i.rotOut.size() > 1 ? i.rotOut[1] : 0);
                } else {
                    for (auto& j : i.rotOut) if (j) indexes.push_back(j);
                }
            }
        }
    }

    int slots_transform = std::min(slots * 2, static_cast<int>(cc->GetCyclotomicOrder()) / 4);
    for (auto& i : indexes) {
        auto j_ = i % slots_transform;
        if (j_ < 0)
            j_ += slots_transform;
        if (j_ > slots_transform / 2)
            j_ += cc->GetCyclotomicOrder() / 4 - slots_transform;
        i = j_;
    }

    if (static_cast<int>(cc->GetRingDimension() / 2) != slots) {
        const int bStep = 4;
        std::vector<int> rotations =
            accumulate_rotation_indices(bStep, slots, static_cast<int>(cc->GetRingDimension() / 2) / slots);
        for (auto idx : rotations) {
            indexes.push_back(idx);
        }
    }

    return indexes;
}

// RawCiphertext.cu
void gen_bootstrap_keys(const LbCC& cc_in, const LbSK& keys, int slots) {
    LbCC cc = cc_in;
    cc->EvalMultKeyGen(keys);

    auto evalKeys = gen_rotation_keys(keys, bootstrap_indexes(cc, slots));
    auto conjKey = fhe_of(cc)->ConjugateKeyGen(keys);

    (*evalKeys)[cc->GetCyclotomicOrder() - 1] = conjKey;

    auto cc_switch = create_switchable_context(cc, 1, 1, static_cast<int>(cc->GetRingDimension() / 2));

    auto [swtch, sk_sparse] = create_context_switching_keys(cc, cc_switch, keys, 32);
    (*evalKeys)[cc->GetCyclotomicOrder() - 2] = swtch.first;
    (*evalKeys)[cc->GetCyclotomicOrder() - 4] =
        swtch.second;  // Use a pair index so no collision with 5^k mod 2N exists

    // sk_sparse and cc_switch are discarded

    lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::InsertEvalAutomorphismKey(
        evalKeys, keys->GetKeyTag());  // Reinsert all keys to add the particular conj key and sse keys
}

std::vector<uint32_t> expected_automorphism_indexes(const LbCC& cc, const std::vector<int32_t>& band,
                                                    const std::vector<uint32_t>& slot_counts) {
    std::set<uint32_t> out;
    const uint32_t M = cc->GetCyclotomicOrder();
    auto scheme = cc->GetScheme();
    // CryptoContextImpl::EvalAtIndexKeyGen maps every rotation step through the scheme's
    // FindAutomorphismIndex (the int32 step travels through the uint32 parameter).
    for (int32_t step : band)
        if (step) out.insert(scheme->FindAutomorphismIndex(static_cast<uint32_t>(step), M));
    for (uint32_t s : slot_counts)
        for (int i : bootstrap_indexes(cc, static_cast<int>(s)))
            if (i) out.insert(scheme->FindAutomorphismIndex(static_cast<uint32_t>(i), M));
    if (!slot_counts.empty()) {   // gen_bootstrap_keys' conj key and ENCAPS switching pair
        out.insert(M - 1);
        out.insert(M - 2);
        out.insert(M - 4);
    }
    return {out.begin(), out.end()};
}

std::map<std::string, std::vector<uint32_t>> automorphism_indexes_in_file(const std::string& path) {
    std::ifstream fr(path, std::ios::binary);
    if (!fr) throw fhe::FHEError("automorphism_indexes_in_file: cannot open '" + path + "'");
    std::map<std::string, std::shared_ptr<LbEvalKeyMap>> omap;
    lbcrypto::Serial::Deserialize(omap, fr, lbcrypto::SerType::BINARY);
    std::map<std::string, std::vector<uint32_t>> out;
    for (const auto& [tag, keys] : omap) {
        std::vector<uint32_t> idx;
        if (keys) for (const auto& [k, v] : *keys) idx.push_back(k);
        out[tag] = std::move(idx);
    }
    return out;
}

std::vector<uint32_t> automorphism_key_indexes(const ClientContext& ctx) {
    std::vector<uint32_t> out;
    if (!ctx.kp.publicKey) return out;
    auto& all = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetAllEvalAutomorphismKeys();
    auto it = all.find(ctx.kp.publicKey->GetKeyTag());
    if (it == all.end() || !it->second) return out;
    for (const auto& [k, v] : *it->second) out.push_back(k);
    return out;
}

}  // namespace perseus_client
