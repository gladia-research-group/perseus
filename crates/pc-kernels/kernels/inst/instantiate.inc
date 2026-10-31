// One (word, ModMul) instantiation of every kernel template plus its KernelTable. Included by exactly one TU per
// instantiation after defining PC_W (word type), PC_MM (strategy template) and PC_TABLE (exported function name).
// The kernels live in an anonymous namespace so the TUs' identical W-only templates never collide at link time.
#include <cmath>
#include "../args.cuh"

namespace {
#include "../arith.cuh"
#include "../ntt.cuh"
#include "../bconv.cuh"
#include "../elem.cuh"
#include "../ks.cuh"
#include "../cts.cuh"
#include "../em.cuh"
#include "../rescale.cuh"
#include "../automorph.cuh"
#include "../encode.cuh"
#include "../encrypt.cuh"
#include "../keygen.cuh"
#include "../bts.cuh"
#include "../spru.cuh"
#include "../pcmm.cuh"
#include "../ccmm.cuh"
#include "../coeffrows.cuh"

using W = PC_W;
using MM = pc::PC_MM<W>;
}  // namespace

extern "C" const KernelTable* PC_TABLE() {
    static constexpr KernelTable T = {
        PC_KERNELS_ABI,
        (uint32_t)sizeof(W),
        MM::KIND,
        (uint32_t)PC_LOGN,
        &pc::ntt::fwd<W>,
        &pc::ntt::inv<W>,
        &pc::bconv::launch<W, MM>,
        &pc::elem::add_l<W>,
        &pc::elem::scale_l<W>,
        &pc::elem::sub_mul_l<W>,
        &pc::elem::copy_l<W>,
        &pc::elem::pt_dot_l<W, MM>,
        &pc::ks::dot_l<W, MM>,
        &pc::ks::regen_l<W, MM>,
        &pc::ks::expand_l<W, MM>,
        &pc::cts::launch<W, MM>,
        &pc::em::tensor_l<W, MM>,
        &pc::em::md_epilogue_l<W>,
        &pc::em::wsum_l<W>,
        &pc::em::add_const_l<W>,
        &pc::em::mul_table_l<W, MM>,
        &pc::rescale::coef_l<W, MM>,
        &pc::rescale::combine_l<W>,
        &pc::automorph::launch<W>,
        &pc::encode::launch<W, MM>,
        &pc::encrypt::launch<W, MM>,
        &pc::keygen::gen_l<W, MM>,
        &pc::keygen::unpack_l<W>,
        &pc::bts::mod_raise_l<W, MM>,
        &pc::bts::small_int_l<W, MM>,
        &pc::spru::extract<W>,
        &pc::spru::fft_pass,
        &pc::spru::scatter<W>,
        &pc::cts::launch_regen<W, MM>,
        &pc::ks::key_l<W, MM>,
        &pc::pcmm::launch<W>,
        &pc::ccmm::launch<W>,
        &pc::em::wsum_k_l<W>,
        &pc::em::combine_to_l<W>,
        &pc::em::md_epilogue_to_l<W>,
        &pc::bts::small_int1_l<W, MM>,
        &pc::coeffrows::launch<W, MM>,
        &pc::em::mul_add_l<W, MM>,
        &pc::rescale::fwd_l<W, MM>,
        &pc::em::mul_add_n_l<W, MM>,
    };
    return &T;
}
