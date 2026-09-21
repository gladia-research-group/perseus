#include "model/mha.h"

#include "attention.h"
#include "model/gpt2.h"
#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "packing/cachemir_filling/cachemir_filling_delta_attention.h"

#include <iostream>
#include <mutex>

std::vector<Op> mha_ops() {
    return {
        { {}, [](Inference& i, PackedCtx& x) {
            const int d = i.size.hidDim;
            i.name_graph_ct_if_absent(x, "mha_block.x");
            i.fhe->bootstrap_hint(x, i.fhe->level_headroom(1), /*account_pending_rescale=*/true);
            i.fhe->level_hint(x, i.fhe->level_headroom(1));
            if (i.complex) {
                auto qv = linear_multi(i, x, {"kv", "q"}, d, d,
                                       /*stream_pt=*/is_cachemir_filling(i.packing));
                cache_kv_push_packed(i, qv[0]);   // fused "kv" linear P=K+iV → one bootstrap → complex caches
                x = qv[1];
            } else {
                auto qkv = linear_multi(i, x, {"k", "v", "q"}, d, d,
                                        /*stream_pt=*/is_cachemir_filling(i.packing));
                cache_kv_push(i, qkv[0], qkv[1]);
                x = qkv[2];
            } }, "qkv" },
        { {}, [](Inference& i, PackedCtx& x) {
            i.fhe->level_hint(x, i.fhe->level_headroom(3));   // ceiling-relative (was 21 = 24-3)
            std::vector<PackedCtx> scores = qkt(i, x);
            scores = attention_softmax_thor(i, std::move(scores), "attn");
            x = softmax_v(i, std::move(scores)); }, "attn_core" },
        { {}, [](Inference& i, PackedCtx& x) {
            i.fhe->bootstrap_hint(x, i.fhe->level_headroom(1), /*account_pending_rescale=*/true);
            i.fhe->level_hint(x, i.fhe->level_headroom(1));
            x = linear(i, x, "out", i.size.hidDim, i.size.hidDim,
                       /*stream_pt=*/is_cachemir_filling(i.packing)); }, "out_proj" },
    };
}

std::vector<Op> mha_ops_token_pair() {
    return {
        { {}, [](Inference& i, PackedCtx& x) {
            i.name_graph_ct_if_absent(x, "mha_block.x");
            i.fhe->bootstrap_hint(x, i.fhe->level_headroom(1), /*account_pending_rescale=*/true);
            i.fhe->level_hint(x, i.fhe->level_headroom(1));
            cachemir_filling::mha_qkv_token_pair(i, x); }, "qkv" },
        { {}, [](Inference& i, PackedCtx& x) {
            x = cachemir_filling::delta_block_enabled()
                    ? cachemir_filling::mha_attn_token_pair_delta(i, x)
                    : mha_attn_token_pair(i, x); }, "attn_core" },
        { {}, [](Inference& i, PackedCtx& x) {
            i.fhe->bootstrap_hint(x, i.fhe->level_headroom(1), /*account_pending_rescale=*/true);
            i.fhe->level_hint(x, i.fhe->level_headroom(1));
            x = linear(i, x, "out", i.size.hidDim, i.size.hidDim,
                       /*stream_pt=*/is_cachemir_filling(i.packing)); }, "out_proj" },
    };
}

PackedCtx mha_block(Inference& inf, PackedCtx& x) {
    WithStep _w(inf, "mha_block");

    if (!inf.use_cache) {
        static std::once_flag _no_cache_warn;
        std::call_once(_no_cache_warn, []() {
            std::cout << "[warning]: cache disabled, be sure you are not generating!"
                      << std::endl;
        });
        prepare_mha_masks(inf);
        prepare_vcache(inf);
    }

    const bool stream = streams_within_block(inf.weight_granularity);
    const Overlap ov  = stream ? device_overlap(inf.mode) : Overlap::Sync;
    const bool token_pair = is_cachemir_filling(inf.packing) && inf.token_pair;
    return run_ops(inf, x, token_pair ? mha_ops_token_pair() : mha_ops(), stream, ov);
}
