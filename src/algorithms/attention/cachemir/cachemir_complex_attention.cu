#include "model/gpt2.h"
#include "inference.h"
#include "packing/cachemir/cachemir_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "ckks_types.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cachemir {

PackedCtx complex_qkt(Inference& inf, const PackedCtx& query) {
    WithStep _w(inf, "qkt");
    const int N = inf.slots, d = inf.size.hidDim, H = inf.size.numHeads, t = N / d, tH = t * H;

    PackedCtx q;
    {                                                   // recognized step → site "tok0" (planned-strict priming)
        WithStep _w2(inf, "q_tok0_mask_mult");
        Ptx tok0_pt = inf.encode_at_cached("tok0", real_head_tok0_mask(inf), query);
        q = inf.fhe->mult(query, tok0_pt);
    }
    {
        WithStep _w2(inf, "q_replicate");
        for (int step = 1; step < t; step *= 2) {
            PackedCtx tmp = inf.fhe->rotate(q, mha_rot(inf, -step));
            inf.fhe->inplace_add(q, tmp);
        }
    }

    auto& kc = inf.cache[inf.scoped("k")];
    const int num_cplx = static_cast<int>(kc.size());
    const int kcount   = inf.k_count();

    bool have_attn = false;
    PackedCtx attn_ct;
    auto accum = [&](PackedCtx r) {
        if (!have_attn) { attn_ct = std::move(r); have_attn = true; return; }

        if (inf.fhe->level_for_ct(attn_ct.ct) != inf.fhe->level_for_ct(r.ct))
            throw std::runtime_error(
                "[qkt_accum] level mismatch attn_ct@" + std::to_string(inf.fhe->level_for_ct(attn_ct.ct)) +
                " vs contribution@" + std::to_string(inf.fhe->level_for_ct(r.ct)) +
                " — endless-decode QKT body is not level-uniform");
        inf.fhe->inplace_add(attn_ct, r);
    };

    auto _scope = inf.fhe->graph_scope_guard();

    for (int gc = 0; gc < num_cplx; ++gc) {
        WithStep _wg(inf, "qkt_group");
        inf.fhe->graph_scope_set("qkt.bucket");   // reset body-local names every iteration
        const int g_even = 2 * gc, g_odd = 2 * gc + 1;

        inf.name_graph_ct(kc[gc], inf.scoped("cache.k." + std::to_string(g_even) +
                                             "-lvl=" + std::to_string(inf.fhe->level_for_ct(kc[gc].ct))));
        WithStep _w2g(inf, "q_dot_k");
        PackedCtx result = inf.fhe->mult(q, kc[gc]);
        _w2g.next("tH_reduce");
        for (int s = tH; s < N; s *= 2) {
            PackedCtx rot = inf.fhe->rotate(result, mha_rot(inf, s));
            inf.fhe->inplace_add(result, rot);
        }
        _w2g.next("gmask_mult");
        PackedCtx conj  = inf.fhe->conjugate(result);
        PackedCtx res_e = inf.fhe->add(result, conj);

        const int n_even = std::min(t, kcount - g_even * t);
        Ptx ge = inf.encode_at_cached(qkt_group_mask_tag(kcount, g_even), res_e,
                                      [&]{ return qkt_group_mask_vec(inf, n_even, g_even); });
        accum(inf.fhe->mult(res_e, ge));

        if (g_odd * t < kcount) {
            PackedCtx res_o = inf.fhe->sub(result, conj);
            const int n_odd = std::min(t, kcount - g_odd * t);
            Ptx go = inf.encode_at_cached_complex(qkt_complex_odd_mask_tag(kcount, g_odd), res_o,
                                                  [&]{ return qkt_complex_odd_mask_vec(inf, n_odd, g_odd); });
            accum(inf.fhe->mult(res_o, go));
        }
    }

    auto& pend = inf.cache[inf.scoped("k.pend")];
    for (int j = 0; j < static_cast<int>(pend.size()); ++j) {
        WithStep _wg(inf, "qkt_group");
        inf.fhe->graph_scope_set("qkt.group");    // reset body-local names every iteration
        const int g = 2 * num_cplx + j;

        inf.name_graph_ct(pend[j], inf.scoped("cache.k." + std::to_string(g) +
                                              "-lvl=" + std::to_string(inf.fhe->level_for_ct(pend[j].ct))));
        WithStep _w2g(inf, "q_dot_k");
        PackedCtx result = inf.fhe->mult(q, pend[j]);
        _w2g.next("tH_reduce");
        for (int s = tH; s < N; s *= 2) {
            PackedCtx rot = inf.fhe->rotate(result, mha_rot(inf, s));
            inf.fhe->inplace_add(result, rot);
        }
        _w2g.next("gmask_mult");                        // recognized step → site "qkt.gmask"
        const int n = std::min(t, kcount - g * t);
        Ptx gm = inf.encode_at_cached(qkt_group_mask_tag(kcount, g), result,
                                      [&]{ return qkt_group_mask_vec(inf, n, g); });
        result = inf.fhe->mult(result, gm);
        { WithStep _wc(inf, "im_cleanse"); inf.fhe->inplace_im_cleanse(result); }
        accum(std::move(result));
    }
    return attn_ct;
}


PackedCtx complex_softmax_v(Inference& inf, const PackedCtx& softmax_scores) {
    WithStep _w(inf, "softmax_v");
    const int N = inf.slots, d = inf.size.hidDim, H = inf.size.numHeads, t = N / d, tH = t * H;
    const int d_head_real = inf.size.getRealDHead();
    auto& vc = inf.cache[inf.scoped("v")];   // complex buckets (d_head/2)

    PackedCtx scores_b = inf.fhe->clone(softmax_scores);
    
    Ptx neg_i = inf.encode_complex_const_at(0.0, -1.0, scores_b);   // −i at the (bootstrapped) score level
    PackedCtx conj_S_all;
    {
        WithStep _wl(inf, "lane_mult");
        PackedCtx rot1 = inf.fhe->rotate(scores_b, mha_rot(inf, tH));   // s_k+1 → lane k
        conj_S_all = inf.fhe->add(scores_b, inf.fhe->mult(rot1, neg_i));
    }
    std::vector<PackedCtx> pair_scores;       // conj_S_j at base, j in [1, d_head_real/2)
    {
        WithStep _wl(inf, "lane_mult");
        std::vector<int32_t> steps;
        steps.reserve(d_head_real / 2 - 1);
        for (int j = 1; j < d_head_real / 2; ++j) steps.push_back(mha_rot(inf, 2 * j * tH));
        if (!steps.empty()) pair_scores = inf.fhe->rotate_hoisted(conj_S_all, steps);
    }
    auto conj_S_at = [&](int j) -> const PackedCtx& { return (j == 0) ? conj_S_all : pair_scores[j - 1]; };

    PackedCtx res;
    {
        WithStep _wm(inf, "lane0_mult");
        const int g_lvl = inf.fhe->level_for_ct(vc[0].ct);
        inf.name_graph_ct(vc[0], inf.scoped("cache.v.0-lvl=" + std::to_string(g_lvl)));
        res = inf.fhe->mult(vc[0], conj_S_at(0));
    }
    bool lanes_batched = false;
    if (d_head_real / 2 > 1) {
        std::vector<const PackedCtx*> vptr, sptr;
        vptr.reserve(d_head_real / 2 - 1);
        sptr.reserve(d_head_real / 2 - 1);
        for (int j = 1; j < d_head_real / 2; ++j) {
            vptr.push_back(&vc[j]);
            sptr.push_back(&conj_S_at(j));
        }
        if (inf.fhe->mult_add_many_usable(res, vptr, sptr)) {
            // Batched lanes: one relinearization for all pair products.
            WithStep _wm(inf, "lane_mult");
            for (int j = 1; j < d_head_real / 2; ++j)
                inf.name_graph_ct(vc[j], inf.scoped("cache.v." + std::to_string(j) + "-lvl=" + std::to_string(inf.fhe->level_for_ct(vc[j].ct))));
            inf.fhe->mult_add_many(res, vptr, sptr);
            lanes_batched = true;
        }
    }
    if (!lanes_batched) {
        for (int j = 1; j < d_head_real / 2; ++j) {
            WithStep _wm(inf, "lane_mult");
            const int g_lvl = inf.fhe->level_for_ct(vc[j].ct);
            inf.name_graph_ct(vc[j], inf.scoped("cache.v." + std::to_string(j) + "-lvl=" + std::to_string(g_lvl)));
            PackedCtx contrib = inf.fhe->mult(vc[j], conj_S_at(j));
            inf.fhe->inplace_add(res, contrib);
        }
    }

    { WithStep _wr(inf, "tok_reduce");
      for (int step = 1; step < t; step *= 2)
          inf.fhe->inplace_add(res, inf.fhe->rotate(res, mha_rot(inf, step))); }
    { WithStep _wt(inf, "tok0_mask_mult");
      inf.fhe->inplace_im_cleanse(res);   // 2*Re; the ×0.5 tok0h head-select mask absorbs the doubling
      Ptx tok0h = inf.encode_at_cached("tok0.h", res, [&] { return real_head_half_mask(inf); });
      res = inf.fhe->mult(res, tok0h); }
    return res;
}

}  // namespace cachemir
