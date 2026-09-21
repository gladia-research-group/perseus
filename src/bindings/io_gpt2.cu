#include "config_loader.h"

#include <algorithm>
#include "npconv.h"
#include "cutmax.h"
#include "encoded_block.h"
#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "weight_loader.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {
constexpr auto kRelease = py::call_guard<py::gil_scoped_release>();
}

// Persistent lm_head tile store (mirrors GPT2Model::lm_tiles_): encode once, reuse per token.
struct LMHeadCache {
    std::vector<EncodedBlock> tiles;
};

void bind_io_gpt2(py::module_& m) {
    m.def("kv_prefetch_first", &gpt2_kv_prefetch_first,
          py::arg("inf"), py::arg("n_blocks"), kRelease,
          "Overlapped KV pipeline: seed block 0's KV reload before the block loop.");
    m.def("kv_block_prologue", &gpt2_kv_block_prologue,
          py::arg("inf"), py::arg("block_idx"), py::arg("n_blocks"), kRelease,
          "Per-block KV prologue: prefetch the next block's KV reload.");
    m.def("kv_finalize_last", &gpt2_kv_finalize_last,
          py::arg("inf"), py::arg("n_blocks"), kRelease,
          "Post-loop: drain + evict the last block's deferred KV offload.");
    m.def("block_release", &gpt2_block_release, py::arg("inf"), py::arg("block_idx"), kRelease,
          "Decode-arm block release: device sync, evict the block weights, offload its KV.");
    m.def("apply_final_ln", &apply_final_ln, py::arg("inf"), py::arg("x"), py::arg("lnf"), kRelease,
          "Install lnf, apply the final LayerNorm (ln_f) to x, evict lnf; returns the normed ct.");
    m.def("extract_token_i_cachemir", &extract_token_i_cachemir,
          py::arg("inf"), py::arg("filling_ct"), py::arg("i"), kRelease,
          "Extract token i of a cachemir_filling group ct into a cachemir single-token ct.");
    m.def("reset_graph_runtime", &gpt2_reset_graph_runtime, py::arg("inf"),
          "Reset the runtime-graph naming state (ct/pt vars + counters) for capture/planned runs.");
    m.def("reset_kv_cache", &gpt2_reset_kv_cache, py::arg("inf"), py::arg("n_blocks"), kRelease,
          "Reset every block's K/V caches for a fresh sequence (also prewarms the pinned arenas).");

    m.def("configure_prefill_phase", [](Inference& inf, int n_tok) {
        inf.packing.kind = PackingKind::CachemirFilling;
        inf.complex      = false;   // filling arm; TP carries the complex payload
        const int t      = inf.slots / inf.size.hidDim;
        inf.n_tok        = (inf.token_pair && n_tok > t) ? t : n_tok;
        if (!inf.token_pair) inf.n_tok_imag = 0;
        inf.weight_store = nullptr;
        inf.weight_granularity = WeightGranularity::Plaintext;
    }, py::arg("inf"), py::arg("n_tok"),
       "Set inf's prefill phase fields: CachemirFilling packing, n_tok, plaintext weights.");
    m.def("configure_decode_phase", [](Inference& inf, bool complex_decode) {
        inf.packing.kind       = PackingKind::Cachemir;
        inf.n_tok              = 1;
        inf.weight_granularity = WeightGranularity::Block;
        inf.token_pair         = false;
        inf.complex            = complex_decode;
    }, py::arg("inf"), py::arg("complex_decode"),
       "Set inf's decode phase fields: Cachemir packing, n_tok=1, block weights, complex flag.");
    m.def("gpt2_prefill",
          [](Inference& inf, PackedCtx x, const weight_loader::WeightStore& store,
             const config_loader::ParsedConfigs& parsed, int n_blocks, bool chunk) {
              return gpt2_prefill(inf, std::move(x), store, parsed, BlockPlans{},
                                  n_blocks, chunk ? PrefillMode::Chunk
                                                  : PrefillMode::SingleShot);
          },
          py::arg("inf"), py::arg("x"), py::arg("store"), py::arg("configs"),
          py::arg("n_blocks"), py::arg("chunk") = true, kRelease,
          "One EAGER prefill chunk over the filling packing (weights encoded per pass, "
          "worker-overlapped). chunk=True skips the final LN (caller LNs after the last "
          "chunk). KV lands in the filling cache: kv_handoff_filling_to_cachemir after "
          "the last chunk.");
    m.def("filling_rot_steps", [](Inference& inf) {
        return gpt2_filling_only_rot_steps(inf.slots, inf.size.hidDim,
                                           inf.size.expDim, inf.size.numHeads);
    }, py::arg("inf"), "The filling-exclusive rotation steps (freeable post-prefill).");

    m.def("lm_head_vocab",
          [](const weight_loader::WeightStore& store) {
              if (!store.has(weight_loader::gpt2_lm_head_name())) throw py::key_error(weight_loader::gpt2_lm_head_name());
              return static_cast<int>(store.meta(weight_loader::gpt2_lm_head_name()).shape[0]);
          },
          py::arg("store"), "Vocab size: rows of the lm_head (wte) weight in the store.");
    m.def("lm_head_tile_width", &lm_head_tile_width, py::arg("inf"), py::arg("vocab"),
          "The lm_head tile width: hidDim when vocab <= hidDim (small head), else slots.");

    py::class_<LMHeadCache>(m, "LMHeadCache")
        .def(py::init<>(),
             "An empty lm_head tile store; filled by lm_head / prepare_feedback_weights.");

    m.def("token_embedding",
          [](const weight_loader::WeightStore& store, int token, int position) {
              auto row_of = [&](const std::string& name, int idx) {
                  const auto& flat  = store.tensor(name);
                  const auto& shape = store.meta(name).shape;
                  const int n = static_cast<int>(shape[0]), d = static_cast<int>(shape[1]);
                  if (idx < 0 || idx >= n)
                      throw std::runtime_error("token_embedding: index " + std::to_string(idx) +
                                               " out of [0," + std::to_string(n) + ") for " + name);
                  return std::make_pair(flat.data() + static_cast<size_t>(idx) * d, d);
              };
              auto [wte, d1] = row_of(weight_loader::gpt2_wte_name(), token);
              auto [wpe, d2] = row_of(weight_loader::gpt2_wpe_name(), position);
              const int d = std::min(d1, d2);
              std::vector<double> e(static_cast<size_t>(d));
              for (int j = 0; j < d; ++j) e[j] = wte[j] + wpe[j];
              return e;
          },
          py::arg("store"), py::arg("token"), py::arg("position"), kRelease,
          "wte[token] + wpe[position] read from the plaintext store (client-side embedding).");

    m.def("prepare_feedback_weights",
          [](Inference& inf, const weight_loader::WeightStore& store, int vocab,
             bool packed_z, LMHeadCache& cache, const BootstrapPlan* plan14) {
              gpt2_prepare_feedback_weights(inf, store, vocab,
                                            lm_head_tile_width(inf, vocab), packed_z,
                                            cache.tiles, plan14);
          },
          py::arg("inf"), py::arg("store"), py::arg("vocab"), py::arg("packed_z"),
          py::arg("cache"), py::arg("plan14") = static_cast<const BootstrapPlan*>(nullptr),
          kRelease,
          "Encode the CutMax one-hot codebook (wte tiles) into cache once; "
          "plan14 = strict-tail encode level.");

    m.def("cutmax_feedback",
          [](Inference& inf, const std::vector<PackedCtx>& tiles,
             const weight_loader::WeightStore& store, int vocab, const CutMaxConfig& cmc,
             LMHeadCache* fb, int position, const BootstrapPlan* plan13,
             const BootstrapPlan* plan14) {
              std::vector<PackedCtx> z;
              double am_s = 0.0;
              PackedCtx x;
              {
                  py::gil_scoped_release nogil;
                  EncodedBlock* fb_blk =
                      (fb && !fb->tiles.empty()) ? &fb->tiles.front() : nullptr;
                  x = gpt2_cutmax_feedback(inf, tiles, store, vocab,
                                           lm_head_tile_width(inf, vocab), cmc, fb_blk,
                                           position, &z, &am_s, plan13, plan14);
              }
              py::object xr = (position >= 0) ? py::cast(x) : py::object(py::none());
              return py::make_tuple(xr, py::cast(z), am_s);
          },
          py::arg("inf"), py::arg("tiles"), py::arg("store"), py::arg("vocab"),
          py::arg("config"), py::arg("feedback") = static_cast<LMHeadCache*>(nullptr),
          py::arg("position") = -1,
          py::arg("plan13") = static_cast<const BootstrapPlan*>(nullptr),
          py::arg("plan14") = static_cast<const BootstrapPlan*>(nullptr),
          "Generation tail: CutMax argmax + codebook feedback + wpe + entry bootstrap. Returns "
          "(next_x | None, z_tiles, argmax_seconds); position < 0 skips the feedback half.");
    m.def("kv_handoff_filling_to_cachemir", &gpt2_kv_handoff_filling_to_cachemir,
          py::arg("inf"), py::arg("n_blocks"), py::arg("m"), kRelease,
          "Convert a filling-packing prefill's KV caches to the cachemir decode layout.");
    m.def("lm_head",
          [](Inference& inf, const PackedCtx& x, const weight_loader::WeightStore& store,
             int vocab, const BootstrapPlan& plan, LMHeadCache* cache) {
              return gpt2_lm_head(inf, x, store, vocab, lm_head_tile_width(inf, vocab),
                                  cache ? &cache->tiles : nullptr, plan);
          },
          py::arg("inf"), py::arg("x"), py::arg("store"), py::arg("vocab"), py::arg("plan"),
          py::arg("cache") = static_cast<LMHeadCache*>(nullptr), kRelease,
          "lm_head logits of x as ciphertext tiles (tiles encoded once into `cache` when given).");
    m.def("decode_lm_head_logits",
          [](Inference& inf, const std::vector<PackedCtx>& tiles, int vocab) {
              std::vector<double> v;
              {
                  py::gil_scoped_release nogil;
                  v = decode_lm_head_logits(inf, tiles, vocab, lm_head_tile_width(inf, vocab));
              }
              return perseus_np::from_vec(v);
          },
          py::arg("inf"), py::arg("tiles"), py::arg("vocab"),
          "Decrypt lm_head logit tiles into a [vocab] float array (uses inf's secret key).");
}
