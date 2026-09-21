#include "encoded_block.h"
#include "model/block_residency.h"
#include "op_sequence.h"
#include "slot_layout.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct PyStage {
    py::object state;                   // None | _core.EncodedBlock (cached-block stage)
    std::vector<std::string> weights;   // inf.w keys (streamed-key stage)
    py::object loader;                  // None | (WeightStore, ParsedConfigs, BootstrapPlan,
                                        // block_idx): encode-on-the-fly stage — the canonical
                                        // loader runs per pass (on the worker under Threaded),
                                        // so no pre-encoded state is held host-side
    py::object compute;                 // callable x -> x (main thread, GIL reacquired)
    py::object release;                 // None | callable() (main thread, GIL reacquired);
                                        // REPLACES the stage's default C++ release
    std::string label = "stage";
};

py::object run_stages(Inference& inf, py::object x, std::vector<PyStage> specs,
                      py::object mode_obj) {
    if (specs.empty()) return x;
    const InferenceMode mode =
        mode_obj.is_none() ? inf.mode : mode_obj.cast<InferenceMode>();

    struct Resolved {
        PyStage* spec;
        EncodedBlock* blk = nullptr;   // owned by spec->state, kept alive by `specs`
        // loader stage: encode-on-the-fly via the canonical loader (run_blocks shape)
        const weight_loader::WeightStore* store = nullptr;
        const config_loader::ParsedConfigs* cfg = nullptr;
        BootstrapPlan plan;
        int block = -1;
        std::shared_ptr<EncodedBlock> slot;   // acquire fills it; install moves it out
    };
    std::vector<Resolved> rs;
    rs.reserve(specs.size());
    for (auto& s : specs) {
        Resolved r{&s};
        if (!s.compute || !PyCallable_Check(s.compute.ptr()))
            throw std::runtime_error("run_stages: stage '" + s.label +
                                     "' compute is not callable");
        if (!s.release.is_none() && !PyCallable_Check(s.release.ptr()))
            throw std::runtime_error("run_stages: stage '" + s.label +
                                     "' release is not callable");
        const int kinds = (!s.state.is_none()) + (!s.weights.empty()) + (!s.loader.is_none());
        if (kinds > 1)
            throw std::runtime_error("run_stages: stage '" + s.label +
                                     "' must pick ONE of state / weights / loader");
        if (!s.state.is_none()) {
            try {
                r.blk = s.state.cast<EncodedBlock*>();
            } catch (const py::cast_error&) {
                throw std::runtime_error("run_stages: stage '" + s.label +
                                         "' state is not an EncodedBlock");
            }
        } else if (!s.loader.is_none()) {
            try {
                auto t = s.loader.cast<py::tuple>();
                if (t.size() != 4) throw py::cast_error();
                r.store = t[0].cast<const weight_loader::WeightStore*>();
                r.cfg   = t[1].cast<const config_loader::ParsedConfigs*>();
                r.plan  = t[2].cast<BootstrapPlan>();
                r.block = t[3].cast<int>();
                r.slot  = std::make_shared<EncodedBlock>();
            } catch (const py::cast_error&) {
                throw std::runtime_error("run_stages: stage '" + s.label +
                                         "' loader must be (WeightStore, ParsedConfigs, "
                                         "BootstrapPlan, block_idx)");
            }
        }
        rs.push_back(r);
    }

    bool any_loader = false;
    for (const auto& r : rs) any_loader |= (r.store != nullptr);

    const bool stream_inner = streams_within_block(inf.weight_granularity);

    const Overlap ov = stream_inner ? Overlap::Sync
                     : any_loader   ? overlap_of(mode)
                                    : device_overlap(mode);

    const bool blk_prefetch = !any_loader;

    for (auto& r : rs)
        if (r.blk) evict_block_from_device(inf, *r.blk);

    py::object* xp = &x;

    std::vector<ResidencyStage> stages;
    stages.reserve(rs.size());
    for (int b = 0; b < static_cast<int>(rs.size()); ++b) {
        Resolved* r = &rs[b];
        ResidencyStage st;
        st.label = r->spec->label;

        if (r->blk && !stream_inner) {
            EncodedBlock* blk = r->blk;
            if (blk_prefetch) {
                st.acquire = [blk](Inference& i, cudaStream_t s) {
                    load_block_to_device(i, *blk, s);
                };
                st.prefetch_cpu = [blk](Inference& i) { cpu_extract_block(i, *blk); };
            } else {
                st.acquire = [blk](Inference& i, cudaStream_t s) {
                    cpu_extract_block(i, *blk);          // main thread, before the upload
                    load_block_to_device(i, *blk, s);
                };
            }
            st.install = [blk](Inference& i) { install_block_state_copy(i, *blk); };
            st.release = [blk](Inference& i) {
                { WithStep _w(i, "block_sync"); cudaDeviceSynchronize(); }
                { WithStep _w(i, "weight_evict"); evict_block_from_device(i, *blk); }
            };
            if (blk_prefetch) {
                st.stage_owner = b; blk->stage_owner = b;
            } else {
                blk->stage_owner = -1;
            }
        } else if (r->blk) {   // stream_inner: the ops stream their own weights
            EncodedBlock* blk = r->blk;
            st.install = [blk](Inference& i) { install_block_state_copy(i, *blk); };
            st.release = [blk](Inference& i) {
                { WithStep _w(i, "block_sync"); cudaDeviceSynchronize(); }
                { WithStep _w(i, "weight_evict"); evict_block_from_device(i, *blk); }
            };
            st.prefetch_next = false;
        } else if (r->store) {   // loader stage — mirrors encode_block_op (run_blocks)
            Resolved* L = r;
            st.acquire = [L](Inference& i, cudaStream_t s) {
                const int nthr = pt_stage_block_threads();
                if (nthr > 0) {
                    i.begin_stage_block();
                    i.pt_stage_hook = [&i, nthr](const std::string&, std::vector<Ptx>& pts) {
                        stage_plaintexts(i, pts, nthr);
                    };
                }
                *L->slot = load_block_state(i, *L->store, *L->cfg, L->plan, L->block, s);
                if (nthr > 0) {
                    i.pt_stage_hook = nullptr;
                    stage_block_weights(i, *L->slot, nthr);
                }
            };
            st.install = [L](Inference& i) { install_block_state(i, std::move(*L->slot)); };
            st.release = [](Inference& i) {
                { WithStep _w(i, "block_sync"); cudaDeviceSynchronize(); }
                { WithStep _w(i, "weight_evict"); evict_block_weights(i); }
                reclaim_host_async(i);   // the per-pass encode churns the host allocator
            };
        } else if (!r->spec->weights.empty()) {
            const std::vector<std::string>* keys = &r->spec->weights;
            st.acquire = [keys](Inference& i, cudaStream_t s) {
                load_weight_keys(i, *keys, s);
            };
            st.release = [keys](Inference& i) {
                { WithStep _w(i, "block_sync"); cudaDeviceSynchronize(); }
                evict_weight_keys(i, *keys);
            };
            st.prefetch_next = false;   // a heavy weighted op can't host the next load
        }

        st.compute = [r, xp](Inference& i) {
            WithStep _w(i, r->spec->label);
            py::gil_scoped_acquire gil;
            *xp = r->spec->compute(*xp);
        };
        if (!r->spec->release.is_none()) {
            st.release = [r](Inference&) {
                py::gil_scoped_acquire gil;
                r->spec->release();
            };
        }
        stages.push_back(std::move(st));
    }

    {
        py::gil_scoped_release nogil;
        run_residency_pipeline(inf, std::move(stages), ov);
        // residency release of the transient mask device copies (mirrors run_cached_blocks)
        { WithStep _w(inf, "enc_cache_evict"); inf.evict_enc_cache_device(); }
        if (any_loader) finish_host_reclaim();   // mirrors run_blocks' callers
    }
    return x;
}

}  // namespace

void bind_pipeline(py::module_& m) {
    py::class_<PyStage>(m, "Stage",
        "One residency-pipeline stage: compute(x) -> x on the main thread while the runner "
        "streams the next stage's weights. Exactly one of state= (an EncodedBlock), weights= "
        "(inf.w keys) or loader= describes what to stream. release=: a callable run on the "
        "main thread after compute; it REPLACES the stage's default release (device sync + "
        "evict), so a stage that passes release= must evict what it acquired itself.")
        .def(py::init([](py::object compute, py::object state,
                         std::vector<std::string> weights, py::object loader,
                         py::object release, std::string label) {
                 PyStage s;
                 s.compute = std::move(compute);
                 s.state = std::move(state);
                 s.weights = std::move(weights);
                 s.loader = std::move(loader);
                 s.release = std::move(release);
                 s.label = std::move(label);
                 return s;
             }),
             py::arg("compute"), py::arg("state") = py::none(),
             py::arg("weights") = std::vector<std::string>{},
             py::arg("loader") = py::none(),
             py::arg("release") = py::none(), py::arg("label") = std::string("stage"),
             "One residency-pipeline stage: `compute` (x -> x) is the Python forward; pick at "
             "most one of `state` (EncodedBlock), `weights` (inf.w keys) or `loader` "
             "((WeightStore, ParsedConfigs, BootstrapPlan, block_idx): encode-on-the-fly); "
             "`release` replaces the stage's default C++ release.")
        .def_readwrite("compute", &PyStage::compute,
                       "Callable x -> x: the stage's Python forward (main thread, GIL held).")
        .def_readwrite("state", &PyStage::state,
                       "None | _core.EncodedBlock: pre-encoded state of a cached-block stage.")
        .def_readwrite("weights", &PyStage::weights,
                       "inf.w keys loaded before / evicted after compute (streamed-key stage).")
        .def_readwrite("loader", &PyStage::loader,
                       "None | (WeightStore, ParsedConfigs, BootstrapPlan, block_idx): "
                       "encode-on-the-fly stage, the canonical loader run per pass.")
        .def_readwrite("release", &PyStage::release,
                       "None | callable(): replaces the stage's default C++ release "
                       "(main thread, GIL held).")
        .def_readwrite("label", &PyStage::label,
                       "Stage name: its compute's WithStep profiler step and the run_stages "
                       "error messages.");

    m.def("set_strict_layout", &slotlayout::set_strict, py::arg("on"),
          "Upgrade slot-layout mismatches from a once-per-weight warning to a throw "
          "([layout_error]).");
    m.def("layout_of", [](const PackedCtx& pc) {
        return std::string(slotlayout::name(slotlayout::get(pc.ct)));
    }, py::arg("x"), "Debug: the ciphertext's tracked slot-layout basis.");

    m.def("run_stages", &run_stages,
          py::arg("inf"), py::arg("x"), py::arg("stages"),
          py::arg("mode") = py::none(),
          "Run stages through the residency pipeline: acquire/install/evict each stage's "
          "weight state around its Python compute. Cached states follow the decode shape "
          "(worker CPU extraction + streamed uploads); loader stages the prefill/encoder "
          "shape (per-pass canonical encode on the worker). mode None = inf.mode.");
}
