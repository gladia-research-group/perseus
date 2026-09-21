#include "fideslib_wrapper.h"
#include <filesystem>
#include <unistd.h>
#include <cerrno>
#include "block_artifact.h"
#include "config_loader.h"
#include "inference.h"
#include "weight_loader.h"

#include "ciphertext-ser.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace {

constexpr auto kRelease = py::call_guard<py::gil_scoped_release>();

using LbCt = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

py::bytes serialize_ct(Inference& inf, const PackedCtx& pc) {
    if (!pc.ct) throw std::runtime_error("serialize_ct: empty ciphertext");
    std::stringstream ss;
    {
        py::gil_scoped_release nogil;
        Ctx ct = pc.ct;
        inf.fhe->sync_ciphertext_cpu_from_device(ct);
        auto& host = std::any_cast<LbCt&>(ct->cpu);
        lbcrypto::Serial::Serialize(host, ss, lbcrypto::SerType::BINARY);
    }
    return py::bytes(ss.str());
}

PackedCtx deserialize_ct(Inference& inf, const py::bytes& data) {
    std::string buf = data;   // copy under GIL
    LbCt host;
    {
        py::gil_scoped_release nogil;
        std::stringstream ss(std::move(buf));
        lbcrypto::Serial::Deserialize(host, ss, lbcrypto::SerType::BINARY);
    }
    if (!host) throw std::runtime_error("deserialize_ct: no ciphertext in payload");
    CC cc = inf.cc();
    Ctx nc = std::make_shared<fideslib::CiphertextImpl<fideslib::DCRTPoly>>(std::move(cc));
    nc->cpu = host;      // host-resident; the GPU shadow loads on first use
    nc->loaded = false;
    nc->gpu = 0;
    return inf.pack(nc);
}

void save_keys(Inference& inf, const std::string& dir) {
    using CCI = fideslib::CryptoContextImpl<fideslib::DCRTPoly>;
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir))
        throw std::runtime_error("save_keys: cannot use bundle directory '" + dir + "': " +
                                 (ec ? ec.message() : std::string("not a directory")));
    auto fail = [](const char* what, const std::string& path) {
        const int e = errno;
        std::string msg = std::string("save_keys: ") + what + " serialization failed for '" +
                          path + "'";
        if (e) msg += std::string(" (") + std::strerror(e) + ")";
        throw std::runtime_error(msg);
    };
    const std::string tmp = ".partial-" + std::to_string(::getpid()) + "-";
    auto commit = [&](const std::string& name, const char* what) {
        std::error_code rc;
        fs::rename(dir + "/" + tmp + name, dir + "/" + name, rc);
        if (rc) throw std::runtime_error(std::string("save_keys: cannot rename ") + what + " into '" +
                                         dir + "/" + name + "': " + rc.message());
    };
    errno = 0;
    if (!fideslib::Serial::SerializeToFile(dir + "/" + tmp + "context.bin", inf.cc(), fideslib::BINARY))
        fail("context", dir + "/" + tmp + "context.bin");
    commit("context.bin", "context");
    {   // FIDESlib writes a device sidecar next to the context file under the same base name
        std::error_code rc;
        if (fs::exists(dir + "/" + tmp + "context.bin.dev", rc))
            commit("context.bin.dev", "context sidecar");
    }
    errno = 0;
    if (!fideslib::Serial::SerializeToFile(dir + "/" + tmp + "public.key", inf.fhe->keys.publicKey,
                                           fideslib::BINARY))
        fail("public key", dir + "/" + tmp + "public.key");
    commit("public.key", "public key");
    errno = 0;
    {
        std::ofstream fm(dir + "/" + tmp + "multkeys.bin", std::ios::binary);
        if (!fm || !CCI::SerializeEvalMultKey(fm, fideslib::BINARY))
            fail("eval-mult key", dir + "/" + tmp + "multkeys.bin");
    }
    commit("multkeys.bin", "eval-mult key");
    errno = 0;
    {
        std::ofstream fr(dir + "/" + tmp + "rotkeys.bin", std::ios::binary);
        if (!fr || !CCI::SerializeEvalAutomorphismKey(fr, fideslib::BINARY))
            fail("rotation key", dir + "/" + tmp + "rotkeys.bin");
    }
    commit("rotkeys.bin", "rotation key");
}

void save_secret_key(Inference& inf, const std::string& path) {
    if (!fideslib::Serial::SerializeToFile(path, inf.fhe->keys.secretKey, fideslib::BINARY))
        throw std::runtime_error("save_secret_key: serialization failed");
}

EncodedBlock load_block_state_file(Inference& inf,
                                   const config_loader::ParsedConfigs& parsed,
                                   const BootstrapPlan& plan, int block_idx,
                                   const std::string& path) {
    EncodedBlock blk = block_artifact::read_block_artifact(inf, path);
    const std::string base = weight_loader::gpt2_block_base(block_idx);
    blk.prefix = base + ".";
    auto need = [](const auto& table, const std::string& key, const char* section) -> const auto& {
        auto it = table.find(key);
        if (it == table.end())
            throw py::key_error("configs.json has no '" + std::string(section) + "' section for " + key);
        return it->second;
    };
    blk.norm_cfg["ln_1"]    = need(parsed.norm,     base + ".ln_1",    "norm");
    blk.norm_cfg["ln_2"]    = need(parsed.norm,     base + ".ln_2",    "norm");
    blk.sm_cfg  ["attn"]    = need(parsed.softmax,  base + ".attn",    "softmax");
    blk.gelu_cfg["mlp.act"] = need(parsed.softgelu, base + ".mlp.act", "softgelu");
    blk.plan = plan;
    return blk;
}

}  // namespace

void bind_serial(py::module_& m) {
    m.def("serialize_ct", &serialize_ct, py::arg("inf"), py::arg("x"),
          "Ciphertext -> bytes (OpenFHE binary), syncing a device-computed value to the "
          "host first. The packing rides out-of-band: deserialize_ct stamps the session's.");
    m.def("deserialize_ct", &deserialize_ct, py::arg("inf"), py::arg("data"),
          "bytes -> PackedCtx in this session's context; host-resident until first use.");
    m.def("save_keys", &save_keys, py::arg("inf"), py::arg("dir"), kRelease,
          "Write the SERVER bundle (context/public/eval keys — no secret material).");
    m.def("save_secret_key", &save_secret_key, py::arg("inf"), py::arg("path"), kRelease,
          "Write the CLIENT's secret key. This file never leaves the client.");
    m.def("encode_block_state_coeff", &block_artifact::encode_block_state_coeff,
          py::arg("inf"), py::arg("store"), py::arg("configs"), py::arg("plan"),
          py::arg("block_idx"), kRelease,
          "load_block_state with the coeff-encode gate armed (needs FHE_PT_COEFF_ENCODE=1): "
          "eligible weight plaintexts come back as 1-limb coeff-staged forms.");
    m.def("save_block_state", &block_artifact::save_block_state, py::arg("inf"), py::arg("state"),
          py::arg("path"), kRelease,
          "Serialize an EncodedBlock's plaintexts (coeff-staged ~0.5 MB each; gate-failed "
          "ones full-size) to one artifact file.");
    m.def("block_state_diff", &block_artifact::block_state_diff, py::arg("a"), py::arg("b"),
          "Compare two EncodedBlocks pt-by-pt: DCRT element equality for coeff-staged, "
          "slot values for full. Returns a summary string (all-zero mismatches = equal).");
    m.def("load_block_state_file", &load_block_state_file, py::arg("inf"),
          py::arg("configs"), py::arg("plan"), py::arg("block_idx"), py::arg("path"),
          kRelease, "Rebuild an EncodedBlock from a save_block_state artifact: weights "
          "from disk, configs/prefix/plan from the canonical parse.");
}
