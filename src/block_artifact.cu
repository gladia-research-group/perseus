// Block artifacts (cachemir_lib): save / read / diff of an EncodedBlock's plaintexts and
// the coeff-encode producer. Moved verbatim out of src/bindings/serial.cu (PB-07); the
// format is documented in include/block_artifact.h.
#include "block_artifact.h"
#include "build_stamp.h"
#include "plan_json.h"

#include "ciphertext-ser.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string lb_serialize_poly(const lbcrypto::DCRTPoly& poly) {
    std::stringstream ss;
    lbcrypto::Serial::Serialize(poly, ss, lbcrypto::SerType::BINARY);
    return ss.str();
}

}  // namespace

namespace block_artifact {

void save_block_state(Inference& inf, const EncodedBlock& blk, const std::string& path) {
    std::ostringstream hdr;
    std::string blobs;
    hdr << "[{\"n\":\"__meta__\",\"format\":3,\"chain\":\"" << perseus_stamp::kChain
        << "\",\"logN\":" << inf.logN << ",\"perseus\":\"" << PERSEUS_VERSION << "\"}";
    bool first = false;
    size_t full = 0, staged = 0;
    for (const auto& [name, pts] : blk.w) {
        for (size_t i = 0; i < pts.size(); ++i) {
            const Ptx& pt = pts[i];
            if (!pt) continue;
            auto& lb = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
            const auto& poly = lb->GetElement<lbcrypto::DCRTPoly>();
            std::string blob;
            if (pt->coeff_staged) {
                blob = lb_serialize_poly(poly);
            } else {

                const auto vals = lb->GetRealPackedValue();
                blob.assign(reinterpret_cast<const char*>(vals.data()),
                            vals.size() * sizeof(double));
            }
            (pt->coeff_staged ? staged : full)++;
            hdr << (first ? "" : ",") << "{\"n\":\"" << name << "\",\"i\":" << i
                << ",\"lv\":" << lb->GetLevel() << ",\"deg\":" << lb->GetNoiseScaleDeg()
                << ",\"sf\":" << std::setprecision(17) << lb->GetScalingFactor()
                << ",\"slots\":" << lb->GetSlots()
                << ",\"cs\":" << (pt->coeff_staged ? 1 : 0)
                << ",\"pre\":" << pt->coeff_prescale_log2
                << ",\"off\":" << blobs.size() << ",\"len\":" << blob.size() << "}";
            first = false;
            blobs += blob;
        }
    }
    hdr << "]";
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_block_state: cannot open " + path);
    const std::string h = hdr.str();
    const uint64_t hlen = h.size();
    f.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
    f.write(h.data(), static_cast<std::streamsize>(h.size()));
    f.write(blobs.data(), static_cast<std::streamsize>(blobs.size()));
    std::fprintf(stderr, "[artifact] %s: %zu coeff + %zu full plaintexts, %.1f MB\n",
                 path.c_str(), staged, full,
                 (sizeof(hlen) + h.size() + blobs.size()) / 1048576.0);
}

EncodedBlock encode_block_state_coeff(Inference& inf,
                                      const weight_loader::WeightStore& store,
                                      const config_loader::ParsedConfigs& parsed,
                                      const BootstrapPlan& plan, int block_idx) {

    auto saved = inf.pt_stage_hook;
    inf.pt_stage_hook = [](const std::string&, std::vector<Ptx>&) {};
    EncodedBlock blk;
    try {
        blk = load_block_state(inf, store, parsed, plan, block_idx, nullptr);
    } catch (...) {
        inf.pt_stage_hook = saved;
        throw;
    }
    inf.pt_stage_hook = saved;
    evict_block_from_device(inf, blk);
    return blk;
}

EncodedBlock read_block_artifact(Inference& inf, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_block_state_file: cannot open " + path);
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), sizeof(hlen));
    std::string h(hlen, '\0');
    f.read(h.data(), static_cast<std::streamsize>(hlen));
    const std::streampos blob0 = f.tellg();
    const auto doc = planjson::parse(h);   // the repo's iterative JSON parser
    EncodedBlock blk;
    for (const auto& e : doc.arr) {
        auto num = [&e](const char* k) {
            const auto* v = e.find(k);
            if (!v) throw std::runtime_error(std::string("artifact header missing ") + k);
            return v->number;
        };
        const std::string name = e.find("n")->str;
        if (name == "__meta__") {
            const auto* chain = e.find("chain");
            if (chain && chain->str != perseus_stamp::kChain)
                throw std::runtime_error("load_block_state_file: " + path + " was encoded on chain " +
                                         chain->str + " but this session is " + perseus_stamp::kChain);
            const auto* ln = e.find("logN");
            if (ln && static_cast<int>(ln->number) != inf.logN)
                throw std::runtime_error("load_block_state_file: " + path + " was encoded at logN=" +
                                         std::to_string(static_cast<int>(ln->number)) +
                                         " but this session is logN=" + std::to_string(inf.logN));
            continue;
        }
        const size_t idx = static_cast<size_t>(num("i"));
        const uint32_t lv = static_cast<uint32_t>(num("lv"));
        const int deg = static_cast<int>(num("deg"));
        const double sf = num("sf");
        const uint32_t slots = static_cast<uint32_t>(num("slots"));
        const bool cs = num("cs") != 0;
        const int pre = static_cast<int>(num("pre"));
        const uint64_t off = static_cast<uint64_t>(num("off"));
        const uint64_t len = static_cast<uint64_t>(num("len"));
        std::string blob(len, '\0');
        f.seekg(blob0 + static_cast<std::streamoff>(off));
        f.read(blob.data(), static_cast<std::streamsize>(len));
        Ptx pt;
        if (!cs) {
            // FULL pt: exact re-encode from the stored slot values (deterministic;
            // the values are what GetRealPackedValue must return downstream).
            std::vector<double> vals(len / sizeof(double));
            std::memcpy(vals.data(), blob.data(), len);
            pt = inf.cc()->MakeCKKSPackedPlaintext(vals, deg, lv);
            auto& lb = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
            lb->SetSlots(slots);
            auto& vec0 = blk.w[name];
            if (vec0.size() <= idx) vec0.resize(idx + 1);
            vec0[idx] = pt;
            continue;
        }
        lbcrypto::DCRTPoly poly;
        {
            std::stringstream ss(std::move(blob));
            lbcrypto::Serial::Deserialize(poly, ss, lbcrypto::SerType::BINARY);
        }

        std::vector<double> dummy(1, 0.0);
        Ptx pt2 = inf.cc()->MakeCKKSPackedPlaintext(dummy, deg, lv);
        pt = pt2;
        auto& lb = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
        lb->GetElement<lbcrypto::DCRTPoly>() = std::move(poly);
        lb->SetSlots(slots);
        if (cs) {

            inf.cc()->MarkCoeffStaged(pt, lv, sf, pre);
        }
        auto& vec = blk.w[name];
        if (vec.size() <= idx) vec.resize(idx + 1);
        vec[idx] = pt;
    }
    return blk;
}

std::string block_state_diff(const EncodedBlock& a, const EncodedBlock& b) {
    std::ostringstream out;
    size_t pts = 0, elem_mismatch = 0, val_mismatch = 0, meta_mismatch = 0;
    if (a.w.size() != b.w.size())
        out << "name-count " << a.w.size() << " vs " << b.w.size() << "; ";
    for (const auto& [name, va] : a.w) {
        auto it = b.w.find(name);
        if (it == b.w.end()) { out << "missing " << name << "; "; continue; }
        const auto& vb = it->second;
        if (va.size() != vb.size()) { out << name << " count; "; continue; }
        for (size_t i = 0; i < va.size(); ++i) {
            const Ptx& pa = va[i]; const Ptx& pb = vb[i];
            if (!pa || !pb) continue;
            ++pts;
            auto& la = std::any_cast<const lbcrypto::Plaintext&>(pa->cpu);
            auto& lc = std::any_cast<const lbcrypto::Plaintext&>(pb->cpu);
            if (pa->coeff_staged != pb->coeff_staged ||
                la->GetLevel() != lc->GetLevel() ||
                la->GetNoiseScaleDeg() != lc->GetNoiseScaleDeg())
                { ++meta_mismatch; continue; }
            if (pa->coeff_staged) {
                if (!(la->GetElement<lbcrypto::DCRTPoly>() ==
                      lc->GetElement<lbcrypto::DCRTPoly>()))
                    ++elem_mismatch;
            } else {
                const auto x = la->GetRealPackedValue();
                const auto y = lc->GetRealPackedValue();
                double d = 0.0;
                for (size_t k = 0; k < std::min(x.size(), y.size()); ++k)
                    d = std::max(d, std::abs(x[k] - y[k]));
                if (x.size() != y.size() || d > 1e-9) ++val_mismatch;
            }
        }
    }
    out << "pts=" << pts << " elem_mismatch=" << elem_mismatch
        << " val_mismatch=" << val_mismatch << " meta_mismatch=" << meta_mismatch;
    return out.str();
}

}  // namespace block_artifact
