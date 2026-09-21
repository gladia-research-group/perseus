#include "ckks_fixture.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace FIDESlib::CKKS { void setArcsineOverride(int v); }

using namespace test_helpers;

namespace {

// Reserve-neutrality + scoping probe. Run the SAME binary in three env
// regimes (scripts/23_arcsine_scope.sh):
//   vanilla                      -> baseline levels/errors (ScopedOn skipped)
//   FIDESLIB_ARCSINE_RESERVE=1   -> the decode go/no-go: ScopedOff must match
//                                   vanilla (level AND error); ScopedOn must
//                                   engage (flat A=100) inside the same process
//   FIDESLIB_ARCSINE=1           -> legacy both-on sanity
// A=100 discriminates: arcsine-off err ~1.6e-2 (cubic), arcsine-on ~1.3e-3.
class ArcsineScopeTest : public CkksFixture {
 protected:
    static void probe(const char* tag) {
        for (double A : {1.0, 100.0}) {
            std::mt19937 gen(12345);
            std::uniform_real_distribution<double> dist(-A, A);
            std::vector<double> v(slots());
            for (auto& x : v) x = dist(gen);

            Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
            const int lvl_in = static_cast<int>(level_of(ct));
            fhe().bootstrap(ct);
            const int lvl_out = static_cast<int>(level_of(ct));
            auto out = decrypt_slots(fhe(), ct);
            out.resize(v.size());
            double emax = 0.0;
            for (size_t i = 0; i < v.size(); ++i)
                emax = std::max(emax, std::fabs(out[i] - v[i]));
            std::cout << std::scientific << std::setprecision(3)
                      << "[arc_scope] " << tag << " A=" << std::fixed
                      << std::setprecision(0) << A << " level " << lvl_in
                      << "->" << lvl_out << std::scientific
                      << std::setprecision(3) << " abs_max=" << emax
                      << std::endl;
        }
    }
};

TEST_F(ArcsineScopeTest, ScopedOff) {
    FIDESlib::CKKS::setArcsineOverride(0);
    probe("scoped_off");
    FIDESlib::CKKS::setArcsineOverride(-1);
}

TEST_F(ArcsineScopeTest, ScopedOn) {
    if (const char* r = std::getenv("ARC_SCOPE_ON_OK"); !(r && *r && *r != '0')) {
        GTEST_SKIP() << "needs a reserved context (set ARC_SCOPE_ON_OK=1)";
    }
    FIDESlib::CKKS::setArcsineOverride(1);
    probe("scoped_on");
    FIDESlib::CKKS::setArcsineOverride(-1);
}

TEST_F(ArcsineScopeTest, EnvDefault) {
    probe("env_default");
}

}  // namespace
