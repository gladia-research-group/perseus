// LD_PRELOAD interposer to localize a caught std::out_of_range ("map::at" /
// "vector::at"). std::map::at calls std::__throw_out_of_range INSIDE the
// instantiated (header) code in OUR binary, so that symbol IS interposable via
// the PLT (unlike __cxa_throw, which map::at reaches internally to libstdc++).
// We print the throw-site backtrace and abort() so a core is also produced.
//   g++ -shared -fPIC -O0 -g diag_throw_trace.cpp -o throw_trace.so
//   LD_PRELOAD=./throw_trace.so ./test_binary
#include <execinfo.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

static void dump(const char* what, const char* msg) {
    void* bt[128];
    int n = backtrace(bt, 128);
    std::fprintf(stderr, "\n=== %s(\"%s\") backtrace ===\n", what, msg ? msg : "?");
    backtrace_symbols_fd(bt, n, 2);
    std::fflush(stderr);
}

extern "C" {
[[noreturn]] void _ZSt20__throw_out_of_rangePKc(const char* s) {       // std::__throw_out_of_range
    dump("__throw_out_of_range", s);
    abort();
}
[[noreturn]] void _ZSt24__throw_out_of_range_fmtPKcz(const char* fmt, ...) {  // ..._fmt(const char*, ...)
    dump("__throw_out_of_range_fmt", fmt);
    abort();
}
}
