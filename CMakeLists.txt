cmake_minimum_required(VERSION 3.25)

# ── GPU architecture (must precede project(): CUDA enablement defines the
#    variable itself — pre-70 default on no-GPU nodes breaks FIDESlib) ───────
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80-real)   # A100 default
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, defaulting to 80-real (A100)")
endif()

project(cuda_cachemir LANGUAGES CXX)

# ── client-only build (perseus._client: OpenFHE only, no CUDA toolchain, no FIDESlib) ──
# A GPU-less client machine configures with -DCACHEMIR_CLIENT_ONLY=ON: CUDA is never
# enabled, the runtime library / executable / _core / tests are skipped, and only the
# `_client` pybind11 module is built (scripts/local_build_client.sh). The default
# configuration is unchanged: CUDA enabled, everything below built as before.
option(CACHEMIR_CLIENT_ONLY "Build only perseus._client (OpenFHE, no CUDA toolchain, no FIDESlib)" OFF)
option(CACHEMIR_BUILD_CLIENT "Build the perseus._client pybind11 module" ${CACHEMIR_CLIENT_ONLY})
if(NOT CACHEMIR_CLIENT_ONLY)
    enable_language(CUDA)
endif()

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ── FIDESlib ───────────────────────────────────────────────────────────────
# FIDESlib wraps OpenFHE 1.4.2 with GPU acceleration + bootstrapping.
# If pre-installed: cmake -DFIDESLIB_ROOT=<prefix> ..
# If not found, fall back to FetchContent (requires internet).
if(DEFINED FIDESLIB_ROOT AND NOT DEFINED fideslib_ROOT)
    set(fideslib_ROOT "${FIDESLIB_ROOT}")
endif()
if(NOT CACHEMIR_CLIENT_ONLY)
    find_package(fideslib QUIET CONFIG
        HINTS ${FIDESLIB_ROOT} /usr/local /opt/fideslib)
endif()

# OpenFHE include dirs (openfhe.h lives under include/openfhe/pke/)
find_package(OpenFHE QUIET CONFIG
    HINTS ${FIDESLIB_ROOT}/lib/OpenFHE ${FIDESLIB_ROOT}/lib/cmake/OpenFHE)
if(OpenFHE_FOUND)
    # OpenFHEConfig.cmake sets OpenFHE_INCLUDE = .../include/openfhe
    set(_openfhe_incdirs ${OpenFHE_INCLUDE}
                         ${OpenFHE_INCLUDE}/pke
                         ${OpenFHE_INCLUDE}/core
                         ${OpenFHE_INCLUDE}/binfhe)
elseif(CACHEMIR_CLIENT_ONLY)
    message(FATAL_ERROR "perseus._client needs the patched OpenFHE of a deps tree: pass "
                        "-DFIDESLIB_ROOT=<deps_n32|deps_n64> (its lib/OpenFHE/OpenFHEConfig.cmake)")
else()
    # Fallback: derive from FIDESLIB_ROOT
    set(_openfhe_incdirs
        ${FIDESLIB_ROOT}/include/openfhe
        ${FIDESLIB_ROOT}/include/openfhe/pke
        ${FIDESLIB_ROOT}/include/openfhe/core
        ${FIDESLIB_ROOT}/include/openfhe/binfhe)
endif()

if(NOT CACHEMIR_CLIENT_ONLY AND NOT fideslib_FOUND)
    message(STATUS "fideslib not found via find_package – fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(
        fideslib
        GIT_REPOSITORY https://github.com/gladia-research-group/FIDESlib32bits.git
        # the submodule's pin: a moving reference here would link a different library than
        # every measured row used, and plans are bound to the binary
        GIT_TAG        3f938c388abd2ade7c73e748373193bafa914c31
    )
    set(FIDESLIB_INSTALL_OPENFHE ON  CACHE BOOL "" FORCE)
    set(FIDESLIB_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
    set(FIDESLIB_BUILD_BENCH     OFF CACHE BOOL "" FORCE)
    set(FIDESLIB_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fideslib)
endif()

find_package(OpenMP REQUIRED)
find_package(LibArchive QUIET)

# ── compile options ────────────────────────────────────────────────────────
set(COMMON_CUDA_FLAGS
    --use_fast_math
    -Xcompiler=-fopenmp
    -lineinfo
)
set(COMMON_CXX_FLAGS -O3 -fopenmp)

if(NOT CACHEMIR_CLIENT_ONLY)
# ── main library (object lib shared by exe + tests) ───────────────────────
add_library(cachemir_lib STATIC
    src/primitives/primitives.cu
    src/primitives/polynomial.cu
    src/slot_layout.cu
    src/algorithms/linear/linear.cu
    src/algorithms/linear/cachemir/cachemir_linear.cu
    src/algorithms/linear/diagonal/diagonal_linear.cu
    src/algorithms/nonlinear/nonlinear.cu
    src/algorithms/nonlinear/token_pair_nonlinear.cu
    src/algorithms/nonlinear/cutmax.cu
    src/algorithms/norm/norm.cu
    src/algorithms/norm/sparse_norm.cu
    src/algorithms/norm/token_pair_norm.cu
    src/algorithms/attention/attention.cu
    src/algorithms/attention/cachemir/cachemir_attention.cu
    src/algorithms/attention/cachemir/cachemir_complex_attention.cu
    src/algorithms/attention/cachemir_filling/cachemir_filling_attention.cu
    src/algorithms/attention/cachemir_filling/cachemir_filling_complex_attention.cu
    src/algorithms/attention/cachemir_filling/cachemir_filling_delta_attention.cu
    src/algorithms/attention/cachemir_filling/cachemir_filling_delta_complex_attention.cu
    src/algorithms/attention/bidirectional/delta_bi_attention.cu
    src/algorithms/attention/bidirectional/complex_bi_attention.cu
    src/packing/cachemir/cachemir_kv_cache.cu
    src/packing/cachemir/cachemir_complex_kv_cache.cu
    src/packing/cachemir/cachemir_norm_utils.cu
    src/packing/cachemir/cachemir_attention_utils.cu
    src/packing/cachemir/cachemir_linear_utils.cu
    src/packing/cachemir/cachemir_rot_indices.cu
    src/packing/diagonal/diagonal_linear_utils.cu
    src/packing/diagonal/diagonal_norm_utils.cu
    src/packing/diagonal/diagonal_rot_indices.cu
    src/packing/cachemir_filling/cachemir_filling_kv_cache.cu
    src/packing/cachemir_filling/cachemir_filling_complex_io.cu
    src/packing/cachemir_filling/cachemir_filling_complex_kv_cache.cu
    src/packing/cachemir_filling/cachemir_filling_rot_indices.cu
    src/encoded_block.cu
    src/block_artifact.cu
    src/residency_pipeline.cu
    src/model/block_residency.cu
    src/model/gpt2/gpt2_model.cu
    src/model/gpt2/gpt2_block.cu
    src/model/gpt2/gpt2_residency.cu
    src/model/gpt2/gpt2_kv_handoff.cu
    src/model/gpt2/gpt2_prefill.cu
    src/model/gpt2/gpt2_decode.cu
    src/model/gpt2/gpt2_head.cu
    src/model/gpt2/gpt2_embedding.cu
    src/model/gpt2/gpt2_io.cu
    src/model/gpt2/gpt2_model_api.cu
    src/model/vit/vit_model.cu
    src/model/bert/bert_model.cu
    src/model/mha.cu
    src/model/mlp.cu
    src/model/layer_norm.cu
    src/app/pipeline.cu
)
target_include_directories(cachemir_lib PUBLIC include ${_openfhe_incdirs})
target_link_libraries(cachemir_lib PUBLIC fideslib::fideslib OpenMP::OpenMP_CXX)
if(LibArchive_FOUND)
    target_link_libraries(cachemir_lib PUBLIC LibArchive::LibArchive)
    # LibArchive::LibArchive marks its include dir SYSTEM; nvcc drops imported-target
    # SYSTEM includes for the CUDA language, so .cu TUs that pull in weight_io.h get the
    # WEIGHT_LOADER_WITH_LIBARCHIVE define but no -I → "archive.h: No such file". Add the
    # include dir as a plain -I too (variable is always set when found).
    # NB: this CMake's FindLibArchive sets only the singular cache var
    # LibArchive_INCLUDE_DIR (plural LibArchive_INCLUDE_DIRS is empty here).
    target_include_directories(cachemir_lib PUBLIC
        ${LibArchive_INCLUDE_DIR} ${LibArchive_INCLUDE_DIRS})
    target_compile_definitions(cachemir_lib PUBLIC WEIGHT_LOADER_WITH_LIBARCHIVE)
else()
    message(FATAL_ERROR
        "LibArchive not found: the exported weights (weights.bin.zip) cannot be read.\n"
        "  Install it (libarchive-dev / libarchive-devel), or pass\n"
        "  -DLibArchive_INCLUDE_DIR=<dir with archive.h> -DLibArchive_LIBRARY=<libarchive.so>.")
endif()
target_compile_options(cachemir_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_FLAGS}>
)
set_target_properties(cachemir_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ── main executable ────────────────────────────────────────────────────────
add_executable(cuda_cachemir src/main.cu src/app/cli.cu)
target_link_libraries(cuda_cachemir PRIVATE cachemir_lib)
target_compile_options(cuda_cachemir PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
)
set_target_properties(cuda_cachemir PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ── python bindings (perseus._core) ──────────────────────────────────────
option(CACHEMIR_BUILD_PYTHON "Build the perseus._core pybind11 module" OFF)
if(CACHEMIR_BUILD_PYTHON)
    set_property(TARGET cachemir_lib PROPERTY POSITION_INDEPENDENT_CODE ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core
        src/bindings/module.cu
        src/bindings/ops.cu
        src/bindings/io.cu
        src/bindings/io_gpt2.cu
        src/bindings/io_encoders.cu
        src/bindings/pipeline.cu
        src/bindings/serial.cu
        src/bindings/session.cu
    )
    target_link_libraries(_core PRIVATE cachemir_lib)
    target_compile_options(_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
    )
    set_target_properties(_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/perseus
    )
endif()
endif()  # NOT CACHEMIR_CLIENT_ONLY

# ── python client bindings (perseus._client): OpenFHE only, no CUDA ──────
# Four plain C++ TUs against the deps tree's patched OpenFHE static archives. Hidden
# visibility (pybind11_add_module's default) plus --exclude-libs keeps the archive's
# symbols — cereal's polymorphic registries in particular — private to this module, so it
# co-loads with _core in one interpreter (measured: without it the second module to load
# breaks on the other's registry).
if(CACHEMIR_BUILD_CLIENT)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_client
        src/bindings/client.cpp
        src/client/client_context.cpp
        src/client/client_bootstrap_keys.cpp
        src/client/client_packing.cpp
    )
    target_include_directories(_client PRIVATE include src/bindings ${_openfhe_incdirs})
    target_compile_options(_client PRIVATE ${COMMON_CXX_FLAGS} -Wno-parentheses)
    target_link_libraries(_client PRIVATE
        OPENFHEpke_static OPENFHEbinfhe_static OPENFHEcore_static OpenMP::OpenMP_CXX)
    target_link_options(_client PRIVATE -Wl,--exclude-libs,ALL)
    set_target_properties(_client PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/perseus
    )
endif()

# ── tests ──────────────────────────────────────────────────────────────────
enable_testing()
option(CACHEMIR_BUILD_TESTS "Build test suite" ON)
if(CACHEMIR_BUILD_TESTS AND NOT CACHEMIR_CLIENT_ONLY)
    # Use GTest if available, otherwise download it
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    add_subdirectory(tests)
endif()
