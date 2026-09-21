cmake_minimum_required(VERSION 3.25)

# ── test helper ──────────────────────────────────────────────────────────
function(add_cachemir_test name)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE cachemir_lib GTest::gtest_main)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
        ${GTEST_INCLUDE_DIRS}
    )
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-fopenmp -lineinfo>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fopenmp>
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

enable_testing()

# Top level = the PUBLISHED, self-contained paper reproduction only: the packing
# table bench + its one helper (test_helpers.h). Everything else lives in dev/.
file(GLOB CACHEMIR_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu")
foreach(src ${CACHEMIR_TESTS})
    get_filename_component(name ${src} NAME_WE)
    add_cachemir_test(${name})
endforeach()

# Dev/experimental test suite (reuses add_cachemir_test above).
add_subdirectory(dev)
