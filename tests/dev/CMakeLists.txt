# Dev/experimental test suite. Auto-discovers dev/test_*.cu and reuses the
# add_cachemir_test() helper from the parent tests/CMakeLists.txt. The parent
# tests/ dir (test_helpers.h) is already on the include path; add this dev/ dir
# too, since the family-scoped helper headers now live here.
file(GLOB CACHEMIR_DEV_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu")
foreach(src ${CACHEMIR_DEV_TESTS})
    get_filename_component(name ${src} NAME_WE)
    add_cachemir_test(${name})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# test_rotation_ext_moddown reaches the FIDESlib-native GPU API (Ciphertext::rotate(idx, moddown),
# GenCryptoContextGPU, ...) which is NOT part of the installed OpenFHE-compat headers, so it needs
# the FIDESlib source tree on its include path. Scoped to this one target.
if(TARGET test_rotation_ext_moddown)
    target_include_directories(test_rotation_ext_moddown PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/FIDESlib/src)
endif()

# test_ks_digit_intt calls FIDESlib::CKKS::setKsDigitIntt (the runtime A/B override for the
# redundant per-digit source INTT), declared in the FIDESlib-native Ciphertext.cuh.
if(TARGET test_ks_digit_intt)
    target_include_directories(test_ks_digit_intt PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/FIDESlib/src)
endif()

# The device magnitude probe's check: a plain executable with its own main that exercises the
# library itself and exits non-zero on a parity failure, registered as a ctest test labelled `gpu`
# since it needs a device (`ctest -R gpu_mag_probe_check`). It includes test_helpers.h, which
# includes gtest, so it links GTest::gtest (not gtest_main) for its headers wherever gtest comes
# from (find_package or the root CMakeLists.txt's FetchContent), not only where a system copy
# exists.
#   gpu_mag_probe_check      the capture's device magnitude probe against the host path
function(add_dev_probe name)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE cachemir_lib GTest::gtest)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/third_party/FIDESlib/src
        ${GTEST_INCLUDE_DIRS})
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp -lineinfo -g>)   # host line info for backtraces
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "gpu")
endfunction()
add_dev_probe(gpu_mag_probe_check)
