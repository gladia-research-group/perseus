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
