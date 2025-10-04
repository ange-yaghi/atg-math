cmake_minimum_required(VERSION 3.30)

# =========================================================
# Project setup
include(FetchContent)
enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(FETCHCONTENT_QUIET OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(atg-math
    VERSION 1.0.0
    LANGUAGES
        CXX
)

set(CMAKE_CXX_STANDARD 20)

# =========================================================
# version info generation

include("GenInfo")
gen_info(${PROJECT_NAME}-info)

# =========================================================
# gtest

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set_property(TARGET gmock PROPERTY FOLDER "gtest")
set_property(TARGET gmock_main PROPERTY FOLDER "gtest")
set_property(TARGET gtest PROPERTY FOLDER "gtest")
set_property(TARGET gtest_main PROPERTY FOLDER "gtest")

# =========================================================
# library target

add_library(atg-math INTERFACE
    # Header files
    include/atg-math/aabb.h
    include/atg-math/base_type.hpp
    include/atg-math/complex.hpp
    include/atg-math/constants.h
    include/atg-math/definitions.h
    include/atg-math/functions.h
    include/atg-math/library.h
    include/atg-math/matrix.h
    include/atg-math/scalar.h
    include/atg-math/vector.h
    include/atg-math/geometry.hpp
)

target_sources(atg-math INTERFACE $<TARGET_OBJECTS:${PROJECT_NAME}-info>)

target_include_directories(atg-math
    INTERFACE
        include
        $<TARGET_PROPERTY:${PROJECT_NAME}-info,INTERFACE_INCLUDE_DIRECTORIES>
)

# gtest - unit tests
add_executable(atg-math-test
    # Source files
    test/library_test.cpp
    test/sanity_test.cpp
    test/vector2_tests.cpp
    test/vector4_tests.cpp
    test/vector8_tests.cpp
    test/version_test.cpp
)

target_link_libraries(atg-math-test PRIVATE
    gtest_main
    atg-math
)

# gtest - benchmarks
add_executable(atg-math-benchmark
    # Source files
    benchmark/empty_benchmark.cpp
)

target_link_libraries(atg-math-benchmark PRIVATE
    gtest_main
    atg-math
)

include(GoogleTest)
gtest_discover_tests(atg-math-test)

set_property(TARGET atg-math PROPERTY FOLDER "atg-math")
set_property(TARGET atg-math-test PROPERTY FOLDER "atg-math")
set_property(TARGET atg-math-benchmark PROPERTY FOLDER "atg-math")
