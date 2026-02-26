cmake_minimum_required(VERSION 3.30)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(ProjectName atg-math)
set(ProjectNamespace "atg_math")

project(atg-math)

set(CMAKE_CXX_STANDARD 20)

# =========================================================
# gtest

include(FetchContent)
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

# -----------------------------------------------------------------------------
# Compile definitions
# -----------------------------------------------------------------------------
add_library(atg-math STATIC
    # Header files
    include/atg-math/aabb.h
    include/atg-math/base_type.hpp
    include/atg-math/complex.hpp
    include/atg-math/conditional.h
    include/atg-math/constants.h
    include/atg-math/definitions.h
    include/atg-math/functions.h
    include/atg-math/geometry.hpp
    include/atg-math/information.h
    include/atg-math/intersection.hpp
    include/atg-math/library.h
    include/atg-math/matrix.h
    include/atg-math/scalar.h
    include/atg-math/vector.h

    # Source files
    src/information.cpp
    src/intersection.cpp
)

target_include_directories(atg-math INTERFACE
    include/
)

target_compile_definitions(atg-math PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})

# gtest - unit tests
enable_testing()

add_executable(atg-math-test
    # Source files
    test/library_test.cpp
    test/sanity_test.cpp
    test/vector2_tests.cpp
    test/vector4_tests.cpp
    test/vector8_tests.cpp
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
