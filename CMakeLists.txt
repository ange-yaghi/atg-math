cmake_minimum_required(VERSION 3.10)

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
    URL
    https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
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
    # Source files
    src/information.cpp

    # Header files
    include/atg_math.h
    include/information.h
    include/library.h
    include/matrix.h
    include/scalar.h
    include/vector.h
)

target_include_directories(atg-math INTERFACE
    include/
)

target_compile_definitions(atg-math PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})

# gtest

enable_testing()

add_executable(atg-math-test
    # Source files
    test/library_test.cpp
    test/sanity_test.cpp
    test/vector_tests.cpp

    # Header files
)

target_link_libraries(atg-math-test
    gtest_main
    atg-math
)

include(GoogleTest)
gtest_discover_tests(atg-math-test)
