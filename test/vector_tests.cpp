#include <gtest/gtest.h>

#include "../include/vector.h"

#include <chrono>
#include <fstream>

TEST(VectorTests, SwizzleTest) {
    atg_math::vec<float, 4, true> v_simd = {1.0f, 2.0f, 3.0f, 4.0f};
    atg_math::vec<float, 4, false> v = v_simd;

    EXPECT_EQ(v_simd.xy().x(), 1.0f);
    EXPECT_EQ(v_simd.xy().y(), 2.0f);
    EXPECT_EQ(v_simd.xy().z(), 0.0f);
    EXPECT_EQ(v_simd.xy().w(), 0.0f);

    EXPECT_EQ(v_simd.yz().x(), 2.0f);
    EXPECT_EQ(v_simd.yz().y(), 3.0f);
    EXPECT_EQ(v_simd.yz().z(), 0.0f);
    EXPECT_EQ(v_simd.yz().w(), 0.0f);

    EXPECT_EQ(v_simd.xz().x(), 1.0f);
    EXPECT_EQ(v_simd.xz().y(), 3.0f);
    EXPECT_EQ(v_simd.xz().z(), 0.0f);
    EXPECT_EQ(v_simd.xz().w(), 0.0f);

    EXPECT_EQ(v_simd.xyz().x(), 1.0f);
    EXPECT_EQ(v_simd.xyz().y(), 2.0f);
    EXPECT_EQ(v_simd.xyz().z(), 3.0f);
    EXPECT_EQ(v_simd.xyz().w(), 0.0f);

    EXPECT_EQ(v.xy().x(), 1.0f);
    EXPECT_EQ(v.xy().y(), 2.0f);
    EXPECT_EQ(v.xy().z(), 0.0f);
    EXPECT_EQ(v.xy().w(), 0.0f);

    EXPECT_EQ(v.yz().x(), 2.0f);
    EXPECT_EQ(v.yz().y(), 3.0f);
    EXPECT_EQ(v.yz().z(), 0.0f);
    EXPECT_EQ(v.yz().w(), 0.0f);

    EXPECT_EQ(v.xz().x(), 1.0f);
    EXPECT_EQ(v.xz().y(), 3.0f);
    EXPECT_EQ(v.xz().z(), 0.0f);
    EXPECT_EQ(v.xz().w(), 0.0f);

    EXPECT_EQ(v.xyz().x(), 1.0f);
    EXPECT_EQ(v.xyz().y(), 2.0f);
    EXPECT_EQ(v.xyz().z(), 3.0f);
    EXPECT_EQ(v.xyz().w(), 0.0f);

    EXPECT_EQ(v_simd.position().w(), 1.0f);
    EXPECT_EQ(v.position().w(), 1.0f);
}

TEST(VectorTests, MultiplicationTestSimd) {
    atg_math::vec<float, 4, true> v_simd = {1.0f, 2.0f, 3.0f, 4.0f};
    atg_math::vec<float, 4, true> l_mul = 2.0f * v_simd;
    atg_math::vec<float, 4, true> r_mul = v_simd * 2.0f;

    EXPECT_EQ(l_mul, r_mul);
}
