#include <gtest/gtest.h>

#include "../include/atg-math/vector.h"

#include <chrono>
#include <fstream>
TEST(VectorTests, SwizzleTest) {
    using vec_simd = atg_math::vec<float, 4, true>;
    using vec_s = atg_math::vec<float, 4, false>;

    vec_simd v_simd = {1.0f, 2.0f, 3.0f, 4.0f};
    vec_s v = v_simd;

    EXPECT_EQ(v_simd.xy(), (vec_simd{1.0f, 2.0f, 0.0f, 0.0f}));
    EXPECT_EQ(v_simd.yz(), (vec_simd{2.0f, 3.0f, 0.0f, 0.0f}));
    EXPECT_EQ(v_simd.xz(), (vec_simd{1.0f, 3.0f, 0.0f, 0.0f}));
    EXPECT_EQ(v_simd.xyz(), (vec_simd{1.0f, 2.0f, 3.0f, 0.0f}));

    EXPECT_EQ(v.xy(), (vec_s{1.0f, 2.0f, 0.0f, 0.0f}));
    EXPECT_EQ(v.yz(), (vec_s{2.0f, 3.0f, 0.0f, 0.0f}));
    EXPECT_EQ(v.xz(), (vec_s{1.0f, 3.0f, 0.0f, 0.0f}));
    EXPECT_EQ(v.xyz(), (vec_s{1.0f, 2.0f, 3.0f, 0.0f}));

    EXPECT_EQ(v_simd.position(), (vec_simd{1.0f, 2.0f, 3.0f, 1.0f}));
    EXPECT_EQ(v.position(), (vec_s{1.0f, 2.0f, 3.0f, 1.0f}));
}

template<typename t_vec>
void multiplicationTest() {
    const t_vec v = {t_vec::t_scalar(1), t_vec::t_scalar(2)};

    EXPECT_EQ(v * t_vec::t_scalar(2),
              (t_vec{t_vec::t_scalar(2), t_vec::t_scalar(4)}));
    EXPECT_EQ(t_vec::t_scalar(2) * v,
              (t_vec{t_vec::t_scalar(2), t_vec::t_scalar(4)}));
}

TEST(Vector2Tests, MultiplicationTest) {
    multiplicationTest<atg_math::vec<float, 2, false>>();
    multiplicationTest<atg_math::vec<double, 2, true>>();
    multiplicationTest<atg_math::vec<int, 2, false>>();
}

template<typename t_vec>
void dotProductTest() {
    const t_vec v0 = {t_vec::t_scalar(1), t_vec::t_scalar(2)};
    const t_vec v1 = {t_vec::t_scalar(3), t_vec::t_scalar(4)};

    EXPECT_EQ(t_vec::t_scalar(v0.dot(v1)), t_vec::t_scalar(11));
}

TEST(Vector2Tests, DotProductTest) {
    dotProductTest<atg_math::vec<float, 2, false>>();
    dotProductTest<atg_math::vec<double, 2, true>>();
    dotProductTest<atg_math::vec<int, 2, false>>();
}

template<typename t_vec>
void negateTest() {
    const t_vec v0 = {t_vec::t_scalar(1), t_vec::t_scalar(2)};
    EXPECT_EQ(-v0, (t_vec{-t_vec::t_scalar(1), -t_vec::t_scalar(2)}));
}

TEST(Vector2Tests, NegateTest) {
    negateTest<atg_math::vec<float, 2, false>>();
    negateTest<atg_math::vec<double, 2, true>>();
    negateTest<atg_math::vec<int, 2, false>>();
}

template<typename t_vec>
void absTest() {
    const t_vec v0 = {-t_vec::t_scalar(1), -t_vec::t_scalar(2)};
    EXPECT_EQ(v0.abs(), (t_vec{t_vec::t_scalar(1), t_vec::t_scalar(2)}));
}

TEST(Vector2Tests, AbsTest) {
    absTest<atg_math::vec<float, 2, false>>();
    absTest<atg_math::vec<double, 2, true>>();
    absTest<atg_math::vec<int, 2, false>>();
}

template<typename t_vec>
void infTest() {
    const t_vec v0 = {t_vec::t_scalar(INFINITY), -t_vec::t_scalar(INFINITY)};
    const t_vec v1 = {0, -t_vec::t_scalar(INFINITY)};
    const t_vec v2 = {0, 0};

    EXPECT_TRUE(atg_math::any(atg_math::isinf(v0)));
    EXPECT_TRUE(atg_math::any(atg_math::isinf(v1)));
    EXPECT_FALSE(atg_math::any(atg_math::isinf(v2)));
}

TEST(Vector2Tests, InfTest) {
    infTest<atg_math::vec<float, 2, false>>();
    infTest<atg_math::vec<double, 2, false>>();
    infTest<atg_math::vec<double, 2, true>>();
}

template<typename t_vec>
void nanTest() {
    const t_vec v0 = {1, 1};
    const t_vec v1 = {0, 1};
    const t_vec v2 = {0, 0};

    EXPECT_TRUE(atg_math::any(atg_math::isnan(v2 * (v0 / v1))));
    EXPECT_TRUE(atg_math::any(atg_math::isnan(v2 * (v0 / v2))));
    EXPECT_FALSE(atg_math::any(atg_math::isnan(v2 * (v0 / v0))));
}

TEST(Vector2Tests, NanTest) {
    nanTest<atg_math::vec<float, 2, false>>();
    nanTest<atg_math::vec<double, 2, false>>();
    nanTest<atg_math::vec<double, 2, true>>();
}
