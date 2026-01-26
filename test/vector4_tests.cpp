#include <gtest/gtest.h>

#include "../include/atg-math/vector.h"

template<typename t_vec>
void scalarMultiplicationTest() {
    const t_vec v = {t_vec::t_scalar(1), t_vec::t_scalar(2), t_vec::t_scalar(3),
                     t_vec::t_scalar(4)};

    EXPECT_EQ(v * t_vec::t_scalar(2),
              (t_vec{t_vec::t_scalar(2), t_vec::t_scalar(4), t_vec::t_scalar(6),
                     t_vec::t_scalar(8)}));
    EXPECT_EQ(t_vec::t_scalar(2) * v,
              (t_vec{t_vec::t_scalar(2), t_vec::t_scalar(4), t_vec::t_scalar(6),
                     t_vec::t_scalar(8)}));
}

TEST(Vector4Tests, ScalarMultiplicationTest) {
    scalarMultiplicationTest<atg_math::vec<float, 4, true>>();
    scalarMultiplicationTest<atg_math::vec<float, 4, false>>();
    scalarMultiplicationTest<atg_math::vec<double, 4, false>>();
    scalarMultiplicationTest<atg_math::vec<double, 4, true>>();
    scalarMultiplicationTest<atg_math::vec<int, 4, false>>();
}

template<typename t_vec>
void absTest() {
    const t_vec v = {-t_vec::t_scalar(1), -t_vec::t_scalar(2),
                     -t_vec::t_scalar(3), -t_vec::t_scalar(4)};
    EXPECT_EQ(v.abs(), (t_vec{t_vec::t_scalar(1), t_vec::t_scalar(2),
                              t_vec::t_scalar(3), t_vec::t_scalar(4)}));
}

TEST(Vector4Tests, AbsoluteTest) {
    absTest<atg_math::vec<float, 4, true>>();
    absTest<atg_math::vec<float, 4, false>>();
    absTest<atg_math::vec<double, 4, false>>();
    absTest<atg_math::vec<double, 4, true>>();
    absTest<atg_math::vec<int, 4, false>>();
}

template<typename t_vec>
void negateTest() {
    const t_vec v = {-t_vec::t_scalar(1), -t_vec::t_scalar(2),
                     -t_vec::t_scalar(3), -t_vec::t_scalar(4)};
    EXPECT_EQ(-v, (t_vec{t_vec::t_scalar(1), t_vec::t_scalar(2),
                         t_vec::t_scalar(3), t_vec::t_scalar(4)}));
}

TEST(Vector4Tests, NegateTest) {
    negateTest<atg_math::vec<float, 4, true>>();
    negateTest<atg_math::vec<float, 4, false>>();
    negateTest<atg_math::vec<double, 4, false>>();
    negateTest<atg_math::vec<double, 4, true>>();
    negateTest<atg_math::vec<int, 4, false>>();
}

TEST(Vector4Tests, LeftShiftTest_vec4f) {
    float data0[] = {0.0f, 1.0f, 2.0f, 3.0f};
    float data1[] = {1.0f, 2.0f, 3.0f, 3.0f};
    atg_math::vec<float, 4, true> v0, v1;
    v0.load(data0);
    v1.load(data1);

    EXPECT_TRUE(bool(v0.l_shift() == v1));
}

TEST(Vector4Tests, LeftShiftTest_vec4d) {
    double data0[] = {0.0, 1.0, 2.0, 3.0};
    double data1[] = {1.0, 2.0, 3.0, 3.0};
    atg_math::vec<double, 4, true> v0, v1;
    v0.load(data0);
    v1.load(data1);

    EXPECT_TRUE(bool(v0.l_shift() == v1));
}

TEST(Vector4Tests, MinMaxComponentTest_vec4f) {
    for (int i = 0; i < 4; ++i) {
        atg_math::vec<float, 4, true> v;
        v = 1.0f;
        v[i] = -1.0f;

        EXPECT_EQ(v.min(), -1.0f);
        EXPECT_EQ(v.max(), 1.0f);

        v = -1.0f;
        v[i] = 1.0f;

        EXPECT_EQ(v.min(), -1.0f);
        EXPECT_EQ(v.max(), 1.0f);
    }
}

TEST(Vector4Tests, Sum_vec4d) {
    atg_math::vec<double, 4, true> s = {1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(double(s.sum()), 10.0);
}

TEST(Vector4Tests, MinMaxComponentTest_vec4d) {
    for (int i = 0; i < 4; ++i) {
        atg_math::vec<double, 4, true> v;
        v = 1.0;
        v[i] = -1.0;

        EXPECT_EQ(v.min(), -1.0);
        EXPECT_EQ(v.max(), 1.0);

        v = -1.0;
        v[i] = 1.0;

        EXPECT_EQ(v.min(), -1.0);
        EXPECT_EQ(v.max(), 1.0);
    }
}

TEST(Vector4Tests, BooleanReduce_vec4d) {
    atg_math::vec<double, 4, true> v0 = 0, v1 = 0, v2 = 1;
    EXPECT_TRUE(bool(v0 == v1));
    EXPECT_FALSE(bool(v0 == v2));
}

TEST(Vector4Tests, BooleanReduce_vec4f) {
    atg_math::vec<float, 4, true> v0 = 0, v1 = 0, v2 = 1;
    EXPECT_TRUE(bool(v0 == v1));
    EXPECT_FALSE(bool(v0 == v2));
}
