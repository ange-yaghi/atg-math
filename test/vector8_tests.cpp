#include "gtest/gtest.h"

#include "../include/atg-math/vector.h"

template<typename t_vec>
void sumTest() {
    t_vec v;
    for (int i = 0; i < 8; ++i) { v.data[i] = t_vec::t_scalar(i); }
    EXPECT_EQ(v.sum(), (t_vec{t_vec::t_scalar(28)}));
}

TEST(Vector8Tests, SumTest) {
    sumTest<atg_math::vec<float, 8, true>>();
    sumTest<atg_math::vec<float, 8, false>>();
    sumTest<atg_math::vec<double, 8, false>>();
    sumTest<atg_math::vec<int, 8, false>>();
}

TEST(Vector8Tests, EqTest) {
    float data0[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float data1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    atg_math::vec<float, 8, true> v0, v1;
    v0.load(data0);
    v1.load(data1);

    EXPECT_TRUE(bool(v0 == v1));

    v0.data[5] = 6.0f;

    EXPECT_FALSE(bool(v0 == v1));
}

TEST(Vector8Tests, MaddTest_1) {
    float data0[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float data1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    atg_math::vec<float, 8, true> v0, v1;
    v0.load(data0);
    v1.load(data1);

    EXPECT_EQ(v0.madd(v0, v1), v0 * v0 + v1);
}

TEST(Vector8Tests, MaddTest_2) {
    float data0[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float data1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    atg_math::vec<float, 8, true> v0, v1;
    v0.load(data0);
    v1.load(data1);

    EXPECT_EQ((v0 + v1).madd(v0, v1), (v0 + v1) * v0 + v1);
}

TEST(Vector8Tests, LeftShiftTest_vec8f) {
    float data0[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 7.0f};
    atg_math::vec<float, 8, true> v0, v1;
    v0.load(data0);
    v1.load(data1);

    EXPECT_TRUE(bool(v0.l_shift() == v1));
}

TEST(Vector8Tests, LeftShiftTest_vec8f_l2) {
    float data[16] = {};
    for (int i = 0; i < 16; ++i) { data[i] = float(i); }
    atg_math::vec<float, 8, true> v0, v1, result;
    v0.load(data + 0);
    v1.load(data + 8);

    result.load(data + 1);
    EXPECT_TRUE(bool(v0.l_shift<1>(v1) == result));

    result.load(data + 2);
    EXPECT_TRUE(bool(v0.l_shift<2>(v1) == result));

    result.load(data + 8 - 1);
    EXPECT_TRUE(bool(v1.r_shift<1>(v0) == result));

    result.load(data + 8 - 2);
    EXPECT_TRUE(bool(v1.r_shift<2>(v0) == result));
}

TEST(Vector8Tests, MinComponentTest_vec8f) {
    for (int i = 0; i < 8; ++i) {
        atg_math::vec<float, 8, true> v;
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

TEST(Vector8Tests, BooleanReduce_vec8f) {
    atg_math::vec<float, 8, true> v0 = 0, v1 = 0, v2 = 0;
    EXPECT_TRUE(bool(v0 == v1));
    EXPECT_TRUE(bool(v0 == v2));
}
