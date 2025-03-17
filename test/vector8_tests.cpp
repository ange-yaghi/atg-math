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

    EXPECT_TRUE(v0 == v1);
    EXPECT_FALSE(v0 != v1);

    v0.data[5] = 6.0f;

    EXPECT_FALSE(v0 == v1);
    EXPECT_TRUE(v0 != v1);
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
