#include "gtest/gtest.h"

#include "../include/vector.h"

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
