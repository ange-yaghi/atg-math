#include "gtest/gtest.h"

#include "../include/vector.h"

template<typename t_vec>
void sumTest() {
    const t_vec v = {t_vec::t_scalar(1), t_vec::t_scalar(2), t_vec::t_scalar(3),
                     t_vec::t_scalar(4), t_vec::t_scalar(5), t_vec::t_scalar(6),
                     t_vec::t_scalar(7), t_vec::t_scalar(8)};
    EXPECT_EQ(v.sum(), (t_vec{t_vec::t_scalar(36)}));
}

TEST(Vector8Tests, SumTest) {
    sumTest<atg_math::vec<float, 8, true>>();
    sumTest<atg_math::vec<float, 8, false>>();
    sumTest<atg_math::vec<double, 8, false>>();
    sumTest<atg_math::vec<int, 8, false>>();
}
