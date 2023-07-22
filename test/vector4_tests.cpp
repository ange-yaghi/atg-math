#include <gtest/gtest.h>

#include "../include/vector.h"

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
    negateTest<atg_math::vec<int, 4, false>>();
}
