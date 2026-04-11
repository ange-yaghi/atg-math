#include <gtest/gtest.h>

#include "../include/atg-math/constants.h"
#include "../include/atg-math/library.h"

#include <chrono>
#include <fstream>

TEST(MatrixTests, OrthogonalInverse) {
    atg_math::mat44_s m;
    atg_math::vec4_s axis = {0.0f, 0.0f, 1.0f};
    atg_math::rotationMatrix(axis, 3.14159f * 0.5f, &m);
    m = atg_math::translationMatrix<atg_math::mat44_s>({1.0f, 2.0f, 3.0f}) * m;

    atg_math::mat44_s m_inv = m.orthogonal_inverse();
    atg_math::mat44_s I = m * m_inv;

    atg_math::vec4_s p = {13, 17, 37, 1};

    const auto p_T_T = m * p;
    const auto p_T_ref = m_inv * p_T_T;
    const auto p_T_test = m.orthogonal_inverse_mul(p_T_T);

    EXPECT_NEAR(p_T_ref.x(), p.x(), 1E-4);
    EXPECT_NEAR(p_T_ref.y(), p.y(), 1E-4);
    EXPECT_NEAR(p_T_ref.z(), p.z(), 1E-4);
    EXPECT_NEAR(p_T_ref.w(), p.w(), 1E-4);

    EXPECT_NEAR(p_T_test.x(), p.x(), 1E-4);
    EXPECT_NEAR(p_T_test.y(), p.y(), 1E-4);
    EXPECT_NEAR(p_T_test.z(), p.z(), 1E-4);
    EXPECT_NEAR(p_T_test.w(), p.w(), 1E-4);
}
