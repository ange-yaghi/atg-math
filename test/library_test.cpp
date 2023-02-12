#include <gtest/gtest.h>

#include "../include/library.h"

#include <chrono>
#include <fstream>

TEST(LibraryTest, RotationMatrixTest) {
    atg_math::mat44_s m;
    atg_math::vec4_s axis = {0.0f, 0.0f, 1.0f};
    atg_math::rotationMatrix(axis, 3.14159f * 0.5f, &m);

    const atg_math::vec4_s in = {10.0f, 10.0f, 0.0f};
    const atg_math::vec4_s out = m * in;
    const atg_math::vec4_s expected = {-10.0f, 10.0f, 0.0};

    EXPECT_NEAR(out.x(), expected.x(), 1E-4);
    EXPECT_NEAR(out.y(), expected.y(), 1E-4);
    EXPECT_NEAR(out.z(), expected.z(), 1E-4);
    EXPECT_NEAR(out.w(), expected.w(), 1E-4);
}

TEST(LibraryTest, CameraTargetTest) {
    const atg_math::vec4_s eye = {10.0f, 0.0f, 0.0f, 1.0f};
    const atg_math::vec4_s up = {0.0f, 0.0f, 1.0f};
    const atg_math::vec4_s target = {0.0f, 0.0f, 0.0f, 1.0f};

    atg_math::mat44_s m;
    atg_math::cameraTarget(eye, target, up, &m);

    const atg_math::vec4_s projected =
            m * atg_math::vec4_s(1.0f, 1.0f, 0.0f, 1.0f);
}

TEST(LibraryTest, FrustumProjectionTest) {
    atg_math::mat44_s m;
    atg_math::frustumPerspective(0.5f, 0.5f, 1.0f, 50.0f, &m);

    const atg_math::vec4_s projected =
            m * atg_math::vec4_s(0.0f, 0.0f, -9.0f, 1.0f);
}

TEST(LibraryTest, MatrixMultiplicationTest) {
    atg_math::mat44_v m;
    atg_math::translationMatrix({1.0f, 0.0f, 0.0f}, &m);

    atg_math::mat44_v I;
    I.set_identity();

    atg_math::mat44_v s = I * m;
    int a = 0;
}
