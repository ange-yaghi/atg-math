#include <gtest/gtest.h>

#include "atg-math/constants.h"
#include "atg-math/library.h"

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

TEST(LibraryTest, InverseCameraTargetTest) {
    const atg_math::vec4_s eye = {10.0f, 10.0f, 10.0f, 1.0f};
    const atg_math::vec4_s up = {0.0f, 0.0f, 1.0f};
    const atg_math::vec4_s target = {0.0f, 0.0f, 0.0f, 1.0f};

    atg_math::mat44_s m, m_inv;
    atg_math::cameraTarget(eye, target, up, &m);
    m_inv = m.orthogonal_inverse();

    const atg_math::mat44_s I = m * m_inv;
    int a = 0;
}

TEST(LibraryTest, FrustumProjectionTest) {
    atg_math::mat44_s m;
    atg_math::frustumPerspective(0.5f, 0.5f, 1.0f, 50.0f, &m);

    const atg_math::vec4_s projected =
            m * atg_math::vec4_s(0.0f, 0.0f, -9.0f, 1.0f);
}

TEST(LibraryTest, FrustumProjectionInverseTest) {
    atg_math::mat44_s m, m_inv;
    atg_math::frustumPerspective(0.5f, 0.5f, 1.0f, 50.0f, &m);
    atg_math::inverseFrustumPerspective(0.5f, 0.5f, 1.0f, 50.0f, &m_inv);

    atg_math::mat44_s I = m * m_inv;
    int a = 0;
}

TEST(LibraryTest, OrthographicProjectionInverseTest) {
    atg_math::mat44_s m, m_inv;
    atg_math::orthographicProjection(10.0f, 5.0f, 1.0f, 50.0f, &m);
    atg_math::inverseOrthographicProjection(10.0f, 5.0f, 1.0f, 50.0f, &m_inv);

    atg_math::mat44_s I = m * m_inv;
    int a = 0;
}

TEST(LibraryTest, MatrixMultiplicationTest) {
    atg_math::mat44_v m;
    atg_math::translationMatrix({1.0f, 0.0f, 0.0f}, &m);

    atg_math::mat44_v I;
    I.set_identity();

    atg_math::mat44_v s = I * m;
    int a = 0;
}

TEST(LibraryTest, CameraTests) {
    atg_math::mat44_s c, m, t;
    atg_math::cameraTarget({0.0f, 0.0f, 10.0f, 1.0f},
                           atg_math::vec4(0.0f).position(),
                           {0.0f, 1.0f, 0.0f, 1.0f}, &c);
    atg_math::frustumPerspective(atg_math::constants_f::pi / 4, 1.0f, 1.0f,
                                 50.0f, &m);
    t = m * c;

    {
        const atg_math::vec4_s vertex =
                atg_math::vec4_s(-1.0f, 1.0f, 9.0f, 1.0f);
        atg_math::vec4_s proj0 = t * vertex;
        proj0 /= proj0.w();

        EXPECT_NEAR(
                float((proj0 - atg_math::vec4{-1.0f, -1.0f, 0.0f, 1.0f}).sum()),
                0.0f, 1E-7f);
    }

    {
        const atg_math::vec4_s vertex =
                atg_math::vec4_s(0.0f, 0.0f, -40.0f, 1.0f);
        atg_math::vec4_s proj0 = t * vertex;
        proj0 /= proj0.w();

        EXPECT_NEAR(
                float((proj0 - atg_math::vec4{0.0f, 0.0f, 1.0f, 1.0f}).sum()),
                0.0f, 1E-7f);
    }
}

TEST(LibraryTest, OrthographicCameraTests) {
    atg_math::mat44_s c, m, t;
    atg_math::cameraTarget({0.0f, 0.0f, 10.0f, 1.0f},
                           atg_math::vec4(0.0f).position(),
                           {0.0f, 1.0f, 0.0f, 1.0f}, &c);
    atg_math::orthographicProjection(10.0f, 5.0f, 1.0f, 50.0f, &m);
    t = m * c;

    {
        const atg_math::vec4_s vertex =
                atg_math::vec4_s(-5.0f, 2.5f, 9.0f, 1.0f);
        atg_math::vec4_s proj0 = t * vertex;
        proj0 /= proj0.w();

        EXPECT_NEAR(
                float((proj0 - atg_math::vec4{-1.0f, -1.0f, 0.0f, 1.0f}).sum()),
                0.0f, 1E-7f);
    }

    {
        const atg_math::vec4_s vertex =
                atg_math::vec4_s(0.0f, 0.0f, -40.0f, 1.0f);
        atg_math::vec4_s proj0 = t * vertex;
        proj0 /= proj0.w();

        EXPECT_NEAR(
                float((proj0 - atg_math::vec4{0.0f, 0.0f, 1.0f, 1.0f}).sum()),
                0.0f, 1E-7f);
    }
}

TEST(LibraryTest, InverseCameraTests) {
    atg_math::mat44_s c, m, t;
    atg_math::cameraTarget({0.0f, 0.0f, 10.0f, 1.0f},
                           atg_math::vec4(0.0f).position(),
                           {0.0f, 1.0f, 0.0f, 1.0f}, &c);
    atg_math::frustumPerspective(1.0f, 1.0f, 1.0f, 10.0f, &m);
    t = m * c;

    {
        const atg_math::vec4_s vertex =
                atg_math::vec4_s(0.0f, 0.0f, 9.0f, 1.0f);
        atg_math::vec4_s proj0 = t * vertex;
        proj0 /= proj0.w();

        EXPECT_NEAR(
                float((proj0 - atg_math::vec4{0.0f, 0.0f, 0.0f, 1.0f}).sum()),
                0.0f, 1E-7f);
    }

    {
        const atg_math::vec4_s vertex =
                atg_math::vec4_s(0.0f, 0.0f, 0.0f, 1.0f);
        atg_math::vec4_s proj0 = t * vertex;
        proj0 /= proj0.w();

        EXPECT_NEAR(
                float((proj0 - atg_math::vec4{0.0f, 0.0f, 1.0f, 1.0f}).sum()),
                0.0f, 1E-7f);
    }
}
