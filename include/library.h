#ifndef ATG_MATH_LIBRARY_H
#define ATG_MATH_LIBRARY_H

#include "matrix.h"

namespace atg_math {

template<typename t_matrix>
void rotationMatrix(const typename t_matrix::t_vec &axis,
                    typename t_matrix::t_scalar angle, t_matrix *target) {
    const typename t_matrix::t_scalar cos_theta = std::cos(angle);
    const typename t_matrix::t_scalar sin_theta = std::sin(angle);
    const typename t_matrix::t_vec t{cos_theta, sin_theta};

    const typename t_matrix::t_vec a0{t_matrix::t_vec(1 - cos_theta)};

    const typename t_matrix::t_vec b0{1, axis.z(), -axis.y()};
    const typename t_matrix::t_vec b1{-axis.z(), 1, axis.x()};
    const typename t_matrix::t_vec b2{axis.y(), -axis.x(), 1};

    const typename t_matrix::t_vec c0 = t.shuffle<0, 1, 1>();
    const typename t_matrix::t_vec c1 = t.shuffle<1, 0, 1>();
    const typename t_matrix::t_vec c2 = t.shuffle<1, 1, 0>();

    const typename t_matrix::t_vec axis_xxx = axis.shuffle<0, 0, 0>();
    const typename t_matrix::t_vec axis_yyy = axis.shuffle<1, 1, 1>();
    const typename t_matrix::t_vec axis_zzz = axis.shuffle<2, 2, 2>();

    target->set_identity();
    target->columns[0] = axis * axis_xxx * a0 + b0 * c0;
    target->columns[1] = axis * axis_yyy * a0 + b1 * c1;
    target->columns[2] = axis * axis_zzz * a0 + b2 * c2;
}

template<typename t_matrix>
void rotationMatrixReference(const typename t_matrix::t_vec &axis,
                             typename t_matrix::t_scalar angle,
                             t_matrix *target) {
    const typename t_matrix::t_scalar cos_theta = std::cos(angle);
    const typename t_matrix::t_scalar sin_theta = std::sin(angle);

    target->set_identity();
    target->columns[0] = {
            cos_theta + axis.x() * axis.x() * (1 - cos_theta),
            axis.y() * axis.x() * (1 - cos_theta) + axis.z() * sin_theta,
            axis.z() * axis.x() * (1 - cos_theta) - axis.y() * sin_theta};
    target->columns[1] = {
            axis.x() * axis.y() * (1 - cos_theta) - axis.z() * sin_theta,
            cos_theta + axis.y() * axis.y() * (1 - cos_theta),
            axis.z() * axis.y() * (1 - cos_theta) + axis.x() * sin_theta};
    target->columns[2] = {
            axis.x() * axis.z() * (1 - cos_theta) + axis.y() * sin_theta,
            axis.y() * axis.z() * (1 - cos_theta) - axis.x() * sin_theta,
            cos_theta + axis.z() * axis.z() * (1 - cos_theta)};
}

template<typename t_matrix>
void translationMatrix(const typename t_matrix::t_vec &translation,
                       t_matrix *target) {
    target->set_identity();
    target->columns[3] = translation;
    target->columns[3].data[3] = 1;
}

template<typename t_matrix>
void scaleMatrix(const typename t_matrix::t_vec &scale, t_matrix *target) {
    target->set_identity();
    target->columns[0].data[0] = scale.x();
    target->columns[1].data[1] = scale.y();
    target->columns[2].data[2] = scale.z();
}

template<typename t_matrix>
void frustumPerspective(typename t_matrix::t_scalar fov_y,
                        typename t_matrix::t_scalar aspect,
                        typename t_matrix::t_scalar near,
                        typename t_matrix::t_scalar far,
                        t_matrix *target) {
    using t_scalar = typename t_matrix::t_scalar;

    const t_scalar sin_fov = std::sin(fov_y / 2);
    const t_scalar cos_fov = std::cos(fov_y / 2);

    const t_scalar height = cos_fov / sin_fov;
    const t_scalar width = height / aspect;

    target->columns[0] = {width, 0, 0, 0};
    target->columns[1] = {0, height, 0, 0};
    target->columns[2] = {0, 0, far / (far - near), -1};
    target->columns[3] = {0, 0, -(far * near) / (far - near), 0};
}

template<typename t_matrix>
void orthographicProjection(typename t_matrix::t_scalar width,
                            typename t_matrix::t_scalar height,
                            typename t_matrix::t_scalar near,
                            typename t_matrix::t_scalar far,
                            t_matrix *transform) {
    using t_scalar = typename t_matrix::t_scalar;

    const t_scalar f_range = 1 / (far - near);
    transform->columns[0] = {2 / width, 0, 0, 0};
    transform->columns[1] = {0, 2 / height, 0, 0};
    transform->columns[2] = {0, 0, 2 * f_range, -f_range * near};
    transform->columns[3] = {0, 0, 0, 1};
}

template<typename t_matrix>
void cameraTarget(typename t_matrix::t_vec eye,
                  typename t_matrix::t_vec target,
                  typename t_matrix::t_vec up, t_matrix *transform) {
    using t_scalar = typename t_matrix::t_scalar;
    using t_vec = typename t_matrix::t_vec;

    const t_vec c2 = (eye - target).normalize();
    const t_vec c0 = c2.cross(up).normalize();
    const t_vec c1 = c0.cross(c2);
    const t_vec n_eye = -eye;

    const t_scalar d0 = (t_scalar) c0.dot(n_eye);
    const t_scalar d1 = (t_scalar) c1.dot(n_eye);
    const t_scalar d2 = (t_scalar) c2.dot(n_eye);

    transform->set_identity();
    transform->columns[0] = c0;
    transform->columns[1] = c1;
    transform->columns[2] = c2;

    transform->columns[0].w() = d0;
    transform->columns[1].w() = d1;
    transform->columns[2].w() = d2;
    transform->set_transpose();
}

}// namespace atg_math

#endif /* ATG_MATH_LIBRARY_H */
