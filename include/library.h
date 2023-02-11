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

}// namespace atg_math

#endif /* ATG_MATH_LIBRARY_H */
