#ifndef ATG_ATG_MATH_GEOMETRY_HPP
#define ATG_ATG_MATH_GEOMETRY_HPP

#include "constants.h"
#include "functions.h"

namespace atg_math {

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar circleArea(t_scalar r) {
    return Constants<t_scalar>::pi * squared(r);
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar circleCircumference(t_scalar r) {
    return Constants<t_scalar>::tau * r;
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar circleRadius(t_scalar area) {
    return sqrt(area * Constants<t_scalar>::pi_inv);
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar
circleCircumferenceToRadius(t_scalar circumference) {
    return circumference * Constants<t_scalar>::tau_inv;
}

}// namespace atg_math

#endif /* ATG_ATG_MATH_GEOMETRY_HPP */
