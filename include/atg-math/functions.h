#ifndef ATG_MATH_FUNCTIONS_H
#define ATG_MATH_FUNCTIONS_H

#include "definitions.h"

#include <cmath>

namespace atg_math {

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar min(t_scalar data0, t_scalar data1) {
    return data0 < data1 ? data0 : data1;
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar max(t_scalar data0, t_scalar data1) {
    return data0 > data1 ? data0 : data1;
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar abs(t_scalar data0) {
    return data0 > t_scalar(0) ? data0 : -data0;
}

template<typename t_scalar>
inline constexpr t_scalar clamp(t_scalar x, t_scalar x0 = t_scalar(0.0),
                                t_scalar x1 = t_scalar(1.0)) {
    return (x > x0) ? ((x < x1) ? x : x1) : x0;
}

template<typename t_scalar>
inline t_scalar clearNanInf(t_scalar x) {
    if (std::isinf(x) || std::isnan(x)) { return t_scalar(0); }
    return x;
}

template<typename t_scalar>
inline constexpr t_scalar ramp(t_scalar x, t_scalar x0, t_scalar x1) {
    return clamp((x - x0) / (x1 - x0));
}

template<typename t_scalar>
inline constexpr t_scalar line(t_scalar x, t_scalar x0, t_scalar x1) {
    return (x - x0) / (x1 - x0);
}

template<typename t_scalar>
inline constexpr bool in_range(t_scalar x, t_scalar x0, t_scalar x1) {
    return (x >= x0) && (x <= x1);
}

template<typename t_scalar>
inline constexpr bool near(t_scalar x0, t_scalar x1, t_scalar distance) {
    return std::abs(x0 - x1) <= distance;
}

template<typename t_scalar>
inline constexpr t_scalar lerp(t_scalar s, t_scalar x0, t_scalar x1) {
    return s * x1 + (t_scalar(1) - s) * x0;
}

}// namespace atg_math

#endif /* ATG_MATH_FUNCTIONS_H */
