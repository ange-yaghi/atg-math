#ifndef ATG_MATH_FUNCTIONS_H
#define ATG_MATH_FUNCTIONS_H

#include "conditional.h"
#include "definitions.h"

#include <cmath>
#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

namespace atg_math {

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar squared(t_scalar s) {
    return s * s;
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar sign(t_scalar s) {
    return (s > 0) ? t_scalar(1) : t_scalar(-1);
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar cubed(t_scalar s) {
    return s * s * s;
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar pow7(t_scalar s) {
    const t_scalar pow_2 = squared(s);
    const t_scalar pow_4 = pow_2 * pow_2;
    return pow_4 * pow_2 * s;
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar pow10(t_scalar s) {
    const t_scalar pow_2 = squared(s);
    const t_scalar pow_4 = pow_2 * pow_2;
    return pow_4 * pow_4 * pow_2;
}

template<typename t_scalar>
FORCE_INLINE t_scalar pow(t_scalar s, t_scalar p) {
    return std::pow(s, p);
}

template<>
FORCE_INLINE double pow<double>(double s, double p) {
    return _mm_cvtsd_f64(
            _mm_exp_pd(_mm_mul_pd(_mm_set1_pd(p), _mm_log_pd(_mm_set1_pd(s)))));
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar min(t_scalar data0, t_scalar data1) {
    return data0 > data1 ? data1 : data0;
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar max(t_scalar data0, t_scalar data1) {
    return data0 < data1 ? data1 : data0;
}

template<typename t_scalar>
FORCE_INLINE constexpr t_scalar abs(t_scalar data0) {
    return data0 > t_scalar(0) ? data0 : -data0;
}

template<typename t_scalar>
FORCE_INLINE t_scalar sqrt(t_scalar s) {
    return std::sqrt(s);
}

template<typename t_scalar>
FORCE_INLINE t_scalar fmadd(t_scalar a, t_scalar b, t_scalar c_add) {
    return std::fma(a, b, c_add);
}

template<typename t_scalar>
FORCE_INLINE t_scalar
ternary(typename conditional_type<t_scalar>::type condition, t_scalar a,
        t_scalar b) {
    return condition ? a : b;
}

template<typename t_scalar>
FORCE_INLINE t_scalar ternary(t_scalar condition, t_scalar a, t_scalar b) {
    return (condition != 0) ? a : b;
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

template<typename t_scalar>
inline constexpr t_scalar lerp_fast(t_scalar s, t_scalar x0, t_scalar x1) {
    return x0 + s * (x1 - x0);
}

template<typename t_scalar>
inline constexpr t_scalar smoothstep(t_scalar s_, t_scalar x0, t_scalar x1) {
    const t_scalar s = ramp(s_, x0, x1);
    return 3 * s * s - 2 * s * s * s;
}

template<typename t_scalar>
inline constexpr t_scalar smoothstep(t_scalar s_) {
    const t_scalar s = clamp(s_);
    return 3 * s * s - 2 * s * s * s;
}

template<typename t_scalar>
inline constexpr t_scalar smootherstep(t_scalar s_) {
    const t_scalar s = clamp(s_);
    const t_scalar s_2 = squared(s);
    const t_scalar s_4 = squared(s_2);
    return t_scalar(6) * s_4 * s - t_scalar(15) * s_4 + t_scalar(10) * s_2 * s;
}

template<typename t_scalar>
inline constexpr t_scalar smootherstep(t_scalar s_, t_scalar x0, t_scalar x1) {
    const t_scalar s = ramp(s_, x0, x1);
    const t_scalar s_2 = squared(s);
    const t_scalar s_4 = squared(s_2);
    return t_scalar(6) * s_4 * s - t_scalar(15) * s_4 + t_scalar(10) * s_2 * s;
}

template<typename t_scalar>
inline constexpr t_scalar heaviside(t_scalar x, t_scalar x0) {
    return (x >= x0) ? t_scalar(1) : t_scalar(0);
}

template<typename t_scalar>
inline constexpr t_scalar rect(t_scalar x, t_scalar x0, t_scalar x1) {
    return heaviside(x, x0) - heaviside(x, x1);
}

template<typename t_scalar>
inline constexpr t_scalar ramp_rect(t_scalar x, t_scalar x0, t_scalar x1,
                                    t_scalar w) {
    return ramp(x, x0 - w, x0) - ramp(x, x1, x1 + w);
}

template<typename t_scalar>
inline constexpr t_scalar smooth_ramp_rect(t_scalar x, t_scalar x0, t_scalar x1,
                                           t_scalar w) {
    return smoothstep(ramp(x, x0 - w, x0) - ramp(x, x1, x1 + w));
}

template<typename t_scalar>
inline constexpr t_scalar avg(t_scalar a, t_scalar b) {
    return t_scalar(0.5) * (a + b);
}

template<typename t_scalar>
inline constexpr t_scalar inv(t_scalar a) {
    return t_scalar(1) / a;
}

template<typename t_scalar>
inline constexpr t_scalar half(t_scalar a) {
    return t_scalar(0.5) * a;
}

template<typename t_scalar>
inline constexpr t_scalar minComponent(t_scalar a) {
    return a;
}

template<typename t_scalar>
inline constexpr t_scalar maxComponent(t_scalar a) {
    return a;
}

}// namespace atg_math

#endif /* ATG_MATH_FUNCTIONS_H */
