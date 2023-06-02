#ifndef ATG_MATH_SCALAR_H
#define ATG_MATH_SCALAR_H

namespace atg_math {

template<typename T_Real>
T_Real clamp(T_Real x, T_Real x_min, T_Real x_max) {
    if (x < x_min) {
        return x_min;
    } else if (x > x_max) {
        return x_max;
    } else {
        return x;
    }
}

}// namespace atg_math

#endif /* ATG_MATH_SCALAR_H */
