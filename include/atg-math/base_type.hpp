#ifndef ATG_ATG_MATH_BASE_TYPE_HPP
#define ATG_ATG_MATH_BASE_TYPE_HPP

#include "definitions.h"

namespace atg_math {

template<typename T_Type>
struct base_type {
    using type = typename T_Type::t_scalar;
};

template<typename T_Type>
constexpr unsigned int type_width() {
    return T_Type::t_size;
}

template<>
struct base_type<double> {
    using type = double;
};

template<>
struct base_type<float> {
    using type = float;
};

template<>
struct base_type<int> {
    using type = int;
};

template<>
FORCE_INLINE constexpr unsigned int type_width<float>() {
    return 1;
}

template<>
FORCE_INLINE constexpr unsigned int type_width<double>() {
    return 1;
}

template<>
FORCE_INLINE constexpr unsigned int type_width<int>() {
    return 1;
}

}// namespace atg_math

#endif /* ATG_ATG_MATH_BASE_TYPE_HPP */
