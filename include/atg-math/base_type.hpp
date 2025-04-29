#ifndef ATG_ATG_MATH_BASE_TYPE_HPP
#define ATG_ATG_MATH_BASE_TYPE_HPP

namespace atg_math {

template<typename T_Type>
struct base_type {
    using type = T_Type::t_scalar;
};

}// namespace atg_math

#endif /* ATG_ATG_MATH_BASE_TYPE_HPP */
