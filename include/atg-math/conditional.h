#ifndef ATG_ATG_MATH_CONDITIONAL_H
#define ATG_ATG_MATH_CONDITIONAL_H

namespace atg_math {

template<typename T_Real>
struct conditional_type {
    using type = T_Real;
};

template<>
struct conditional_type<double> {
    using type = bool;
};

template<>
struct conditional_type<float> {
    using type = bool;
};

template<>
struct conditional_type<int> {
    using type = bool;
};

inline bool h_and(bool input) { return input; }
inline bool h_or(bool input) { return input; }

}// namespace atg_math

#endif /* ATG_ATG_MATH_CONDITIONAL_H */
