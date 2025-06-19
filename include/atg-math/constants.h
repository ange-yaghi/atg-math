#ifndef ATG_MATH_CONSTANTS_H
#define ATG_MATH_CONSTANTS_H

#include "base_type.hpp"

#define ATG_MATH_PI_DIGITS                                                     \
    3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679L
#define ATG_MATH_E_DIGITS                                                      \
    2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274L

#define ATG_MATH_CONSTANT(name, value)                                         \
    static constexpr T_Scalar name = T_Scalar((value))

namespace atg_math {

template<typename T_Scalar_>
class Constants {
    using T_Scalar = base_type<T_Scalar_>::type;

private:
    Constants() = delete;
    Constants(const Constants &) = delete;

public:
    ATG_MATH_CONSTANT(pi, ATG_MATH_PI_DIGITS);
    ATG_MATH_CONSTANT(pi_inv, 1.0L / ATG_MATH_PI_DIGITS);
    ATG_MATH_CONSTANT(tau, ATG_MATH_PI_DIGITS * 2.0L);
    ATG_MATH_CONSTANT(e, ATG_MATH_E_DIGITS);
};

using constants_f = Constants<float>;
using constants_d = Constants<double>;

}// namespace atg_math

#undef ATG_MATH_PI_DIGITS
#undef ATG_MATH_E_DIGITS
#undef ATG_MATH_CONSTANT

#endif /* ATG_MATH_CONSTANTS_H */
