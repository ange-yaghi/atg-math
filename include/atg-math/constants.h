#ifndef ATG_MATH_CONSTANTS_H
#define ATG_MATH_CONSTANTS_H

#define ATG_MATH_PI_DIGITS                                                     \
    3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
#define ATG_MATH_E_DIGITS                                                      \
    2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274

#define ATG_MATH_CONSTANTS_CONSTANT(name, value)                               \
    template<typename T_Return = double>                                       \
    inline constexpr T_Return name() {                                         \
        return static_cast<T_Return>((value));                                 \
    }                                                                          \
    constexpr double name##_d = name();                                        \
    constexpr float name##_f = name<float>()

namespace atg_math {
namespace constants {

ATG_MATH_CONSTANTS_CONSTANT(pi, ATG_MATH_PI_DIGITS);
ATG_MATH_CONSTANTS_CONSTANT(tau, ATG_MATH_PI_DIGITS * 2.0L);
ATG_MATH_CONSTANTS_CONSTANT(e, ATG_MATH_E_DIGITS);

}// namespace constants
}// namespace atg_math

#endif /* ATG_MATH_CONSTANTS_H */
