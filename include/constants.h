#ifndef ATG_MATH_CONSTANTS_H
#define ATG_MATH_CONSTANTS_H

#define ATG_MATH_PI_DIGITS                                                     \
    3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
#define ATG_MATH_E_DIGITS                                                      \
    2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274

namespace atg_math {
namespace constants {

constexpr float pi_f = float(ATG_MATH_PI_DIGITS);
constexpr double pi_d = double(ATG_MATH_PI_DIGITS);
constexpr double pi = pi_d;

constexpr float e_f = float(ATG_MATH_E_DIGITS);
constexpr double e_d = double(ATG_MATH_E_DIGITS);
constexpr double e = e_d;

}// namespace constants
}// namespace atg_math

#endif /* ATG_MATH_CONSTANTS_H */
