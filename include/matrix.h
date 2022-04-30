#ifndef ATG_MATH_MATRIX_H
#define ATG_MATH_MATRIX_H

#include "vector.h"

namespace atg_math {
    template<typename t_scalar, unsigned int t_size, bool t_enable_simd>
    struct matrix { /* void */ };

    template<typename t_scalar, unsigned int t_size>
    struct matrix<t_scalar, t_size, false> {
        typedef vec<t_scalar, t_size, false> t_vec;

        t_vec columns[t_size];
        
        t_vec operator*(const t_vec &v) {
            t_vec result;
            for (unsigned int i = 0; i < t_size; ++i) {
                result += v * columns[i];
            }

            return result;
        }
    };
} /* namespace atg_math */

#endif /* ATG_MATH_MATRIX_H */
