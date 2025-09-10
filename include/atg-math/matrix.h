#ifndef ATG_MATH_MATRIX_H
#define ATG_MATH_MATRIX_H

#include "vector.h"

#include <assert.h>

namespace atg_math {
template<typename t_scalar, unsigned int t_size, bool t_enable_simd>
struct matrix { /* void */
};

#define NON_SIMD_TEMPLATE template<typename t_scalar, unsigned int t_size>
#define NON_SIMD_MATRIX matrix<t_scalar, t_size, false>
#define NON_SIMD_VECTOR vec<t_scalar, t_size, false>

NON_SIMD_TEMPLATE
NON_SIMD_MATRIX &generic_set_identity(NON_SIMD_MATRIX *target) {
    for (unsigned int i = 0; i < t_size; ++i) {
        for (unsigned int j = 0; j < t_size; ++j) {
            target->columns[i].data[j] = t_scalar((i == j) ? 1 : 0);
        }
    }

    return *target;
}

NON_SIMD_TEMPLATE
NON_SIMD_MATRIX &generic_set_transpose(const NON_SIMD_MATRIX &source,
                                       NON_SIMD_MATRIX *target) {
    for (unsigned int i = 0; i < t_size; ++i) {
        for (unsigned int j = i; j < t_size; ++j) {
            const t_scalar temp = source.columns[i].data[j];
            target->columns[i].data[j] = source.columns[j].data[i];
            target->columns[j].data[i] = temp;
        }
    }

    return *target;
}

NON_SIMD_TEMPLATE
NON_SIMD_VECTOR generic_vector_multiply(const NON_SIMD_MATRIX &mat,
                                        const NON_SIMD_VECTOR &v) {
    NON_SIMD_VECTOR result = mat.columns[0] * v.data[0];
    for (unsigned int i = 1; i < t_size; ++i) {
        result += mat.columns[i] * v.data[i];
    }

    return result;
}

NON_SIMD_TEMPLATE
NON_SIMD_MATRIX &generic_matrix_multiply(const NON_SIMD_MATRIX &mat0,
                                         const NON_SIMD_MATRIX &mat1,
                                         NON_SIMD_MATRIX *target) {
    for (unsigned int col = 0; col < t_size; ++col) {
        for (unsigned int row = 0; row < t_size; ++row) {
            t_scalar dot = 0;
            for (unsigned int i = 0; i < t_size; ++i) {
                dot += mat0.columns[i].data[row] * mat1.columns[col].data[i];
            }

            target->columns[col].data[row] = dot;
        }
    }

    return *target;
}

template<typename t_scalar_, unsigned int t_size>
struct matrix<t_scalar_, t_size, false> {
    typedef t_scalar_ t_scalar;
    typedef vec<t_scalar, t_size, false> t_vec;
    typedef matrix<t_scalar, t_size, false> t_matrix;

    t_vec columns[t_size];

    inline t_matrix &set_identity() { return generic_set_identity(this); }
    inline t_matrix &set_transpose(t_matrix *target) const {
        return generic_set_transpose(*this, target);
    }
    inline t_matrix &set_transpose() { return set_transpose(this); }
    inline t_matrix transpose() const {
        t_matrix result;
        return set_transpose(&result);
    }

    inline t_vec operator*(const t_vec &v) const {
        return generic_vector_multiply(*this, v);
    }
    inline t_matrix &mul(const t_matrix &m, t_matrix *target) const {
        assert(&m != target);
        assert(this != target);

        return generic_matrix_multiply(*this, m, target);
    }
    inline t_matrix operator*(const t_matrix &v) const {
        t_matrix result;
        return mul(v, &result);
    }
    inline t_matrix operator*=(const t_matrix &v) {
        return *this = (*this) * v;
    }

    // temp
    // Only works for 4x4 matrices
    inline t_matrix orthogonal_inverse() const {
        const t_matrix r = {
                columns[0],
                columns[1],
                columns[2],
                {t_scalar(0), t_scalar(0), t_scalar(0), t_scalar(1)}};
        const t_matrix r_inv = r.transpose();

        t_matrix t_inv;
        t_inv.set_identity();
        t_inv.columns[3] = (-columns[3]).position();

        return r_inv * t_inv;
    }
    // end-temp
};// namespace atg_math

template<typename t_scalar_>
struct matrix<t_scalar_, 4, true> {
    typedef t_scalar_ t_scalar;
    typedef vec<t_scalar, 4, true> t_vec;
    typedef matrix<t_scalar, 4, true> t_matrix;

    t_vec columns[4];

    inline matrix() {}
    inline matrix(const t_vec &c0, const t_vec &c1, const t_vec &c2,
                  const t_vec &c3) {
        set(c0, c1, c2, c3);
    }

    inline void set(const t_vec &c0, const t_vec &c1, const t_vec &c2,
                    const t_vec &c3) {
        columns[0] = c0;
        columns[1] = c1;
        columns[2] = c2;
        columns[3] = c3;
    }

    inline t_matrix &set_identity() {
        columns[0] = {t_scalar(1), t_scalar(0), t_scalar(0), t_scalar(0)};
        columns[1] = {t_scalar(0), t_scalar(1), t_scalar(0), t_scalar(0)};
        columns[2] = {t_scalar(0), t_scalar(0), t_scalar(1), t_scalar(0)};
        columns[3] = {t_scalar(0), t_scalar(0), t_scalar(0), t_scalar(1)};

        return *this;
    }

    inline t_matrix &set_transpose(t_matrix *target) const {
        const __m128 t0 = _mm_shuffle_ps(columns[0], columns[1], 0x44);
        const __m128 t1 = _mm_shuffle_ps(columns[0], columns[1], 0xEE);
        const __m128 t2 = _mm_shuffle_ps(columns[2], columns[3], 0x44);
        const __m128 t3 = _mm_shuffle_ps(columns[2], columns[3], 0xEE);

        target->columns[0] = _mm_shuffle_ps(t0, t2, 0x88);
        target->columns[1] = _mm_shuffle_ps(t0, t2, 0xDD);
        target->columns[2] = _mm_shuffle_ps(t1, t3, 0x88);
        target->columns[3] = _mm_shuffle_ps(t1, t3, 0xDD);

        return *target;
    }

    inline t_matrix &set_transpose() { return set_transpose(this); }
    inline t_matrix transpose() {
        t_matrix result;
        return set_transpose(&result);
    }

    inline t_vec operator*(const t_vec &v) const {
        const t_vec v_x = _mm_shuffle_ps(
                v.data_v, v.data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_X, ATG_MATH_S_X, ATG_MATH_S_X,
                                      ATG_MATH_S_X));
        const t_vec v_y = _mm_shuffle_ps(
                v.data_v, v.data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_Y, ATG_MATH_S_Y,
                                      ATG_MATH_S_Y));
        const t_vec v_z = _mm_shuffle_ps(
                v.data_v, v.data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Z, ATG_MATH_S_Z, ATG_MATH_S_Z,
                                      ATG_MATH_S_Z));
        const t_vec v_w = _mm_shuffle_ps(
                v.data_v, v.data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_W, ATG_MATH_S_W, ATG_MATH_S_W,
                                      ATG_MATH_S_W));

        return v_x * columns[0] + v_y * columns[1] + v_z * columns[2] +
               v_w * columns[3];
    }

    inline t_matrix &mul(const t_matrix &m, t_matrix *target) const {
        assert(&m != target);
        assert(this != target);

        target->columns[0] = (*this) * m.columns[0];
        target->columns[1] = (*this) * m.columns[1];
        target->columns[2] = (*this) * m.columns[2];
        target->columns[3] = (*this) * m.columns[3];

        return *target;
    }

    inline t_matrix operator*(const t_matrix &v) const {
        t_matrix result;
        return mul(v, &result);
    }

    inline t_matrix operator*=(const t_matrix &v) {
        return *this = (*this) * v;
    }
};

typedef matrix<float, 3, false> mat33_s;
typedef matrix<float, 4, false> mat44_s, mat_s;
typedef matrix<double, 3, false> dmat33_s;
typedef matrix<double, 4, false> dmat44_s, dmat_s;

typedef matrix<float, 4, true> mat44_v;

typedef mat33_s mat3;
typedef mat44_s mat4;
} /* namespace atg_math */

#endif /* ATG_MATH_MATRIX_H */
