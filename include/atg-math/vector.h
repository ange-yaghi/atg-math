#ifndef ATG_MATH_VECTOR_H
#define ATG_MATH_VECTOR_H

#include "base_type.hpp"
#include "definitions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

namespace atg_math {
template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
struct vec {};

#define ATG_MATH_ALIAS(name, index)                                            \
    FORCE_INLINE constexpr t_scalar &name() { return data[index]; }            \
    FORCE_INLINE constexpr t_scalar name() const { return data[index]; }

#define ATG_MATH_DEFINE_T_VEC typedef vec<t_scalar_, t_size, false> t_vec
#define ATG_MATH_DEFINE_T_SCALAR typedef t_scalar_ t_scalar

#define ATG_MATH_VEC_DEFINES(size, simd)                                       \
    static constexpr unsigned int t_size = size;                               \
    static constexpr bool t_simd = simd;                                       \
    ATG_MATH_DEFINE_T_VEC;                                                     \
    ATG_MATH_DEFINE_T_SCALAR;

#define ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(op)                            \
    FORCE_INLINE t_vec operator op(const t_vec &b) const {                     \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = data[i] op b.data[i];                             \
        }                                                                      \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_MADD()                                                 \
    FORCE_INLINE t_vec madd(const t_vec &m, const t_vec &a) const {            \
        return (*this) * m + a;                                                \
    }

#define ATG_MATH_DEFINE_SCALAR_OPERATOR(op)                                    \
    FORCE_INLINE t_vec operator op(t_scalar b) const {                         \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = data[i] op b;                                     \
        }                                                                      \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_LEFT_SCALAR_OPERATOR(op)                               \
    template<typename t_scalar, unsigned int t_size, bool t_simd>              \
    FORCE_INLINE vec<t_scalar, t_size, t_simd> operator*(                      \
            typename vec<t_scalar, t_size, t_simd>::t_scalar left,             \
            const vec<t_scalar, t_size, t_simd> &right) {                      \
        return right op left;                                                  \
    }

#define ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(op)                        \
    FORCE_INLINE bool operator op(const t_vec &b) const {                      \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            if (!(data[i] op b.data[i])) { return false; }                     \
        }                                                                      \
        return true;                                                           \
    }

#define ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(op, name)                   \
    FORCE_INLINE t_vec compare_##name(const t_vec &b) const {                  \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            if (data[i] op b.data[i]) {                                        \
                result.data[i] = t_scalar(1);                                  \
            } else {                                                           \
                result.data[i] = t_scalar(0);                                  \
            }                                                                  \
        }                                                                      \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_SIGN                                                   \
    FORCE_INLINE t_vec sign() const {                                          \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = data[i] < 0 ? t_scalar(-1) : t_scalar(1);         \
        }                                                                      \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_NEGATED_BOOLEAN_COMPARISON_OPERATOR(op)                \
    FORCE_INLINE bool operator op(const t_vec &b) const {                      \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            if (data[i] op b.data[i]) { return true; }                         \
        }                                                                      \
        return false;                                                          \
    }

#define ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(full_op, op)                       \
    FORCE_INLINE t_vec &operator full_op(const t_vec & b) {                    \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            data[i] = data[i] op b.data[i];                                    \
        }                                                                      \
        return *this;                                                          \
    }

#define ATG_MATH_DEFINE_NEGATE_OPERATOR                                        \
    FORCE_INLINE t_vec operator-() const {                                     \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = -data[i];                                         \
        }                                                                      \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_INDEX_OPERATOR()                                       \
    FORCE_INLINE t_scalar operator[](unsigned int i) const { return data[i]; }
#define ATG_MATH_DEFINE_INDEX_OPERATOR_MODIFIABLE()                            \
    FORCE_INLINE t_scalar &operator[](unsigned int i) { return data[i]; }

#define ATG_MATH_DEFINE_POSITIVE_OPERATOR                                      \
    FORCE_INLINE t_vec operator+() const { return *this; }

#define ATG_MATH_DEFINE_DOT_PRODUCT                                            \
    FORCE_INLINE t_vec dot(const t_vec &b) const {                             \
        t_scalar result = 0;                                                   \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result += data[i] * b.data[i];                                     \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_SUM                                                    \
    FORCE_INLINE t_vec sum() const {                                           \
        t_scalar result = 0;                                                   \
        for (unsigned int i = 0; i < t_size; ++i) { result += data[i]; }       \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_COMPONENT_MIN                                          \
    FORCE_INLINE t_vec min(const t_vec &b) const {                             \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = std::min(data[i], b.data[i]);                     \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_COMPONENT_MAX                                          \
    FORCE_INLINE t_vec max(const t_vec &b) const {                             \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = std::max(data[i], b.data[i]);                     \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_ABS                                                    \
    FORCE_INLINE t_vec abs() const {                                           \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = std::abs(data[i]);                                \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_SQRT                                                   \
    FORCE_INLINE t_vec sqrt() const {                                          \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = std::sqrt(data[i]);                               \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

template<typename T_Data>
inline T_Data fp_bitwise_not(T_Data a) {
    union U {
        T_Data data;
        unsigned long long bits;
    };

    U u0, result;
    u0.data = a;
    result.bits = ~u0.bits;
    return result.data;
}

template<typename T_Data>
inline T_Data fp_bitwise_or(T_Data a, T_Data b) {
    union U {
        T_Data data;
        unsigned long long bits;
    };

    U u0, u1, result;
    u0.data = a;
    u1.data = b;
    result.bits = u0.bits | u1.bits;
    return result.data;
}

template<typename T_Data>
inline T_Data fp_bitwise_and(T_Data a, T_Data b) {
    union U {
        T_Data data;
        unsigned long long bits;
    };

    U u0, u1, result;
    u0.data = a;
    u1.data = b;
    result.bits = u0.bits & u1.bits;
    return result.data;
}

template<typename T_Data>
inline T_Data fp_fill_ones() {
    union U {
        T_Data data;
        unsigned long long bits;
    };

    U result;
    result.bits = ~((unsigned long long) (0));
    return result.data;
}

template<typename T_Data>
inline T_Data fp_fill_zeroes() {
    union U {
        T_Data data;
        unsigned long long bits;
    };

    U result;
    result.bits = ((unsigned long long) (0));
    return result.data;
}

#define ATG_MATH_COMPARE_GE_MASK                                               \
    FORCE_INLINE t_vec compare_ge_mask(const t_vec &b) const {                 \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = (data[i] >= b.data[i])                            \
                                     ? fp_fill_ones<t_scalar>()                \
                                     : fp_fill_zeroes<t_scalar>();             \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_AND_MASK                                                      \
    FORCE_INLINE t_vec and_mask(const t_vec &b) const {                        \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = fp_bitwise_and(data[i], b.data[i]);               \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_AND_NOT_MASK                                                  \
    FORCE_INLINE t_vec and_not_mask(const t_vec &b) const {                    \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] =                                                   \
                    fp_bitwise_and(data[i], fp_bitwise_not(b.data[i]));        \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_BITWISE_OR                                                    \
    FORCE_INLINE t_vec bitwise_or(const t_vec &b) const {                      \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = fp_bitwise_or(data[i], b.data[i]);                \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_MAGNITUDE_SQUARED                                      \
    FORCE_INLINE t_vec magnitude_squared() const { return dot(*this); }

#define ATG_MATH_DEFINE_MAGNITUDE                                              \
    FORCE_INLINE t_vec magnitude() const { return magnitude_squared().sqrt(); }

#define ATG_MATH_DEFINE_NORMALIZE                                              \
    FORCE_INLINE t_vec normalize() const { return (*this) / magnitude(); }

#define ATG_MATH_DEFINE_CONVERSION_CONSTRUCTOR                                 \
    template<typename t_b_type>                                                \
    vec(const t_b_type &b) {                                                   \
        constexpr unsigned int l =                                             \
                (t_size < t_b_type::t_size) ? t_size : t_b_type::t_size;       \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            data[i] = t_scalar(b.data[i]);                                     \
        }                                                                      \
                                                                               \
        for (unsigned int i = l; i < t_size; ++i) { data[i] = t_scalar(0); }   \
    }

#define ATG_MATH_DEFINE_CONVERSION                                             \
    template<typename t_b_type>                                                \
    FORCE_INLINE t_vec operator=(const t_b_type &b) {                          \
        constexpr unsigned int l =                                             \
                (t_size < t_b_type::t_size) ? t_size : t_b_type::t_size;       \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            data[i] = t_scalar(b.data[i]);                                     \
        }                                                                      \
                                                                               \
        for (unsigned int i = l; i < t_size; ++i) { data[i] = t_scalar(0); }   \
                                                                               \
        return *this;                                                          \
    }                                                                          \
                                                                               \
    FORCE_INLINE t_vec operator=(const t_scalar &b) {                          \
        for (unsigned int i = 0; i < t_size; ++i) { data[i] = b; }             \
                                                                               \
        return *this;                                                          \
    }

#define ATG_MATH_DEFINE_SCALAR_CONSTRUCTOR                                     \
    vec(int s) {                                                               \
        for (unsigned int i = 0; i < t_size; ++i) { data[i] = t_scalar(s); }   \
    }                                                                          \
    vec(float s) {                                                             \
        for (unsigned int i = 0; i < t_size; ++i) { data[i] = t_scalar(s); }   \
    }                                                                          \
    vec(double s) {                                                            \
        for (unsigned int i = 0; i < t_size; ++i) { data[i] = t_scalar(s); }   \
    }                                                                          \
    vec(unsigned int s) {                                                      \
        for (unsigned int i = 0; i < t_size; ++i) { data[i] = t_scalar(s); }   \
    }

#define ATG_MATH_DEFINE_DEFAULT_CONSTRUCTOR                                    \
    vec() {}

#define ATG_MATH_DEFINE_LOAD                                                   \
    FORCE_INLINE void load(const t_scalar *data) {                             \
        for (unsigned int i = 0; i < t_size; ++i) { this->data[i] = data[i]; } \
    }

#define ATG_MATH_DEFINE_EXTRACT                                                \
    FORCE_INLINE void extract(t_scalar *data) {                                \
        for (unsigned int i = 0; i < t_size; ++i) { data[i] = this->data[i]; } \
    }

#define ATG_MATH_DEFINE_EXPLICIT_SCALAR_CONVERSION                             \
    FORCE_INLINE explicit operator t_scalar() const { return data[0]; }

#define ATG_MATH_S_X 0
#define ATG_MATH_S_Y 1
#define ATG_MATH_S_Z 2
#define ATG_MATH_S_W 3
#define ATG_MATH_M128_SHUFFLE(p0, p1, p2, p3)                                  \
    _MM_SHUFFLE((p3), (p2), (p1), (p0))

template<typename t_scalar_>
struct vec<t_scalar_, 1, false> {
    ATG_MATH_VEC_DEFINES(1, false)
    ATG_MATH_ALIAS(s, 0)

    t_scalar data[t_size];

    ATG_MATH_DEFINE_CONVERSION_CONSTRUCTOR
    ATG_MATH_DEFINE_CONVERSION
    ATG_MATH_DEFINE_SCALAR_CONSTRUCTOR
    ATG_MATH_DEFINE_DEFAULT_CONSTRUCTOR
    ATG_MATH_DEFINE_LOAD
    ATG_MATH_DEFINE_EXTRACT

    ATG_MATH_DEFINE_EXPLICIT_SCALAR_CONVERSION

    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(+)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(-)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(*)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(/)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(==)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(<=)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(>=)
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(==, eq)
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(<=, le)
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(>=, ge)
    ATG_MATH_DEFINE_NEGATED_BOOLEAN_COMPARISON_OPERATOR(!=)
    ATG_MATH_DEFINE_MADD()

    ATG_MATH_DEFINE_NEGATE_OPERATOR
    ATG_MATH_DEFINE_POSITIVE_OPERATOR
    ATG_MATH_DEFINE_SQRT
    ATG_MATH_DEFINE_SIGN

    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(+=, +)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(-=, -)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(*=, *)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(/=, /)

    ATG_MATH_DEFINE_COMPONENT_MAX
    ATG_MATH_DEFINE_COMPONENT_MIN
    ATG_MATH_DEFINE_ABS
    ATG_MATH_DEFINE_SUM

    ATG_MATH_COMPARE_GE_MASK
    ATG_MATH_AND_MASK
    ATG_MATH_AND_NOT_MASK
    ATG_MATH_BITWISE_OR
};

template<typename t_scalar_>
struct vec<t_scalar_, 2, false> {
    ATG_MATH_VEC_DEFINES(2, false)

    inline constexpr vec(t_scalar x, t_scalar y) {
        data[0] = x;
        data[1] = y;
    }

    ATG_MATH_ALIAS(x, 0)
    ATG_MATH_ALIAS(y, 1)

    ATG_MATH_ALIAS(s, 0)
    ATG_MATH_ALIAS(t, 1)

    ATG_MATH_ALIAS(u, 0)
    ATG_MATH_ALIAS(v, 1)

    ATG_MATH_ALIAS(w, 0)
    ATG_MATH_ALIAS(h, 1)

    t_scalar data[t_size];

    ATG_MATH_DEFINE_CONVERSION_CONSTRUCTOR
    ATG_MATH_DEFINE_CONVERSION
    ATG_MATH_DEFINE_SCALAR_CONSTRUCTOR
    ATG_MATH_DEFINE_DEFAULT_CONSTRUCTOR
    ATG_MATH_DEFINE_LOAD
    ATG_MATH_DEFINE_EXTRACT

    ATG_MATH_DEFINE_EXPLICIT_SCALAR_CONVERSION

    ATG_MATH_DEFINE_INDEX_OPERATOR()
    ATG_MATH_DEFINE_INDEX_OPERATOR_MODIFIABLE()

    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(+)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(-)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(*)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(/)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(==)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(<=)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(>=)
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(==, eq);
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(<=, le);
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(>=, ge);
    ATG_MATH_DEFINE_NEGATED_BOOLEAN_COMPARISON_OPERATOR(!=)

    ATG_MATH_DEFINE_NEGATE_OPERATOR
    ATG_MATH_DEFINE_POSITIVE_OPERATOR

    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(+=, +)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(-=, -)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(*=, *)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(/=, /)

    ATG_MATH_DEFINE_SUM
    ATG_MATH_DEFINE_DOT_PRODUCT

    ATG_MATH_DEFINE_COMPONENT_MIN
    ATG_MATH_DEFINE_COMPONENT_MAX

    ATG_MATH_DEFINE_ABS
    ATG_MATH_DEFINE_MAGNITUDE
    ATG_MATH_DEFINE_MAGNITUDE_SQUARED
    ATG_MATH_DEFINE_NORMALIZE
    ATG_MATH_DEFINE_SIGN
    ATG_MATH_DEFINE_SQRT

    ATG_MATH_COMPARE_GE_MASK
    ATG_MATH_AND_MASK
    ATG_MATH_AND_NOT_MASK
    ATG_MATH_BITWISE_OR

    FORCE_INLINE t_vec orthogonal() const { return {-y(), x()}; }
    FORCE_INLINE t_vec yx() const { return {y(), x()}; }
};

template<typename t_scalar_>
struct vec<t_scalar_, 3, false> {
    ATG_MATH_VEC_DEFINES(3, false)

    inline constexpr vec(t_scalar x, t_scalar y, t_scalar z = 0) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }

    ATG_MATH_ALIAS(x, 0)
    ATG_MATH_ALIAS(y, 1)
    ATG_MATH_ALIAS(z, 2)

    ATG_MATH_ALIAS(u, 0)
    ATG_MATH_ALIAS(v, 1)
    ATG_MATH_ALIAS(w, 2)

    ATG_MATH_ALIAS(r, 0)
    ATG_MATH_ALIAS(g, 1)
    ATG_MATH_ALIAS(b, 2)

    t_scalar data[t_size];

    t_vec cross(const t_vec &b) const {
        return {y() * b.z() - z() * b.y(), z() * b.x() - x() * b.z(),
                x() * b.y() - y() * b.x()};
    }

    template<int i0 = 0, int i1 = 0, int i2 = 2, int i3 = 3>
    t_vec shuffle() const {
        return {data[i0], data[i1], data[i2]};
    }

    ATG_MATH_DEFINE_CONVERSION_CONSTRUCTOR
    ATG_MATH_DEFINE_CONVERSION
    ATG_MATH_DEFINE_SCALAR_CONSTRUCTOR
    ATG_MATH_DEFINE_DEFAULT_CONSTRUCTOR
    ATG_MATH_DEFINE_LOAD
    ATG_MATH_DEFINE_EXTRACT

    ATG_MATH_DEFINE_INDEX_OPERATOR()
    ATG_MATH_DEFINE_INDEX_OPERATOR_MODIFIABLE()

    ATG_MATH_DEFINE_EXPLICIT_SCALAR_CONVERSION

    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(+)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(-)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(*)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(/)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(==)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(<=)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(>=)
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(==, eq);
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(<=, le);
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(>=, ge);
    ATG_MATH_DEFINE_NEGATED_BOOLEAN_COMPARISON_OPERATOR(!=)
    ATG_MATH_DEFINE_MADD()

    ATG_MATH_DEFINE_NEGATE_OPERATOR
    ATG_MATH_DEFINE_POSITIVE_OPERATOR

    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(+=, +)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(-=, -)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(*=, *)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(/=, /)

    ATG_MATH_DEFINE_SUM
    ATG_MATH_DEFINE_DOT_PRODUCT
    ATG_MATH_DEFINE_COMPONENT_MIN
    ATG_MATH_DEFINE_COMPONENT_MAX

    ATG_MATH_DEFINE_ABS
    ATG_MATH_DEFINE_MAGNITUDE_SQUARED
    ATG_MATH_DEFINE_MAGNITUDE
    ATG_MATH_DEFINE_NORMALIZE
    ATG_MATH_DEFINE_SIGN
    ATG_MATH_DEFINE_SQRT

    ATG_MATH_COMPARE_GE_MASK
    ATG_MATH_AND_MASK
    ATG_MATH_AND_NOT_MASK
    ATG_MATH_BITWISE_OR
};

template<typename t_scalar_>
struct vec<t_scalar_, 4, false> {
    ATG_MATH_VEC_DEFINES(4, false)

    inline constexpr vec(t_scalar x, t_scalar y, t_scalar z, t_scalar w) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }

    inline constexpr vec(t_scalar x, t_scalar y) {
        data[0] = x;
        data[1] = y;
        data[2] = data[3] = 0;
    }

    inline constexpr vec(t_scalar x, t_scalar y, t_scalar z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = 0;
    }

    ATG_MATH_ALIAS(x, 0)
    ATG_MATH_ALIAS(y, 1)
    ATG_MATH_ALIAS(z, 2)
    ATG_MATH_ALIAS(w, 3)

    ATG_MATH_ALIAS(r, 0)
    ATG_MATH_ALIAS(g, 1)
    ATG_MATH_ALIAS(b, 2)
    ATG_MATH_ALIAS(a, 3)

    t_scalar data[t_size];

    t_vec cross(const t_vec &b) const {
        return {y() * b.z() - z() * b.y(), z() * b.x() - x() * b.z(),
                x() * b.y() - y() * b.x(), 0};
    }

    template<int i0 = 0, int i1 = 1, int i2 = 2, int i3 = 3>
    t_vec shuffle() const {
        return {data[i0], data[i1], data[i2], data[i3]};
    }

    ATG_MATH_DEFINE_CONVERSION_CONSTRUCTOR
    ATG_MATH_DEFINE_CONVERSION
    ATG_MATH_DEFINE_SCALAR_CONSTRUCTOR
    ATG_MATH_DEFINE_DEFAULT_CONSTRUCTOR
    ATG_MATH_DEFINE_LOAD
    ATG_MATH_DEFINE_EXTRACT

    ATG_MATH_DEFINE_EXPLICIT_SCALAR_CONVERSION

    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(+)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(-)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(*)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(/)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(==)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(<=)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(>=)
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(==, eq);
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(<=, le);
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(>=, ge);
    ATG_MATH_DEFINE_NEGATED_BOOLEAN_COMPARISON_OPERATOR(!=)
    ATG_MATH_DEFINE_MADD()

    ATG_MATH_DEFINE_NEGATE_OPERATOR
    ATG_MATH_DEFINE_POSITIVE_OPERATOR

    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(+=, +)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(-=, -)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(*=, *)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(/=, /)

    ATG_MATH_DEFINE_SUM
    ATG_MATH_DEFINE_DOT_PRODUCT
    ATG_MATH_DEFINE_COMPONENT_MIN
    ATG_MATH_DEFINE_COMPONENT_MAX

    ATG_MATH_DEFINE_ABS
    ATG_MATH_DEFINE_MAGNITUDE_SQUARED
    ATG_MATH_DEFINE_MAGNITUDE
    ATG_MATH_DEFINE_NORMALIZE
    ATG_MATH_DEFINE_SIGN
    ATG_MATH_DEFINE_SQRT

    ATG_MATH_COMPARE_GE_MASK
    ATG_MATH_AND_MASK
    ATG_MATH_AND_NOT_MASK
    ATG_MATH_BITWISE_OR

    FORCE_INLINE t_vec xy() const {
        return {x(), y(), t_scalar(0), t_scalar(0)};
    }
    FORCE_INLINE t_vec yz() const {
        return {y(), z(), t_scalar(0), t_scalar(0)};
    }
    FORCE_INLINE t_vec xz() const {
        return {x(), z(), t_scalar(0), t_scalar(0)};
    }
    FORCE_INLINE t_vec xyz() const { return {x(), y(), z(), t_scalar(0)}; }
    FORCE_INLINE t_vec position() const { return {x(), y(), z(), t_scalar(1)}; }
};

template<typename t_scalar_, unsigned int t_size_>
struct vec<t_scalar_, t_size_, false> {
    static constexpr unsigned int t_size = t_size_;
    static constexpr bool t_simd = false;

    ATG_MATH_DEFINE_T_VEC;
    ATG_MATH_DEFINE_T_SCALAR;

    t_scalar data[t_size];

    ATG_MATH_DEFINE_CONVERSION_CONSTRUCTOR
    ATG_MATH_DEFINE_CONVERSION
    ATG_MATH_DEFINE_SCALAR_CONSTRUCTOR
    ATG_MATH_DEFINE_DEFAULT_CONSTRUCTOR
    ATG_MATH_DEFINE_LOAD
    ATG_MATH_DEFINE_EXTRACT

    ATG_MATH_DEFINE_EXPLICIT_SCALAR_CONVERSION

    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(+)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(-)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(*)
    ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(/)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(==)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(<=)
    ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(>=)
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(==, eq);
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(<=, le);
    ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(>=, ge);
    ATG_MATH_DEFINE_NEGATED_BOOLEAN_COMPARISON_OPERATOR(!=)
    ATG_MATH_DEFINE_MADD()

    ATG_MATH_DEFINE_INDEX_OPERATOR()
    ATG_MATH_DEFINE_INDEX_OPERATOR_MODIFIABLE()

    ATG_MATH_DEFINE_NEGATE_OPERATOR
    ATG_MATH_DEFINE_POSITIVE_OPERATOR

    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(+=, +)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(-=, -)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(*=, *)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(/=, /)

    ATG_MATH_DEFINE_SUM
    ATG_MATH_DEFINE_DOT_PRODUCT
    ATG_MATH_DEFINE_COMPONENT_MIN
    ATG_MATH_DEFINE_COMPONENT_MAX

    ATG_MATH_DEFINE_ABS
    ATG_MATH_DEFINE_MAGNITUDE_SQUARED
    ATG_MATH_DEFINE_MAGNITUDE
    ATG_MATH_DEFINE_NORMALIZE
    ATG_MATH_DEFINE_SQRT
    ATG_MATH_DEFINE_SIGN

    ATG_MATH_COMPARE_GE_MASK
    ATG_MATH_AND_MASK
    ATG_MATH_AND_NOT_MASK
    ATG_MATH_BITWISE_OR
};

template<>
struct vec<float, 4, true> {
    using t_scalar = float;
    static constexpr unsigned int t_size = 4;
    static constexpr bool t_simd = true;
    typedef vec<float, 4, true> t_vec;

    FORCE_INLINE constexpr vec() { data_v = {0.0f, 0.0f, 0.0f, 0.0f}; }
    FORCE_INLINE constexpr vec(const __m128 &v) : data_v(v) {}
    FORCE_INLINE constexpr vec(float x, float y, float z = 0, float w = 0) {
        data_v = {x, y, z, w};
    }
    FORCE_INLINE constexpr vec(float s) { data_v = {s, s, s, s}; }
    FORCE_INLINE constexpr vec(const vec<float, 4, false> &v) {
        data_v = {v.x(), v.y(), v.z(), v.w()};
    }
    FORCE_INLINE constexpr vec(const vec<float, 2, false> &v) {
        data_v = {v.x(), v.y(), 0.0f, 0.0f};
    }

    ATG_MATH_ALIAS(x, 0)
    ATG_MATH_ALIAS(y, 1)
    ATG_MATH_ALIAS(z, 2)
    ATG_MATH_ALIAS(w, 3)

    ATG_MATH_ALIAS(r, 0)
    ATG_MATH_ALIAS(g, 1)
    ATG_MATH_ALIAS(b, 2)
    ATG_MATH_ALIAS(a, 3)

    template<int i0 = 0, int i1 = 1, int i2 = 2, int i3 = 3>
    FORCE_INLINE t_vec shuffle() const {
        return _mm_shuffle_ps(data_v, data_v,
                              ATG_MATH_M128_SHUFFLE(i0, i1, i2, i3));
    }

    union {
        __m128 data_v;
        float data[4];
        int mask[4];
    };

    FORCE_INLINE explicit operator float() const {
        return _mm_cvtss_f32(data_v);
    }

    FORCE_INLINE operator __m128() const { return data_v; }

    FORCE_INLINE float operator[](unsigned int i) const { return data[i]; }
    FORCE_INLINE float &operator[](unsigned int i) { return data[i]; }

    FORCE_INLINE t_vec operator-() const {
        const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(~0x7FFFFFFF));
        return _mm_xor_ps(data_v, mask);
    }

    FORCE_INLINE t_vec operator+() const { return *this; }

    FORCE_INLINE t_vec operator+(const t_vec &b) const {
        return _mm_add_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator-(const t_vec &b) const {
        return _mm_sub_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator*(const t_vec &b) const {
        return _mm_mul_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator/(const t_vec &b) const {
        return _mm_div_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec madd(const t_vec &m, const t_vec &a) const {
        return _mm_fmadd_ps(data_v, m, a);
    }

    FORCE_INLINE t_vec compare_eq(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpeq_ps(data_v, b.data_v);
        return _mm_and_ps(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_neq(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpneq_ps(data_v, b.data_v);
        return _mm_and_ps(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_le(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmple_ps(data_v, b.data_v);
        return _mm_and_ps(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_ge(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpge_ps(data_v, b.data_v);
        return _mm_and_ps(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE explicit operator bool() const {
        return mask[0] != 0 && mask[1] != 0 && mask[2] != 0 && mask[3] != 0;
    }

    FORCE_INLINE t_vec operator==(const t_vec &b) const {
        return _mm_cmpeq_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator<=(const t_vec &b) const {
        return _mm_cmple_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator>=(const t_vec &b) const {
        return _mm_cmpge_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator<(const t_vec &b) const {
        return _mm_cmplt_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator>(const t_vec &b) const {
        return _mm_cmpgt_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator!=(const t_vec &b) const {
        return _mm_cmpeq_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator+=(const t_vec &b) {
        return data_v = _mm_add_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator-=(const t_vec &b) {
        return data_v = _mm_sub_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator*=(const t_vec &b) {
        return data_v = _mm_mul_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator/=(const t_vec &b) {
        return data_v = _mm_div_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec sum() const {
        const __m128 t1 = _mm_shuffle_ps(
                data_v, data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Z, ATG_MATH_S_W, ATG_MATH_S_X,
                                      ATG_MATH_S_Y));
        const __m128 t2 = _mm_add_ps(data_v, t1);
        const __m128 t3 = _mm_shuffle_ps(
                t2, t2,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_X, ATG_MATH_S_W,
                                      ATG_MATH_S_Z));
        return _mm_add_ps(t3, t2);
    }

    FORCE_INLINE t_vec dot(const t_vec &b) const {
        const __m128 t0 = _mm_mul_ps(data_v, b.data_v);
        return t_vec(t0).sum();
    }

    FORCE_INLINE t_vec cross(const t_vec &b) const {
        const __m128 t1 = _mm_shuffle_ps(
                data_v, data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_Z, ATG_MATH_S_X,
                                      ATG_MATH_S_W));
        const __m128 t2 = _mm_shuffle_ps(
                b.data_v, b.data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Z, ATG_MATH_S_X, ATG_MATH_S_Y,
                                      ATG_MATH_S_W));
        const __m128 t3 = _mm_shuffle_ps(
                data_v, data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Z, ATG_MATH_S_X, ATG_MATH_S_Y,
                                      ATG_MATH_S_W));
        const __m128 t4 = _mm_shuffle_ps(
                b.data_v, b.data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_Z, ATG_MATH_S_X,
                                      ATG_MATH_S_W));

        return _mm_sub_ps(_mm_mul_ps(t1, t2), _mm_mul_ps(t3, t4));
    }

    FORCE_INLINE t_vec min(const t_vec &b) const {
        return _mm_min_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec max(const t_vec &b) const {
        return _mm_max_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec abs() const {
        const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        return _mm_and_ps(data_v, mask);
    }

    FORCE_INLINE t_vec magnitude_squared() const { return dot(*this); }
    FORCE_INLINE t_vec sqrt() const { return _mm_sqrt_ps(data_v); }
    FORCE_INLINE t_vec magnitude() const { return magnitude_squared().sqrt(); }
    FORCE_INLINE t_vec normalize() const {
        return _mm_div_ps(data_v, magnitude());
    }

    FORCE_INLINE t_vec xy() const {
        return _mm_shuffle_ps(data_v, {0.0f, 0.0f, 0.0f, 0.0f},
                              ATG_MATH_M128_SHUFFLE(ATG_MATH_S_X, ATG_MATH_S_Y,
                                                    ATG_MATH_S_Z,
                                                    ATG_MATH_S_W));
    }

    FORCE_INLINE t_vec yz() const {
        return _mm_shuffle_ps(data_v, {0.0f, 0.0f, 0.0f, 0.0f},
                              ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_Z,
                                                    ATG_MATH_S_Z,
                                                    ATG_MATH_S_W));
    }

    FORCE_INLINE t_vec xz() const {
        return _mm_shuffle_ps(data_v, {0.0f, 0.0f, 0.0f, 0.0f},
                              ATG_MATH_M128_SHUFFLE(ATG_MATH_S_X, ATG_MATH_S_Z,
                                                    ATG_MATH_S_Z,
                                                    ATG_MATH_S_W));
    }

    FORCE_INLINE t_vec xyz() const {
        return _mm_and_ps(data_v,
                          _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1)));
    }

    FORCE_INLINE t_vec position() const {
        return _mm_or_ps(_mm_and_ps(data_v, _mm_castsi128_ps(_mm_set_epi32(
                                                    0, -1, -1, -1))),
                         {0.0f, 0.0f, 0.0f, 1.0f});
    }

    FORCE_INLINE void load(float *data) { data_v = _mm_load_ps(data); }
    FORCE_INLINE void extract(float *data) const { _mm_store_ps(data, data_v); }

    FORCE_INLINE t_vec exp() const { return _mm_exp_ps(data_v); }
    FORCE_INLINE t_vec log() const { return _mm_log_ps(data_v); }
    FORCE_INLINE t_vec pow(const t_vec &p) const { return (p * log()).exp(); }

    FORCE_INLINE t_vec and_mask(const t_vec &mask) const {
        return _mm_and_ps(data_v, mask);
    }

    FORCE_INLINE t_vec bitwise_or(const t_vec &b) const {
        return _mm_or_ps(data_v, b);
    }

    FORCE_INLINE t_vec and_not_mask(const t_vec &mask) const {
        return _mm_andnot_ps(mask, data_v);
    }
};

template<>
struct vec<float, 8, true> {
    using t_scalar = float;
    static constexpr unsigned int t_size = 8;
    static constexpr bool t_simd = true;
    typedef vec<float, 8, true> t_vec;

    FORCE_INLINE constexpr vec() { data_v = {0, 0, 0, 0, 0, 0, 0, 0}; }
    FORCE_INLINE constexpr vec(const __m256 &v) : data_v(v) {}
    FORCE_INLINE constexpr vec(float s) { data_v = {s, s, s, s, s, s, s, s}; }
    FORCE_INLINE constexpr vec(float s0, float s1, float s2, float s3, float s4,
                               float s5, float s6, float s7) {
        data_v = {s0, s1, s2, s3, s4, s5, s6, s7};
    }

    union {
        __m256 data_v;
        float data[8];
        int mask[8];
    };

    FORCE_INLINE float operator[](unsigned int i) const { return data[i]; }
    FORCE_INLINE float &operator[](unsigned int i) { return data[i]; }

    FORCE_INLINE explicit operator float() const {
        return _mm256_cvtss_f32(data_v);
    }
    FORCE_INLINE operator __m256() const { return data_v; }

    FORCE_INLINE t_vec operator-() const {
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x7FFFFFFF));
        return _mm256_xor_ps(data_v, mask);
    }

    FORCE_INLINE t_vec operator+() const { return *this; }

    FORCE_INLINE t_vec operator+(const t_vec &b) const {
        return _mm256_add_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator-(const t_vec &b) const {
        return _mm256_sub_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator*(const t_vec &b) const {
        return _mm256_mul_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator/(const t_vec &b) const {
        return _mm256_div_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec madd(const t_vec &m, const t_vec &a) const {
        return _mm256_fmadd_ps(data_v, m, a);
    }

    FORCE_INLINE t_vec compare_eq(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_ps(data_v, b.data_v, _CMP_EQ_OQ);
        return _mm256_and_ps(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_neq(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_ps(data_v, b.data_v, _CMP_NEQ_OQ);
        return _mm256_and_ps(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_le(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_ps(data_v, b.data_v, _CMP_LE_OQ);
        return _mm256_and_ps(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_ge(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_ps(data_v, b.data_v, _CMP_GE_OQ);
        return _mm256_and_ps(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_ge_mask(const t_vec &b) const {
        return _mm256_cmp_ps(data_v, b.data_v, _CMP_GE_OQ);
    }

    FORCE_INLINE t_vec and_mask(const t_vec &mask) const {
        return _mm256_and_ps(data_v, mask);
    }

    FORCE_INLINE t_vec bitwise_or(const t_vec &b) const {
        return _mm256_or_ps(data_v, b);
    }

    FORCE_INLINE t_vec and_not_mask(const t_vec &mask) const {
        return _mm256_andnot_ps(mask, data_v);
    }

    FORCE_INLINE explicit operator bool() const {
        for (int i = 0; i < 8; ++i) {
            if (mask[i] == 0) { return false; }
        }

        return true;
    }

    FORCE_INLINE t_vec operator==(const t_vec &b) const {
        return _mm256_cmp_ps(data_v, b.data_v, _CMP_EQ_OQ);
    }

    FORCE_INLINE t_vec operator<=(const t_vec &b) const {
        return _mm256_cmp_ps(data_v, b.data_v, _CMP_LE_OQ);
    }

    FORCE_INLINE t_vec operator<(const t_vec &b) const {
        return _mm256_cmp_ps(data_v, b.data_v, _CMP_LT_OQ);
    }

    FORCE_INLINE t_vec operator>(const t_vec &b) const {
        return _mm256_cmp_ps(data_v, b.data_v, _CMP_GT_OQ);
    }

    FORCE_INLINE t_vec operator>=(const t_vec &b) const {
        return _mm256_cmp_ps(data_v, b.data_v, _CMP_GE_OQ);
    }

    FORCE_INLINE t_vec operator!=(const t_vec &b) const {
        return _mm256_cmp_ps(data_v, b.data_v, _CMP_NEQ_OQ);
    }

    FORCE_INLINE t_vec operator+=(const t_vec &b) {
        return data_v = _mm256_add_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator-=(const t_vec &b) {
        return data_v = _mm256_sub_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator*=(const t_vec &b) {
        return data_v = _mm256_mul_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator/=(const t_vec &b) {
        return data_v = _mm256_div_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec min(const t_vec &b) const {
        return _mm256_min_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec max(const t_vec &b) const {
        return _mm256_max_ps(data_v, b.data_v);
    }

    FORCE_INLINE t_vec abs() const {
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        return _mm256_and_ps(data_v, mask);
    }

    FORCE_INLINE t_vec sign() const {
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x7FFFFFFF));
        return _mm256_or_ps(_mm256_set1_ps(1.0f), _mm256_and_ps(data_v, mask));
    }

    FORCE_INLINE t_vec sum() const {
        __m256 sum = data_v;
        const __m256 s0 = _mm256_shuffle_ps(
                data_v, data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_Z, ATG_MATH_S_W,
                                      ATG_MATH_S_X));
        sum = _mm256_add_ps(sum, s0);

        const __m256 s1 = _mm256_shuffle_ps(
                sum, sum,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Z, ATG_MATH_S_W, ATG_MATH_S_X,
                                      ATG_MATH_S_Y));

        sum = _mm256_add_ps(sum, s1);

        const __m256 s2 = _mm256_permute2f128_ps(sum, sum, 0x01);
        return _mm256_add_ps(s2, sum);
    }

    FORCE_INLINE t_vec sqrt() const { return _mm256_sqrt_ps(data_v); }

    FORCE_INLINE void load(const float *data) { data_v = _mm256_load_ps(data); }
    FORCE_INLINE void extract(float *data) { _mm256_store_ps(data, data_v); }

    FORCE_INLINE t_vec exp() const { return _mm256_exp_ps(data_v); }
    FORCE_INLINE t_vec log() const { return _mm256_log_ps(data_v); }
    FORCE_INLINE t_vec pow(const t_vec &p) const { return (p * log()).exp(); }
};

template<>
struct vec<double, 2, true> {
    using t_scalar = double;
    static constexpr unsigned int t_size = 2;
    static constexpr bool t_simd = true;
    typedef vec<double, 2, true> t_vec;

    FORCE_INLINE constexpr vec() { data[0] = data[1] = 0; }
    FORCE_INLINE constexpr vec(const __m128d &v) : data_v(v) {}
    FORCE_INLINE constexpr vec(double x, double y) { data_v = {x, y}; }
    FORCE_INLINE vec(double s) { data_v = _mm_set_pd1(s); }

    ATG_MATH_ALIAS(x, 0)
    ATG_MATH_ALIAS(y, 1)

    ATG_MATH_ALIAS(u, 0)
    ATG_MATH_ALIAS(v, 1)

    ATG_MATH_ALIAS(w, 0)
    ATG_MATH_ALIAS(h, 1)

    template<int i0 = 0, int i1 = 1>
    FORCE_INLINE t_vec shuffle() const {
        return _mm_shuffle_pd(data_v, data_v,
                              ATG_MATH_M128_SHUFFLE(i0, i1, 0, 0));
    }

    union {
        double data[2];
        __m128d data_v;
        int mask[2];
    };

    FORCE_INLINE explicit operator double() const {
        return _mm_cvtsd_f64(data_v);
    }

    FORCE_INLINE operator __m128d() const { return data_v; }

    FORCE_INLINE t_vec operator-() const {
        const __m128d mask =
                _mm_castsi128_pd(_mm_set1_epi64x(~0x7FFFFFFFFFFFFFFF));
        return _mm_xor_pd(data_v, mask);
    }

    FORCE_INLINE t_vec operator+() const { return *this; }

    FORCE_INLINE t_vec operator+(const t_vec &b) const {
        return _mm_add_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator-(const t_vec &b) const {
        return _mm_sub_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator*(const t_vec &b) const {
        return _mm_mul_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator/(const t_vec &b) const {
        return _mm_div_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec compare_eq(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpeq_pd(data_v, b.data_v);
        return _mm_and_pd(cmp_mask, t_vec(1.0));
    }

    FORCE_INLINE t_vec compare_neq(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpneq_pd(data_v, b.data_v);
        return _mm_and_pd(cmp_mask, t_vec(1.0));
    }

    FORCE_INLINE t_vec compare_le(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmple_pd(data_v, b.data_v);
        return _mm_and_pd(cmp_mask, t_vec(1.0));
    }

    FORCE_INLINE t_vec compare_ge(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpge_pd(data_v, b.data_v);
        return _mm_and_pd(cmp_mask, t_vec(1.0));
    }

    FORCE_INLINE bool operator==(const t_vec &b) const {
        const t_vec cmp = _mm_cmpeq_pd(data_v, b.data_v);
        return !(cmp.x() == 0 || cmp.y() == 0);
    }

    FORCE_INLINE bool operator<=(const t_vec &b) const {
        const t_vec cmp = _mm_cmple_pd(data_v, b.data_v);
        return !(cmp.x() == 0 || cmp.y() == 0);
    }

    FORCE_INLINE bool operator>=(const t_vec &b) const {
        const t_vec cmp = _mm_cmpge_pd(data_v, b.data_v);
        return !(cmp.x() == 0 || cmp.y() == 0);
    }

    FORCE_INLINE bool operator!=(const t_vec &b) const {
        const t_vec cmp = _mm_cmpeq_pd(data_v, b.data_v);
        return cmp.x() == 0 && cmp.y() == 0;
    }

    FORCE_INLINE t_vec operator+=(const t_vec &b) {
        return data_v = _mm_add_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator-=(const t_vec &b) {
        return data_v = _mm_sub_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator*=(const t_vec &b) {
        return data_v = _mm_mul_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator/=(const t_vec &b) {
        return data_v = _mm_div_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec sum() const {
        const __m128d t1 = _mm_shuffle_pd(
                data_v, data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_X, 0, 0));
        return _mm_add_pd(data_v, t1);
    }

    FORCE_INLINE t_vec dot(const t_vec &b) const {
        const __m128d t0 = _mm_mul_pd(data_v, b.data_v);
        return t_vec(t0).sum();
    }

    FORCE_INLINE t_vec min(const t_vec &b) const {
        return _mm_min_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec max(const t_vec &b) const {
        return _mm_max_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec abs() const {
        const __m128d mask =
                _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        return _mm_and_pd(data_v, mask);
    }

    FORCE_INLINE t_vec magnitude_squared() const { return dot(*this); }
    FORCE_INLINE t_vec sqrt() const { return _mm_sqrt_pd(data_v); }
    FORCE_INLINE t_vec magnitude() const { return magnitude_squared().sqrt(); }
    FORCE_INLINE t_vec normalize() const {
        return _mm_div_pd(data_v, magnitude());
    }

    FORCE_INLINE void load(double *data) { data_v = _mm_load_pd(data); }
    FORCE_INLINE void extract(double *data) { _mm_store_pd(data, data_v); }

    FORCE_INLINE t_vec exp() const { return _mm_exp_pd(data_v); }
    FORCE_INLINE t_vec log() const { return _mm_log_pd(data_v); }
    FORCE_INLINE t_vec pow(const t_vec &p) const { return (p * log()).exp(); }
};

template<>
struct vec<double, 4, true> {
    using t_scalar = double;
    static constexpr unsigned int t_size = 4;
    static constexpr bool t_simd = true;
    typedef vec<double, 4, true> t_vec;

    FORCE_INLINE constexpr vec() { data_v = {0, 0, 0, 0}; }
    FORCE_INLINE constexpr vec(const __m256d &v) : data_v(v) {}
    FORCE_INLINE constexpr vec(double x, double y, double z = 0, double w = 0) {
        data_v = {x, y, z, w};
    }
    FORCE_INLINE constexpr vec(double s) { data_v = {s, s, s, s}; }

    ATG_MATH_ALIAS(x, 0)
    ATG_MATH_ALIAS(y, 1)
    ATG_MATH_ALIAS(z, 2)
    ATG_MATH_ALIAS(w, 3)

    ATG_MATH_ALIAS(r, 0)
    ATG_MATH_ALIAS(g, 1)
    ATG_MATH_ALIAS(b, 2)
    ATG_MATH_ALIAS(a, 3)

    template<int i0 = 0, int i1 = 1, int i2 = 2, int i3 = 3>
    FORCE_INLINE t_vec shuffle() const {
        return _mm_shuffle_ps(data_v, data_v,
                              ATG_MATH_M128_SHUFFLE(i0, i1, i2, i3));
    }

    union {
        __m256d data_v;
        double data[4];
        int64_t mask[4];
    };

    FORCE_INLINE explicit operator double() const {
        return _mm256_cvtsd_f64(data_v);
    }

    FORCE_INLINE double operator[](size_t index) const { return data[index]; }
    FORCE_INLINE double &operator[](size_t index) { return data[index]; }

    FORCE_INLINE operator __m256d() const { return data_v; }

    FORCE_INLINE t_vec operator-() const {
        const __m256d mask =
                _mm256_castsi256_pd(_mm256_set1_epi64x(~0x7FFFFFFFFFFFFFFF));
        return _mm256_xor_pd(data_v, mask);
    }

    FORCE_INLINE t_vec operator+() const { return *this; }

    FORCE_INLINE t_vec operator+(const t_vec &b) const {
        return _mm256_add_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator-(const t_vec &b) const {
        return _mm256_sub_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator*(const t_vec &b) const {
        return _mm256_mul_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator/(const t_vec &b) const {
        return _mm256_div_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec madd(const t_vec &m, const t_vec &a) const {
        return _mm256_fmadd_pd(data_v, m, a);
    }

    FORCE_INLINE t_vec compare_eq(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_pd(data_v, b.data_v, _CMP_EQ_OQ);
        return _mm256_and_pd(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_eq_mask(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_EQ_OQ);
    }

    FORCE_INLINE t_vec compare_neq(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_pd(data_v, b.data_v, _CMP_NEQ_OQ);
        return _mm256_and_pd(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_neq_mask(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_NEQ_OQ);
    }

    FORCE_INLINE t_vec compare_le(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_pd(data_v, b.data_v, _CMP_LE_OQ);
        return _mm256_and_pd(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_le_mask(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_LE_OQ);
    }

    FORCE_INLINE t_vec compare_ge(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_pd(data_v, b.data_v, _CMP_GE_OQ);
        return _mm256_and_pd(cmp_mask, t_vec(1.0f));
    }

    FORCE_INLINE t_vec compare_ge_mask(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_GE_OQ);
    }

    FORCE_INLINE t_vec select(const t_vec &b, const t_vec &mask) const {
        return _mm256_or_pd(_mm256_and_pd(data_v, mask),
                            _mm256_andnot_pd(mask, b));
    }

    FORCE_INLINE t_vec bitwise_and(const t_vec &b) const {
        return _mm256_and_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec bitwise_or(const t_vec &b) const {
        return _mm256_or_pd(data_v, b.data_v);
    }

    FORCE_INLINE explicit operator bool() const {
        return x() != 0 && y() != 0 && z() != 0 && w() != 0;
    }

    FORCE_INLINE t_vec operator==(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_EQ_OQ);
    }

    FORCE_INLINE t_vec operator<=(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_LE_OQ);
    }

    FORCE_INLINE t_vec operator>=(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_GE_OQ);
    }

    FORCE_INLINE t_vec operator<(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_LT_OQ);
    }

    FORCE_INLINE t_vec operator>(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_GT_OQ);
    }

    FORCE_INLINE t_vec operator!=(const t_vec &b) const {
        return _mm256_cmp_pd(data_v, b.data_v, _CMP_NEQ_OQ);
    }

    FORCE_INLINE t_vec operator+=(const t_vec &b) {
        return data_v = _mm256_add_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator-=(const t_vec &b) {
        return data_v = _mm256_sub_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator*=(const t_vec &b) {
        return data_v = _mm256_mul_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec operator/=(const t_vec &b) {
        return data_v = _mm256_div_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec sum() const {
        const __m256d t1 = _mm256_shuffle_pd(data_v, data_v, 0b11);
        const __m256d t2 = _mm256_add_pd(data_v, t1);
        const __m256d t3 = _mm256_permute2f128_pd(t2, t2, 0x01);
        return _mm256_add_pd(t3, t2);
    }

    FORCE_INLINE t_vec dot(const t_vec &b) const {
        const __m256d t0 = _mm256_mul_pd(data_v, b.data_v);
        return t_vec(t0).sum();
    }

    FORCE_INLINE t_vec cross(const t_vec &b) const {
        const __m256d t1 = _mm256_permute4x64_pd(
                data_v, ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_Z,
                                              ATG_MATH_S_X, ATG_MATH_S_W));
        const __m256d t2 = _mm256_permute4x64_pd(
                b.data_v, ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Z, ATG_MATH_S_X,
                                                ATG_MATH_S_Y, ATG_MATH_S_W));
        const __m256d t3 = _mm256_permute4x64_pd(
                data_v, ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Z, ATG_MATH_S_X,
                                              ATG_MATH_S_Y, ATG_MATH_S_W));
        const __m256d t4 = _mm256_permute4x64_pd(
                b.data_v, ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_Z,
                                                ATG_MATH_S_X, ATG_MATH_S_W));

        return _mm256_sub_pd(_mm256_mul_pd(t1, t2), _mm256_mul_pd(t3, t4));
    }

    FORCE_INLINE t_vec min(const t_vec &b) const {
        return _mm256_min_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec max(const t_vec &b) const {
        return _mm256_max_pd(data_v, b.data_v);
    }

    FORCE_INLINE t_vec abs() const {
        const __m256d mask =
                _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        return _mm256_and_pd(data_v, mask);
    }

    FORCE_INLINE t_vec sign() const {
        const __m256d mask =
                _mm256_castsi256_pd(_mm256_set1_epi64x(~0x7FFFFFFFFFFFFFFF));
        return _mm256_or_pd(_mm256_set1_pd(1.0), _mm256_and_pd(data_v, mask));
    }

    FORCE_INLINE t_vec magnitude_squared() const { return dot(*this); }
    FORCE_INLINE t_vec sqrt() const { return _mm256_sqrt_pd(data_v); }
    FORCE_INLINE t_vec magnitude() const { return magnitude_squared().sqrt(); }
    FORCE_INLINE t_vec normalize() const {
        return _mm256_div_pd(data_v, magnitude());
    }

    FORCE_INLINE t_vec xy() const {
        return _mm256_shuffle_pd(data_v, {0.0, 0.0, 0.0, 0.0}, 0b0101);
    }

    FORCE_INLINE t_vec yz() const {
        const __m256d s0 = _mm256_permute4x64_pd(
                data_v, ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_Z,
                                              ATG_MATH_S_Y, ATG_MATH_S_Z));
        return _mm256_and_pd(
                s0, _mm256_castsi256_pd(_mm256_set_epi64x(0, 0, -1, -1)));
    }

    FORCE_INLINE t_vec xz() const {
        const __m256d s0 = _mm256_permute4x64_pd(
                data_v, ATG_MATH_M128_SHUFFLE(ATG_MATH_S_X, ATG_MATH_S_Z,
                                              ATG_MATH_S_X, ATG_MATH_S_Z));
        return _mm256_and_pd(
                s0, _mm256_castsi256_pd(_mm256_set_epi64x(0, 0, -1, -1)));
    }

    FORCE_INLINE t_vec xyz() const {
        return _mm256_and_pd(
                data_v, _mm256_castsi256_pd(_mm256_set_epi64x(0, -1, -1, -1)));
    }

    FORCE_INLINE t_vec position() const {
        return _mm256_or_pd(
                _mm256_and_pd(data_v, _mm256_castsi256_pd(_mm256_set_epi64x(
                                              0, -1, -1, -1))),
                {0.0, 0.0, 0.0, 1.0});
    }

    FORCE_INLINE t_vec and_mask(const t_vec &mask) const {
        return _mm256_and_pd(mask, data_v);
    }

    FORCE_INLINE t_vec and_not_mask(const t_vec &mask) const {
        return _mm256_andnot_pd(mask, data_v);
    }

    FORCE_INLINE void load(double *data) { data_v = _mm256_load_pd(data); }
    FORCE_INLINE void extract(double *data) { _mm256_store_pd(data, data_v); }

    FORCE_INLINE t_vec exp() const { return _mm256_exp_pd(data_v); }
    FORCE_INLINE t_vec log() const { return _mm256_log_pd(data_v); }
    FORCE_INLINE t_vec pow(const t_vec &p) const { return (p * log()).exp(); }
};

template<>
struct vec<double, 8, true> {
    using t_scalar = double;
    static constexpr unsigned int t_size = 8;
    static constexpr bool t_simd = true;
    typedef vec<double, 8, true> t_vec;

    FORCE_INLINE constexpr vec() : data0(double(0)), data1(double(0)) {}
    FORCE_INLINE constexpr vec(const vec<double, 4, true> &v0,
                               const vec<double, 4, true> &v1)
        : data0(v0.data_v), data1(v1.data_v) {}
    FORCE_INLINE constexpr vec(t_scalar s) : data0(s), data1(s) {}

    vec<double, 4, true> data0, data1;

    FORCE_INLINE explicit operator double() const { return double(data0); }
    FORCE_INLINE double operator[](size_t index) const {
        return index <= 3 ? data0.data[index] : data1.data[index - 4];
    }
    FORCE_INLINE double &operator[](size_t index) {
        return index <= 3 ? data0.data[index] : data1.data[index - 4];
    }

    FORCE_INLINE t_vec operator-() const { return {-data0, -data1}; }
    FORCE_INLINE t_vec operator+() const { return *this; }

    FORCE_INLINE t_vec operator+(const t_vec &b) const {
        return {data0 + b.data0, data1 + b.data1};
    }

    FORCE_INLINE t_vec operator-(const t_vec &b) const {
        return {data0 - b.data0, data1 - b.data1};
    }

    FORCE_INLINE t_vec operator*(const t_vec &b) const {
        return {data0 * b.data0, data1 * b.data1};
    }

    FORCE_INLINE t_vec operator/(const t_vec &b) const {
        return {data0 / b.data0, data1 / b.data1};
    }

    FORCE_INLINE t_vec madd(const t_vec &m, const t_vec &a) const {
        return {data0.madd(m.data0, a.data0), data1.madd(m.data1, a.data1)};
    }

    FORCE_INLINE t_vec compare_eq(const t_vec &b) const {
        return {data0.compare_eq(b.data0), data1.compare_eq(b.data1)};
    }

    FORCE_INLINE t_vec compare_eq_mask(const t_vec &b) const {
        return {data0.compare_eq_mask(b.data0), data1.compare_eq_mask(b.data1)};
    }

    FORCE_INLINE t_vec compare_neq(const t_vec &b) const {
        return {data0.compare_neq(b.data0), data1.compare_neq(b.data1)};
    }

    FORCE_INLINE t_vec compare_neq_mask(const t_vec &b) const {
        return {data0.compare_neq_mask(b.data0),
                data1.compare_neq_mask(b.data1)};
    }

    FORCE_INLINE t_vec compare_le(const t_vec &b) const {
        return {data0.compare_le(b.data0), data1.compare_le(b.data1)};
    }

    FORCE_INLINE t_vec compare_le_mask(const t_vec &b) const {
        return {data0.compare_le_mask(b.data0), data1.compare_le_mask(b.data1)};
    }

    FORCE_INLINE t_vec compare_ge(const t_vec &b) const {
        return {data0.compare_ge(b.data0), data1.compare_ge(b.data1)};
    }

    FORCE_INLINE t_vec compare_ge_mask(const t_vec &b) const {
        return {data0.compare_ge_mask(b.data0), data1.compare_ge_mask(b.data1)};
    }

    FORCE_INLINE t_vec select(const t_vec &b, const t_vec &mask) const {
        return {data0.select(b.data0, mask.data0),
                data1.select(b.data1, mask.data1)};
    }

    FORCE_INLINE bool operator==(const t_vec &b) const {
        return data0 == b.data0 && data1 == b.data1;
    }

    FORCE_INLINE bool operator<=(const t_vec &b) const {
        return data0 <= b.data0 && data1 <= b.data1;
    }

    FORCE_INLINE bool operator>=(const t_vec &b) const {
        return data0 >= b.data0 && data1 >= b.data1;
    }

    FORCE_INLINE bool operator!=(const t_vec &b) const {
        return data0 != b.data0 || data1 != b.data1;
    }

    FORCE_INLINE t_vec operator+=(const t_vec &b) {
        return {data0 += b.data0, data1 += b.data1};
    }

    FORCE_INLINE t_vec operator-=(const t_vec &b) {
        return {data0 -= b.data0, data1 -= b.data1};
    }

    FORCE_INLINE t_vec operator*=(const t_vec &b) {
        return {data0 *= b.data0, data1 *= b.data1};
    }

    FORCE_INLINE t_vec operator/=(const t_vec &b) {
        return {data0 /= b.data0, data1 /= b.data1};
    }

    FORCE_INLINE t_vec bitwise_and(const t_vec &b) const {
        return {data0.bitwise_and(b.data0), data1.bitwise_and(b.data1)};
    }

    FORCE_INLINE t_vec bitwise_or(const t_vec &b) const {
        return {data0.bitwise_or(b.data0), data1.bitwise_or(b.data1)};
    }

    FORCE_INLINE t_vec sum() const {
        const auto s = data0.sum() + data1.sum();
        return {s, s};
    }

    FORCE_INLINE t_vec dot(const t_vec &b) const {
        const auto d = data0.dot(b.data0) + data1.dot(b.data1);
        return {d, d};
    }

    FORCE_INLINE t_vec min(const t_vec &b) const {
        return {data0.min(b.data0), data1.min(b.data1)};
    }

    FORCE_INLINE t_vec max(const t_vec &b) const {
        return {data0.max(b.data0), data1.max(b.data1)};
    }

    FORCE_INLINE t_vec abs() const { return {data0.abs(), data1.abs()}; }
    FORCE_INLINE t_vec sign() const { return {data0.sign(), data1.sign()}; }
    FORCE_INLINE t_vec magnitude_squared() const { return dot(*this); }
    FORCE_INLINE t_vec sqrt() const { return {data0.sqrt(), data1.sqrt()}; }
    FORCE_INLINE t_vec magnitude() const { return magnitude_squared().sqrt(); }
    FORCE_INLINE void load(double *data) {
        this->data0.load(data);
        this->data1.load(data + 4);
    }

    FORCE_INLINE void extract(double *data) {
        this->data0.extract(data);
        this->data1.extract(data + 4);
    }

    FORCE_INLINE t_vec exp() const { return {data0.exp(), data1.exp()}; }
    FORCE_INLINE t_vec log() const { return {data0.log(), data1.log()}; }
    FORCE_INLINE t_vec pow(const t_vec &p) const {
        return {data0.pow(p.data0), data1.pow(p.data1)};
    }

    FORCE_INLINE t_vec and_mask(const t_vec &mask) const {
        return {_mm256_and_pd(mask.data0, data0),
                _mm256_and_pd(mask.data1, data1)};
    }

    FORCE_INLINE t_vec and_not_mask(const t_vec &mask) const {
        return {_mm256_andnot_pd(mask.data0, data0),
                _mm256_andnot_pd(mask.data1, data1)};
    }
};

template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
FORCE_INLINE vec<t_scalar_, t_size, t_enable_simd>
sqrt(const vec<t_scalar_, t_size, t_enable_simd> &v) {
    return v.sqrt();
}

template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
FORCE_INLINE vec<t_scalar_, t_size, t_enable_simd>
fmadd(const vec<t_scalar_, t_size, t_enable_simd> &a,
      const vec<t_scalar_, t_size, t_enable_simd> &b,
      const vec<t_scalar_, t_size, t_enable_simd> &c_add) {
    return a.madd(b, c_add);
}

template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
FORCE_INLINE vec<t_scalar_, t_size, t_enable_simd>
abs(const vec<t_scalar_, t_size, t_enable_simd> &v) {
    return v.abs();
}

template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
FORCE_INLINE vec<t_scalar_, t_size, t_enable_simd>
min(const vec<t_scalar_, t_size, t_enable_simd> &a,
    const vec<t_scalar_, t_size, t_enable_simd> &b) {
    return a.min(b);
}

template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
FORCE_INLINE vec<t_scalar_, t_size, t_enable_simd>
max(const vec<t_scalar_, t_size, t_enable_simd> &a,
    const vec<t_scalar_, t_size, t_enable_simd> &b) {
    return a.max(b);
}

template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
FORCE_INLINE vec<t_scalar_, t_size, t_enable_simd>
pow(const vec<t_scalar_, t_size, t_enable_simd> &a,
    const vec<t_scalar_, t_size, t_enable_simd> &b) {
    return a.pow(b);
}

template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
FORCE_INLINE vec<t_scalar_, t_size, t_enable_simd>
ternary(const vec<t_scalar_, t_size, t_enable_simd> &condition,
        const vec<t_scalar_, t_size, t_enable_simd> &a,
        const vec<t_scalar_, t_size, t_enable_simd> &b) {
    return a.and_mask(condition).bitwise_or(b.and_not_mask(condition));
}

ATG_MATH_DEFINE_LEFT_SCALAR_OPERATOR(*);

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
FORCE_INLINE unsigned int type_width<float>() {
    return 1;
}

template<>
FORCE_INLINE unsigned int type_width<double>() {
    return 1;
}

template<>
FORCE_INLINE unsigned int type_width<int>() {
    return 1;
}

typedef vec<float, 2, false> vec2_s;
typedef vec<int, 2, false> ivec2_s;
typedef vec<double, 2, false> dvec2_s;

typedef vec<float, 3, false> vec3_s;
typedef vec<uint8_t, 3, false> rgb_24;
typedef vec<int, 3, false> ivec3_s;
typedef vec<double, 3, false> dvec3_s;

typedef vec<float, 4, false> vec4_s, quat_s;
typedef vec<uint8_t, 4, false> rgba_32;
typedef vec<int, 4, false> ivec4_s;
typedef vec<double, 4, false> dvec4_s, dquat_s;

typedef vec<float, 4, true> vec4_v, quat_v;
typedef vec<float, 8, true> vec8_v;
typedef vec<double, 2, true> dvec2_v;
typedef vec<double, 4, true> dvec4_v;
typedef vec<double, 8, true> dvec8_v;

template<unsigned int t_size>
using vec_s = vec<float, t_size, false>;
template<unsigned int t_size>
using dvec_s = vec<double, t_size, false>;
template<unsigned int t_size>
using ivec_s = vec<int, t_size, false>;

#if ATG_MATH_USE_INTRINSICS
using vec4 = vec4_v;
#else
using vec4 = vec4_s;
#endif// ATG_MATH_USE_INTRINSICS

using vec2 = vec2_s;
using ivec2 = ivec2_s;
using dvec2 = dvec2_s;

using vec3 = vec3_s;
using ivec3 = ivec3_s;
using dvec3 = dvec3_s;

using quat = quat_s;
using ivec4 = ivec4_s;
using dvec4 = dvec4_s;
} /* namespace atg_math */

#endif /* ATG_MATH_ATG_MATH_H */
