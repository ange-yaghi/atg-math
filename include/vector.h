#ifndef ATG_MATH_VECTOR_H
#define ATG_MATH_VECTOR_H

#include <cmath>
#include <cstdint>

#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

namespace atg_math {
template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
struct vec {};

#define ATG_MATH_ALIAS(name, index)                                            \
    inline t_scalar &name() { return data[index]; }                            \
    inline t_scalar name() const { return data[index]; }

#define ATG_MATH_DEFINE_T_VEC typedef vec<t_scalar_, t_size, false> t_vec
#define ATG_MATH_DEFINE_T_SCALAR typedef t_scalar_ t_scalar

#define ATG_MATH_VEC_DEFINES(size)                                             \
    static constexpr unsigned int t_size = size;                               \
    ATG_MATH_DEFINE_T_VEC;                                                     \
    ATG_MATH_DEFINE_T_SCALAR;

#define ATG_MATH_DEFINE_COMPONENT_WISE_OPERATOR(op)                            \
    inline t_vec operator op(const t_vec &b) const {                           \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = data[i] op b.data[i];                             \
        }                                                                      \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_SCALAR_OPERATOR(op)                                    \
    inline t_vec operator op(t_scalar b) const {                               \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = data[i] op b;                                     \
        }                                                                      \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_LEFT_SCALAR_OPERATOR(op)                               \
    template<typename t_scalar, unsigned int t_size, bool t_simd>              \
    inline vec<t_scalar, t_size, t_simd> operator*(                            \
            typename vec<t_scalar, t_size, t_simd>::t_scalar left,             \
            const vec<t_scalar, t_size, t_simd> &right) {                      \
        return right op left;                                                  \
    }

#define ATG_MATH_DEFINE_BOOLEAN_COMPARISON_OPERATOR(op)                        \
    inline bool operator op(const t_vec &b) const {                            \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            if (!(data[i] op b.data[i])) { return false; }                     \
        }                                                                      \
        return true;                                                           \
    }

#define ATG_MATH_DEFINE_VECTOR_COMPARISON_OPERATOR(op, name)                   \
    inline t_vec compare_##name(const t_vec &b) const {                        \
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

#define ATG_MATH_DEFINE_NEGATED_BOOLEAN_COMPARISON_OPERATOR(op)                \
    inline bool operator op(const t_vec &b) const {                            \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            if (data[i] op b.data[i]) { return true; }                         \
        }                                                                      \
        return false;                                                          \
    }

#define ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(full_op, op)                       \
    inline t_vec &operator full_op(const t_vec &b) {                           \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            data[i] = data[i] op b.data[i];                                    \
        }                                                                      \
        return *this;                                                          \
    }

#define ATG_MATH_DEFINE_NEGATE_OPERATOR                                        \
    inline t_vec operator-() const {                                           \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = -data[i];                                         \
        }                                                                      \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_POSITIVE_OPERATOR                                      \
    inline t_vec operator+() const { return *this; }

#define ATG_MATH_DEFINE_DOT_PRODUCT                                            \
    inline t_vec dot(const t_vec &b) const {                                   \
        t_scalar result = 0;                                                   \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result += data[i] * b.data[i];                                     \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_SUM                                                    \
    inline t_vec sum() const {                                                 \
        t_scalar result = 0;                                                   \
        for (unsigned int i = 0; i < t_size; ++i) { result += data[i]; }       \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_COMPONENT_MIN                                          \
    inline t_vec min(const t_vec &b) const {                                   \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = std::min(data[i], b.data[i]);                     \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_COMPONENT_MAX                                          \
    inline t_vec max(const t_vec &b) const {                                   \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = std::max(data[i], b.data[i]);                     \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_ABS                                                    \
    inline t_vec abs() const {                                                 \
        t_vec result;                                                          \
        for (unsigned int i = 0; i < t_size; ++i) {                            \
            result.data[i] = std::abs(data[i]);                                \
        }                                                                      \
                                                                               \
        return result;                                                         \
    }

#define ATG_MATH_DEFINE_MAGNITUDE_SQUARED                                      \
    inline t_scalar magnitude_squared() const { return t_scalar(dot(*this)); }

#define ATG_MATH_DEFINE_MAGNITUDE                                              \
    inline t_scalar magnitude() const { return std::sqrt(magnitude_squared()); }

#define ATG_MATH_DEFINE_NORMALIZE                                              \
    inline t_vec normalize() const { return (*this) / magnitude(); }

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
    inline t_vec operator=(const t_b_type &b) {                                \
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
    inline t_vec operator=(const t_scalar &b) {                                \
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

#define ATG_MATH_DEFINE_EXPLICIT_SCALAR_CONVERSION                             \
    inline explicit operator t_scalar() const { return data[0]; }

#define ATG_MATH_S_X 0
#define ATG_MATH_S_Y 1
#define ATG_MATH_S_Z 2
#define ATG_MATH_S_W 3
#define ATG_MATH_M128_SHUFFLE(p0, p1, p2, p3)                                  \
    _MM_SHUFFLE((p3), (p2), (p1), (p0))

template<typename t_scalar_>
struct vec<t_scalar_, 1, false> {
    ATG_MATH_VEC_DEFINES(1)

    vec(t_scalar s) { data[0] = s; }

    ATG_MATH_ALIAS(s, 0)

    t_scalar data[t_size];

    ATG_MATH_DEFINE_CONVERSION_CONSTRUCTOR
    ATG_MATH_DEFINE_CONVERSION
    ATG_MATH_DEFINE_DEFAULT_CONSTRUCTOR

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

    ATG_MATH_DEFINE_NEGATE_OPERATOR
    ATG_MATH_DEFINE_POSITIVE_OPERATOR

    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(+=, +)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(-=, -)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(*=, *)
    ATG_MATH_DEFINE_ASSIGNMENT_OPERATOR(/=, /)
};

template<typename t_scalar_>
struct vec<t_scalar_, 2, false> {
    ATG_MATH_VEC_DEFINES(2)

    vec(t_scalar x, t_scalar y) {
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
};

template<typename t_scalar_>
struct vec<t_scalar_, 3, false> {
    ATG_MATH_VEC_DEFINES(3)

    vec(t_scalar x, t_scalar y, t_scalar z = 0) {
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
};

template<typename t_scalar_>
struct vec<t_scalar_, 4, false> {
    ATG_MATH_VEC_DEFINES(4)

    vec(t_scalar x, t_scalar y, t_scalar z, t_scalar w) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }

    vec(t_scalar x, t_scalar y) {
        data[0] = x;
        data[1] = y;
        data[2] = data[3] = 0;
    }

    vec(t_scalar x, t_scalar y, t_scalar z) {
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

    inline t_vec xy() const { return {x(), y(), t_scalar(0), t_scalar(0)}; }
    inline t_vec yz() const { return {y(), z(), t_scalar(0), t_scalar(0)}; }
    inline t_vec xz() const { return {x(), z(), t_scalar(0), t_scalar(0)}; }
    inline t_vec xyz() const { return {x(), y(), z(), t_scalar(0)}; }
    inline t_vec position() const { return {x(), y(), z(), t_scalar(1)}; }
};

template<typename t_scalar_, unsigned int t_size>
struct vec<t_scalar_, t_size, false> {
    ATG_MATH_DEFINE_T_VEC;
    ATG_MATH_DEFINE_T_SCALAR;

    t_scalar data[t_size];

    ATG_MATH_DEFINE_CONVERSION_CONSTRUCTOR
    ATG_MATH_DEFINE_CONVERSION
    ATG_MATH_DEFINE_SCALAR_CONSTRUCTOR
    ATG_MATH_DEFINE_DEFAULT_CONSTRUCTOR

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
};

template<>
struct vec<float, 4, true> {
    using t_scalar = float;
    static constexpr unsigned int t_size = 4;
    typedef vec<float, 4, true> t_vec;

    inline vec() { data[0] = data[1] = data[2] = data[3] = 0; }
    inline vec(const __m128 &v) : data_v(v) {}
    inline vec(float x, float y, float z = 0, float w = 0) {
        data_v = _mm_set_ps(w, z, y, x);
    }
    inline vec(float s) { data_v = _mm_set_ps1(s); }

    ATG_MATH_ALIAS(x, 0)
    ATG_MATH_ALIAS(y, 1)
    ATG_MATH_ALIAS(z, 2)
    ATG_MATH_ALIAS(w, 3)

    ATG_MATH_ALIAS(r, 0)
    ATG_MATH_ALIAS(g, 1)
    ATG_MATH_ALIAS(b, 2)
    ATG_MATH_ALIAS(a, 3)

    template<int i0 = 0, int i1 = 1, int i2 = 2, int i3 = 3>
    inline t_vec shuffle() const {
        return _mm_shuffle_ps(data_v, data_v,
                              ATG_MATH_M128_SHUFFLE(i0, i1, i2, i3));
    }

    union {
        int mask[4];
        float data[4];
        __m128 data_v;
    };

    inline explicit operator float() const { return _mm_cvtss_f32(data_v); }

    inline operator __m128() const { return data_v; }

    inline t_vec operator-() const {
        const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(~0x7FFFFFFF));
        return _mm_xor_ps(data_v, mask);
    }

    inline t_vec operator+() const { return *this; }

    inline t_vec operator+(const t_vec &b) const {
        return _mm_add_ps(data_v, b.data_v);
    }

    inline t_vec operator-(const t_vec &b) const {
        return _mm_sub_ps(data_v, b.data_v);
    }

    inline t_vec operator*(const t_vec &b) const {
        return _mm_mul_ps(data_v, b.data_v);
    }

    inline t_vec operator/(const t_vec &b) const {
        return _mm_div_ps(data_v, b.data_v);
    }

    inline t_vec compare_eq(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpeq_ps(data_v, b.data_v);
        return _mm_and_ps(cmp_mask, t_vec(1.0f));
    }

    inline t_vec compare_neq(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpneq_ps(data_v, b.data_v);
        return _mm_and_ps(cmp_mask, t_vec(1.0f));
    }

    inline t_vec compare_le(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmple_ps(data_v, b.data_v);
        return _mm_and_ps(cmp_mask, t_vec(1.0f));
    }

    inline t_vec compare_ge(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpge_ps(data_v, b.data_v);
        return _mm_and_ps(cmp_mask, t_vec(1.0f));
    }

    inline bool operator==(const t_vec &b) const {
        const t_vec cmp = _mm_cmpeq_ps(data_v, b.data_v);
        return !(cmp.x() == 0 || cmp.y() == 0 || cmp.z() == 0 || cmp.w() == 0);
    }

    inline bool operator<=(const t_vec &b) const {
        const t_vec cmp = _mm_cmple_ps(data_v, b.data_v);
        return !(cmp.x() == 0 || cmp.y() == 0 || cmp.z() == 0 || cmp.w() == 0);
    }

    inline bool operator>=(const t_vec &b) const {
        const t_vec cmp = _mm_cmpge_ps(data_v, b.data_v);
        return !(cmp.x() == 0 || cmp.y() == 0 || cmp.z() == 0 || cmp.w() == 0);
    }

    inline bool operator!=(const t_vec &b) const {
        const t_vec cmp = _mm_cmpeq_ps(data_v, b.data_v);
        return cmp.x() == 0 && cmp.y() == 0 && cmp.z() == 0 && cmp.w() == 0;
    }

    inline t_vec operator+=(const t_vec &b) {
        return data_v = _mm_add_ps(data_v, b.data_v);
    }

    inline t_vec operator-=(const t_vec &b) {
        return data_v = _mm_sub_ps(data_v, b.data_v);
    }

    inline t_vec operator*=(const t_vec &b) {
        return data_v = _mm_mul_ps(data_v, b.data_v);
    }

    inline t_vec operator/=(const t_vec &b) {
        return data_v = _mm_div_ps(data_v, b.data_v);
    }

    inline t_vec sum() const {
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

    inline t_vec dot(const t_vec &b) const {
        const __m128 t0 = _mm_mul_ps(data_v, b.data_v);
        return t_vec(t0).sum();
    }

    inline t_vec cross(const t_vec &b) const {
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

    inline t_vec min(const t_vec &b) const {
        return _mm_min_ps(data_v, b.data_v);
    }

    inline t_vec max(const t_vec &b) const {
        return _mm_max_ps(data_v, b.data_v);
    }

    inline t_vec abs() const {
        const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        return _mm_and_ps(data_v, mask);
    }

    inline t_vec magnitude_squared() const { return dot(*this); }
    inline t_vec sqrt() const { return _mm_sqrt_ps(data_v); }
    inline t_vec magnitude() const { return magnitude_squared().sqrt(); }
    inline t_vec normalize() const { return _mm_div_ps(data_v, magnitude()); }

    inline t_vec xy() const {
        return _mm_shuffle_ps(data_v, {0.0f, 0.0f, 0.0f, 0.0f},
                              ATG_MATH_M128_SHUFFLE(ATG_MATH_S_X, ATG_MATH_S_Y,
                                                    ATG_MATH_S_Z,
                                                    ATG_MATH_S_W));
    }

    inline t_vec yz() const {
        return _mm_shuffle_ps(data_v, {0.0f, 0.0f, 0.0f, 0.0f},
                              ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_Z,
                                                    ATG_MATH_S_Z,
                                                    ATG_MATH_S_W));
    }

    inline t_vec xz() const {
        return _mm_shuffle_ps(data_v, {0.0f, 0.0f, 0.0f, 0.0f},
                              ATG_MATH_M128_SHUFFLE(ATG_MATH_S_X, ATG_MATH_S_Z,
                                                    ATG_MATH_S_Z,
                                                    ATG_MATH_S_W));
    }

    inline t_vec xyz() const {
        return _mm_and_ps(data_v,
                          _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1)));
    }

    inline t_vec position() const {
        return _mm_or_ps(_mm_and_ps(data_v, _mm_castsi128_ps(_mm_set_epi32(
                                                    0, -1, -1, -1))),
                         {0.0f, 0.0f, 0.0f, 1.0f});
    }
};

template<>
struct vec<float, 8, true> {
    using t_scalar = float;
    static constexpr unsigned int t_size = 8;
    typedef vec<float, 8, true> t_vec;

    inline vec() { data_v = _mm256_set1_ps(0.0f); }
    inline vec(const __m256 &v) : data_v(v) {}
    inline vec(float s) { data_v = _mm256_set1_ps(s); }

    union {
        int mask[8];
        float data[8];
        __m256 data_v;
    };

    inline explicit operator float() const { return _mm256_cvtss_f32(data_v); }
    inline operator __m256() const { return data_v; }

    inline t_vec operator-() const {
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x7FFFFFFF));
        return _mm256_xor_ps(data_v, mask);
    }

    inline t_vec operator+() const { return *this; }

    inline t_vec operator+(const t_vec &b) const {
        return _mm256_add_ps(data_v, b.data_v);
    }

    inline t_vec operator-(const t_vec &b) const {
        return _mm256_sub_ps(data_v, b.data_v);
    }

    inline t_vec operator*(const t_vec &b) const {
        return _mm256_mul_ps(data_v, b.data_v);
    }

    inline t_vec operator/(const t_vec &b) const {
        return _mm256_div_ps(data_v, b.data_v);
    }

    inline t_vec compare_eq(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_ps(data_v, b.data_v, _CMP_EQ_OQ);
        return _mm256_and_ps(cmp_mask, t_vec(1.0f));
    }

    inline t_vec compare_neq(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_ps(data_v, b.data_v, _CMP_NEQ_OQ);
        return _mm256_and_ps(cmp_mask, t_vec(1.0f));
    }

    inline t_vec compare_le(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_ps(data_v, b.data_v, _CMP_LE_OQ);
        return _mm256_and_ps(cmp_mask, t_vec(1.0f));
    }

    inline t_vec compare_ge(const t_vec &b) const {
        const t_vec cmp_mask = _mm256_cmp_ps(data_v, b.data_v, _CMP_GE_OQ);
        return _mm256_and_ps(cmp_mask, t_vec(1.0f));
    }

    inline bool operator==(const t_vec &b) const {
        const t_vec cmp = _mm256_cmp_ps(data_v, b.data_v, _CMP_EQ_OQ);
        for (int i = 0; i < 8; ++i) {
            if (cmp.data[0] == 0) { return false; }
        }

        return true;
    }

    inline bool operator<=(const t_vec &b) const {
        const t_vec cmp = _mm256_cmp_ps(data_v, b.data_v, _CMP_LE_OQ);
        for (int i = 0; i < 8; ++i) {
            if (cmp.data[0] == 0) { return false; }
        }

        return true;
    }

    inline bool operator>=(const t_vec &b) const {
        const t_vec cmp = _mm256_cmp_ps(data_v, b.data_v, _CMP_GE_OQ);
        for (int i = 0; i < 8; ++i) {
            if (cmp.data[0] == 0) { return false; }
        }

        return true;
    }

    inline bool operator!=(const t_vec &b) const {
        const t_vec cmp = _mm256_cmp_ps(data_v, b.data_v, _CMP_NEQ_OQ);
        for (int i = 0; i < 8; ++i) {
            if (cmp.data[0] == 0) { return false; }
        }

        return true;
    }

    inline t_vec operator+=(const t_vec &b) {
        return data_v = _mm256_add_ps(data_v, b.data_v);
    }

    inline t_vec operator-=(const t_vec &b) {
        return data_v = _mm256_sub_ps(data_v, b.data_v);
    }

    inline t_vec operator*=(const t_vec &b) {
        return data_v = _mm256_mul_ps(data_v, b.data_v);
    }

    inline t_vec operator/=(const t_vec &b) {
        return data_v = _mm256_div_ps(data_v, b.data_v);
    }

    inline t_vec min(const t_vec &b) const {
        return _mm256_min_ps(data_v, b.data_v);
    }

    inline t_vec max(const t_vec &b) const {
        return _mm256_max_ps(data_v, b.data_v);
    }

    inline t_vec abs() const {
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        return _mm256_and_ps(data_v, mask);
    }

    inline t_vec sum() const {
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

    inline t_vec sqrt() const { return _mm256_sqrt_ps(data_v); }
};

template<>
struct vec<double, 2, true> {
    using t_scalar = double;
    static constexpr unsigned int t_size = 2;
    typedef vec<double, 2, true> t_vec;

    inline vec() { data[0] = data[1] = 0; }
    inline vec(const __m128d &v) : data_v(v) {}
    inline vec(double x, double y) { data_v = _mm_set_pd(x, y); }
    inline vec(double s) { data_v = _mm_set_pd1(s); }

    ATG_MATH_ALIAS(x, 0)
    ATG_MATH_ALIAS(y, 1)

    ATG_MATH_ALIAS(u, 0)
    ATG_MATH_ALIAS(v, 1)

    ATG_MATH_ALIAS(w, 0)
    ATG_MATH_ALIAS(h, 1)

    template<int i0 = 0, int i1 = 1>
    inline t_vec shuffle() const {
        return _mm_shuffle_pd(data_v, data_v,
                              ATG_MATH_M128_SHUFFLE(i0, i1, 0, 0));
    }

    union {
        int mask[2];
        double data[2];
        __m128d data_v;
    };

    inline explicit operator double() const { return _mm_cvtsd_f64(data_v); }

    inline operator __m128d() const { return data_v; }

    inline t_vec operator-() const {
        const __m128d mask =
                _mm_castsi128_pd(_mm_set1_epi64x(~0x7FFFFFFFFFFFFFFF));
        return _mm_xor_pd(data_v, mask);
    }

    inline t_vec operator+() const { return *this; }

    inline t_vec operator+(const t_vec &b) const {
        return _mm_add_pd(data_v, b.data_v);
    }

    inline t_vec operator-(const t_vec &b) const {
        return _mm_sub_pd(data_v, b.data_v);
    }

    inline t_vec operator*(const t_vec &b) const {
        return _mm_mul_pd(data_v, b.data_v);
    }

    inline t_vec operator/(const t_vec &b) const {
        return _mm_div_pd(data_v, b.data_v);
    }

    inline t_vec compare_eq(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpeq_pd(data_v, b.data_v);
        return _mm_and_pd(cmp_mask, t_vec(1.0));
    }

    inline t_vec compare_neq(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpneq_pd(data_v, b.data_v);
        return _mm_and_pd(cmp_mask, t_vec(1.0));
    }

    inline t_vec compare_le(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmple_pd(data_v, b.data_v);
        return _mm_and_pd(cmp_mask, t_vec(1.0));
    }

    inline t_vec compare_ge(const t_vec &b) const {
        const t_vec cmp_mask = _mm_cmpge_pd(data_v, b.data_v);
        return _mm_and_pd(cmp_mask, t_vec(1.0));
    }

    inline bool operator==(const t_vec &b) const {
        const t_vec cmp = _mm_cmpeq_pd(data_v, b.data_v);
        return !(cmp.x() == 0 || cmp.y() == 0);
    }

    inline bool operator<=(const t_vec &b) const {
        const t_vec cmp = _mm_cmple_pd(data_v, b.data_v);
        return !(cmp.x() == 0 || cmp.y() == 0);
    }

    inline bool operator>=(const t_vec &b) const {
        const t_vec cmp = _mm_cmpge_pd(data_v, b.data_v);
        return !(cmp.x() == 0 || cmp.y() == 0);
    }

    inline bool operator!=(const t_vec &b) const {
        const t_vec cmp = _mm_cmpeq_pd(data_v, b.data_v);
        return cmp.x() == 0 && cmp.y() == 0;
    }

    inline t_vec operator+=(const t_vec &b) {
        return data_v = _mm_add_pd(data_v, b.data_v);
    }

    inline t_vec operator-=(const t_vec &b) {
        return data_v = _mm_sub_pd(data_v, b.data_v);
    }

    inline t_vec operator*=(const t_vec &b) {
        return data_v = _mm_mul_pd(data_v, b.data_v);
    }

    inline t_vec operator/=(const t_vec &b) {
        return data_v = _mm_div_pd(data_v, b.data_v);
    }

    inline t_vec sum() const {
        const __m128d t1 = _mm_shuffle_pd(
                data_v, data_v,
                ATG_MATH_M128_SHUFFLE(ATG_MATH_S_Y, ATG_MATH_S_X, 0, 0));
        return _mm_add_pd(data_v, t1);
    }

    inline t_vec dot(const t_vec &b) const {
        const __m128d t0 = _mm_mul_pd(data_v, b.data_v);
        return t_vec(t0).sum();
    }

    inline t_vec min(const t_vec &b) const {
        return _mm_min_pd(data_v, b.data_v);
    }

    inline t_vec max(const t_vec &b) const {
        return _mm_max_pd(data_v, b.data_v);
    }

    inline t_vec abs() const {
        const __m128d mask =
                _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        return _mm_and_pd(data_v, mask);
    }

    inline t_vec magnitude_squared() const { return dot(*this); }
    inline t_vec sqrt() const { return _mm_sqrt_pd(data_v); }
    inline t_vec magnitude() const { return magnitude_squared().sqrt(); }
    inline t_vec normalize() const { return _mm_div_pd(data_v, magnitude()); }
};

ATG_MATH_DEFINE_LEFT_SCALAR_OPERATOR(*);

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
