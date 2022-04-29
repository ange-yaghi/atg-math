#ifndef ATG_MATH_VECTOR_H
#define ATG_MATH_VECTOR_H

#include <cmath>
#include <cstdint>
#include <xmmintrin.h>
#include <emmintrin.h>

namespace atg_math {
    template<typename t_scalar, unsigned int t_size, bool t_enable_simd>
    struct vec { /* void */ };

#define DEFINE_T_VEC \
        typedef vec<t_scalar, t_size, false> t_vec \

#define VEC_DEFINES(size)  \
        static constexpr unsigned int t_size = size; \
        DEFINE_T_VEC;

#define DEFINE_COMPONENT_WISE_OPERATOR(op) \
        inline t_vec operator op(const t_vec &b) const { \
            t_vec result; \
            for (unsigned int i = 0; i < t_size; ++i) { \
                result.data[i] = data[i] op b.data[i]; \
            } \
            return result; \
        }

#define DEFINE_ASSIGNMENT_OPERATOR(full_op, op) \
        inline t_vec &operator full_op(const t_vec &b) { \
            for (unsigned int i = 0; i < t_size; ++i) { \
                data[i] = data[i] op b.data[i]; \
            } \
            return *this; \
        }

#define DEFINE_NEGATE_OPERATOR \
        inline t_vec operator-() const { \
            t_vec result; \
            for (unsigned int i = 0; i < t_size; ++i) { \
                result.data[i] = -data[i]; \
            } \
            return result; \
        }

#define DEFINE_POSITIVE_OPERATOR \
        inline t_vec operator+() const { \
            return *this; \
        }

#define DEFINE_DOT_PRODUCT \
        inline t_vec dot(const t_vec &b) const { \
            t_scalar result = 0; \
            for (unsigned int i = 0; i < t_size; ++i) { \
                result += data[i] * b.data[i]; \
            } \
            \
            return result; \
        }

#define DEFINE_COMPONENT_MIN \
        inline t_vec min(const t_vec &b) const { \
            t_vec result; \
            for (unsigned int i = 0; i < t_size; ++i) { \
                result.data[i] = std::min(data[i], b.data[i]); \
            } \
            \
            return result; \
        }

#define DEFINE_COMPONENT_MAX \
        inline t_vec max(const t_vec &b) const { \
            t_vec result; \
            for (unsigned int i = 0; i < t_size; ++i) { \
                result.data[i] = std::max(data[i], b.data[i]); \
            } \
            \
            return result; \
        }

#define DEFINE_ABS \
        inline t_vec abs() const { \
            t_vec result; \
            for (unsigned int i = 0; i < t_size; ++i) { \
                result.data[i] = std::abs(data[i]); \
            } \
            \
            return result; \
        }

#define DEFINE_MAGNITUDE_SQUARED \
        inline t_scalar magnitude_squared() const { \
            return dot(*this); \
        }

#define DEFINE_MAGNITUDE \
        inline t_scalar magnitude() const { \
            return std::sqrt(magnitude_squared()); \
        }

#define DEFINE_NORMALIZE \
        inline t_scalar normalize() const { \
            return (*this) / magnitude(); \
        }

#define DEFINE_CONVERSION \
        template<typename t_b_type> \
        inline t_vec operator=(const t_b_type &b) { \
            constexpr unsigned int l = (t_size < t_b_type::t_size) \
                ? t_size \
                : t_b_type::t_size; \
            for (unsigned int i = 0; i < t_size; ++i) { \
                data[i] = b.data[i]; \
            } \
            \
            for (unsigned int i = l; i < t_size; ++i) { \
                data[i] = 0; \
            } \
            \
            return *this; \
        } \
        \
        template<> \
        inline t_vec operator=(const t_scalar &b) { \
            for (unsigned int i = 0; i < t_size; ++i) { \
                data[i] = b; \
            } \
            \
            return *this; \
        }

#define DEFINE_SCALAR_CONSTRUCTOR \
        vec(t_scalar s) { \
            for (unsigned int i = 0; i < t_size; ++i) { \
                data[i] = s; \
            } \
        }

#define DEFINE_DEFAULT_CONSTRUCTOR \
        vec() { \
            /* void */ \
        }

#define DEFINE_EXPLICIT_SCALAR_CONVERSION \
        inline explicit operator float() const { return data[0]; }

#define S_X 0
#define S_Y 1
#define S_Z 2
#define S_W 3
#define M128_SHUFFLE(p0, p1, p2, p3) _MM_SHUFFLE((p3), (p2), (p1), (p0))

    template<typename t_scalar>
    struct vec<t_scalar, 1, false> {
        VEC_DEFINES(1)

        vec(t_scalar s) : s(s) {
            /* void */
        }

        union {
            struct { t_scalar s; };
            t_scalar data[t_size];
        };

        DEFINE_CONVERSION
        DEFINE_DEFAULT_CONSTRUCTOR

        DEFINE_EXPLICIT_SCALAR_CONVERSION

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)

        DEFINE_NEGATE_OPERATOR
        DEFINE_POSITIVE_OPERATOR

        DEFINE_ASSIGNMENT_OPERATOR(+=, +)
        DEFINE_ASSIGNMENT_OPERATOR(-=, -)
        DEFINE_ASSIGNMENT_OPERATOR(*=, *)
        DEFINE_ASSIGNMENT_OPERATOR(/=, /)
    };

    template<typename t_scalar>
    struct vec<t_scalar, 2, false> {
        VEC_DEFINES(2)

        vec(t_scalar x, t_scalar y) : x(x), y(y) {
            /* void */
        }

        union {
            struct { t_scalar s, t; };
            struct { t_scalar u, v; };
            struct { t_scalar x, y; };
            t_scalar data[t_size];
        };

        DEFINE_CONVERSION
        DEFINE_SCALAR_CONSTRUCTOR
        DEFINE_DEFAULT_CONSTRUCTOR

        DEFINE_EXPLICIT_SCALAR_CONVERSION

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)

        DEFINE_NEGATE_OPERATOR
        DEFINE_POSITIVE_OPERATOR

        DEFINE_ASSIGNMENT_OPERATOR(+=, +)
        DEFINE_ASSIGNMENT_OPERATOR(-=, -)
        DEFINE_ASSIGNMENT_OPERATOR(*=, *)
        DEFINE_ASSIGNMENT_OPERATOR(/=, /)

        DEFINE_DOT_PRODUCT

        DEFINE_COMPONENT_MIN
        DEFINE_COMPONENT_MAX

        DEFINE_ABS
        DEFINE_MAGNITUDE
        DEFINE_MAGNITUDE_SQUARED
        DEFINE_NORMALIZE
    };

    template<typename t_scalar>
    struct vec<t_scalar, 3, false> {
        VEC_DEFINES(3);

        vec(t_scalar x, t_scalar y, t_scalar z) : x(x), y(y), z(z) {
            /* void */
        }

        union {
            struct { t_scalar u, v, w; };
            struct { t_scalar x, y, z; };
            struct { t_scalar r, g, b; };
            t_scalar data[t_size];
        };

        t_vec cross(const t_vec &b) const {
            return {
                y * b.z - z * b.y,
                z * b.x - x * b.z,
                x * b.y - y * b.x
            };
        }

        DEFINE_CONVERSION
        DEFINE_SCALAR_CONSTRUCTOR
        DEFINE_DEFAULT_CONSTRUCTOR

        DEFINE_EXPLICIT_SCALAR_CONVERSION

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)
        
        DEFINE_NEGATE_OPERATOR
        DEFINE_POSITIVE_OPERATOR

        DEFINE_ASSIGNMENT_OPERATOR(+=, +)
        DEFINE_ASSIGNMENT_OPERATOR(-=, -)
        DEFINE_ASSIGNMENT_OPERATOR(*=, *)
        DEFINE_ASSIGNMENT_OPERATOR(/=, /)

        DEFINE_DOT_PRODUCT
        DEFINE_COMPONENT_MIN
        DEFINE_COMPONENT_MAX

        DEFINE_ABS
        DEFINE_MAGNITUDE_SQUARED
        DEFINE_MAGNITUDE
        DEFINE_NORMALIZE
    };

    template<typename t_scalar>
    struct vec<t_scalar, 4, false> {
        VEC_DEFINES(4);

        vec(t_scalar x, t_scalar y, t_scalar z, t_scalar w) : x(x), y(y), z(z), w(w) {
            /* void */
        }

        union {
            struct { t_scalar x, y, z, w; };
            struct { t_scalar r, g, b, a; };
            t_scalar data[t_size];
        };

        t_vec cross(const t_vec &b) const {
            return {
                y * b.z - z * b.y,
                z * b.x - x * b.z,
                x * b.y - y * b.x
            };
        }

        DEFINE_CONVERSION
        DEFINE_SCALAR_CONSTRUCTOR
        DEFINE_DEFAULT_CONSTRUCTOR

        DEFINE_EXPLICIT_SCALAR_CONVERSION

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)

        DEFINE_NEGATE_OPERATOR
        DEFINE_POSITIVE_OPERATOR

        DEFINE_ASSIGNMENT_OPERATOR(+=, +)
        DEFINE_ASSIGNMENT_OPERATOR(-=, -)
        DEFINE_ASSIGNMENT_OPERATOR(*=, *)
        DEFINE_ASSIGNMENT_OPERATOR(/=, /)

        DEFINE_DOT_PRODUCT
        DEFINE_COMPONENT_MIN
        DEFINE_COMPONENT_MAX

        DEFINE_ABS
        DEFINE_MAGNITUDE_SQUARED
        DEFINE_MAGNITUDE
        DEFINE_NORMALIZE
    };

    template<typename t_scalar, unsigned int t_size>
    struct vec<t_scalar, t_size, false> {
        DEFINE_T_VEC;

        t_scalar data[t_size];

        DEFINE_CONVERSION
        DEFINE_SCALAR_CONSTRUCTOR
        DEFINE_DEFAULT_CONSTRUCTOR

        DEFINE_EXPLICIT_SCALAR_CONVERSION

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)

        DEFINE_NEGATE_OPERATOR
        DEFINE_POSITIVE_OPERATOR

        DEFINE_ASSIGNMENT_OPERATOR(+=, +)
        DEFINE_ASSIGNMENT_OPERATOR(-=, -)
        DEFINE_ASSIGNMENT_OPERATOR(*=, *)
        DEFINE_ASSIGNMENT_OPERATOR(/=, /)

        DEFINE_DOT_PRODUCT
        DEFINE_COMPONENT_MIN
        DEFINE_COMPONENT_MAX

        DEFINE_ABS
        DEFINE_MAGNITUDE_SQUARED
        DEFINE_MAGNITUDE
        DEFINE_NORMALIZE
    };

    template<>
    struct vec<float, 4, true> {
        typedef vec<float, 4, true> t_vec;

        inline vec() { /* void */ }
        inline vec(const __m128 &v) : data_v(v) { /* void */ }
        inline vec(float x, float y, float z, float w = 0) {
            data_v = _mm_set_ps(w, z, y, x);
        }
        inline vec(float s) {
            data_v = _mm_set_ps1(s);
        }

        union {
            struct { float x, y, z, w; };
            struct { float r, g, b, a; };
            float data[4];
            __m128 data_v;
        };

        inline explicit operator float() const {
            return _mm_cvtss_f32(data_v);
        }

        inline t_vec operator-() const {
            const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
            return _mm_xor_ps(data_v, mask);
        }

        inline t_vec operator+() const {
            return *this;
        }

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
            const __m128 t1 = _mm_shuffle_ps(data_v, data_v, M128_SHUFFLE(S_Z, S_W, S_X, S_Y));
            const __m128 t2 = _mm_add_ps(data_v, t1);
            const __m128 t3 = _mm_shuffle_ps(t2, t2, M128_SHUFFLE(S_Y, S_X, S_Z, S_W));
            return _mm_add_ps(t3, t2);
        }

        inline t_vec dot(const t_vec &b) const {
            const __m128 t0 = _mm_mul_ps(data_v, b.data_v);
            return t_vec(t0).sum();
        }

        inline t_vec cross(const t_vec &b) const {  
            const __m128 t1 = _mm_shuffle_ps(data_v, data_v, M128_SHUFFLE(S_Y, S_Z, S_X, S_W));
            const __m128 t2 = _mm_shuffle_ps(b.data_v, b.data_v, M128_SHUFFLE(S_Z, S_X, S_Y, S_W));
            const __m128 t1_t2 = _mm_mul_ps(t1, t2);

            const __m128 t3 = _mm_shuffle_ps(data_v, data_v, M128_SHUFFLE(S_Z, S_X, S_Y, S_W));
            const __m128 t4 = _mm_shuffle_ps(b.data_v, b.data_v, M128_SHUFFLE(S_Y, S_Z, S_X, S_W));
            const __m128 t3_t4 = _mm_mul_ps(t3, t4);

            return _mm_sub_ps(t1_t2, t3_t4);
        }

        inline t_vec min(const t_vec &b) const {
            return _mm_min_ps(data_v, b.data_v);
        }

        inline t_vec max(const t_vec &b) const {
            return _mm_max_ps(data_v, b.data_v);
        }

        inline t_vec abs() const {
            return max(-(*this));
        }

        inline t_vec magnitude_squared() const {
            return dot(*this);
        }

        inline t_vec sqrt() const {
            return _mm_sqrt_ps(data_v);
        }

        inline t_vec magnitude() const {
            return magnitude_squared().sqrt();
        }
    };

    typedef vec<float, 2, false> vec2_s;
    typedef vec<int, 2, false> ivec2_s;
    typedef vec<double, 2, false> dvec2_s;

    typedef vec<float, 3, false> vec3_s;
    typedef vec<uint8_t, 3, false> rgb_24;
    typedef vec<int, 3, false> ivec3_s;
    typedef vec<double, 3, false> dvec3_s;

    typedef vec<float, 4, false> vec4_s;
    typedef vec<uint8_t, 4, false> rgba_32;
    typedef vec<int, 4, false> ivec4_s;
    typedef vec<double, 4, false> dvec4_s;

    typedef vec<float, 4, true> vec4_v;

    template<unsigned int t_size> using vec_s = vec<float, t_size, false>;
    template<unsigned int t_size> using dvec_s = vec<double, t_size, false>;
    template<unsigned int t_size> using ivec_s = vec<int, t_size, false>;
} /* namespace atg_math */

#endif /* ATG_MATH_ATG_MATH_H */
