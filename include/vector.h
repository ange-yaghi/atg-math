#ifndef ATG_MATH_VECTOR_H
#define ATG_MATH_VECTOR_H

#include <cmath>
#include <cstdint>

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

#define DEFINE_DOT_PRODUCT \
        inline t_scalar dot(const t_vec &b) const { \
            t_scalar result = 0; \
            for (unsigned int i = 0; i < t_size; ++i) { \
                result += data[i] * b.data[i]; \
            } \
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

#define DEFINE_CONVERSION \
        template<typename t_b_type> \
        t_vec operator=(const t_b_type &b) { \
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
        t_vec operator=(const t_scalar &b) { \
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

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)

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

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)

        DEFINE_ASSIGNMENT_OPERATOR(+=, +)
        DEFINE_ASSIGNMENT_OPERATOR(-=, -)
        DEFINE_ASSIGNMENT_OPERATOR(*=, *)
        DEFINE_ASSIGNMENT_OPERATOR(/=, /)

        DEFINE_DOT_PRODUCT
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

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)

        DEFINE_ASSIGNMENT_OPERATOR(+=, +)
        DEFINE_ASSIGNMENT_OPERATOR(-=, -)
        DEFINE_ASSIGNMENT_OPERATOR(*=, *)
        DEFINE_ASSIGNMENT_OPERATOR(/=, /)

        DEFINE_DOT_PRODUCT
        DEFINE_MAGNITUDE_SQUARED
        DEFINE_MAGNITUDE
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

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)

        DEFINE_ASSIGNMENT_OPERATOR(+=, +)
        DEFINE_ASSIGNMENT_OPERATOR(-=, -)
        DEFINE_ASSIGNMENT_OPERATOR(*=, *)
        DEFINE_ASSIGNMENT_OPERATOR(/=, /)

        DEFINE_DOT_PRODUCT
        DEFINE_MAGNITUDE_SQUARED
        DEFINE_MAGNITUDE
    };

    template<typename t_scalar, unsigned int t_size>
    struct vec<t_scalar, t_size, false> {
        DEFINE_T_VEC;

        t_scalar data[t_size];

        DEFINE_CONVERSION
        DEFINE_SCALAR_CONSTRUCTOR
        DEFINE_DEFAULT_CONSTRUCTOR

        DEFINE_COMPONENT_WISE_OPERATOR(+)
        DEFINE_COMPONENT_WISE_OPERATOR(-)
        DEFINE_COMPONENT_WISE_OPERATOR(*)
        DEFINE_COMPONENT_WISE_OPERATOR(/)

        DEFINE_ASSIGNMENT_OPERATOR(+=, +)
        DEFINE_ASSIGNMENT_OPERATOR(-=, -)
        DEFINE_ASSIGNMENT_OPERATOR(*=, *)
        DEFINE_ASSIGNMENT_OPERATOR(/=, /)

        DEFINE_DOT_PRODUCT
        DEFINE_MAGNITUDE_SQUARED
        DEFINE_MAGNITUDE
    };

    typedef vec<float, 2, false> vec2_s;
    typedef vec<int, 2, false> ivec2_s;
    typedef vec<double, 2, false> dvec2_s;

    typedef vec<float, 3, false> vec3_s;
    typedef vec<uint8_t, 3, false> rgb24;
    typedef vec<int, 3, false> ivec3_s;
    typedef vec<double, 3, false> dvec3_s;

    typedef vec<float, 4, false> vec4_s;
    typedef vec<uint8_t, 4, false> rgba32;
    typedef vec<int, 4, false> ivec4_s;
    typedef vec<double, 4, false> dvec4_s;

    template<unsigned int t_size> using vec_s = vec<float, t_size, false>;
    template<unsigned int t_size> using dvec_s = vec<double, t_size, false>;
    template<unsigned int t_size> using ivec_s = vec<int, t_size, false>;
};

#endif /* ATG_MATH_ATG_MATH_H */
