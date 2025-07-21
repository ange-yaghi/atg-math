#ifndef ATG_ATG_MATH_COMPLEX_HPP
#define ATG_ATG_MATH_COMPLEX_HPP

#include "functions.h"

namespace atg_math {
template<typename T_Real>
class complex {
public:
    inline complex() : m_r(0), m_i(0) {}
    inline complex(T_Real r, T_Real i = 0) : m_r(r), m_i(i) {}
    inline ~complex() {}

    inline explicit operator T_Real() const { return abs(); }
    inline T_Real re() const { return m_r; }
    inline T_Real im() const { return m_i; }

    inline bool operator==(const complex &x) const {
        return (m_r == x.m_r && m_i == x.m_i);
    }

    inline complex operator+(const complex &b) const {
        return complex(m_r + b.m_r, m_i + b.m_i);
    }

    inline complex operator+=(const complex &b) { return *this = *this + b; }

    inline complex operator-(const complex &b) const {
        return complex(m_r - b.m_r, m_i - b.m_i);
    }

    inline complex operator-=(const complex &b) { return *this = *this - b; }

    inline complex operator*(const complex &b) const {
        return complex(m_r * b.m_r - m_i * b.m_i, m_r * b.m_i + m_i * b.m_r);
    }

    inline complex operator*=(const complex &b) { return *this = *this * b; }

    inline complex operator*(const T_Real &v) const {
        return complex(m_r * v, m_i * v);
    }

    inline complex operator*=(const T_Real &b) { return *this = *this * b; }

    inline complex operator/(const T_Real &v) const {
        return complex(m_r / v, m_i / v);
    }

    inline complex operator/=(const T_Real &b) { return *this = *this / b; }

    inline complex operator/(const complex &b) const {
        return (*this) * b.inverse();
    }

    inline complex operator/=(const complex &b) { return *this = *this / b; }

    inline complex conjugate() const { return complex(m_r, -m_i); }
    inline T_Real abs() const { return sqrt(abs_2()); }
    inline T_Real abs_2() const { return squared(m_r) + squared(m_i); }
    inline complex inverse() const { return conjugate() / abs_2(); }

private:
    T_Real m_r, m_i;
};

template<typename T_Real>
inline complex<T_Real> abs(const complex<T_Real> &v) {
    return v.abs();
}

template<typename T_Real>
inline complex<T_Real> exp_i(const T_Real &v) {
    return complex<T_Real>(std::cos(v), std::sin(v));
}

template<typename T_Real>
inline complex<T_Real> exp(const complex<T_Real> &x) {
    return std::exp(x.re()) * exp_i(x.im());
}

}// namespace atg_math

#endif /* ATG_ATG_MATH_COMPLEX_HPP */
