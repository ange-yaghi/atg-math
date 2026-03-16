#ifndef ATG_ATG_MATH_INTERSECTION_HPP
#define ATG_ATG_MATH_INTERSECTION_HPP

#include "vector.h"

namespace atg_math {

template<typename T_Vec>
T_Vec linePlaneIntersection(const T_Vec &l0, const T_Vec &l, const T_Vec &p0,
                            const T_Vec &n) {
    const T_Vec d = (p0 - l0).dot(n) / (l.dot(n));
    return l0 + l * d;
}

template<typename T_Vec>
T_Vec rayPlaneIntersection(const T_Vec &l0, const T_Vec &l, const T_Vec &p0,
                           const T_Vec &n, bool *intersects) {
    const T_Vec d = (p0 - l0).dot(n) / (l.dot(n));
    *intersects = bool(d > T_Vec(0));
    return l0 + l * d;
}

}// namespace atg_math

#endif /* ATG_ATG_MATH_INTERSECTION_HPP */
