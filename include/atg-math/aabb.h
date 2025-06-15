#ifndef ATG_MATH_AABB_H
#define ATG_MATH_AABB_H

#include "functions.h"
#include "vector.h"

namespace atg_math {

template<typename t_scalar_, unsigned int t_size, bool t_enable_simd>
struct aabb {};

template<typename t_scalar_, bool t_enable_simd>
struct aabb<t_scalar_, 2, t_enable_simd> {
    using vec = vec<t_scalar_, 2, t_enable_simd>;
    using t_scalar = vec::t_scalar;

    static constexpr t_scalar l = t_scalar(0.0);
    static constexpr t_scalar r = t_scalar(1.0);
    static constexpr t_scalar m = t_scalar(0.5);
    static constexpr t_scalar t = t_scalar(1.0);
    static constexpr t_scalar b = t_scalar(0.0);
    static constexpr t_scalar z = t_scalar(0.0);

    static constexpr vec c = {m, m};
    static constexpr vec tl = {l, t}, tr = {r, t}, tm = {m, t};
    static constexpr vec bl = {l, b}, br = {r, b}, bm = {m, b};
    static constexpr vec lm = {l, m}, rm = {r, m};

    vec m0;
    vec m1;

    inline constexpr aabb() {}
    inline constexpr aabb(const vec &m0, const vec &m1) : m0(m0), m1(m1) {}
    inline constexpr aabb(t_scalar x0, t_scalar x1, t_scalar y0, t_scalar y1) {
        const vec p0 = {x0, y0};
        const vec p1 = {x1, y1};
        m0 = p0.min(p1);
        m1 = p0.max(p1);
    }

    inline constexpr aabb(const vec &pos, const vec &size, const vec &ref) {
        m0 = z;
        m1 = size;
        setPosition(pos, ref);
    }

    inline bool overlaps(const vec &p) const { return (p >= m0 && p <= m1); }
    inline vec normalize(const vec &g) const { return (g - m0) / size(); }
    inline aabb reorient() const { return {m0.min(m1), m0.max(m1)}; }

    inline aabb add(const aabb &b) const {
        return {m0.min(b.m0), m1.max(b.m1)};
    }

    inline aabb add(const vec &b) const { return {m0.min(b), m1.max(b)}; }

    inline void addReflexive(const vec &b) {
        m0 = m0.min(b);
        m1 = m1.max(b);
    }

    inline aabb clamp(const aabb &b) {
        aabb clamped = {m0.max(b.m0), m1.min(b.m1)};
        if (clamped.width() <= 0) {
            const t_scalar center =
                    t_scalar(0.5) * (clamped.m0.x() + clamped.m1.x());
            clamped.m0.x() = clamped.m1.x() = center;
        }

        if (clamped.height() <= 0) {
            const t_scalar center =
                    t_scalar(0.5) * (clamped.m0.y() + clamped.m1.y());
            clamped.m0.y() = clamped.m1.y() = center;
        }

        return clamped;
    }

    inline aabb clampHeight(t_scalar minHeight, t_scalar maxHeight,
                            const vec &origin = c) {
        const t_scalar newHeight =
                atg_math::clamp(height(), minHeight, maxHeight);
        return aabb(width(), newHeight, position(origin), origin);
    }

    inline aabb clampWidth(t_scalar minWidth, t_scalar maxWidth,
                           const vec &origin = c) {
        const t_scalar newWidth = atg_math::clamp(width(), minWidth, maxWidth);
        return aabb(newWidth, height(), position(origin), origin);
    }

    inline aabb move(const vec &delta) {
        m0 += delta;
        m1 += delta;

        return *this;
    }

    inline void setPosition(const vec &pos, const vec &ref = tl) {
        move(pos - position(ref));
    }

    inline vec center() const { return position(c); }

    inline vec position(const vec &ref = tl) const {
        const vec offset = ref * size();
        return m0 + offset;
    }

    inline t_scalar left() const { return m0.x(); }
    inline t_scalar right() const { return m1.x(); }
    inline t_scalar top() const { return m1.y(); }
    inline t_scalar bottom() const { return m0.y(); }
    inline t_scalar center_h() const { return (m0.x() + m1.x()) / 2; }
    inline t_scalar center_v() const { return (m0.y() + m1.y()) / 2; }

    inline vec bottomLeft() const { return {left(), bottom()}; }
    inline vec bottomRight() const { return {right(), bottom()}; }
    inline vec topLeft() const { return {left(), top()}; }
    inline vec topRight() const { return {right(), top()}; }

    inline t_scalar width() const { return m1.x() - m0.x(); }
    inline t_scalar height() const { return m1.y() - m0.y(); }
    inline vec size() const { return vec(width(), height()); }

    inline aabb inset(t_scalar amount) const {
        return {m0 + amount, m1 - amount};
    }

    inline aabb horizontalInset(t_scalar amount) const {
        return {m0 + vec{amount, z}, m1 - vec{amount, z}};
    }

    inline aabb verticalInset(t_scalar amount) const {
        return {m0 + vec{z, amount}, m1 - vec{z, amount}};
    }

    inline aabb bottomInset(t_scalar amount) const {
        return {m0 + vec{z, amount}, m1};
    }

    inline aabb topInset(t_scalar amount) const {
        return {m0, m1 - vec{z, amount}};
    }

    inline aabb leftInset(t_scalar amount) const {
        return {m0 + vec{amount, z}, m1};
    }

    inline aabb rightInset(t_scalar amount) const {
        return {m0, m1 - vec{amount, z}};
    }

    inline aabb grow(t_scalar amount) const { return inset(-amount); }

    inline aabb verticalSegment(t_scalar s0, t_scalar s1 = z) const {
        const t_scalar s_min = std::fmin(s0, s1);
        const t_scalar s_max = std::fmax(s0, s1);

        return {m0 + vec(z, height() * s_min),
                m0 + vec(z, height() * s_max) + vec(width(), z)};
    }

    inline aabb horizontalSegment(t_scalar s0, t_scalar s1 = z) const {
        const t_scalar s_min = std::fmin(s0, s1);
        const t_scalar s_max = std::fmax(s0, s1);

        return {m0 + vec(width() * s_min, z),
                m0 + vec(width() * s_max, z) + vec(z, height())};
    }

    inline void verticalSplit(t_scalar s, aabb *bottom, aabb *top) const {
        *top = {m0 + vec(z, s * height()), m1};
        *bottom = {m0, top->bottomRight()};
    }

    inline void horizontalSplit(t_scalar s, aabb *left, aabb *right) const {
        *left = {m0, m1 - vec((t_scalar(1) - s) * width(), z)};
        *right = {left->bottomRight(), m1};
    }

    inline aabb scale(t_scalar s, const vec &origin = c) const {
        const vec o = position(origin);
        return aabb(width() * s, height() * s, o, origin);
    }

    inline aabb scale_x(t_scalar s, const vec &origin = c) const {
        const vec o = position(origin);
        return aabb(width() * s, height(), o, origin);
    }

    inline aabb scale_y(t_scalar s, const vec &origin = c) const {
        const vec o = position(origin);
        return aabb(width(), height() * s, o, origin);
    }

    inline aabb pixelPerfect() const {
        return aabb({std::round(m0.x()), std::round(m0.y())},
                    {std::round(m1.x()), std::round(m1.y())});
    }

    inline vec x_range() const { return {m0[0], m1[0]}; }
    inline vec y_range() const { return {m0[1], m1[1]}; }
};

struct Grid {
    inline Grid() { h_cells = v_cells = 0; }
    inline Grid(int h, int v) : h_cells(h), v_cells(v) {}

    int h_cells;
    int v_cells;

    template<typename t_scalar, unsigned int t_size, bool t_enable_simd>
    aabb<t_scalar, t_size, t_enable_simd>
    get(const aabb<t_scalar, t_size, t_enable_simd> &a, int x, int y, int w = 1,
        int h = 1) const {
        using t_aabb = aabb<t_scalar, t_size, t_enable_simd>;

        const t_scalar cellWidth = a.width() / h_cells;
        const t_scalar cellHeight = a.height() / v_cells;

        const t_scalar width = cellWidth * w;
        const t_scalar height = cellHeight * h;

        const t_aabb::vec p0 = a.position(t_aabb::tl) +
                               t_aabb::vec(x * cellWidth, -y * cellHeight);
        return t_aabb(p0, {width, height}, t_aabb::tl);
    }
};

using aabb2 = aabb<float, 2, false>;

}// namespace atg_math

#endif /* ATG_MATH_AABB_H */
