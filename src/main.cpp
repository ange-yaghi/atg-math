#include "../include/atg_math.h"

#include <stdlib.h>
#include <random>
#include <iostream>
#include <chrono>

template<typename t_vec>
t_vec *generate_random_array(int seed, int test_size) {
    //std::default_random_engine engine;
    //engine.seed(seed);
    
    //std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    size_t size = sizeof(t_vec);

    void *buffer = _aligned_malloc(sizeof(t_vec) * test_size, 16);
    t_vec *data = new(buffer) t_vec[test_size];

    //for (int i = 0; i < test_size; ++i) {
    //    data[i] = t_vec(dist(engine), dist(engine), dist(engine), dist(engine));
    //}

    for (int i = 0; i < test_size; ++i) {
        data[i] = t_vec(i, i + 1, i + 2, i + 3) * 0.0001f;
    }

    return data;
}

__m128 *generate_random_array_m128(int seed, int test_size) {
    //std::default_random_engine engine;
    //engine.seed(seed);

    //std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    void *buffer = _aligned_malloc(sizeof(__m128) * test_size, 16);
    __m128 *data = new(buffer) __m128[test_size];

    //for (int i = 0; i < test_size; ++i) {
    //    data[i] = { dist(engine), dist(engine), dist(engine), dist(engine) };
    //}

    for (int i = 0; i < test_size; ++i) {
        data[i] = {
            0.0001f * (float)i,
            0.0001f * ((float)i + 1),
            0.0001f * ((float)i + 2),
            0.0001f * ((float)i + 3) };
    }

    return data;
}

void do_calculations_simd(int test_size) {
    atg_math::vec4_v *in0 = generate_random_array<atg_math::vec4_v>(0, test_size);
    atg_math::vec4_v *in1 = generate_random_array<atg_math::vec4_v>(1, test_size);
    atg_math::vec4_v *result = generate_random_array<atg_math::vec4_v>(0, test_size);

    atg_math::vec4_v two = 2.0f;
    for (int i = 0; i < test_size; ++i) {
        atg_math::vec4_v t = 1.0f;
        for (int j = 0; j < 100; ++j) {
            const atg_math::vec4_v a = in0[i];
            const atg_math::vec4_v b = in1[i];

            t = t * ((a + b) * two + (a * b));
        }

        result[i] = t;
    }

    atg_math::vec4_v sum = 0;
    for (int i = 0; i < test_size; ++i) {
        sum += result[i];
    }

    std::cout << sum.x << ", " << sum.y << ", " << sum.z << ", " << sum.w << "\n";

    _aligned_free(in0);
    _aligned_free(in1);
    _aligned_free(result);
}

void do_calculations_scalar(int test_size) {
    atg_math::vec4_s *ref_in0 = generate_random_array<atg_math::vec4_s>(0, test_size);
    atg_math::vec4_s *ref_in1 = generate_random_array<atg_math::vec4_s>(1, test_size);
    atg_math::vec4_s *ref_out = generate_random_array<atg_math::vec4_s>(0, test_size);

    for (int i = 0; i < test_size; ++i) {
        const atg_math::vec4_s a = ref_in0[i];
        const atg_math::vec4_s b = ref_in1[i];

        atg_math::vec4_s t = 1.0f;
        for (int j = 0; j < 100; ++j) {
            t = t * ((a + b) * 2.0f + (a * b));
        }

        ref_out[i] = t;
    }

    atg_math::vec4_s sum = 0;
    for (int i = 0; i < test_size; ++i) {
        sum += ref_out[i];
    }

    std::cout << sum.x << ", " << sum.y << ", " << sum.z << ", " << sum.w << "\n";

    _aligned_free(ref_in0);
    _aligned_free(ref_in1);
    _aligned_free(ref_out);
}

void do_calculations_bare_simd(int test_size) {
    __m128 *simd_in0 = generate_random_array_m128(0, test_size);
    __m128 *simd_in1 = generate_random_array_m128(1, test_size);
    __m128 *simd_out = generate_random_array_m128(0, test_size);

    const __m128 two = _mm_set_ps1(2.0f);
    for (int i = 0; i < test_size; ++i) {
        
        __m128 t = _mm_set_ps1(1.0f);
        for (int j = 0; j < 100; ++j) {
            const __m128 a = simd_in0[i];
            const __m128 b = simd_in1[i];

            __m128 temp = _mm_add_ps(a, b);
            temp = _mm_mul_ps(two, temp);

            const __m128 a_b = _mm_mul_ps(a, b);
            temp = _mm_add_ps(temp, a_b);

            t = _mm_mul_ps(t, temp);
        }

        simd_out[i] = t;
    }

    __m128 sum = { 0, 0, 0, 0 };
    for (int i = 0; i < test_size; ++i) {
        sum = _mm_add_ps(sum, simd_out[i]);
    }

    std::cout << sum.m128_f32[0] << ", " << sum.m128_f32[1] << ", " << sum.m128_f32[2] << ", " << sum.m128_f32[3] << "\n";

    _aligned_free(simd_in0);
    _aligned_free(simd_in1);
    _aligned_free(simd_out);
}

int main() {
    auto t0 = std::chrono::steady_clock::now();

    const int test_size = 4000000;
    float d = (float)atg_math::vec4_v(1.0f, 2.0f, 3.0f).dot(atg_math::vec4_v(2.0f, 3.0f, -1.0f));

    float mag = (float)atg_math::vec4_v(1.0f, 1.0f, 1.0f, 1.0f).magnitude_squared();

    atg_math::vec4_v v1(1.0f, 2.0f, -2.0f, -1.0f);
    atg_math::vec4_v v2(-5.0f, 2.0f, 2.0f, -1.0f);

    auto c = v1.cross(v2);
    float dot1 = (float)v1.dot(c);
    float dot2 = (float)v2.dot(c);

    auto t1 = std::chrono::steady_clock::now();

    std::cout << "Time: " << std::chrono::duration<double>(t1 - t0).count() << "\n";

    return 0;
}
