// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SIMD_X86_SSE4_H_
#define PIK_SIMD_X86_SSE4_H_

// 128-bit SSE4 vectors and operations.

#include "pik/simd/compiler_specific.h"
#include "pik/simd/shared.h"
#include "pik/simd/targets.h"
#include "pik/simd/util.h"

#if SIMD_ENABLE & SIMD_SSE4
#include <smmintrin.h>

namespace pik {

// On X86, it is cheaper to use small vectors (prefixes of larger registers)
// when possible; this also reduces the number of overloaded functions.
template <class Target>
struct PartTargetT<1, Target> {
  using type = SSE4;
};

template <typename T>
struct raw_sse4 {
  using type = __m128i;
};
template <>
struct raw_sse4<float> {
  using type = __m128;
};
template <>
struct raw_sse4<double> {
  using type = __m128d;
};

// Returned by set_shift_*_count, also used by AVX2; do not use directly.
template <typename T, size_t N>
struct shift_left_count {
  __m128i raw;
};

template <typename T, size_t N>
struct shift_right_count {
  __m128i raw;
};

// Returned by set_table_indices for use by table_lookup_lanes.
template <typename T>
struct permute_sse4 {
  __m128i raw;
};

template <typename T, size_t N = SSE4::NumLanes<T>()>
class vec_sse4 {
  using Raw = typename raw_sse4<T>::type;

 public:
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4() {}
  vec_sse4(const vec_sse4&) = default;
  vec_sse4& operator=(const vec_sse4&) = default;
  SIMD_ATTR_SSE4 SIMD_INLINE explicit vec_sse4(const Raw raw) : raw(raw) {}

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator*=(const vec_sse4 other) {
    return *this = (*this * other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator/=(const vec_sse4 other) {
    return *this = (*this / other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator+=(const vec_sse4 other) {
    return *this = (*this + other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator-=(const vec_sse4 other) {
    return *this = (*this - other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator&=(const vec_sse4 other) {
    return *this = (*this & other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator|=(const vec_sse4 other) {
    return *this = (*this | other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator^=(const vec_sse4 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

template <typename T, size_t N>
struct VecT<T, N, SSE4> {
  using type = vec_sse4<T, N>;
};

using u8x16 = vec_sse4<uint8_t, 16>;
using u16x8 = vec_sse4<uint16_t, 8>;
using u32x4 = vec_sse4<uint32_t, 4>;
using u64x2 = vec_sse4<uint64_t, 2>;
using i8x16 = vec_sse4<int8_t, 16>;
using i16x8 = vec_sse4<int16_t, 8>;
using i32x4 = vec_sse4<int32_t, 4>;
using i64x2 = vec_sse4<int64_t, 2>;
using f32x4 = vec_sse4<float, 4>;
using f64x2 = vec_sse4<double, 2>;

using u8x8 = vec_sse4<uint8_t, 8>;
using u16x4 = vec_sse4<uint16_t, 4>;
using u32x2 = vec_sse4<uint32_t, 2>;
using i8x8 = vec_sse4<int8_t, 8>;
using i16x4 = vec_sse4<int16_t, 4>;
using i32x2 = vec_sse4<int32_t, 2>;
using f32x2 = vec_sse4<float, 2>;
using f64x1 = vec_sse4<double, 1>;

using u8x4 = vec_sse4<uint8_t, 4>;
using i8x4 = vec_sse4<int8_t, 4>;
using f32x1 = vec_sse4<float, 1>;

// ------------------------------ Cast

SIMD_ATTR_SSE4 SIMD_INLINE __m128i BitCastToInteger(__m128i v) { return v; }
SIMD_ATTR_SSE4 SIMD_INLINE __m128i BitCastToInteger(__m128 v) {
  return _mm_castps_si128(v);
}
SIMD_ATTR_SSE4 SIMD_INLINE __m128i BitCastToInteger(__m128d v) {
  return _mm_castpd_si128(v);
}

// cast_to_u8
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> cast_to_u8(
    Desc<uint8_t, N, SSE4>, vec_sse4<T, N / sizeof(T)> v) {
  return vec_sse4<uint8_t, N>(BitCastToInteger(v.raw));
}

// Cannot rely on function overloading because return types differ.
template <typename T>
struct BitCastFromIntegerSSE4 {
  SIMD_ATTR_SSE4 SIMD_INLINE __m128i operator()(__m128i v) { return v; }
};
template <>
struct BitCastFromIntegerSSE4<float> {
  SIMD_ATTR_SSE4 SIMD_INLINE __m128 operator()(__m128i v) {
    return _mm_castsi128_ps(v);
  }
};
template <>
struct BitCastFromIntegerSSE4<double> {
  SIMD_ATTR_SSE4 SIMD_INLINE __m128d operator()(__m128i v) {
    return _mm_castsi128_pd(v);
  }
};

// cast_u8_to
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> cast_u8_to(
    Desc<T, N, SSE4>, vec_sse4<uint8_t, N * sizeof(T)> v) {
  return vec_sse4<T, N>(BitCastFromIntegerSSE4<T>()(v.raw));
}

// cast_to
template <typename T, size_t N, typename FromT>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> cast_to(
    Desc<T, N, SSE4> d, vec_sse4<FromT, N * sizeof(T) / sizeof(FromT)> v) {
  const auto u8 = cast_to_u8(Desc<uint8_t, N * sizeof(T), SSE4>(), v);
  return cast_u8_to(d, u8);
}

// ------------------------------ Set

// Returns an all-zero vector/part.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> setzero(Desc<T, N, SSE4>) {
  return vec_sse4<T, N>(_mm_setzero_si128());
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> setzero(Desc<float, N, SSE4>) {
  return vec_sse4<float, N>(_mm_setzero_ps());
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> setzero(Desc<double, N, SSE4>) {
  return vec_sse4<double, N>(_mm_setzero_pd());
}

// Returns a vector/part with all lanes set to "t".
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> set1(Desc<uint8_t, N, SSE4>,
                                                     const uint8_t t) {
  return vec_sse4<uint8_t, N>(_mm_set1_epi8(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> set1(Desc<uint16_t, N, SSE4>,
                                                      const uint16_t t) {
  return vec_sse4<uint16_t, N>(_mm_set1_epi16(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> set1(Desc<uint32_t, N, SSE4>,
                                                      const uint32_t t) {
  return vec_sse4<uint32_t, N>(_mm_set1_epi32(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> set1(Desc<uint64_t, N, SSE4>,
                                                      const uint64_t t) {
  return vec_sse4<uint64_t, N>(_mm_set1_epi64x(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> set1(Desc<int8_t, N, SSE4>,
                                                    const int8_t t) {
  return vec_sse4<int8_t, N>(_mm_set1_epi8(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> set1(Desc<int16_t, N, SSE4>,
                                                     const int16_t t) {
  return vec_sse4<int16_t, N>(_mm_set1_epi16(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> set1(Desc<int32_t, N, SSE4>,
                                                     const int32_t t) {
  return vec_sse4<int32_t, N>(_mm_set1_epi32(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t, N> set1(Desc<int64_t, N, SSE4>,
                                                     const int64_t t) {
  return vec_sse4<int64_t, N>(_mm_set1_epi64x(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> set1(Desc<float, N, SSE4>,
                                                   const float t) {
  return vec_sse4<float, N>(_mm_set1_ps(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> set1(Desc<double, N, SSE4>,
                                                    const double t) {
  return vec_sse4<double, N>(_mm_set1_pd(t));
}

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <typename T, size_t N, typename T2>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> iota(Desc<T, N, SSE4> d,
                                               const T2 first) {
  SIMD_ALIGN T lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

SIMD_DIAGNOSTICS(push)
SIMD_DIAGNOSTICS_OFF(disable : 4700, ignored "-Wuninitialized")

// Returns a vector with uninitialized elements.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> undefined(Desc<T, N, SSE4>) {
#ifdef __clang__
  return vec_sse4<T, N>(_mm_undefined_si128());
#else
  __m128i raw;
  return vec_sse4<T, N>(raw);
#endif
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> undefined(Desc<float, N, SSE4>) {
#ifdef __clang__
  return vec_sse4<float, N>(_mm_undefined_ps());
#else
  __m128 raw;
  return vec_sse4<float, N>(raw);
#endif
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> undefined(
    Desc<double, N, SSE4>) {
#ifdef __clang__
  return vec_sse4<double, N>(_mm_undefined_pd());
#else
  __m128d raw;
  return vec_sse4<double, N>(raw);
#endif
}

SIMD_DIAGNOSTICS(pop)

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> operator+(
    const vec_sse4<uint8_t, N> a, const vec_sse4<uint8_t, N> b) {
  return vec_sse4<uint8_t, N>(_mm_add_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> operator+(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_add_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> operator+(
    const vec_sse4<uint32_t, N> a, const vec_sse4<uint32_t, N> b) {
  return vec_sse4<uint32_t, N>(_mm_add_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> operator+(
    const vec_sse4<uint64_t, N> a, const vec_sse4<uint64_t, N> b) {
  return vec_sse4<uint64_t, N>(_mm_add_epi64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> operator+(
    const vec_sse4<int8_t, N> a, const vec_sse4<int8_t, N> b) {
  return vec_sse4<int8_t, N>(_mm_add_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> operator+(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_add_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> operator+(
    const vec_sse4<int32_t, N> a, const vec_sse4<int32_t, N> b) {
  return vec_sse4<int32_t, N>(_mm_add_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t, N> operator+(
    const vec_sse4<int64_t, N> a, const vec_sse4<int64_t, N> b) {
  return vec_sse4<int64_t, N>(_mm_add_epi64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator+(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_add_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator+(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_add_pd(a.raw, b.raw));
}

// ------------------------------ Subtraction

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> operator-(
    const vec_sse4<uint8_t, N> a, const vec_sse4<uint8_t, N> b) {
  return vec_sse4<uint8_t, N>(_mm_sub_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> operator-(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_sub_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> operator-(
    const vec_sse4<uint32_t, N> a, const vec_sse4<uint32_t, N> b) {
  return vec_sse4<uint32_t, N>(_mm_sub_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> operator-(
    const vec_sse4<uint64_t, N> a, const vec_sse4<uint64_t, N> b) {
  return vec_sse4<uint64_t, N>(_mm_sub_epi64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> operator-(
    const vec_sse4<int8_t, N> a, const vec_sse4<int8_t, N> b) {
  return vec_sse4<int8_t, N>(_mm_sub_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> operator-(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_sub_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> operator-(
    const vec_sse4<int32_t, N> a, const vec_sse4<int32_t, N> b) {
  return vec_sse4<int32_t, N>(_mm_sub_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t, N> operator-(
    const vec_sse4<int64_t, N> a, const vec_sse4<int64_t, N> b) {
  return vec_sse4<int64_t, N>(_mm_sub_epi64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator-(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_sub_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator-(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_sub_pd(a.raw, b.raw));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> saturated_add(
    const vec_sse4<uint8_t, N> a, const vec_sse4<uint8_t, N> b) {
  return vec_sse4<uint8_t, N>(_mm_adds_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> saturated_add(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_adds_epu16(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> saturated_add(
    const vec_sse4<int8_t, N> a, const vec_sse4<int8_t, N> b) {
  return vec_sse4<int8_t, N>(_mm_adds_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> saturated_add(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_adds_epi16(a.raw, b.raw));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> saturated_subtract(
    const vec_sse4<uint8_t, N> a, const vec_sse4<uint8_t, N> b) {
  return vec_sse4<uint8_t, N>(_mm_subs_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> saturated_subtract(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_subs_epu16(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> saturated_subtract(
    const vec_sse4<int8_t, N> a, const vec_sse4<int8_t, N> b) {
  return vec_sse4<int8_t, N>(_mm_subs_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> saturated_subtract(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_subs_epi16(a.raw, b.raw));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> average_round(
    const vec_sse4<uint8_t, N> a, const vec_sse4<uint8_t, N> b) {
  return vec_sse4<uint8_t, N>(_mm_avg_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> average_round(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_avg_epu16(a.raw, b.raw));
}

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> abs(
    const vec_sse4<int8_t, N> v) {
  return vec_sse4<int8_t, N>(_mm_abs_epi8(v.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> abs(
    const vec_sse4<int16_t, N> v) {
  return vec_sse4<int16_t, N>(_mm_abs_epi16(v.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> abs(
    const vec_sse4<int32_t, N> v) {
  return vec_sse4<int32_t, N>(_mm_abs_epi32(v.raw));
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> shift_left(
    const vec_sse4<uint16_t, N> v) {
  return vec_sse4<uint16_t, N>(_mm_slli_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> shift_right(
    const vec_sse4<uint16_t, N> v) {
  return vec_sse4<uint16_t, N>(_mm_srli_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> shift_left(
    const vec_sse4<uint32_t, N> v) {
  return vec_sse4<uint32_t, N>(_mm_slli_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> shift_right(
    const vec_sse4<uint32_t, N> v) {
  return vec_sse4<uint32_t, N>(_mm_srli_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> shift_left(
    const vec_sse4<uint64_t, N> v) {
  return vec_sse4<uint64_t, N>(_mm_slli_epi64(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> shift_right(
    const vec_sse4<uint64_t, N> v) {
  return vec_sse4<uint64_t, N>(_mm_srli_epi64(v.raw, kBits));
}

// Signed (no i64 shift_right)
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> shift_left(
    const vec_sse4<int16_t, N> v) {
  return vec_sse4<int16_t, N>(_mm_slli_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> shift_right(
    const vec_sse4<int16_t, N> v) {
  return vec_sse4<int16_t, N>(_mm_srai_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> shift_left(
    const vec_sse4<int32_t, N> v) {
  return vec_sse4<int32_t, N>(_mm_slli_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> shift_right(
    const vec_sse4<int32_t, N> v) {
  return vec_sse4<int32_t, N>(_mm_srai_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t, N> shift_left(
    const vec_sse4<int64_t, N> v) {
  return vec_sse4<int64_t, N>(_mm_slli_epi64(v.raw, kBits));
}

// ------------------------------ Shift lanes by same variable #bits

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE shift_left_count<T, N> set_shift_left_count(
    Desc<T, N, SSE4>, const int bits) {
  return shift_left_count<T, N>{_mm_cvtsi32_si128(bits)};
}

// Same as shift_left_count on x86, but different on ARM.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE shift_right_count<T, N> set_shift_right_count(
    Desc<T, N, SSE4>, const int bits) {
  return shift_right_count<T, N>{_mm_cvtsi32_si128(bits)};
}

// Unsigned (no u8)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> shift_left_same(
    const vec_sse4<uint16_t, N> v, const shift_left_count<uint16_t, N> bits) {
  return vec_sse4<uint16_t, N>(_mm_sll_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> shift_right_same(
    const vec_sse4<uint16_t, N> v, const shift_right_count<uint16_t, N> bits) {
  return vec_sse4<uint16_t, N>(_mm_srl_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> shift_left_same(
    const vec_sse4<uint32_t, N> v, const shift_left_count<uint32_t, N> bits) {
  return vec_sse4<uint32_t, N>(_mm_sll_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> shift_right_same(
    const vec_sse4<uint32_t, N> v, const shift_right_count<uint32_t, N> bits) {
  return vec_sse4<uint32_t, N>(_mm_srl_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> shift_left_same(
    const vec_sse4<uint64_t, N> v, const shift_left_count<uint64_t, N> bits) {
  return vec_sse4<uint64_t, N>(_mm_sll_epi64(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> shift_right_same(
    const vec_sse4<uint64_t, N> v, const shift_right_count<uint64_t, N> bits) {
  return vec_sse4<uint64_t, N>(_mm_srl_epi64(v.raw, bits.raw));
}

// Signed (no i8,i64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> shift_left_same(
    const vec_sse4<int16_t, N> v, const shift_left_count<int16_t, N> bits) {
  return vec_sse4<int16_t, N>(_mm_sll_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> shift_right_same(
    const vec_sse4<int16_t, N> v, const shift_right_count<int16_t, N> bits) {
  return vec_sse4<int16_t, N>(_mm_sra_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> shift_left_same(
    const vec_sse4<int32_t, N> v, const shift_left_count<int32_t, N> bits) {
  return vec_sse4<int32_t, N>(_mm_sll_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> shift_right_same(
    const vec_sse4<int32_t, N> v, const shift_right_count<int32_t, N> bits) {
  return vec_sse4<int32_t, N>(_mm_sra_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t, N> shift_left_same(
    const vec_sse4<int64_t, N> v, const shift_left_count<int64_t, N> bits) {
  return vec_sse4<int64_t, N>(_mm_sll_epi64(v.raw, bits.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

#if SIMD_TARGET_VALUE == SIMD_AVX2

// Unsigned (no u8,u16)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> operator<<(
    const vec_sse4<uint32_t, N> v, const vec_sse4<uint32_t, N> bits) {
  return vec_sse4<uint32_t, N>(_mm_sllv_epi32(v.raw, bits));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> operator>>(
    const vec_sse4<uint32_t, N> v, const vec_sse4<uint32_t, N> bits) {
  return vec_sse4<uint32_t, N>(_mm_srlv_epi32(v.raw, bits));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> operator<<(
    const vec_sse4<uint64_t, N> v, const vec_sse4<uint64_t, N> bits) {
  return vec_sse4<uint64_t, N>(_mm_sllv_epi64(v.raw, bits));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> operator>>(
    const vec_sse4<uint64_t, N> v, const vec_sse4<uint64_t, N> bits) {
  return vec_sse4<uint64_t, N>(_mm_srlv_epi64(v.raw, bits));
}

// Signed (no i8,i16,i64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> operator<<(
    const vec_sse4<int32_t, N> v, const vec_sse4<int32_t, N> bits) {
  return vec_sse4<int32_t, N>(_mm_sllv_epi32(v.raw, bits));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> operator>>(
    const vec_sse4<int32_t, N> v, const vec_sse4<int32_t, N> bits) {
  return vec_sse4<int32_t, N>(_mm_srav_epi32(v.raw, bits));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t, N> operator<<(
    const vec_sse4<int64_t, N> v, const vec_sse4<int64_t, N> bits) {
  return vec_sse4<int64_t, N>(_mm_sllv_epi64(v.raw, bits));
}

#endif  // SIMD_TARGET_VALUE == SIMD_AVX2

// ------------------------------ Minimum

// Unsigned (no u64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> min(
    const vec_sse4<uint8_t, N> a, const vec_sse4<uint8_t, N> b) {
  return vec_sse4<uint8_t, N>(_mm_min_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> min(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_min_epu16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> min(
    const vec_sse4<uint32_t, N> a, const vec_sse4<uint32_t, N> b) {
  return vec_sse4<uint32_t, N>(_mm_min_epu32(a.raw, b.raw));
}

// Signed (no i64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> min(
    const vec_sse4<int8_t, N> a, const vec_sse4<int8_t, N> b) {
  return vec_sse4<int8_t, N>(_mm_min_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> min(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_min_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> min(
    const vec_sse4<int32_t, N> a, const vec_sse4<int32_t, N> b) {
  return vec_sse4<int32_t, N>(_mm_min_epi32(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> min(const vec_sse4<float, N> a,
                                                  const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_min_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> min(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_min_pd(a.raw, b.raw));
}

// ------------------------------ Maximum

// Unsigned (no u64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> max(
    const vec_sse4<uint8_t, N> a, const vec_sse4<uint8_t, N> b) {
  return vec_sse4<uint8_t, N>(_mm_max_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> max(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_max_epu16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> max(
    const vec_sse4<uint32_t, N> a, const vec_sse4<uint32_t, N> b) {
  return vec_sse4<uint32_t, N>(_mm_max_epu32(a.raw, b.raw));
}

// Signed (no i64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> max(
    const vec_sse4<int8_t, N> a, const vec_sse4<int8_t, N> b) {
  return vec_sse4<int8_t, N>(_mm_max_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> max(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_max_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> max(
    const vec_sse4<int32_t, N> a, const vec_sse4<int32_t, N> b) {
  return vec_sse4<int32_t, N>(_mm_max_epi32(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> max(const vec_sse4<float, N> a,
                                                  const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_max_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> max(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_max_pd(a.raw, b.raw));
}

// Returns the closest value to v within [lo, hi].
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> clamp(const vec_sse4<T, N> v,
                                                const vec_sse4<T, N> lo,
                                                const vec_sse4<T, N> hi) {
  return min(max(lo, v), hi);
}

// ------------------------------ Integer multiplication

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> operator*(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_mullo_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> operator*(
    const vec_sse4<uint32_t, N> a, const vec_sse4<uint32_t, N> b) {
  return vec_sse4<uint32_t, N>(_mm_mullo_epi32(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> operator*(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_mullo_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> operator*(
    const vec_sse4<int32_t, N> a, const vec_sse4<int32_t, N> b) {
  return vec_sse4<int32_t, N>(_mm_mullo_epi32(a.raw, b.raw));
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> mul_high(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_mulhi_epu16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> mul_high(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_mulhi_epi16(a.raw, b.raw));
}

}  // namespace ext

// Returns (((a * b) >> 14) + 1) >> 1.
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> mul_high_round(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_mulhrs_epi16(a.raw, b.raw));
}

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t> mul_even(
    const vec_sse4<int32_t> a, const vec_sse4<int32_t> b) {
  return vec_sse4<int64_t>(_mm_mul_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t> mul_even(
    const vec_sse4<uint32_t> a, const vec_sse4<uint32_t> b) {
  return vec_sse4<uint64_t>(_mm_mul_epu32(a.raw, b.raw));
}

// ------------------------------ Floating-point negate

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> neg(const vec_sse4<float, N> v) {
  const Part<float, N, SSE4> df;
  const Part<uint32_t, N, SSE4> du;
  const auto sign = cast_to(df, set1(du, 0x80000000u));
  return v ^ sign;
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> neg(
    const vec_sse4<double, N> v) {
  const Part<double, N, SSE4> df;
  const Part<uint64_t, N, SSE4> du;
  const auto sign = cast_to(df, set1(du, 0x8000000000000000ull));
  return v ^ sign;
}

// ------------------------------ Floating-point mul / div

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator*(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_mul_ps(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, 1> operator*(
    const vec_sse4<float, 1> a, const vec_sse4<float, 1> b) {
  return vec_sse4<float, 1>(_mm_mul_ss(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator*(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_mul_pd(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, 1> operator*(
    const vec_sse4<double, 1> a, const vec_sse4<double, 1> b) {
  return vec_sse4<double, 1>(_mm_mul_sd(a.raw, b.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator/(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_div_ps(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, 1> operator/(
    const vec_sse4<float, 1> a, const vec_sse4<float, 1> b) {
  return vec_sse4<float, 1>(_mm_div_ss(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator/(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_div_pd(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, 1> operator/(
    const vec_sse4<double, 1> a, const vec_sse4<double, 1> b) {
  return vec_sse4<double, 1>(_mm_div_sd(a.raw, b.raw));
}

// Approximate reciprocal
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> approximate_reciprocal(
    const vec_sse4<float, N> v) {
  return vec_sse4<float, N>(_mm_rcp_ps(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, 1> approximate_reciprocal(
    const vec_sse4<float, 1> v) {
  return vec_sse4<float, 1>(_mm_rcp_ss(v.raw));
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> mul_add(
    const vec_sse4<float, N> mul, const vec_sse4<float, N> x,
    const vec_sse4<float, N> add) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  return vec_sse4<float, N>(_mm_fmadd_ps(mul.raw, x.raw, add.raw));
#else
  return mul * x + add;
#endif
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> mul_add(
    const vec_sse4<double, N> mul, const vec_sse4<double, N> x,
    const vec_sse4<double, N> add) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  return vec_sse4<double, N>(_mm_fmadd_pd(mul.raw, x.raw, add.raw));
#else
  return mul * x + add;
#endif
}

// Returns add - mul * x
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> nmul_add(
    const vec_sse4<float, N> mul, const vec_sse4<float, N> x,
    const vec_sse4<float, N> add) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  return vec_sse4<float, N>(_mm_fnmadd_ps(mul.raw, x.raw, add.raw));
#else
  return add - mul * x;
#endif
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> nmul_add(
    const vec_sse4<double, N> mul, const vec_sse4<double, N> x,
    const vec_sse4<double, N> add) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  return vec_sse4<double, N>(_mm_fnmadd_pd(mul.raw, x.raw, add.raw));
#else
  return add - mul * x;
#endif
}

// Returns x + add
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> fadd(
    vec_sse4<float, N> x, const vec_sse4<float, N> k1,
    const vec_sse4<float, N> add) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfmadd132ps %2, %1, %0"
               : "+x"(x.raw)
               : "x"(add.raw), "x"(k1.raw));
  return x;
#else
  return vec_sse4<float, N>(_mm_fmadd_ps(x.raw, k1.raw, add.raw));
#endif
#else
  return x + add;
#endif
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> fadd(
    vec_sse4<double, N> x, const vec_sse4<double, N> k1,
    const vec_sse4<double, N> add) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfmadd132pd %2, %1, %0"
               : "+x"(x.raw)
               : "x"(add.raw), "x"(k1.raw));
  return x;
#else
  return vec_sse4<double, N>(_mm_fmadd_pd(x.raw, k1.raw, add.raw));
#endif
#else
  return x + add;
#endif
}

// Returns x - sub
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> fsub(
    vec_sse4<float, N> x, const vec_sse4<float, N> k1,
    const vec_sse4<float, N> sub) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfmsub132ps %2, %1, %0"
               : "+x"(x.raw)
               : "x"(sub.raw), "x"(k1.raw));
  return x;
#else
  return vec_sse4<float, N>(_mm_fmsub_ps(x.raw, k1.raw, sub.raw));
#endif
#else
  return x - sub;
#endif
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> fsub(
    vec_sse4<double, N> x, const vec_sse4<double, N> k1,
    const vec_sse4<double, N> sub) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfmsub132pd %2, %1, %0"
               : "+x"(x.raw)
               : "x"(sub.raw), "x"(k1.raw));
  return x;
#else
  return vec_sse4<double, N>(_mm_fmsub_pd(x.raw, k1.raw, sub.raw));
#endif
#else
  return x - sub;
#endif
}

// Returns -sub + x (clobbers sub register)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> fnadd(
    vec_sse4<float, N> sub, const vec_sse4<float, N> k1,
    const vec_sse4<float, N> x) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfnmadd132ps %2, %1, %0"
               : "+x"(sub.raw)
               : "x"(x.raw), "x"(k1.raw));
  return sub;
#else
  return vec_sse4<float, N>(_mm_fnmadd_ps(sub.raw, k1.raw, x.raw));
#endif
#else
  return x - sub;
#endif
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> fnadd(
    vec_sse4<double, N> sub, const vec_sse4<double, N> k1,
    const vec_sse4<double, N> x) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfnmadd132pd %2, %1, %0"
               : "+x"(sub.raw)
               : "x"(x.raw), "x"(k1.raw));
  return sub;
#else
  return vec_sse4<double, N>(_mm_fnmadd_pd(sub.raw, k1.raw, x.raw));
#endif
#else
  return x - sub;
#endif
}

// Slightly more expensive on ARM (extra negate)
namespace ext {

// Returns mul * x - sub
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> mul_subtract(
    const vec_sse4<float, N> mul, const vec_sse4<float, N> x,
    const vec_sse4<float, N> sub) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  return vec_sse4<float, N>(_mm_fmsub_ps(mul.raw, x.raw, sub.raw));
#else
  return mul * x - sub;
#endif
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> mul_subtract(
    const vec_sse4<double, N> mul, const vec_sse4<double, N> x,
    const vec_sse4<double, N> sub) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  return vec_sse4<double, N>(_mm_fmsub_pd(mul.raw, x.raw, sub.raw));
#else
  return mul * x - sub;
#endif
}

// Returns -mul * x - sub
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> nmul_subtract(
    const vec_sse4<float, N> mul, const vec_sse4<float, N> x,
    const vec_sse4<float, N> sub) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  return vec_sse4<float, N>(_mm_fnmsub_ps(mul.raw, x.raw, sub.raw));
#else
  return neg(mul) * x - sub;
#endif
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> nmul_subtract(
    const vec_sse4<double, N> mul, const vec_sse4<double, N> x,
    const vec_sse4<double, N> sub) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  return vec_sse4<double, N>(_mm_fnmsub_pd(mul.raw, x.raw, sub.raw));
#else
  return neg(mul) * x - sub;
#endif
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Full precision square root
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> sqrt(const vec_sse4<float, N> v) {
  return vec_sse4<float, N>(_mm_sqrt_ps(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, 1> sqrt(const vec_sse4<float, 1> v) {
  return vec_sse4<float, 1>(_mm_sqrt_ss(v.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> sqrt(
    const vec_sse4<double, N> v) {
  return vec_sse4<double, N>(_mm_sqrt_pd(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, 1> sqrt(
    const vec_sse4<double, 1> v) {
  return vec_sse4<double, 1>(_mm_sqrt_sd(_mm_setzero_pd(), v.raw));
}

// Approximate reciprocal square root
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> approximate_reciprocal_sqrt(
    const vec_sse4<float, N> v) {
  return vec_sse4<float, N>(_mm_rsqrt_ps(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, 1> approximate_reciprocal_sqrt(
    const vec_sse4<float, 1> v) {
  return vec_sse4<float, 1>(_mm_rsqrt_ss(v.raw));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer, ties to even
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> round(
    const vec_sse4<float, N> v) {
  return vec_sse4<float, N>(
      _mm_round_ps(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> round(
    const vec_sse4<double, N> v) {
  return vec_sse4<double, N>(
      _mm_round_pd(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Toward zero, aka truncate
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> trunc(
    const vec_sse4<float, N> v) {
  return vec_sse4<float, N>(
      _mm_round_ps(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> trunc(
    const vec_sse4<double, N> v) {
  return vec_sse4<double, N>(
      _mm_round_pd(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}

// Toward +infinity, aka ceiling
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> ceil(const vec_sse4<float, N> v) {
  return vec_sse4<float, N>(
      _mm_round_ps(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> ceil(
    const vec_sse4<double, N> v) {
  return vec_sse4<double, N>(
      _mm_round_pd(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}

// Toward -infinity, aka floor
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> floor(
    const vec_sse4<float, N> v) {
  return vec_sse4<float, N>(
      _mm_round_ps(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> floor(
    const vec_sse4<double, N> v) {
  return vec_sse4<double, N>(
      _mm_round_pd(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> operator==(
    const vec_sse4<uint8_t, N> a, const vec_sse4<uint8_t, N> b) {
  return vec_sse4<uint8_t, N>(_mm_cmpeq_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> operator==(
    const vec_sse4<uint16_t, N> a, const vec_sse4<uint16_t, N> b) {
  return vec_sse4<uint16_t, N>(_mm_cmpeq_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, N> operator==(
    const vec_sse4<uint32_t, N> a, const vec_sse4<uint32_t, N> b) {
  return vec_sse4<uint32_t, N>(_mm_cmpeq_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, N> operator==(
    const vec_sse4<uint64_t, N> a, const vec_sse4<uint64_t, N> b) {
  return vec_sse4<uint64_t, N>(_mm_cmpeq_epi64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> operator==(
    const vec_sse4<int8_t, N> a, const vec_sse4<int8_t, N> b) {
  return vec_sse4<int8_t, N>(_mm_cmpeq_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> operator==(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_cmpeq_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> operator==(
    const vec_sse4<int32_t, N> a, const vec_sse4<int32_t, N> b) {
  return vec_sse4<int32_t, N>(_mm_cmpeq_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t, N> operator==(
    const vec_sse4<int64_t, N> a, const vec_sse4<int64_t, N> b) {
  return vec_sse4<int64_t, N>(_mm_cmpeq_epi64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator==(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_cmpeq_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator==(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_cmpeq_pd(a.raw, b.raw));
}

// ------------------------------ Strict inequality

// Signed/float <
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> operator<(
    const vec_sse4<int8_t, N> a, const vec_sse4<int8_t, N> b) {
  return vec_sse4<int8_t, N>(_mm_cmpgt_epi8(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> operator<(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_cmpgt_epi16(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> operator<(
    const vec_sse4<int32_t, N> a, const vec_sse4<int32_t, N> b) {
  return vec_sse4<int32_t, N>(_mm_cmpgt_epi32(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator<(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_cmplt_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator<(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_cmplt_pd(a.raw, b.raw));
}

// Signed/float >
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> operator>(
    const vec_sse4<int8_t, N> a, const vec_sse4<int8_t, N> b) {
  return vec_sse4<int8_t, N>(_mm_cmpgt_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> operator>(
    const vec_sse4<int16_t, N> a, const vec_sse4<int16_t, N> b) {
  return vec_sse4<int16_t, N>(_mm_cmpgt_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> operator>(
    const vec_sse4<int32_t, N> a, const vec_sse4<int32_t, N> b) {
  return vec_sse4<int32_t, N>(_mm_cmpgt_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator>(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_cmpgt_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator>(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_cmpgt_pd(a.raw, b.raw));
}

// ------------------------------ Weak inequality

// Float <= >=
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator<=(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_cmple_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator<=(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_cmple_pd(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator>=(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_cmpge_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator>=(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_cmpge_pd(a.raw, b.raw));
}

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> operator&(const vec_sse4<T, N> a,
                                                    const vec_sse4<T, N> b) {
  return vec_sse4<T, N>(_mm_and_si128(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator&(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_and_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator&(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_and_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> andnot(const vec_sse4<T, N> not_mask,
                                                 const vec_sse4<T, N> mask) {
  return vec_sse4<T, N>(_mm_andnot_si128(not_mask.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> andnot(
    const vec_sse4<float, N> not_mask, const vec_sse4<float, N> mask) {
  return vec_sse4<float, N>(_mm_andnot_ps(not_mask.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> andnot(
    const vec_sse4<double, N> not_mask, const vec_sse4<double, N> mask) {
  return vec_sse4<double, N>(_mm_andnot_pd(not_mask.raw, mask.raw));
}

// ------------------------------ Bitwise OR

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> operator|(const vec_sse4<T, N> a,
                                                    const vec_sse4<T, N> b) {
  return vec_sse4<T, N>(_mm_or_si128(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator|(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_or_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator|(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_or_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise XOR

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> operator^(const vec_sse4<T, N> a,
                                                    const vec_sse4<T, N> b) {
  return vec_sse4<T, N>(_mm_xor_si128(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> operator^(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b) {
  return vec_sse4<float, N>(_mm_xor_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> operator^(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b) {
  return vec_sse4<double, N>(_mm_xor_pd(a.raw, b.raw));
}

// ------------------------------ Select/blend

// Returns a mask for use by select().
// blendv_ps/pd only check the sign bit, so this is a no-op on x86.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> condition_from_sign(
    const vec_sse4<T, N> v) {
  return v;
}

// Returns mask ? b : a. "mask" must either have been returned by
// selector_from_mask, or callers must ensure its lanes are T(0) or ~T(0).
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> select(const vec_sse4<T, N> a,
                                                 const vec_sse4<T, N> b,
                                                 const vec_sse4<T, N> mask) {
  return vec_sse4<T, N>(_mm_blendv_epi8(a.raw, b.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> select(
    const vec_sse4<float, N> a, const vec_sse4<float, N> b,
    const vec_sse4<float, N> mask) {
  return vec_sse4<float, N>(_mm_blendv_ps(a.raw, b.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, N> select(
    const vec_sse4<double, N> a, const vec_sse4<double, N> b,
    const vec_sse4<double, N> mask) {
  return vec_sse4<double, N>(_mm_blendv_pd(a.raw, b.raw, mask.raw));
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> load(Full<T, SSE4>,
                                            const T* SIMD_RESTRICT aligned) {
  return vec_sse4<T>(_mm_load_si128(reinterpret_cast<const __m128i*>(aligned)));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> load(
    Full<float, SSE4>, const float* SIMD_RESTRICT aligned) {
  return vec_sse4<float>(_mm_load_ps(aligned));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> load(
    Full<double, SSE4>, const double* SIMD_RESTRICT aligned) {
  return vec_sse4<double>(_mm_load_pd(aligned));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> load_unaligned(
    Full<T, SSE4>, const T* SIMD_RESTRICT p) {
  return vec_sse4<T>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> load_unaligned(
    Full<float, SSE4>, const float* SIMD_RESTRICT p) {
  return vec_sse4<float>(_mm_loadu_ps(p));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> load_unaligned(
    Full<double, SSE4>, const double* SIMD_RESTRICT p) {
  return vec_sse4<double>(_mm_loadu_pd(p));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, 8 / sizeof(T)> load(
    Desc<T, 8 / sizeof(T), SSE4>, const T* SIMD_RESTRICT p) {
  return vec_sse4<T, 8 / sizeof(T)>(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, 2> load(
    Desc<float, 2, SSE4>, const float* SIMD_RESTRICT p) {
  const __m128 hi = _mm_setzero_ps();
  return vec_sse4<float, 2>(
      _mm_loadl_pi(hi, reinterpret_cast<const __m64*>(p)));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, 1> load(
    Desc<double, 1, SSE4>, const double* SIMD_RESTRICT p) {
  const __m128d hi = _mm_setzero_pd();
  return vec_sse4<double, 1>(_mm_loadl_pd(hi, p));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, 4 / sizeof(T)> load(
    Desc<T, 4 / sizeof(T), SSE4>, const T* SIMD_RESTRICT p) {
  // TODO(janwas): load_ss?
  int32_t bits;
  CopyBytes<4>(p, &bits);
  return vec_sse4<T, 4 / sizeof(T)>(_mm_cvtsi32_si128(bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, 1> load(
    Desc<float, 1, SSE4>, const float* SIMD_RESTRICT p) {
  return vec_sse4<float, 1>(_mm_load_ss(p));
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> load_dup128(
    Full<T, SSE4> d, const T* const SIMD_RESTRICT p) {
  return load_unaligned(d, p);
}

// ------------------------------ Store

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<T> v, Full<T, SSE4>,
                                      T* SIMD_RESTRICT aligned) {
  _mm_store_si128(reinterpret_cast<__m128i*>(aligned), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<float> v,
                                      Full<float, SSE4>,
                                      float* SIMD_RESTRICT aligned) {
  _mm_store_ps(aligned, v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<double> v,
                                      Full<double, SSE4>,
                                      double* SIMD_RESTRICT aligned) {
  _mm_store_pd(aligned, v.raw);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned(const vec_sse4<T> v,
                                                Full<T, SSE4>,
                                                T* SIMD_RESTRICT p) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned(const vec_sse4<float> v,
                                                Full<float, SSE4>,
                                                float* SIMD_RESTRICT p) {
  _mm_storeu_ps(p, v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned(const vec_sse4<double> v,
                                                Full<double, SSE4>,
                                                double* SIMD_RESTRICT p) {
  _mm_storeu_pd(p, v.raw);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<T, 8 / sizeof(T)> v,
                                      Desc<T, 8 / sizeof(T), SSE4>,
                                      T* SIMD_RESTRICT p) {
  _mm_storel_epi64(reinterpret_cast<__m128i*>(p), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<float, 2> v,
                                      Desc<float, 2, SSE4>,
                                      float* SIMD_RESTRICT p) {
  _mm_storel_pi(reinterpret_cast<__m64*>(p), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<double, 1> v,
                                      Desc<double, 1, SSE4>,
                                      double* SIMD_RESTRICT p) {
  _mm_storel_pd(p, v.raw);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<T, 4 / sizeof(T)> v,
                                      Desc<T, 4 / sizeof(T), SSE4>,
                                      T* SIMD_RESTRICT p) {
  // _mm_storeu_si32 is documented but unavailable in Clang; CopyBytes generates
  // bad code; type-punning is unsafe; this actually generates MOVD.
  _mm_store_ss(reinterpret_cast<float * SIMD_RESTRICT>(p),
               _mm_castsi128_ps(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<float, 1> v,
                                      Desc<float, 1, SSE4>,
                                      float* SIMD_RESTRICT p) {
  _mm_store_ss(p, v.raw);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void stream(const vec_sse4<T> v, Full<T, SSE4>,
                                       T* SIMD_RESTRICT aligned) {
  _mm_stream_si128(reinterpret_cast<__m128i*>(aligned), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void stream(const vec_sse4<float> v,
                                       Full<float, SSE4>,
                                       float* SIMD_RESTRICT aligned) {
  _mm_stream_ps(aligned, v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void stream(const vec_sse4<double> v,
                                       Full<double, SSE4>,
                                       double* SIMD_RESTRICT aligned) {
  _mm_stream_pd(aligned, v.raw);
}

// ------------------------------ Gather

#if SIMD_TARGET_VALUE == SIMD_AVX2

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> gather_offset_impl(
    char (&sizeof_t)[4], Full<T, SSE4>, const T* SIMD_RESTRICT base,
    const vec_sse4<int32_t> offset) {
  return vec_sse4<T>(_mm_i32gather_epi32(reinterpret_cast<const int32_t*>(base),
                                         offset.raw, 1));
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> gather_index_impl(
    char (&sizeof_t)[4], Full<T, SSE4>, const T* SIMD_RESTRICT base,
    const vec_sse4<int32_t> index) {
  return vec_sse4<T>(_mm_i32gather_epi32(reinterpret_cast<const int32_t*>(base),
                                         index.raw, 4));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> gather_offset_impl(
    char (&sizeof_t)[8], Full<T, SSE4>, const T* SIMD_RESTRICT base,
    const vec_sse4<int64_t> offset) {
  return vec_sse4<T>(_mm_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), offset.raw, 1));
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> gather_index_impl(
    char (&sizeof_t)[8], Full<T, SSE4>, const T* SIMD_RESTRICT base,
    const vec_sse4<int64_t> index) {
  return vec_sse4<T>(_mm_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), index.raw, 8));
}

template <typename T, typename Offset>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> gather_offset(
    Full<T, SSE4> d, const T* SIMD_RESTRICT base,
    const vec_sse4<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "SVE requires same size base/ofs");
  char sizeof_t[sizeof(T)];
  return gather_offset_impl(sizeof_t, d, base, offset);
}
template <typename T, typename Index>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> gather_index(
    Full<T, SSE4> d, const T* SIMD_RESTRICT base, const vec_sse4<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "SVE requires same size base/idx");
  char sizeof_t[sizeof(T)];
  return gather_index_impl(sizeof_t, d, base, index);
}

template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> gather_offset<float>(
    Full<float, SSE4>, const float* SIMD_RESTRICT base,
    const vec_sse4<int32_t> offset) {
  return vec_sse4<float>(_mm_i32gather_ps(base, offset.raw, 1));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> gather_index<float>(
    Full<float, SSE4>, const float* SIMD_RESTRICT base,
    const vec_sse4<int32_t> index) {
  return vec_sse4<float>(_mm_i32gather_ps(base, index.raw, 4));
}

template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> gather_offset<double>(
    Full<double, SSE4>, const double* SIMD_RESTRICT base,
    const vec_sse4<int64_t> offset) {
  return vec_sse4<double>(_mm_i64gather_pd(base, offset.raw, 1));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> gather_index<double>(
    Full<double, SSE4>, const double* SIMD_RESTRICT base,
    const vec_sse4<int64_t> index) {
  return vec_sse4<double>(_mm_i64gather_pd(base, index.raw, 8));
}

}  // namespace ext

#endif  // SIMD_TARGET_VALUE == SIMD_AVX2

// ================================================== SWIZZLE

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> shift_left_bytes(const vec_sse4<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  return vec_sse4<T>(_mm_slli_si128(v.raw, kBytes));
}

template <int kLanes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> shift_left_lanes(const vec_sse4<T> v) {
  return shift_left_bytes<kLanes * sizeof(T)>(v);
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> shift_right_bytes(const vec_sse4<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  return vec_sse4<T>(_mm_srli_si128(v.raw, kBytes));
}

template <int kLanes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> shift_right_lanes(const vec_sse4<T> v) {
  return shift_right_bytes<kLanes * sizeof(T)>(v);
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> combine_shift_right_bytes(
    const vec_sse4<T> hi, const vec_sse4<T> lo) {
  const Full<uint8_t, SSE4> d8;
  const vec_sse4<uint8_t> extracted_bytes(
      _mm_alignr_epi8(cast_to(d8, hi).raw, cast_to(d8, lo).raw, kBytes));
  return cast_to(Full<T, SSE4>(), extracted_bytes);
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t> broadcast(
    const vec_sse4<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m128i lo = _mm_shufflelo_epi16(v.raw, 0x55 * kLane);
    return vec_sse4<uint16_t>(_mm_unpacklo_epi64(lo, lo));
  } else {
    const __m128i hi = _mm_shufflehi_epi16(v.raw, 0x55 * (kLane - 4));
    return vec_sse4<uint16_t>(_mm_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> broadcast(
    const vec_sse4<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return vec_sse4<uint32_t>(_mm_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t> broadcast(
    const vec_sse4<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return vec_sse4<uint64_t>(_mm_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Signed
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t> broadcast(
    const vec_sse4<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m128i lo = _mm_shufflelo_epi16(v.raw, 0x55 * kLane);
    return vec_sse4<int16_t>(_mm_unpacklo_epi64(lo, lo));
  } else {
    const __m128i hi = _mm_shufflehi_epi16(v.raw, 0x55 * (kLane - 4));
    return vec_sse4<int16_t>(_mm_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> broadcast(
    const vec_sse4<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return vec_sse4<int32_t>(_mm_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t> broadcast(
    const vec_sse4<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return vec_sse4<int64_t>(_mm_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Float
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> broadcast(const vec_sse4<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return vec_sse4<float>(_mm_shuffle_ps(v.raw, v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> broadcast(
    const vec_sse4<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return vec_sse4<double>(_mm_shuffle_pd(v.raw, v.raw, 3 * kLane));
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> table_lookup_bytes(
    const vec_sse4<T> bytes, const vec_sse4<TI> from) {
  return vec_sse4<T>(_mm_shuffle_epi8(bytes.raw, from.raw));
}

// ------------------------------ Hard-coded shuffles

// Notation: let vec_sse4<int32_t> have lanes 3,2,1,0 (0 is least-significant).
// shuffle_0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// combine_shift_right_bytes but the shuffle_abcd notation is more convenient.

// Swap 64-bit halves
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> shuffle_1032(
    const vec_sse4<uint32_t> v) {
  return vec_sse4<uint32_t>(_mm_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> shuffle_1032(
    const vec_sse4<int32_t> v) {
  return vec_sse4<int32_t>(_mm_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> shuffle_1032(
    const vec_sse4<float> v) {
  return vec_sse4<float>(_mm_shuffle_ps(v.raw, v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t> shuffle_01(
    const vec_sse4<uint64_t> v) {
  return vec_sse4<uint64_t>(_mm_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t> shuffle_01(
    const vec_sse4<int64_t> v) {
  return vec_sse4<int64_t>(_mm_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> shuffle_01(
    const vec_sse4<double> v) {
  return vec_sse4<double>(_mm_shuffle_pd(v.raw, v.raw, 1));
}

// Rotate right 32 bits
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> shuffle_0321(
    const vec_sse4<uint32_t> v) {
  return vec_sse4<uint32_t>(_mm_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> shuffle_0321(
    const vec_sse4<int32_t> v) {
  return vec_sse4<int32_t>(_mm_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> shuffle_0321(
    const vec_sse4<float> v) {
  return vec_sse4<float>(_mm_shuffle_ps(v.raw, v.raw, 0x39));
}
// Rotate left 32 bits
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> shuffle_2103(
    const vec_sse4<uint32_t> v) {
  return vec_sse4<uint32_t>(_mm_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> shuffle_2103(
    const vec_sse4<int32_t> v) {
  return vec_sse4<int32_t>(_mm_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> shuffle_2103(
    const vec_sse4<float> v) {
  return vec_sse4<float>(_mm_shuffle_ps(v.raw, v.raw, 0x93));
}

// Reverse
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> shuffle_0123(
    const vec_sse4<uint32_t> v) {
  return vec_sse4<uint32_t>(_mm_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> shuffle_0123(
    const vec_sse4<int32_t> v) {
  return vec_sse4<int32_t>(_mm_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> shuffle_0123(
    const vec_sse4<float> v) {
  return vec_sse4<float>(_mm_shuffle_ps(v.raw, v.raw, 0x1B));
}

// ------------------------------ Permute (runtime variable)

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE permute_sse4<T> set_table_indices(const Full<T, SSE4> d,
                                                       const int32_t* idx) {
  const Full<uint8_t, SSE4> d8;
  SIMD_ALIGN uint8_t control[d8.N];
  for (size_t idx_byte = 0; idx_byte < d8.N; ++idx_byte) {
    const size_t idx_lane = idx_byte / sizeof(T);
    const size_t mod = idx_byte % sizeof(T);
    control[idx_byte] = idx[idx_lane] * sizeof(T) + mod;
  }
  return permute_sse4<T>{load(d8, control).raw};
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> table_lookup_lanes(
    const vec_sse4<uint32_t> v, const permute_sse4<uint32_t> idx) {
  return table_lookup_bytes(v, vec_sse4<uint8_t>(idx.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> table_lookup_lanes(
    const vec_sse4<int32_t> v, const permute_sse4<int32_t> idx) {
  return table_lookup_bytes(v, vec_sse4<uint8_t>(idx.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> table_lookup_lanes(
    const vec_sse4<float> v, const permute_sse4<float> idx) {
  const Full<int32_t, SSE4> di;
  const Full<float, SSE4> df;
  return cast_to(
      df, table_lookup_bytes(cast_to(di, v), vec_sse4<uint8_t>(idx.raw)));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t> interleave_lo(
    const vec_sse4<uint8_t> a, const vec_sse4<uint8_t> b) {
  return vec_sse4<uint8_t>(_mm_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t> interleave_lo(
    const vec_sse4<uint16_t> a, const vec_sse4<uint16_t> b) {
  return vec_sse4<uint16_t>(_mm_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> interleave_lo(
    const vec_sse4<uint32_t> a, const vec_sse4<uint32_t> b) {
  return vec_sse4<uint32_t>(_mm_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t> interleave_lo(
    const vec_sse4<uint64_t> a, const vec_sse4<uint64_t> b) {
  return vec_sse4<uint64_t>(_mm_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t> interleave_lo(
    const vec_sse4<int8_t> a, const vec_sse4<int8_t> b) {
  return vec_sse4<int8_t>(_mm_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t> interleave_lo(
    const vec_sse4<int16_t> a, const vec_sse4<int16_t> b) {
  return vec_sse4<int16_t>(_mm_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> interleave_lo(
    const vec_sse4<int32_t> a, const vec_sse4<int32_t> b) {
  return vec_sse4<int32_t>(_mm_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t> interleave_lo(
    const vec_sse4<int64_t> a, const vec_sse4<int64_t> b) {
  return vec_sse4<int64_t>(_mm_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> interleave_lo(
    const vec_sse4<float> a, const vec_sse4<float> b) {
  return vec_sse4<float>(_mm_unpacklo_ps(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> interleave_lo(
    const vec_sse4<double> a, const vec_sse4<double> b) {
  return vec_sse4<double>(_mm_unpacklo_pd(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t> interleave_hi(
    const vec_sse4<uint8_t> a, const vec_sse4<uint8_t> b) {
  return vec_sse4<uint8_t>(_mm_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t> interleave_hi(
    const vec_sse4<uint16_t> a, const vec_sse4<uint16_t> b) {
  return vec_sse4<uint16_t>(_mm_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> interleave_hi(
    const vec_sse4<uint32_t> a, const vec_sse4<uint32_t> b) {
  return vec_sse4<uint32_t>(_mm_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t> interleave_hi(
    const vec_sse4<uint64_t> a, const vec_sse4<uint64_t> b) {
  return vec_sse4<uint64_t>(_mm_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t> interleave_hi(
    const vec_sse4<int8_t> a, const vec_sse4<int8_t> b) {
  return vec_sse4<int8_t>(_mm_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t> interleave_hi(
    const vec_sse4<int16_t> a, const vec_sse4<int16_t> b) {
  return vec_sse4<int16_t>(_mm_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> interleave_hi(
    const vec_sse4<int32_t> a, const vec_sse4<int32_t> b) {
  return vec_sse4<int32_t>(_mm_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t> interleave_hi(
    const vec_sse4<int64_t> a, const vec_sse4<int64_t> b) {
  return vec_sse4<int64_t>(_mm_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> interleave_hi(
    const vec_sse4<float> a, const vec_sse4<float> b) {
  return vec_sse4<float>(_mm_unpackhi_ps(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> interleave_hi(
    const vec_sse4<double> a, const vec_sse4<double> b) {
  return vec_sse4<double>(_mm_unpackhi_pd(a.raw, b.raw));
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t> zip_lo(
    const vec_sse4<uint8_t> a, const vec_sse4<uint8_t> b) {
  return vec_sse4<uint16_t>(_mm_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> zip_lo(
    const vec_sse4<uint16_t> a, const vec_sse4<uint16_t> b) {
  return vec_sse4<uint32_t>(_mm_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t> zip_lo(
    const vec_sse4<uint32_t> a, const vec_sse4<uint32_t> b) {
  return vec_sse4<uint64_t>(_mm_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t> zip_lo(const vec_sse4<int8_t> a,
                                                    const vec_sse4<int8_t> b) {
  return vec_sse4<int16_t>(_mm_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> zip_lo(const vec_sse4<int16_t> a,
                                                    const vec_sse4<int16_t> b) {
  return vec_sse4<int32_t>(_mm_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t> zip_lo(const vec_sse4<int32_t> a,
                                                    const vec_sse4<int32_t> b) {
  return vec_sse4<int64_t>(_mm_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t> zip_hi(
    const vec_sse4<uint8_t> a, const vec_sse4<uint8_t> b) {
  return vec_sse4<uint16_t>(_mm_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> zip_hi(
    const vec_sse4<uint16_t> a, const vec_sse4<uint16_t> b) {
  return vec_sse4<uint32_t>(_mm_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t> zip_hi(
    const vec_sse4<uint32_t> a, const vec_sse4<uint32_t> b) {
  return vec_sse4<uint64_t>(_mm_unpackhi_epi32(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t> zip_hi(const vec_sse4<int8_t> a,
                                                    const vec_sse4<int8_t> b) {
  return vec_sse4<int16_t>(_mm_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> zip_hi(const vec_sse4<int16_t> a,
                                                    const vec_sse4<int16_t> b) {
  return vec_sse4<int32_t>(_mm_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t> zip_hi(const vec_sse4<int32_t> a,
                                                    const vec_sse4<int32_t> b) {
  return vec_sse4<int64_t>(_mm_unpackhi_epi32(a.raw, b.raw));
}

// ------------------------------ Parts

// Returns a part with value "t".
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, 1> set_part(
    Desc<uint16_t, 1, SSE4>, const uint16_t t) {
  return vec_sse4<uint16_t, 1>(_mm_cvtsi32_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, 1> set_part(Desc<int16_t, 1, SSE4>,
                                                         const int16_t t) {
  return vec_sse4<int16_t, 1>(_mm_cvtsi32_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t, 1> set_part(Desc<uint32_t, 1, SSE4>,
                                                     const uint32_t t) {
  return vec_sse4<uint32_t, 1>(_mm_cvtsi32_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, 1> set_part(Desc<int32_t, 1, SSE4>,
                                                    const int32_t t) {
  return vec_sse4<int32_t, 1>(_mm_cvtsi32_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, 1> set_part(Desc<float, 1, SSE4>,
                                                  const float t) {
  return vec_sse4<float, 1>(_mm_set_ss(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t, 1> set_part(Desc<uint64_t, 1, SSE4>,
                                                     const uint64_t t) {
  return vec_sse4<uint64_t, 1>(_mm_cvtsi64_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t, 1> set_part(Desc<int64_t, 1, SSE4>,
                                                    const int64_t t) {
  return vec_sse4<int64_t, 1>(_mm_cvtsi64_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, 1> set_part(Desc<double, 1, SSE4>,
                                                   const double t) {
  return vec_sse4<double, 1>(_mm_set_sd(t));
}

// Gets the single value stored in a vector/part.
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE uint16_t get_part(Desc<uint16_t, 1, SSE4>,
                                             const vec_sse4<uint16_t, N> v) {
  return _mm_cvtsi128_si32(v.raw) & 0xFFFF;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE int16_t get_part(Desc<int16_t, 1, SSE4>,
                                            const vec_sse4<int16_t, N> v) {
  return _mm_cvtsi128_si32(v.raw) & 0xFFFF;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE uint32_t get_part(Desc<uint32_t, 1, SSE4>,
                                        const vec_sse4<uint32_t, N> v) {
  return _mm_cvtsi128_si32(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE int32_t get_part(Desc<int32_t, 1, SSE4>,
                                       const vec_sse4<int32_t, N> v) {
  return _mm_cvtsi128_si32(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE float get_part(Desc<float, 1, SSE4>,
                                     const vec_sse4<float, N> v) {
  return _mm_cvtss_f32(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE uint64_t get_part(Desc<uint64_t, 1, SSE4>,
                                        const vec_sse4<uint64_t, N> v) {
  return _mm_cvtsi128_si64(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE int64_t get_part(Desc<int64_t, 1, SSE4>,
                                       const vec_sse4<int64_t, N> v) {
  return _mm_cvtsi128_si64(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE double get_part(Desc<double, 1, SSE4>,
                                      const vec_sse4<double, N> v) {
  return _mm_cvtsd_f64(v.raw);
}

// Returns part of a vector (unspecified whether upper or lower).
template <typename T, size_t N, size_t VN>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> any_part(Desc<T, N, SSE4>,
                                                   const vec_sse4<T, VN> v) {
  return vec_sse4<T, N>(v.raw);
}

// Returns full vector with the given part's lane broadcasted. Note that
// callers cannot use broadcast directly because part lane order is undefined.
template <int kLane, typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> broadcast_part(Full<T, SSE4>,
                                                      const vec_sse4<T, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return broadcast<kLane>(vec_sse4<T>(v.raw));
}

// Returns upper/lower half of a vector.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, 8 / sizeof(T)> get_half(
    Lower, const vec_sse4<T> v) {
  return vec_sse4<T, 8 / sizeof(T)>(v.raw);
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, 8 / sizeof(T)> lower_half(
    const vec_sse4<T> v) {
  return get_half(Lower(), v);
}

// These copy hi into lo (smaller instruction encoding than shifts).
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, 8 / sizeof(T)> get_half(
    Upper, const vec_sse4<T> v) {
  return vec_sse4<T, 8 / sizeof(T)>(_mm_unpackhi_epi64(v.raw, v.raw));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, 2> get_half(
    Upper, const vec_sse4<float> v) {
  return vec_sse4<float, 2>(_mm_movehl_ps(v.raw, v.raw));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double, 1> get_half(
    Upper, const vec_sse4<double> v) {
  return vec_sse4<double, 1>(_mm_unpackhi_pd(v.raw, v.raw));
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, 8 / sizeof(T)> upper_half(
    const vec_sse4<T> v) {
  return get_half(Upper(), v);
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> concat_lo_lo(const vec_sse4<T> hi,
                                                    const vec_sse4<T> lo) {
  const Full<uint64_t, SSE4> d64;
  return cast_to(Full<T, SSE4>(),
                 interleave_lo(cast_to(d64, lo), cast_to(d64, hi)));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> concat_hi_hi(const vec_sse4<T> hi,
                                                    const vec_sse4<T> lo) {
  const Full<uint64_t, SSE4> d64;
  return cast_to(Full<T, SSE4>(),
                 interleave_hi(cast_to(d64, lo), cast_to(d64, hi)));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves)
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> concat_lo_hi(const vec_sse4<T> hi,
                                                    const vec_sse4<T> lo) {
  return combine_shift_right_bytes<8>(hi, lo);
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> concat_hi_lo(const vec_sse4<T> hi,
                                                    const vec_sse4<T> lo) {
  return vec_sse4<T>(_mm_blend_epi16(hi.raw, lo.raw, 0x0F));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> concat_hi_lo(
    const vec_sse4<float> hi, const vec_sse4<float> lo) {
  return vec_sse4<float>(_mm_blend_ps(hi.raw, lo.raw, 3));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> concat_hi_lo(
    const vec_sse4<double> hi, const vec_sse4<double> lo) {
  return vec_sse4<double>(_mm_blend_pd(hi.raw, lo.raw, 1));
}

// ------------------------------ Odd/even lanes

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> odd_even_impl(char (&sizeof_t)[1],
                                                     const vec_sse4<T> a,
                                                     const vec_sse4<T> b) {
  const Full<T, SSE4> d;
  const Full<uint8_t, SSE4> d8;
  SIMD_ALIGN constexpr uint8_t mask[16] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
                                           0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
  return select(a, b, cast_to(d, load(d8, mask)));
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> odd_even_impl(char (&sizeof_t)[2],
                                                     const vec_sse4<T> a,
                                                     const vec_sse4<T> b) {
  return vec_sse4<T>(_mm_blend_epi16(a.raw, b.raw, 0x55));
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> odd_even_impl(char (&sizeof_t)[4],
                                                     const vec_sse4<T> a,
                                                     const vec_sse4<T> b) {
  return vec_sse4<T>(_mm_blend_epi16(a.raw, b.raw, 0x33));
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> odd_even_impl(char (&sizeof_t)[8],
                                                     const vec_sse4<T> a,
                                                     const vec_sse4<T> b) {
  return vec_sse4<T>(_mm_blend_epi16(a.raw, b.raw, 0x0F));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> odd_even(const vec_sse4<T> a,
                                                const vec_sse4<T> b) {
  char sizeof_t[sizeof(T)];
  return odd_even_impl(sizeof_t, a, b);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float> odd_even<float>(
    const vec_sse4<float> a, const vec_sse4<float> b) {
  return vec_sse4<float>(_mm_blend_ps(a.raw, b.raw, 5));
}

template <>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> odd_even<double>(
    const vec_sse4<double> a, const vec_sse4<double> b) {
  return vec_sse4<double>(_mm_blend_pd(a.raw, b.raw, 1));
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<double> convert_to(
    Full<double, SSE4>, const vec_sse4<float, 2> v) {
  return vec_sse4<double>(_mm_cvtps_pd(v.raw));
}

// Unsigned: zero-extend.
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t> convert_to(
    Full<uint16_t, SSE4>, const vec_sse4<uint8_t, 8> v) {
  return vec_sse4<uint16_t>(_mm_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> convert_to(
    Full<uint32_t, SSE4>, const vec_sse4<uint8_t, 4> v) {
  return vec_sse4<uint32_t>(_mm_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t> convert_to(
    Full<int16_t, SSE4>, const vec_sse4<uint8_t, 8> v) {
  return vec_sse4<int16_t>(_mm_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> convert_to(
    Full<int32_t, SSE4>, const vec_sse4<uint8_t, 4> v) {
  return vec_sse4<int32_t>(_mm_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> convert_to(
    Full<uint32_t, SSE4>, const vec_sse4<uint16_t, 4> v) {
  return vec_sse4<uint32_t>(_mm_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> convert_to(
    Full<int32_t, SSE4>, const vec_sse4<uint16_t, 4> v) {
  return vec_sse4<int32_t>(_mm_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t> convert_to(
    Full<uint64_t, SSE4>, const vec_sse4<uint32_t, 2> v) {
  return vec_sse4<uint64_t>(_mm_cvtepu32_epi64(v.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint32_t> u32_from_u8(
    const vec_sse4<uint8_t> v) {
  return vec_sse4<uint32_t>(_mm_cvtepu8_epi32(v.raw));
}

// Signed: replicate sign bit.
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t> convert_to(
    Full<int16_t, SSE4>, const vec_sse4<int8_t, 8> v) {
  return vec_sse4<int16_t>(_mm_cvtepi8_epi16(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> convert_to(
    Full<int32_t, SSE4>, const vec_sse4<int8_t, 4> v) {
  return vec_sse4<int32_t>(_mm_cvtepi8_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t> convert_to(
    Full<int32_t, SSE4>, const vec_sse4<int16_t, 4> v) {
  return vec_sse4<int32_t>(_mm_cvtepi16_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int64_t> convert_to(
    Full<int64_t, SSE4>, const vec_sse4<int32_t, 2> v) {
  return vec_sse4<int64_t>(_mm_cvtepi32_epi64(v.raw));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t, N> convert_to(
    Part<uint16_t, N, SSE4>, const vec_sse4<int32_t, N> v) {
  return vec_sse4<uint16_t, N>(_mm_packus_epi32(v.raw, v.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> convert_to(
    Part<uint8_t, N, SSE4>, const vec_sse4<int32_t> v) {
  const __m128i u16 = _mm_packus_epi32(v.raw, v.raw);
  return vec_sse4<uint8_t, N>(_mm_packus_epi16(u16, u16));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, N> convert_to(
    Part<uint8_t, N, SSE4>, const vec_sse4<int16_t> v) {
  return vec_sse4<uint8_t, N>(_mm_packus_epi16(v.raw, v.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t, N> convert_to(
    Part<int16_t, N, SSE4>, const vec_sse4<int32_t> v) {
  return vec_sse4<int16_t, N>(_mm_packs_epi32(v.raw, v.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> convert_to(
    Part<int8_t, N, SSE4>, const vec_sse4<int32_t> v) {
  const __m128i i16 = _mm_packs_epi32(v.raw, v.raw);
  return vec_sse4<int8_t, N>(_mm_packs_epi16(i16, i16));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int8_t, N> convert_to(
    Part<int8_t, N, SSE4>, const vec_sse4<int16_t> v) {
  return vec_sse4<int8_t, N>(_mm_packs_epi16(v.raw, v.raw));
}

// For already range-limited input [0, 255].
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint8_t, 4> u8_from_u32(
    const vec_sse4<uint32_t> v) {
  const Full<uint32_t, SSE4> d32;
  const Full<uint8_t, SSE4> d8;
  SIMD_ALIGN static constexpr uint32_t k8From32[4] = {0x0C080400u, 0x0C080400u,
                                                      0x0C080400u, 0x0C080400u};
  // Replicate bytes into all 32 bit lanes for any_part.
  const auto quad = table_lookup_bytes(v, load(d32, k8From32));
  return any_part(Part<uint8_t, 4, SSE4>(), cast_to(d8, quad));
}

// ------------------------------ Convert i32 <=> f32

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<float, N> convert_to(
    Part<float, N, SSE4>, const vec_sse4<int32_t, N> v) {
  return vec_sse4<float, N>(_mm_cvtepi32_ps(v.raw));
}
// Truncates (rounds toward zero).
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> convert_to(
    Part<int32_t, N, SSE4>, const vec_sse4<float, N> v) {
  return vec_sse4<int32_t, N>(_mm_cvttps_epi32(v.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int32_t, N> nearest_int(
    const vec_sse4<float, N> v) {
  return vec_sse4<int32_t, N>(_mm_cvtps_epi32(v.raw));
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ movemask

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..15 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_ATTR_SSE4 SIMD_INLINE uint32_t movemask(const vec_sse4<uint8_t> v) {
  return _mm_movemask_epi8(v.raw);
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_ATTR_SSE4 SIMD_INLINE uint32_t movemask(const vec_sse4<float> v) {
  return _mm_movemask_ps(v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE uint32_t movemask(const vec_sse4<double> v) {
  return _mm_movemask_pd(v.raw);
}

// ------------------------------ all_zero

// Returns whether all lanes are equal to zero. Supported for all integer V.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE bool all_zero(const vec_sse4<T> v) {
  return static_cast<bool>(_mm_testz_si128(v.raw, v.raw));
}

// ------------------------------ minpos

// Returns index and min value in lanes 1 and 0.
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint16_t> minpos(
    const vec_sse4<uint16_t> v) {
  return vec_sse4<uint16_t>(_mm_minpos_epu16(v.raw));
}

// ------------------------------ Horizontal sum (reduction)

// Returns 64-bit sums of 8-byte groups.
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<uint64_t> sums_of_u8x8(
    const vec_sse4<uint8_t> v) {
  return vec_sse4<uint64_t>(_mm_sad_epu8(v.raw, _mm_setzero_si128()));
}

// Returns N sums of differences of byte quadruplets, starting from byte offset
// i = [0, N) in window (11 consecutive bytes) and idx_ref * 4 in ref.
template <int idx_ref>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<int16_t> mpsadbw(
    const vec_sse4<uint8_t> window, const vec_sse4<uint8_t> ref) {
  return vec_sse4<int16_t>(_mm_mpsadbw_epu8(window.raw, ref.raw, idx_ref));
}

// For u32/i32/f32.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> horz_sum_impl(
    char (&sizeof_t)[4], const vec_sse4<T, N> v3210) {
  const vec_sse4<T> v1032 = shuffle_1032(v3210);
  const vec_sse4<T> v31_20_31_20 = v3210 + v1032;
  const vec_sse4<T> v20_31_20_31 = shuffle_0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}

// For u64/i64/f64.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> horz_sum_impl(
    char (&sizeof_t)[8], const vec_sse4<T, N> v10) {
  const vec_sse4<T> v01 = shuffle_01(v10);
  return v10 + v01;
}

// Supported for u/i/f 32/64. Returns the sum in each lane.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> sum_of_lanes(const vec_sse4<T, N> v) {
  char sizeof_t[sizeof(T)];
  return horz_sum_impl(sizeof_t, v);
}

}  // namespace ext

// TODO(janwas): wrappers for all intrinsics (in x86 namespace).
}  // namespace pik

#endif  // SIMD_ENABLE & SIMD_SSE4
#endif  // PIK_SIMD_X86_SSE4_H_
