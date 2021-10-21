// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SIMD_X86_AVX2_H_
#define PIK_SIMD_X86_AVX2_H_

// 256-bit AVX2 vectors and operations.
// WARNING: most operations do not cross 128-bit block boundaries. In
// particular, "broadcast", pack and zip behavior may be surprising.

#include "pik/simd/compiler_specific.h"
#include "pik/simd/shared.h"
#include "pik/simd/targets.h"
#include "pik/simd/x86_sse4.h"

#if SIMD_ENABLE & SIMD_AVX2
#include <immintrin.h>

namespace pik {

template <class Target>
struct PartTargetT<2, Target> {
  using type = AVX2;
};

template <typename T>
struct raw_avx2 {
  using type = __m256i;
};
template <>
struct raw_avx2<float> {
  using type = __m256;
};
template <>
struct raw_avx2<double> {
  using type = __m256d;
};

// Returned by set_table_indices for use by table_lookup_lanes.
template <typename T>
struct permute_avx2 {
  __m256i raw;
};

template <typename T, size_t N = AVX2::NumLanes<T>()>
class vec_avx2 {
  using Raw = typename raw_avx2<T>::type;

 public:
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2() {}
  vec_avx2(const vec_avx2&) = default;
  vec_avx2& operator=(const vec_avx2&) = default;
  SIMD_ATTR_AVX2 SIMD_INLINE explicit vec_avx2(const Raw raw) : raw(raw) {}

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator*=(const vec_avx2 other) {
    return *this = (*this * other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator/=(const vec_avx2 other) {
    return *this = (*this / other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator+=(const vec_avx2 other) {
    return *this = (*this + other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator-=(const vec_avx2 other) {
    return *this = (*this - other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator&=(const vec_avx2 other) {
    return *this = (*this & other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator|=(const vec_avx2 other) {
    return *this = (*this | other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator^=(const vec_avx2 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

template <typename T, size_t N>
struct VecT<T, N, AVX2> {
  using type = vec_avx2<T, N>;
};

using u8x32 = vec_avx2<uint8_t, 32>;
using u16x16 = vec_avx2<uint16_t, 16>;
using u32x8 = vec_avx2<uint32_t, 8>;
using u64x4 = vec_avx2<uint64_t, 4>;
using i8x32 = vec_avx2<int8_t, 32>;
using i16x16 = vec_avx2<int16_t, 16>;
using i32x8 = vec_avx2<int32_t, 8>;
using i64x4 = vec_avx2<int64_t, 4>;
using f32x8 = vec_avx2<float, 8>;
using f64x4 = vec_avx2<double, 4>;

// ------------------------------ Cast

SIMD_ATTR_AVX2 SIMD_INLINE __m256i BitCastToInteger(__m256i v) { return v; }
SIMD_ATTR_AVX2 SIMD_INLINE __m256i BitCastToInteger(__m256 v) {
  return _mm256_castps_si256(v);
}
SIMD_ATTR_AVX2 SIMD_INLINE __m256i BitCastToInteger(__m256d v) {
  return _mm256_castpd_si256(v);
}

// cast_to_u8
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> cast_to_u8(
    Desc<uint8_t, N, AVX2>, vec_avx2<T, N / sizeof(T)> v) {
  return vec_avx2<uint8_t, N>(BitCastToInteger(v.raw));
}

// Cannot rely on function overloading because return types differ.
template <typename T>
struct BitCastFromIntegerAVX2 {
  SIMD_ATTR_AVX2 SIMD_INLINE __m256i operator()(__m256i v) { return v; }
};
template <>
struct BitCastFromIntegerAVX2<float> {
  SIMD_ATTR_AVX2 SIMD_INLINE __m256 operator()(__m256i v) {
    return _mm256_castsi256_ps(v);
  }
};
template <>
struct BitCastFromIntegerAVX2<double> {
  SIMD_ATTR_AVX2 SIMD_INLINE __m256d operator()(__m256i v) {
    return _mm256_castsi256_pd(v);
  }
};

// cast_u8_to
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> cast_u8_to(
    Desc<T, N, AVX2>, vec_avx2<uint8_t, N * sizeof(T)> v) {
  return vec_avx2<T, N>(BitCastFromIntegerAVX2<T>()(v.raw));
}

// cast_to
template <typename T, size_t N, typename FromT>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> cast_to(
    Desc<T, N, AVX2> d, vec_avx2<FromT, N * sizeof(T) / sizeof(FromT)> v) {
  const auto u8 = cast_to_u8(Desc<uint8_t, N * sizeof(T), AVX2>(), v);
  return cast_u8_to(d, u8);
}

// ------------------------------ Set

// Returns an all-zero vector.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> setzero(Desc<T, N, AVX2>) {
  return vec_avx2<T, N>(_mm256_setzero_si256());
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> setzero(Desc<float, N, AVX2>) {
  return vec_avx2<float, N>(_mm256_setzero_ps());
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> setzero(Desc<double, N, AVX2>) {
  return vec_avx2<double, N>(_mm256_setzero_pd());
}

template <typename T, size_t N, typename T2>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> iota(Desc<T, N, AVX2> d,
                                               const T2 first) {
  SIMD_ALIGN T lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

// Returns a vector with all lanes set to "t".
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> set1(Desc<uint8_t, N, AVX2>,
                                                     const uint8_t t) {
  return vec_avx2<uint8_t, N>(_mm256_set1_epi8(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> set1(Desc<uint16_t, N, AVX2>,
                                                      const uint16_t t) {
  return vec_avx2<uint16_t, N>(_mm256_set1_epi16(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> set1(Desc<uint32_t, N, AVX2>,
                                                      const uint32_t t) {
  return vec_avx2<uint32_t, N>(_mm256_set1_epi32(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> set1(Desc<uint64_t, N, AVX2>,
                                                      const uint64_t t) {
  return vec_avx2<uint64_t, N>(_mm256_set1_epi64x(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> set1(Desc<int8_t, N, AVX2>,
                                                    const int8_t t) {
  return vec_avx2<int8_t, N>(_mm256_set1_epi8(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> set1(Desc<int16_t, N, AVX2>,
                                                     const int16_t t) {
  return vec_avx2<int16_t, N>(_mm256_set1_epi16(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> set1(Desc<int32_t, N, AVX2>,
                                                     const int32_t t) {
  return vec_avx2<int32_t, N>(_mm256_set1_epi32(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t, N> set1(Desc<int64_t, N, AVX2>,
                                                     const int64_t t) {
  return vec_avx2<int64_t, N>(_mm256_set1_epi64x(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> set1(Desc<float, N, AVX2>,
                                                   const float t) {
  return vec_avx2<float, N>(_mm256_set1_ps(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> set1(Desc<double, N, AVX2>,
                                                    const double t) {
  return vec_avx2<double, N>(_mm256_set1_pd(t));
}

SIMD_DIAGNOSTICS(push)
SIMD_DIAGNOSTICS_OFF(disable : 4700, ignored "-Wuninitialized")

// Returns a vector with uninitialized elements.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> undefined(Desc<T, N, AVX2>) {
#ifdef __clang__
  return vec_avx2<T, N>(_mm256_undefined_si256());
#else
  __m256i raw;
  return vec_avx2<T, N>(raw);
#endif
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> undefined(Desc<float, N, AVX2>) {
#ifdef __clang__
  return vec_avx2<float, N>(_mm256_undefined_ps());
#else
  __m256 raw;
  return vec_avx2<float, N>(raw);
#endif
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> undefined(
    Desc<double, N, AVX2>) {
#ifdef __clang__
  return vec_avx2<double, N>(_mm256_undefined_pd());
#else
  __m256d raw;
  return vec_avx2<double, N>(raw);
#endif
}

SIMD_DIAGNOSTICS(pop)

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> operator+(
    const vec_avx2<uint8_t, N> a, const vec_avx2<uint8_t, N> b) {
  return vec_avx2<uint8_t, N>(_mm256_add_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> operator+(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_add_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> operator+(
    const vec_avx2<uint32_t, N> a, const vec_avx2<uint32_t, N> b) {
  return vec_avx2<uint32_t, N>(_mm256_add_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> operator+(
    const vec_avx2<uint64_t, N> a, const vec_avx2<uint64_t, N> b) {
  return vec_avx2<uint64_t, N>(_mm256_add_epi64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> operator+(
    const vec_avx2<int8_t, N> a, const vec_avx2<int8_t, N> b) {
  return vec_avx2<int8_t, N>(_mm256_add_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> operator+(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_add_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> operator+(
    const vec_avx2<int32_t, N> a, const vec_avx2<int32_t, N> b) {
  return vec_avx2<int32_t, N>(_mm256_add_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t, N> operator+(
    const vec_avx2<int64_t, N> a, const vec_avx2<int64_t, N> b) {
  return vec_avx2<int64_t, N>(_mm256_add_epi64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator+(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_add_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator+(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_add_pd(a.raw, b.raw));
}

// ------------------------------ Subtraction

// Unsigned
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> operator-(
    const vec_avx2<uint8_t, N> a, const vec_avx2<uint8_t, N> b) {
  return vec_avx2<uint8_t, N>(_mm256_sub_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> operator-(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_sub_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> operator-(
    const vec_avx2<uint32_t, N> a, const vec_avx2<uint32_t, N> b) {
  return vec_avx2<uint32_t, N>(_mm256_sub_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> operator-(
    const vec_avx2<uint64_t, N> a, const vec_avx2<uint64_t, N> b) {
  return vec_avx2<uint64_t, N>(_mm256_sub_epi64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> operator-(
    const vec_avx2<int8_t, N> a, const vec_avx2<int8_t, N> b) {
  return vec_avx2<int8_t, N>(_mm256_sub_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> operator-(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_sub_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> operator-(
    const vec_avx2<int32_t, N> a, const vec_avx2<int32_t, N> b) {
  return vec_avx2<int32_t, N>(_mm256_sub_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t, N> operator-(
    const vec_avx2<int64_t, N> a, const vec_avx2<int64_t, N> b) {
  return vec_avx2<int64_t, N>(_mm256_sub_epi64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator-(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_sub_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator-(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_sub_pd(a.raw, b.raw));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> saturated_add(
    const vec_avx2<uint8_t, N> a, const vec_avx2<uint8_t, N> b) {
  return vec_avx2<uint8_t, N>(_mm256_adds_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> saturated_add(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_adds_epu16(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> saturated_add(
    const vec_avx2<int8_t, N> a, const vec_avx2<int8_t, N> b) {
  return vec_avx2<int8_t, N>(_mm256_adds_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> saturated_add(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_adds_epi16(a.raw, b.raw));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> saturated_subtract(
    const vec_avx2<uint8_t, N> a, const vec_avx2<uint8_t, N> b) {
  return vec_avx2<uint8_t, N>(_mm256_subs_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> saturated_subtract(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_subs_epu16(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> saturated_subtract(
    const vec_avx2<int8_t, N> a, const vec_avx2<int8_t, N> b) {
  return vec_avx2<int8_t, N>(_mm256_subs_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> saturated_subtract(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_subs_epi16(a.raw, b.raw));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> average_round(
    const vec_avx2<uint8_t, N> a, const vec_avx2<uint8_t, N> b) {
  return vec_avx2<uint8_t, N>(_mm256_avg_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> average_round(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_avg_epu16(a.raw, b.raw));
}

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> abs(
    const vec_avx2<int8_t, N> v) {
  return vec_avx2<int8_t, N>(_mm256_abs_epi8(v.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> abs(
    const vec_avx2<int16_t, N> v) {
  return vec_avx2<int16_t, N>(_mm256_abs_epi16(v.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> abs(
    const vec_avx2<int32_t, N> v) {
  return vec_avx2<int32_t, N>(_mm256_abs_epi32(v.raw));
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> shift_left(
    const vec_avx2<uint16_t, N> v) {
  return vec_avx2<uint16_t, N>(_mm256_slli_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> shift_right(
    const vec_avx2<uint16_t, N> v) {
  return vec_avx2<uint16_t, N>(_mm256_srli_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> shift_left(
    const vec_avx2<uint32_t, N> v) {
  return vec_avx2<uint32_t, N>(_mm256_slli_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> shift_right(
    const vec_avx2<uint32_t, N> v) {
  return vec_avx2<uint32_t, N>(_mm256_srli_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> shift_left(
    const vec_avx2<uint64_t, N> v) {
  return vec_avx2<uint64_t, N>(_mm256_slli_epi64(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> shift_right(
    const vec_avx2<uint64_t, N> v) {
  return vec_avx2<uint64_t, N>(_mm256_srli_epi64(v.raw, kBits));
}

// Signed (no i64 shift_right)
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> shift_left(
    const vec_avx2<int16_t, N> v) {
  return vec_avx2<int16_t, N>(_mm256_slli_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> shift_right(
    const vec_avx2<int16_t, N> v) {
  return vec_avx2<int16_t, N>(_mm256_srai_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> shift_left(
    const vec_avx2<int32_t, N> v) {
  return vec_avx2<int32_t, N>(_mm256_slli_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> shift_right(
    const vec_avx2<int32_t, N> v) {
  return vec_avx2<int32_t, N>(_mm256_srai_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t, N> shift_left(
    const vec_avx2<int64_t, N> v) {
  return vec_avx2<int64_t, N>(_mm256_slli_epi64(v.raw, kBits));
}

// ------------------------------ Shift lanes by same variable #bits

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE shift_left_count<T, N> set_shift_left_count(
    Desc<T, N, AVX2>, const int bits) {
  return shift_left_count<T, N>{_mm_cvtsi32_si128(bits)};
}

// Same as shift_left_count on x86, but different on ARM.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE shift_right_count<T, N> set_shift_right_count(
    Desc<T, N, AVX2>, const int bits) {
  return shift_right_count<T, N>{_mm_cvtsi32_si128(bits)};
}

// Unsigned (no u8)
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> shift_left_same(
    const vec_avx2<uint16_t, N> v, const shift_left_count<uint16_t, N> bits) {
  return vec_avx2<uint16_t, N>(_mm256_sll_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> shift_right_same(
    const vec_avx2<uint16_t, N> v, const shift_right_count<uint16_t, N> bits) {
  return vec_avx2<uint16_t, N>(_mm256_srl_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> shift_left_same(
    const vec_avx2<uint32_t, N> v, const shift_left_count<uint32_t, N> bits) {
  return vec_avx2<uint32_t, N>(_mm256_sll_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> shift_right_same(
    const vec_avx2<uint32_t, N> v, const shift_right_count<uint32_t, N> bits) {
  return vec_avx2<uint32_t, N>(_mm256_srl_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> shift_left_same(
    const vec_avx2<uint64_t, N> v, const shift_left_count<uint64_t, N> bits) {
  return vec_avx2<uint64_t, N>(_mm256_sll_epi64(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> shift_right_same(
    const vec_avx2<uint64_t, N> v, const shift_right_count<uint64_t, N> bits) {
  return vec_avx2<uint64_t, N>(_mm256_srl_epi64(v.raw, bits.raw));
}

// Signed (no i8,i64)
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> shift_left_same(
    const vec_avx2<int16_t, N> v, const shift_left_count<int16_t, N> bits) {
  return vec_avx2<int16_t, N>(_mm256_sll_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> shift_right_same(
    const vec_avx2<int16_t, N> v, const shift_right_count<int16_t, N> bits) {
  return vec_avx2<int16_t, N>(_mm256_sra_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> shift_left_same(
    const vec_avx2<int32_t, N> v, const shift_left_count<int32_t, N> bits) {
  return vec_avx2<int32_t, N>(_mm256_sll_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> shift_right_same(
    const vec_avx2<int32_t, N> v, const shift_right_count<int32_t, N> bits) {
  return vec_avx2<int32_t, N>(_mm256_sra_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t, N> shift_left_same(
    const vec_avx2<int64_t, N> v, const shift_left_count<int64_t, N> bits) {
  return vec_avx2<int64_t, N>(_mm256_sll_epi64(v.raw, bits.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> operator<<(
    const vec_avx2<uint32_t, N> v, const vec_avx2<uint32_t, N> bits) {
  return vec_avx2<uint32_t, N>(_mm256_sllv_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> operator>>(
    const vec_avx2<uint32_t, N> v, const vec_avx2<uint32_t, N> bits) {
  return vec_avx2<uint32_t, N>(_mm256_srlv_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> operator<<(
    const vec_avx2<uint64_t, N> v, const vec_avx2<uint64_t, N> bits) {
  return vec_avx2<uint64_t, N>(_mm256_sllv_epi64(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> operator>>(
    const vec_avx2<uint64_t, N> v, const vec_avx2<uint64_t, N> bits) {
  return vec_avx2<uint64_t, N>(_mm256_srlv_epi64(v.raw, bits.raw));
}

// Signed (no i8,i16,i64)
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> operator<<(
    const vec_avx2<int32_t, N> v, const vec_avx2<int32_t, N> bits) {
  return vec_avx2<int32_t, N>(_mm256_sllv_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> operator>>(
    const vec_avx2<int32_t, N> v, const vec_avx2<int32_t, N> bits) {
  return vec_avx2<int32_t, N>(_mm256_srav_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t, N> operator<<(
    const vec_avx2<int64_t, N> v, const vec_avx2<int64_t, N> bits) {
  return vec_avx2<int64_t, N>(_mm256_sllv_epi64(v.raw, bits.raw));
}

// ------------------------------ Minimum

// Unsigned (no u64)
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> min(
    const vec_avx2<uint8_t, N> a, const vec_avx2<uint8_t, N> b) {
  return vec_avx2<uint8_t, N>(_mm256_min_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> min(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_min_epu16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> min(
    const vec_avx2<uint32_t, N> a, const vec_avx2<uint32_t, N> b) {
  return vec_avx2<uint32_t, N>(_mm256_min_epu32(a.raw, b.raw));
}

// Signed (no i64)
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> min(
    const vec_avx2<int8_t, N> a, const vec_avx2<int8_t, N> b) {
  return vec_avx2<int8_t, N>(_mm256_min_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> min(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_min_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> min(
    const vec_avx2<int32_t, N> a, const vec_avx2<int32_t, N> b) {
  return vec_avx2<int32_t, N>(_mm256_min_epi32(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> min(const vec_avx2<float, N> a,
                                                  const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_min_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> min(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_min_pd(a.raw, b.raw));
}

// ------------------------------ Maximum

// Unsigned (no u64)
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> max(
    const vec_avx2<uint8_t, N> a, const vec_avx2<uint8_t, N> b) {
  return vec_avx2<uint8_t, N>(_mm256_max_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> max(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_max_epu16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> max(
    const vec_avx2<uint32_t, N> a, const vec_avx2<uint32_t, N> b) {
  return vec_avx2<uint32_t, N>(_mm256_max_epu32(a.raw, b.raw));
}

// Signed (no i64)
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> max(
    const vec_avx2<int8_t, N> a, const vec_avx2<int8_t, N> b) {
  return vec_avx2<int8_t, N>(_mm256_max_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> max(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_max_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> max(
    const vec_avx2<int32_t, N> a, const vec_avx2<int32_t, N> b) {
  return vec_avx2<int32_t, N>(_mm256_max_epi32(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> max(const vec_avx2<float, N> a,
                                                  const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_max_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> max(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_max_pd(a.raw, b.raw));
}

// Returns the closest value to v within [lo, hi].
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> clamp(const vec_avx2<T, N> v,
                                                const vec_avx2<T, N> lo,
                                                const vec_avx2<T, N> hi) {
  return min(max(lo, v), hi);
}

// ------------------------------ Integer multiplication

// Unsigned
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> operator*(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_mullo_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> operator*(
    const vec_avx2<uint32_t, N> a, const vec_avx2<uint32_t, N> b) {
  return vec_avx2<uint32_t, N>(_mm256_mullo_epi32(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> operator*(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_mullo_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> operator*(
    const vec_avx2<int32_t, N> a, const vec_avx2<int32_t, N> b) {
  return vec_avx2<int32_t, N>(_mm256_mullo_epi32(a.raw, b.raw));
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> mul_high(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_mulhi_epu16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> mul_high(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_mulhi_epi16(a.raw, b.raw));
}

}  // namespace ext

// Returns (((a * b) >> 14) + 1) >> 1.
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> mul_high_round(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_mulhrs_epi16(a.raw, b.raw));
}

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t> mul_even(
    const vec_avx2<int32_t> a, const vec_avx2<int32_t> b) {
  return vec_avx2<int64_t>(_mm256_mul_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t> mul_even(
    const vec_avx2<uint32_t> a, const vec_avx2<uint32_t> b) {
  return vec_avx2<uint64_t>(_mm256_mul_epu32(a.raw, b.raw));
}

// ------------------------------ Floating-point negate

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> neg(
    const vec_avx2<float, N> v) {
  const Part<float, N, AVX2> df;
  const Part<uint32_t, N, AVX2> du;
  const auto sign = cast_to(df, set1(du, 0x80000000u));
  return v ^ sign;
}

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> neg(
    const vec_avx2<double, N> v) {
  const Part<double, N, AVX2> df;
  const Part<uint64_t, N, AVX2> du;
  const auto sign = cast_to(df, set1(du, 0x8000000000000000ull));
  return v ^ sign;
}

// ------------------------------ Floating-point mul / div

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator*(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_mul_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator*(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_mul_pd(a.raw, b.raw));
}

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator/(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_div_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator/(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_div_pd(a.raw, b.raw));
}

// Approximate reciprocal
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> approximate_reciprocal(
    const vec_avx2<float, N> v) {
  return vec_avx2<float, N>(_mm256_rcp_ps(v.raw));
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> mul_add(
    const vec_avx2<float, N> mul, const vec_avx2<float, N> x,
    const vec_avx2<float, N> add) {
  return vec_avx2<float, N>(_mm256_fmadd_ps(mul.raw, x.raw, add.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> mul_add(
    const vec_avx2<double, N> mul, const vec_avx2<double, N> x,
    const vec_avx2<double, N> add) {
  return vec_avx2<double, N>(_mm256_fmadd_pd(mul.raw, x.raw, add.raw));
}

// Returns add - mul * x
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> nmul_add(
    const vec_avx2<float, N> mul, const vec_avx2<float, N> x,
    const vec_avx2<float, N> add) {
  return vec_avx2<float, N>(_mm256_fnmadd_ps(mul.raw, x.raw, add.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> nmul_add(
    const vec_avx2<double, N> mul, const vec_avx2<double, N> x,
    const vec_avx2<double, N> add) {
  return vec_avx2<double, N>(_mm256_fnmadd_pd(mul.raw, x.raw, add.raw));
}

// Expresses addition/subtraction as FMA for higher throughput (but also
// higher latency) on HSW/BDW. Requires inline assembly because clang > 6
// 'optimizes' FMA by 1.0 to addition/subtraction. x86 offers 132, 213, 231
// forms (1=F, 2=M, 3=A); the first is also the destination.

// Returns x + add
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> fadd(
    vec_avx2<float, N> x, const vec_avx2<float, N> k1,
    const vec_avx2<float, N> add) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfmadd132ps %2, %1, %0"
               : "+x"(x.raw)
               : "x"(add.raw), "x"(k1.raw));
  return x;
#else
  return vec_avx2<float, N>(_mm256_fmadd_ps(k1.raw, x.raw, add.raw));
#endif
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> fadd(
    vec_avx2<double, N> x, const vec_avx2<double, N> k1,
    const vec_avx2<double, N> add) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfmadd132pd %2, %1, %0"
               : "+x"(x.raw)
               : "x"(add.raw), "x"(k1.raw));
  return x;
#else
  return vec_avx2<double, N>(_mm256_fmadd_pd(k1.raw, x.raw, add.raw));
#endif
}

// Returns x - sub
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> fsub(
    vec_avx2<float, N> x, const vec_avx2<float, N> k1,
    const vec_avx2<float, N> sub) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfmsub132ps %2, %1, %0"
               : "+x"(x.raw)
               : "x"(sub.raw), "x"(k1.raw));
  return x;
#else
  return vec_avx2<float, N>(_mm256_fmsub_ps(k1.raw, x.raw, sub.raw));
#endif
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> fsub(
    vec_avx2<double, N> x, const vec_avx2<double, N> k1,
    const vec_avx2<double, N> sub) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfmsub132pd %2, %1, %0"
               : "+x"(x.raw)
               : "x"(sub.raw), "x"(k1.raw));
  return x;
#else
  return vec_avx2<double, N>(_mm256_fmsub_pd(k1.raw, x.raw, sub.raw));
#endif
}

// Returns -sub + x (clobbers sub register)
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> fnadd(
    vec_avx2<float, N> sub, const vec_avx2<float, N> k1,
    const vec_avx2<float, N> x) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfnmadd132ps %2, %1, %0"
               : "+x"(sub.raw)
               : "x"(x.raw), "x"(k1.raw));
  return x;
#else
  return vec_avx2<float, N>(_mm256_fnmadd_ps(sub.raw, k1.raw, x.raw));
#endif
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> fnadd(
    vec_avx2<double, N> sub, const vec_avx2<double, N> k1,
    const vec_avx2<double, N> x) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm volatile("vfnmadd132pd %2, %1, %0"
               : "+x"(sub.raw)
               : "x"(x.raw), "x"(k1.raw));
  return x;
#else
  return vec_avx2<double, N>(_mm256_fnmadd_pd(sub.raw, k1.raw, x.raw));
#endif
}

// Slightly more expensive on ARM (extra negate)
namespace ext {

// Returns mul * x - sub
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> mul_subtract(
    const vec_avx2<float, N> mul, const vec_avx2<float, N> x,
    const vec_avx2<float, N> sub) {
  return vec_avx2<float, N>(_mm256_fmsub_ps(mul.raw, x.raw, sub.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> mul_subtract(
    const vec_avx2<double, N> mul, const vec_avx2<double, N> x,
    const vec_avx2<double, N> sub) {
  return vec_avx2<double, N>(_mm256_fmsub_pd(mul.raw, x.raw, sub.raw));
}

// Returns -mul * x - sub
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> nmul_subtract(
    const vec_avx2<float, N> mul, const vec_avx2<float, N> x,
    const vec_avx2<float, N> sub) {
  return vec_avx2<float, N>(_mm256_fnmsub_ps(mul.raw, x.raw, sub.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> nmul_subtract(
    const vec_avx2<double, N> mul, const vec_avx2<double, N> x,
    const vec_avx2<double, N> sub) {
  return vec_avx2<double, N>(_mm256_fnmsub_pd(mul.raw, x.raw, sub.raw));
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Full precision square root
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> sqrt(const vec_avx2<float, N> v) {
  return vec_avx2<float, N>(_mm256_sqrt_ps(v.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> sqrt(
    const vec_avx2<double, N> v) {
  return vec_avx2<double, N>(_mm256_sqrt_pd(v.raw));
}

// Approximate reciprocal square root
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> approximate_reciprocal_sqrt(
    const vec_avx2<float, N> v) {
  return vec_avx2<float, N>(_mm256_rsqrt_ps(v.raw));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer, tie to even
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> round(
    const vec_avx2<float, N> v) {
  return vec_avx2<float, N>(
      _mm256_round_ps(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> round(
    const vec_avx2<double, N> v) {
  return vec_avx2<double, N>(
      _mm256_round_pd(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Toward zero, aka truncate
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> trunc(
    const vec_avx2<float, N> v) {
  return vec_avx2<float, N>(
      _mm256_round_ps(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> trunc(
    const vec_avx2<double, N> v) {
  return vec_avx2<double, N>(
      _mm256_round_pd(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}

// Toward +infinity, aka ceiling
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> ceil(const vec_avx2<float, N> v) {
  return vec_avx2<float, N>(
      _mm256_round_ps(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> ceil(
    const vec_avx2<double, N> v) {
  return vec_avx2<double, N>(
      _mm256_round_pd(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}

// Toward -infinity, aka floor
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> floor(
    const vec_avx2<float, N> v) {
  return vec_avx2<float, N>(
      _mm256_round_ps(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> floor(
    const vec_avx2<double, N> v) {
  return vec_avx2<double, N>(
      _mm256_round_pd(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t, N> operator==(
    const vec_avx2<uint8_t, N> a, const vec_avx2<uint8_t, N> b) {
  return vec_avx2<uint8_t, N>(_mm256_cmpeq_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t, N> operator==(
    const vec_avx2<uint16_t, N> a, const vec_avx2<uint16_t, N> b) {
  return vec_avx2<uint16_t, N>(_mm256_cmpeq_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t, N> operator==(
    const vec_avx2<uint32_t, N> a, const vec_avx2<uint32_t, N> b) {
  return vec_avx2<uint32_t, N>(_mm256_cmpeq_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t, N> operator==(
    const vec_avx2<uint64_t, N> a, const vec_avx2<uint64_t, N> b) {
  return vec_avx2<uint64_t, N>(_mm256_cmpeq_epi64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> operator==(
    const vec_avx2<int8_t, N> a, const vec_avx2<int8_t, N> b) {
  return vec_avx2<int8_t, N>(_mm256_cmpeq_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> operator==(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_cmpeq_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> operator==(
    const vec_avx2<int32_t, N> a, const vec_avx2<int32_t, N> b) {
  return vec_avx2<int32_t, N>(_mm256_cmpeq_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t, N> operator==(
    const vec_avx2<int64_t, N> a, const vec_avx2<int64_t, N> b) {
  return vec_avx2<int64_t, N>(_mm256_cmpeq_epi64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator==(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_cmp_ps(a.raw, b.raw, _CMP_EQ_OQ));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator==(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_cmp_pd(a.raw, b.raw, _CMP_EQ_OQ));
}

// ------------------------------ Strict inequality

// Signed/float <
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> operator<(
    const vec_avx2<int8_t, N> a, const vec_avx2<int8_t, N> b) {
  return vec_avx2<int8_t, N>(_mm256_cmpgt_epi8(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> operator<(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_cmpgt_epi16(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> operator<(
    const vec_avx2<int32_t, N> a, const vec_avx2<int32_t, N> b) {
  return vec_avx2<int32_t, N>(_mm256_cmpgt_epi32(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t, N> operator<(
    const vec_avx2<int64_t, N> a, const vec_avx2<int64_t, N> b) {
  return vec_avx2<int64_t, N>(_mm256_cmpgt_epi64(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator<(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_cmp_ps(a.raw, b.raw, _CMP_LT_OQ));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator<(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_cmp_pd(a.raw, b.raw, _CMP_LT_OQ));
}

// Signed/float >
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t, N> operator>(
    const vec_avx2<int8_t, N> a, const vec_avx2<int8_t, N> b) {
  return vec_avx2<int8_t, N>(_mm256_cmpgt_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t, N> operator>(
    const vec_avx2<int16_t, N> a, const vec_avx2<int16_t, N> b) {
  return vec_avx2<int16_t, N>(_mm256_cmpgt_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> operator>(
    const vec_avx2<int32_t, N> a, const vec_avx2<int32_t, N> b) {
  return vec_avx2<int32_t, N>(_mm256_cmpgt_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t, N> operator>(
    const vec_avx2<int64_t, N> a, const vec_avx2<int64_t, N> b) {
  return vec_avx2<int64_t, N>(_mm256_cmpgt_epi64(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator>(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_cmp_ps(a.raw, b.raw, _CMP_GT_OQ));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator>(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_cmp_pd(a.raw, b.raw, _CMP_GT_OQ));
}

// ------------------------------ Weak inequality

// Float <= >=
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator<=(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_cmp_ps(a.raw, b.raw, _CMP_LE_OQ));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator<=(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_cmp_pd(a.raw, b.raw, _CMP_LE_OQ));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator>=(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_cmp_ps(a.raw, b.raw, _CMP_GE_OQ));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator>=(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_cmp_pd(a.raw, b.raw, _CMP_GE_OQ));
}

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> operator&(const vec_avx2<T, N> a,
                                                    const vec_avx2<T, N> b) {
  return vec_avx2<T, N>(_mm256_and_si256(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator&(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_and_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator&(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_and_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> andnot(const vec_avx2<T, N> not_mask,
                                                 const vec_avx2<T, N> mask) {
  return vec_avx2<T, N>(_mm256_andnot_si256(not_mask.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> andnot(
    const vec_avx2<float, N> not_mask, const vec_avx2<float, N> mask) {
  return vec_avx2<float, N>(_mm256_andnot_ps(not_mask.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> andnot(
    const vec_avx2<double, N> not_mask, const vec_avx2<double, N> mask) {
  return vec_avx2<double, N>(_mm256_andnot_pd(not_mask.raw, mask.raw));
}

// ------------------------------ Bitwise OR

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> operator|(const vec_avx2<T, N> a,
                                                    const vec_avx2<T, N> b) {
  return vec_avx2<T, N>(_mm256_or_si256(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator|(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_or_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator|(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_or_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise XOR

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> operator^(const vec_avx2<T, N> a,
                                                    const vec_avx2<T, N> b) {
  return vec_avx2<T, N>(_mm256_xor_si256(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> operator^(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b) {
  return vec_avx2<float, N>(_mm256_xor_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> operator^(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b) {
  return vec_avx2<double, N>(_mm256_xor_pd(a.raw, b.raw));
}

// ------------------------------ Select/blend

// Returns a mask for use by select().
// blendv_ps/pd only check the sign bit, so this is a no-op on x86.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> condition_from_sign(
    const vec_avx2<T, N> v) {
  return v;
}

// Returns mask ? b : a. "mask" must either have been returned by
// selector_from_mask, or callers must ensure its lanes are T(0) or ~T(0).
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> select(const vec_avx2<T, N> a,
                                                 const vec_avx2<T, N> b,
                                                 const vec_avx2<T, N> mask) {
  return vec_avx2<T, N>(_mm256_blendv_epi8(a.raw, b.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> select(
    const vec_avx2<float, N> a, const vec_avx2<float, N> b,
    const vec_avx2<float, N> mask) {
  return vec_avx2<float, N>(_mm256_blendv_ps(a.raw, b.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double, N> select(
    const vec_avx2<double, N> a, const vec_avx2<double, N> b,
    const vec_avx2<double, N> mask) {
  return vec_avx2<double, N>(_mm256_blendv_pd(a.raw, b.raw, mask.raw));
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> load(Full<T, AVX2>,
                                            const T* SIMD_RESTRICT aligned) {
  return vec_avx2<T>(
      _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned)));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> load(
    Full<float, AVX2>, const float* SIMD_RESTRICT aligned) {
  return vec_avx2<float>(_mm256_load_ps(aligned));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> load(
    Full<double, AVX2>, const double* SIMD_RESTRICT aligned) {
  return vec_avx2<double>(_mm256_load_pd(aligned));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> load_unaligned(
    Full<T, AVX2>, const T* SIMD_RESTRICT p) {
  return vec_avx2<T>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> load_unaligned(
    Full<float, AVX2>, const float* SIMD_RESTRICT p) {
  return vec_avx2<float>(_mm256_loadu_ps(p));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> load_unaligned(
    Full<double, AVX2>, const double* SIMD_RESTRICT p) {
  return vec_avx2<double>(_mm256_loadu_pd(p));
}

// Loads 128 bit and duplicates into both 128-bit halves. This avoids the
// 3-cycle cost of moving data between 128-bit halves and avoids port 5.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> load_dup128(
    Full<T, AVX2>, const T* const SIMD_RESTRICT p) {
  // Clang 3.9 generates VINSERTF128 which is slower, but inline assembly leads
  // to "invalid output size for constraint" without -mavx2:
  // https://gcc.godbolt.org/z/-Jt_-F
#if (SIMD_COMPILER != SIMD_COMPILER_MSVC) && defined(__AVX2__)
  __m256i out;
  asm volatile("vbroadcasti128 %1, %[reg]" : [reg] "=x"(out) : "m"(p[0]));
  return vec_avx2<T>(out);
#else
  return vec_avx2<T>(
      _mm256_broadcastsi128_si256(load_unaligned(Full<T, SSE4>(), p).raw));
#endif
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> load_dup128(
    Full<float, AVX2>, const float* const SIMD_RESTRICT p) {
#if (SIMD_COMPILER != SIMD_COMPILER_MSVC) && defined(__AVX2__)
  __m256 out;
  asm volatile("vbroadcastf128 %1, %[reg]" : [reg] "=x"(out) : "m"(p[0]));
  return vec_avx2<float>(out);
#else
  return vec_avx2<float>(_mm256_broadcast_ps((const __m128*)p));
#endif
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> load_dup128(
    Full<double, AVX2>, const double* const SIMD_RESTRICT p) {
#if (SIMD_COMPILER != SIMD_COMPILER_MSVC) && defined(__AVX2__)
  __m256d out;
  asm volatile("vbroadcastf128 %1, %[reg]" : [reg] "=x"(out) : "m"(p[0]));
  return vec_avx2<double>(out);
#else
  return vec_avx2<double>(_mm256_broadcast_pd((const __m128d*)p));
#endif
}

// ------------------------------ Store

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store(const vec_avx2<T> v, Full<T, AVX2>,
                                      T* SIMD_RESTRICT aligned) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(aligned), v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void store(const vec_avx2<float> v,
                                      Full<float, AVX2>,
                                      float* SIMD_RESTRICT aligned) {
  _mm256_store_ps(aligned, v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void store(const vec_avx2<double> v,
                                      Full<double, AVX2>,
                                      double* SIMD_RESTRICT aligned) {
  _mm256_store_pd(aligned, v.raw);
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned(const vec_avx2<T> v,
                                                Full<T, AVX2>,
                                                T* SIMD_RESTRICT p) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned(const vec_avx2<float> v,
                                                Full<float, AVX2>,
                                                float* SIMD_RESTRICT p) {
  _mm256_storeu_ps(p, v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned(const vec_avx2<double> v,
                                                Full<double, AVX2>,
                                                double* SIMD_RESTRICT p) {
  _mm256_storeu_pd(p, v.raw);
}

// ------------------------------ Non-temporal stores

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE void stream(const vec_avx2<T, N> v, Full<T, AVX2>,
                                       T* SIMD_RESTRICT aligned) {
  _mm256_stream_si256(reinterpret_cast<__m256i*>(aligned), v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void stream(const vec_avx2<float> v,
                                       Full<float, AVX2>,
                                       float* SIMD_RESTRICT aligned) {
  _mm256_stream_ps(aligned, v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void stream(const vec_avx2<double> v,
                                       Full<double, AVX2>,
                                       double* SIMD_RESTRICT aligned) {
  _mm256_stream_pd(aligned, v.raw);
}

// ------------------------------ Gather

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> gather_offset_impl(
    char (&sizeof_t)[4], Full<T, AVX2>, const T* SIMD_RESTRICT base,
    const vec_avx2<int32_t> offset) {
  return vec_avx2<T>(_mm256_i32gather_epi32(
      reinterpret_cast<const int32_t*>(base), offset.raw, 1));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> gather_index_impl(
    char (&sizeof_t)[4], Full<T, AVX2>, const T* SIMD_RESTRICT base,
    const vec_avx2<int32_t> index) {
  return vec_avx2<T>(_mm256_i32gather_epi32(
      reinterpret_cast<const int32_t*>(base), index.raw, 4));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> gather_offset_impl(
    char (&sizeof_t)[8], Full<T, AVX2>, const T* SIMD_RESTRICT base,
    const vec_avx2<int64_t> offset) {
  return vec_avx2<T>(_mm256_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), offset.raw, 1));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> gather_index_impl(
    char (&sizeof_t)[8], Full<T, AVX2>, const T* SIMD_RESTRICT base,
    const vec_avx2<int64_t> index) {
  return vec_avx2<T>(_mm256_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), index.raw, 8));
}

template <typename T, typename Offset>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> gather_offset(
    Full<T, AVX2> d, const T* SIMD_RESTRICT base,
    const vec_avx2<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "SVE requires same size base/ofs");
  char sizeof_t[sizeof(T)];
  return gather_offset_impl(sizeof_t, d, base, offset);
}
template <typename T, typename Index>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> gather_index(
    Full<T, AVX2> d, const T* SIMD_RESTRICT base, const vec_avx2<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "SVE requires same size base/idx");
  char sizeof_t[sizeof(T)];
  return gather_index_impl(sizeof_t, d, base, index);
}

template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> gather_offset<float>(
    Full<float, AVX2>, const float* SIMD_RESTRICT base,
    const vec_avx2<int32_t> offset) {
  return vec_avx2<float>(_mm256_i32gather_ps(base, offset.raw, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> gather_index<float>(
    Full<float, AVX2>, const float* SIMD_RESTRICT base,
    const vec_avx2<int32_t> index) {
  return vec_avx2<float>(_mm256_i32gather_ps(base, index.raw, 4));
}

template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> gather_offset<double>(
    Full<double, AVX2>, const double* SIMD_RESTRICT base,
    const vec_avx2<int64_t> offset) {
  return vec_avx2<double>(_mm256_i64gather_pd(base, offset.raw, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> gather_index<double>(
    Full<double, AVX2>, const double* SIMD_RESTRICT base,
    const vec_avx2<int64_t> index) {
  return vec_avx2<double>(_mm256_i64gather_pd(base, index.raw, 8));
}

}  // namespace ext

// ================================================== SWIZZLE

// ------------------------------ Extract half

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<T> get_half(Lower, vec_avx2<T> v) {
  return vec_sse4<T>(_mm256_castsi256_si128(v.raw));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<float> get_half(Lower, vec_avx2<float> v) {
  return vec_sse4<float>(_mm256_castps256_ps128(v.raw));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<double> get_half(Lower,
                                                     vec_avx2<double> v) {
  return vec_sse4<double>(_mm256_castpd256_pd128(v.raw));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<T> lower_half(const vec_avx2<T> v) {
  return get_half(Lower(), v);
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<T> get_half(Upper, const vec_avx2<T> v) {
  return vec_sse4<T>(_mm256_extracti128_si256(v.raw, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<float> get_half(Upper,
                                                    const vec_avx2<float> v) {
  return vec_sse4<float>(_mm256_extractf128_ps(v.raw, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<double> get_half(Upper,
                                                     const vec_avx2<double> v) {
  return vec_sse4<double>(_mm256_extractf128_pd(v.raw, 1));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<T> upper_half(const vec_avx2<T> v) {
  return get_half(Upper(), v);
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> shift_left_bytes(
    const vec_avx2<T, N> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  // This is the same operation as _mm256_bslli_epi128.
  return vec_avx2<T, N>(_mm256_slli_si256(v.raw, kBytes));
}

template <int kLanes, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> shift_left_lanes(
    const vec_avx2<T, N> v) {
  return shift_left_bytes<kLanes * sizeof(T)>(v);
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> shift_right_bytes(
    const vec_avx2<T, N> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  // This is the same operation as _mm256_bsrli_epi128.
  return vec_avx2<T, N>(_mm256_srli_si256(v.raw, kBytes));
}

template <int kLanes, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> shift_right_lanes(
    const vec_avx2<T, N> v) {
  return shift_right_bytes<kLanes * sizeof(T)>(v);
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> combine_shift_right_bytes(
    const vec_avx2<T, N> hi, const vec_avx2<T, N> lo) {
  const Full<uint8_t, AVX2> d8;
  const vec_avx2<uint8_t> extracted_bytes(
      _mm256_alignr_epi8(cast_to(d8, hi).raw, cast_to(d8, lo).raw, kBytes));
  return cast_to(Full<T, AVX2>(), extracted_bytes);
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t> broadcast(
    const vec_avx2<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m256i lo = _mm256_shufflelo_epi16(v.raw, 0x55 * kLane);
    return vec_avx2<uint16_t>(_mm256_unpacklo_epi64(lo, lo));
  } else {
    const __m256i hi = _mm256_shufflehi_epi16(v.raw, 0x55 * (kLane - 4));
    return vec_avx2<uint16_t>(_mm256_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> broadcast(
    const vec_avx2<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return vec_avx2<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t> broadcast(
    const vec_avx2<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return vec_avx2<uint64_t>(_mm256_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Signed
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t> broadcast(
    const vec_avx2<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m256i lo = _mm256_shufflelo_epi16(v.raw, 0x55 * kLane);
    return vec_avx2<int16_t>(_mm256_unpacklo_epi64(lo, lo));
  } else {
    const __m256i hi = _mm256_shufflehi_epi16(v.raw, 0x55 * (kLane - 4));
    return vec_avx2<int16_t>(_mm256_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> broadcast(
    const vec_avx2<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return vec_avx2<int32_t>(_mm256_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t> broadcast(
    const vec_avx2<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return vec_avx2<int64_t>(_mm256_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Float
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> broadcast(const vec_avx2<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return vec_avx2<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> broadcast(
    const vec_avx2<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return vec_avx2<double>(_mm256_shuffle_pd(v.raw, v.raw, 15 * kLane));
}

// ------------------------------ Hard-coded shuffles

// Notation: let vec_avx2<int32_t> have lanes 7,6,5,4,3,2,1,0 (0 is
// least-significant). shuffle_0321 rotates four-lane blocks one lane to the
// right (the previous least-significant lane is now most-significant =>
// 47650321). These could also be implemented via combine_shift_right_bytes but
// the shuffle_abcd notation is more convenient.

// Swap 64-bit halves
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> shuffle_1032(
    const vec_avx2<uint32_t> v) {
  return vec_avx2<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> shuffle_1032(
    const vec_avx2<int32_t> v) {
  return vec_avx2<int32_t>(_mm256_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> shuffle_1032(
    const vec_avx2<float> v) {
  // Shorter encoding than _mm256_permute_ps.
  return vec_avx2<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t> shuffle_01(
    const vec_avx2<uint64_t> v) {
  return vec_avx2<uint64_t>(_mm256_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t> shuffle_01(
    const vec_avx2<int64_t> v) {
  return vec_avx2<int64_t>(_mm256_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> shuffle_01(
    const vec_avx2<double> v) {
  // Shorter encoding than _mm256_permute_pd.
  return vec_avx2<double>(_mm256_shuffle_pd(v.raw, v.raw, 5));
}

// Rotate right 32 bits
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> shuffle_0321(
    const vec_avx2<uint32_t> v) {
  return vec_avx2<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> shuffle_0321(
    const vec_avx2<int32_t> v) {
  return vec_avx2<int32_t>(_mm256_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> shuffle_0321(
    const vec_avx2<float> v) {
  return vec_avx2<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x39));
}
// Rotate left 32 bits
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> shuffle_2103(
    const vec_avx2<uint32_t> v) {
  return vec_avx2<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> shuffle_2103(
    const vec_avx2<int32_t> v) {
  return vec_avx2<int32_t>(_mm256_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> shuffle_2103(
    const vec_avx2<float> v) {
  return vec_avx2<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x93));
}

// Reverse
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> shuffle_0123(
    const vec_avx2<uint32_t> v) {
  return vec_avx2<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> shuffle_0123(
    const vec_avx2<int32_t> v) {
  return vec_avx2<int32_t>(_mm256_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> shuffle_0123(
    const vec_avx2<float> v) {
  return vec_avx2<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x1B));
}

// ------------------------------ Permute (runtime variable)

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE permute_avx2<T> set_table_indices(const Full<T, AVX2>,
                                                       const int32_t* idx) {
  return permute_avx2<T>{load_unaligned(Full<int32_t, AVX2>(), idx).raw};
}

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> table_lookup_lanes(
    const vec_avx2<uint32_t> v, const permute_avx2<uint32_t> idx) {
  return vec_avx2<uint32_t>(_mm256_permutevar8x32_epi32(v.raw, idx.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> table_lookup_lanes(
    const vec_avx2<int32_t> v, const permute_avx2<int32_t> idx) {
  return vec_avx2<int32_t>(_mm256_permutevar8x32_epi32(v.raw, idx.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> table_lookup_lanes(
    const vec_avx2<float> v, const permute_avx2<float> idx) {
  return vec_avx2<float>(_mm256_permutevar8x32_ps(v.raw, idx.raw));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t> interleave_lo(
    const vec_avx2<uint8_t> a, const vec_avx2<uint8_t> b) {
  return vec_avx2<uint8_t>(_mm256_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t> interleave_lo(
    const vec_avx2<uint16_t> a, const vec_avx2<uint16_t> b) {
  return vec_avx2<uint16_t>(_mm256_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> interleave_lo(
    const vec_avx2<uint32_t> a, const vec_avx2<uint32_t> b) {
  return vec_avx2<uint32_t>(_mm256_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t> interleave_lo(
    const vec_avx2<uint64_t> a, const vec_avx2<uint64_t> b) {
  return vec_avx2<uint64_t>(_mm256_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t> interleave_lo(
    const vec_avx2<int8_t> a, const vec_avx2<int8_t> b) {
  return vec_avx2<int8_t>(_mm256_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t> interleave_lo(
    const vec_avx2<int16_t> a, const vec_avx2<int16_t> b) {
  return vec_avx2<int16_t>(_mm256_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> interleave_lo(
    const vec_avx2<int32_t> a, const vec_avx2<int32_t> b) {
  return vec_avx2<int32_t>(_mm256_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t> interleave_lo(
    const vec_avx2<int64_t> a, const vec_avx2<int64_t> b) {
  return vec_avx2<int64_t>(_mm256_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> interleave_lo(
    const vec_avx2<float> a, const vec_avx2<float> b) {
  return vec_avx2<float>(_mm256_unpacklo_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> interleave_lo(
    const vec_avx2<double> a, const vec_avx2<double> b) {
  return vec_avx2<double>(_mm256_unpacklo_pd(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint8_t> interleave_hi(
    const vec_avx2<uint8_t> a, const vec_avx2<uint8_t> b) {
  return vec_avx2<uint8_t>(_mm256_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t> interleave_hi(
    const vec_avx2<uint16_t> a, const vec_avx2<uint16_t> b) {
  return vec_avx2<uint16_t>(_mm256_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> interleave_hi(
    const vec_avx2<uint32_t> a, const vec_avx2<uint32_t> b) {
  return vec_avx2<uint32_t>(_mm256_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t> interleave_hi(
    const vec_avx2<uint64_t> a, const vec_avx2<uint64_t> b) {
  return vec_avx2<uint64_t>(_mm256_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int8_t> interleave_hi(
    const vec_avx2<int8_t> a, const vec_avx2<int8_t> b) {
  return vec_avx2<int8_t>(_mm256_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t> interleave_hi(
    const vec_avx2<int16_t> a, const vec_avx2<int16_t> b) {
  return vec_avx2<int16_t>(_mm256_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> interleave_hi(
    const vec_avx2<int32_t> a, const vec_avx2<int32_t> b) {
  return vec_avx2<int32_t>(_mm256_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t> interleave_hi(
    const vec_avx2<int64_t> a, const vec_avx2<int64_t> b) {
  return vec_avx2<int64_t>(_mm256_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> interleave_hi(
    const vec_avx2<float> a, const vec_avx2<float> b) {
  return vec_avx2<float>(_mm256_unpackhi_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> interleave_hi(
    const vec_avx2<double> a, const vec_avx2<double> b) {
  return vec_avx2<double>(_mm256_unpackhi_pd(a.raw, b.raw));
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t> zip_lo(
    const vec_avx2<uint8_t> a, const vec_avx2<uint8_t> b) {
  return vec_avx2<uint16_t>(_mm256_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> zip_lo(
    const vec_avx2<uint16_t> a, const vec_avx2<uint16_t> b) {
  return vec_avx2<uint32_t>(_mm256_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t> zip_lo(
    const vec_avx2<uint32_t> a, const vec_avx2<uint32_t> b) {
  return vec_avx2<uint64_t>(_mm256_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t> zip_lo(const vec_avx2<int8_t> a,
                                                    const vec_avx2<int8_t> b) {
  return vec_avx2<int16_t>(_mm256_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> zip_lo(const vec_avx2<int16_t> a,
                                                    const vec_avx2<int16_t> b) {
  return vec_avx2<int32_t>(_mm256_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t> zip_lo(const vec_avx2<int32_t> a,
                                                    const vec_avx2<int32_t> b) {
  return vec_avx2<int64_t>(_mm256_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t> zip_hi(
    const vec_avx2<uint8_t> a, const vec_avx2<uint8_t> b) {
  return vec_avx2<uint16_t>(_mm256_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> zip_hi(
    const vec_avx2<uint16_t> a, const vec_avx2<uint16_t> b) {
  return vec_avx2<uint32_t>(_mm256_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t> zip_hi(
    const vec_avx2<uint32_t> a, const vec_avx2<uint32_t> b) {
  return vec_avx2<uint64_t>(_mm256_unpackhi_epi32(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t> zip_hi(const vec_avx2<int8_t> a,
                                                    const vec_avx2<int8_t> b) {
  return vec_avx2<int16_t>(_mm256_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> zip_hi(const vec_avx2<int16_t> a,
                                                    const vec_avx2<int16_t> b) {
  return vec_avx2<int32_t>(_mm256_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t> zip_hi(const vec_avx2<int32_t> a,
                                                    const vec_avx2<int32_t> b) {
  return vec_avx2<int64_t>(_mm256_unpackhi_epi32(a.raw, b.raw));
}

// ------------------------------ Parts

// Returns part of a vector (unspecified whether upper or lower).
template <typename T, size_t N, size_t VN>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> any_part(Desc<T, N, AVX2>,
                                                   const vec_avx2<T, VN> v) {
  return vec_avx2<T, N>(v.raw);  // shrink AVX2
}
template <typename T, size_t N, size_t VN>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<T, N> any_part(Desc<T, N, SSE4>,
                                                   const vec_avx2<T, VN> v) {
  return vec_sse4<T, N>(_mm256_castsi256_si128(v.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<float, N> any_part(Desc<float, N, SSE4>,
                                                       vec_avx2<float> v) {
  return vec_sse4<float, N>(_mm256_castps256_ps128(v.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<double, N> any_part(Desc<double, N, SSE4>,
                                                        vec_avx2<double> v) {
  return vec_sse4<double, N>(_mm256_castpd256_pd128(v.raw));
}

// Gets the single value stored in a vector/part.
template <typename T, size_t N, class Target, size_t VN>
SIMD_ATTR_AVX2 SIMD_INLINE T get_part(Desc<T, N, Target>,
                                      const vec_avx2<T, VN> v) {
  const Part<T, 1, AVX2> d;
  return get_part(d, any_part(d, v));
}

// Returns full vector with the given part's lane broadcasted. Note that
// callers cannot use broadcast directly because part lane order is undefined.
template <int kLane, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> broadcast_part(Full<T, AVX2>,
                                                      const vec_sse4<T, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  const auto v128 = broadcast<kLane>(vec_sse4<T>(v.raw));
  // Same as _mm256_castsi128_si256, but with guaranteed zero-extension.
  const auto lo = _mm256_zextsi128_si256(v128.raw);
  // Same instruction as _mm256_permute2f128_si256.
  return vec_avx2<T>(_mm256_permute2x128_si256(lo, lo, 0));
}
template <int kLane, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> broadcast_part(
    Full<float, AVX2>, const vec_sse4<float, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  const auto v128 = broadcast<kLane>(vec_sse4<float>(v.raw)).raw;
  // Same as _mm256_castps128_ps256, but with guaranteed zero-extension.
  const auto lo = _mm256_zextps128_ps256(v128);
  return vec_avx2<float>(_mm256_permute2f128_ps(lo, lo, 0));
}
template <int kLane, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> broadcast_part(
    Full<double, AVX2>, const vec_sse4<double, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  const auto v128 = broadcast<kLane>(vec_sse4<double>(v.raw)).raw;
  // Same as _mm256_castpd128_pd256, but with guaranteed zero-extension.
  const auto lo = _mm256_zextpd128_pd256(v128);
  return vec_avx2<double>(_mm256_permute2f128_pd(lo, lo, 0));
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> concat_lo_lo(const vec_avx2<T> hi,
                                                    const vec_avx2<T> lo) {
  return vec_avx2<T>(_mm256_permute2x128_si256(lo.raw, hi.raw, 0x20));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> concat_lo_lo(
    const vec_avx2<float> hi, const vec_avx2<float> lo) {
  return vec_avx2<float>(_mm256_permute2f128_ps(lo.raw, hi.raw, 0x20));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> concat_lo_lo(
    const vec_avx2<double> hi, const vec_avx2<double> lo) {
  return vec_avx2<double>(_mm256_permute2f128_pd(lo.raw, hi.raw, 0x20));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> concat_hi_hi(const vec_avx2<T> hi,
                                                    const vec_avx2<T> lo) {
  return vec_avx2<T>(_mm256_permute2x128_si256(lo.raw, hi.raw, 0x31));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> concat_hi_hi(
    const vec_avx2<float> hi, const vec_avx2<float> lo) {
  return vec_avx2<float>(_mm256_permute2f128_ps(lo.raw, hi.raw, 0x31));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> concat_hi_hi(
    const vec_avx2<double> hi, const vec_avx2<double> lo) {
  return vec_avx2<double>(_mm256_permute2f128_pd(lo.raw, hi.raw, 0x31));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves / swap blocks)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> concat_lo_hi(const vec_avx2<T> hi,
                                                    const vec_avx2<T> lo) {
  return vec_avx2<T>(_mm256_permute2x128_si256(lo.raw, hi.raw, 0x21));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> concat_lo_hi(
    const vec_avx2<float> hi, const vec_avx2<float> lo) {
  return vec_avx2<float>(_mm256_permute2f128_ps(lo.raw, hi.raw, 0x21));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> concat_lo_hi(
    const vec_avx2<double> hi, const vec_avx2<double> lo) {
  return vec_avx2<double>(_mm256_permute2f128_pd(lo.raw, hi.raw, 0x21));
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> concat_hi_lo(const vec_avx2<T> hi,
                                                    const vec_avx2<T> lo) {
  return vec_avx2<T>(_mm256_blend_epi32(hi.raw, lo.raw, 0x0F));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> concat_hi_lo(
    const vec_avx2<float> hi, const vec_avx2<float> lo) {
  return vec_avx2<float>(_mm256_blend_ps(hi.raw, lo.raw, 0x0F));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> concat_hi_lo(
    const vec_avx2<double> hi, const vec_avx2<double> lo) {
  return vec_avx2<double>(_mm256_blend_pd(hi.raw, lo.raw, 3));
}

// ------------------------------ Odd/even lanes

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> odd_even_impl(char (&sizeof_t)[1],
                                                     const vec_avx2<T> a,
                                                     const vec_avx2<T> b) {
  const Full<T, AVX2> d;
  const Full<uint8_t, AVX2> d8;
  SIMD_ALIGN constexpr uint8_t mask[16] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
                                           0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
  return select(a, b, cast_to(d, load_dup128(d8, mask)));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> odd_even_impl(char (&sizeof_t)[2],
                                                     const vec_avx2<T> a,
                                                     const vec_avx2<T> b) {
  return vec_avx2<T>(_mm256_blend_epi16(a.raw, b.raw, 0x55));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> odd_even_impl(char (&sizeof_t)[4],
                                                     const vec_avx2<T> a,
                                                     const vec_avx2<T> b) {
  return vec_avx2<T>(_mm256_blend_epi32(a.raw, b.raw, 0x55));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> odd_even_impl(char (&sizeof_t)[8],
                                                     const vec_avx2<T> a,
                                                     const vec_avx2<T> b) {
  return vec_avx2<T>(_mm256_blend_epi32(a.raw, b.raw, 0x33));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> odd_even(const vec_avx2<T> a,
                                                const vec_avx2<T> b) {
  char sizeof_t[sizeof(T)];
  return odd_even_impl(sizeof_t, a, b);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float> odd_even<float>(
    const vec_avx2<float> a, const vec_avx2<float> b) {
  return vec_avx2<float>(_mm256_blend_ps(a.raw, b.raw, 0x55));
}

template <>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> odd_even<double>(
    const vec_avx2<double> a, const vec_avx2<double> b) {
  return vec_avx2<double>(_mm256_blend_pd(a.raw, b.raw, 5));
}

// ================================================== CONVERT

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI, size_t N, size_t NI>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> table_lookup_bytes(
    const vec_avx2<T, N> bytes, const vec_avx2<TI, NI> from) {
  return vec_avx2<T, N>(_mm256_shuffle_epi8(bytes.raw, from.raw));
}

// ------------------------------ Promotions (part w/ narrow lanes -> full)

SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<double> convert_to(
    Full<double, AVX2>, const vec_sse4<float, 4> v) {
  return vec_avx2<double>(_mm256_cvtps_pd(v.raw));
}

// Unsigned: zero-extend.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo would be faster.
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint16_t> convert_to(Full<uint16_t, AVX2>,
                                                         const u8x16 v) {
  return vec_avx2<uint16_t>(_mm256_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> convert_to(Full<uint32_t, AVX2>,
                                                         const u8x8 v) {
  return vec_avx2<uint32_t>(_mm256_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t> convert_to(Full<int16_t, AVX2>,
                                                        const u8x16 v) {
  return vec_avx2<int16_t>(_mm256_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> convert_to(Full<int32_t, AVX2>,
                                                        const u8x8 v) {
  return vec_avx2<int32_t>(_mm256_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> convert_to(Full<uint32_t, AVX2>,
                                                         const u16x8 v) {
  return vec_avx2<uint32_t>(_mm256_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> convert_to(Full<int32_t, AVX2>,
                                                        const u16x8 v) {
  return vec_avx2<int32_t>(_mm256_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t> convert_to(Full<uint64_t, AVX2>,
                                                         const u32x4 v) {
  return vec_avx2<uint64_t>(_mm256_cvtepu32_epi64(v.raw));
}

// Special case for "v" with all blocks equal (e.g. from broadcast_block or
// load_dup128): single-cycle latency instead of 3.
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint32_t> u32_from_u8(
    const vec_avx2<uint8_t> v) {
  const Full<uint32_t, AVX2> d32;
  SIMD_ALIGN static constexpr uint32_t k32From8[8] = {
      0xFFFFFF00UL, 0xFFFFFF01UL, 0xFFFFFF02UL, 0xFFFFFF03UL,
      0xFFFFFF04UL, 0xFFFFFF05UL, 0xFFFFFF06UL, 0xFFFFFF07UL};
  return table_lookup_bytes(cast_to(d32, v), load(d32, k32From8));
}

// Signed: replicate sign bit.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo followed by
// signed shift would be faster.
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t> convert_to(Full<int16_t, AVX2>,
                                                        const i8x16 v) {
  return vec_avx2<int16_t>(_mm256_cvtepi8_epi16(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> convert_to(Full<int32_t, AVX2>,
                                                        const i8x8 v) {
  return vec_avx2<int32_t>(_mm256_cvtepi8_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t> convert_to(Full<int32_t, AVX2>,
                                                        const i16x8 v) {
  return vec_avx2<int32_t>(_mm256_cvtepi16_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int64_t> convert_to(Full<int64_t, AVX2>,
                                                        const i32x4 v) {
  return vec_avx2<int64_t>(_mm256_cvtepi32_epi64(v.raw));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE VT<uint16_t, N, AVX2> convert_to(
    Part<uint16_t, N, AVX2>, const vec_avx2<int32_t> v) {
  const __m256i u16 = _mm256_packus_epi32(v.raw, v.raw);
  // Concatenating lower halves of both 128-bit blocks afterward is more
  // efficient than an extra input with low block = high block of v.
  return VT<uint16_t, N, AVX2>(
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(u16, 0x88)));
}

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE VT<uint8_t, N, AVX2> convert_to(
    Part<uint8_t, N, AVX2>, const vec_avx2<int32_t> v) {
  const __m256i u16_blocks = _mm256_packus_epi32(v.raw, v.raw);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i u16_concat = _mm256_permute4x64_epi64(u16_blocks, 0x88);
  const __m128i u16 = _mm256_castsi256_si128(u16_concat);
  return VT<uint8_t, N, AVX2>(_mm_packus_epi16(u16, u16));
}

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE VT<int16_t, N, AVX2> convert_to(
    Part<int16_t, N, AVX2>, const vec_avx2<int32_t> v) {
  const __m256i i16 = _mm256_packs_epi32(v.raw, v.raw);
  return VT<int16_t, N, AVX2>(
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(i16, 0x88)));
}

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE VT<int8_t, N, AVX2> convert_to(
    Part<int8_t, N, AVX2>, const vec_avx2<int32_t> v) {
  const __m256i i16_blocks = _mm256_packs_epi32(v.raw, v.raw);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i i16_concat = _mm256_permute4x64_epi64(i16_blocks, 0x88);
  const __m128i i16 = _mm256_castsi256_si128(i16_concat);
  return VT<int8_t, N, AVX2>(_mm_packs_epi16(i16, i16));
}

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE VT<uint8_t, N, AVX2> convert_to(
    Part<uint8_t, N, AVX2>, const vec_avx2<int16_t> v) {
  const __m256i u8 = _mm256_packus_epi16(v.raw, v.raw);
  return VT<uint8_t, N, AVX2>(
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(u8, 0x88)));
}

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE VT<int8_t, N, AVX2> convert_to(
    Part<int8_t, N, AVX2>, const vec_avx2<int16_t> v) {
  const __m256i i8 = _mm256_packs_epi16(v.raw, v.raw);
  return VT<int8_t, N, AVX2>(
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(i8, 0x88)));
}

// For already range-limited input [0, 255].
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<uint8_t, 8> u8_from_u32(
    const vec_avx2<uint32_t> v) {
  const Full<uint32_t, AVX2> d32;
  SIMD_ALIGN static constexpr uint32_t k8From32[8] = {
      0x0C080400u, ~0u, ~0u, ~0u, ~0u, 0x0C080400u, ~0u, ~0u};
  // Place first four bytes in lo[0], remainding 4 in hi[1].
  const auto quad = table_lookup_bytes(v, load(d32, k8From32));
  // Interleave both quadruplets - OR instead of unpack reduces port5 pressure.
  const auto lo = get_half(Lower(), quad);
  const auto hi = get_half(Upper(), quad);
  const auto pair = get_half(Lower(), lo | hi);
  return cast_to(Part<uint8_t, 8, SSE4>(), pair);
}

// ------------------------------ Convert i32 <=> f32

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<float, N> convert_to(
    Part<float, N, AVX2>, const vec_avx2<int32_t, N> v) {
  return vec_avx2<float, N>(_mm256_cvtepi32_ps(v.raw));
}
// Truncates (rounds toward zero).
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> convert_to(
    Part<int32_t, N, AVX2>, const vec_avx2<float, N> v) {
  return vec_avx2<int32_t, N>(_mm256_cvttps_epi32(v.raw));
}

template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int32_t, N> nearest_int(
    const vec_avx2<float, N> v) {
  return vec_avx2<int32_t, N>(_mm256_cvtps_epi32(v.raw));
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ movemask

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..31 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_ATTR_AVX2 SIMD_INLINE uint32_t movemask(const vec_avx2<uint8_t> v) {
  return _mm256_movemask_epi8(v.raw);
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_ATTR_AVX2 SIMD_INLINE uint32_t movemask(const vec_avx2<float> v) {
  return _mm256_movemask_ps(v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE uint32_t movemask(const vec_avx2<double> v) {
  return _mm256_movemask_pd(v.raw);
}

// ------------------------------ all_zero

// Returns whether all lanes are equal to zero. Supported for all integer V.
// (Floating-point VTESTP* only test the sign bit!)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE bool all_zero(const vec_avx2<T> v) {
  return static_cast<bool>(_mm256_testz_si256(v.raw, v.raw));
}

// ------------------------------ Horizontal sum (reduction)

// Returns 64-bit sums of 8-byte groups.
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<uint64_t> sums_of_u8x8(
    const vec_avx2<uint8_t> v) {
  return vec_avx2<uint64_t>(_mm256_sad_epu8(v.raw, _mm256_setzero_si256()));
}

// Returns N sums of differences of byte quadruplets, starting from byte offset
// i = [0, N) in window (11 consecutive bytes) and idx_ref * 4 in ref.
// This version computes two independent SAD with separate idx_ref.
template <int idx_ref1, int idx_ref0>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<int16_t> mpsadbw2(
    const vec_avx2<uint8_t> window, const vec_avx2<uint8_t> ref) {
  return vec_avx2<int16_t>(
      _mm256_mpsadbw_epu8(window.raw, ref.raw, (idx_ref1 << 3) + idx_ref0));
}

// Returns sum{lane[i]} in each lane. "v3210" is a replicated 128-bit block.
// Same logic as x86_sse4.h, but with vec_avx2 arguments.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> horz_sum_impl(
    char (&sizeof_t)[4], const vec_avx2<T, N> v3210) {
  const auto v1032 = shuffle_1032(v3210);
  const auto v31_20_31_20 = v3210 + v1032;
  const auto v20_31_20_31 = shuffle_0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> horz_sum_impl(
    char (&sizeof_t)[8], const vec_avx2<T, N> v10) {
  const auto v01 = shuffle_01(v10);
  return v10 + v01;
}

// Supported for {uif}32x8, {uif}64x4. Returns the sum in each lane.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> sum_of_lanes(
    const vec_avx2<T, N> vHL) {
  const vec_avx2<T, N> vLH = concat_lo_hi(vHL, vHL);
  char sizeof_t[sizeof(T)];
  return horz_sum_impl(sizeof_t, vLH + vHL);
}

}  // namespace ext

// TODO(janwas): wrappers for all intrinsics (in x86 namespace).
}  // namespace pik

#endif  // SIMD_ENABLE & SIMD_AVX2
#endif  // PIK_SIMD_X86_AVX2_H_
