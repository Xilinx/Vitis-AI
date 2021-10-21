// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SIMD_ARM64_NEON_H_
#define PIK_SIMD_ARM64_NEON_H_

// 128-bit ARM64 NEON vectors and operations.

#include "pik/simd/compiler_specific.h"
#include "pik/simd/shared.h"
#include "pik/simd/targets.h"

#if SIMD_ENABLE & SIMD_ARM8
#include <arm_neon.h>

namespace pik {

template <typename T, size_t N>
struct raw_arm8;

// 128
template <>
struct raw_arm8<uint8_t, 16> {
  using type = uint8x16_t;
};

template <>
struct raw_arm8<uint16_t, 8> {
  using type = uint16x8_t;
};

template <>
struct raw_arm8<uint32_t, 4> {
  using type = uint32x4_t;
};

template <>
struct raw_arm8<uint64_t, 2> {
  using type = uint64x2_t;
};

template <>
struct raw_arm8<int8_t, 16> {
  using type = int8x16_t;
};

template <>
struct raw_arm8<int16_t, 8> {
  using type = int16x8_t;
};

template <>
struct raw_arm8<int32_t, 4> {
  using type = int32x4_t;
};

template <>
struct raw_arm8<int64_t, 2> {
  using type = int64x2_t;
};

template <>
struct raw_arm8<float, 4> {
  using type = float32x4_t;
};

template <>
struct raw_arm8<double, 2> {
  using type = float64x2_t;
};

// 64
template <>
struct raw_arm8<uint8_t, 8> {
  using type = uint8x8_t;
};

template <>
struct raw_arm8<uint16_t, 4> {
  using type = uint16x4_t;
};

template <>
struct raw_arm8<uint32_t, 2> {
  using type = uint32x2_t;
};

template <>
struct raw_arm8<uint64_t, 1> {
  using type = uint64x1_t;
};

template <>
struct raw_arm8<int8_t, 8> {
  using type = int8x8_t;
};

template <>
struct raw_arm8<int16_t, 4> {
  using type = int16x4_t;
};

template <>
struct raw_arm8<int32_t, 2> {
  using type = int32x2_t;
};

template <>
struct raw_arm8<int64_t, 1> {
  using type = int64x1_t;
};

template <>
struct raw_arm8<float, 2> {
  using type = float32x2_t;
};

template <>
struct raw_arm8<double, 1> {
  using type = float64x1_t;
};

// 32 (same as 64)
template <>
struct raw_arm8<uint8_t, 4> {
  using type = uint8x8_t;
};

template <>
struct raw_arm8<uint16_t, 2> {
  using type = uint16x4_t;
};

template <>
struct raw_arm8<uint32_t, 1> {
  using type = uint32x2_t;
};

template <>
struct raw_arm8<int8_t, 4> {
  using type = int8x8_t;
};

template <>
struct raw_arm8<int16_t, 2> {
  using type = int16x4_t;
};

template <>
struct raw_arm8<int32_t, 1> {
  using type = int32x2_t;
};

template <>
struct raw_arm8<float, 1> {
  using type = float32x2_t;
};

// Returned by set_table_indices for use by table_lookup_lanes.
template <typename T>
struct permute_sse4 {
  uint8x16_t raw;
};

template <typename T, size_t N = ARM8::NumLanes<T>()>
class vec_arm8 {
  using Raw = typename raw_arm8<T, N>::type;

 public:
  SIMD_INLINE vec_arm8() {}
  vec_arm8(const vec_arm8&) = default;
  vec_arm8& operator=(const vec_arm8&) = default;
  SIMD_INLINE explicit vec_arm8(const Raw raw) : raw(raw) {}

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_INLINE vec_arm8& operator*=(const vec_arm8 other) {
    return *this = (*this * other);
  }
  SIMD_INLINE vec_arm8& operator/=(const vec_arm8 other) {
    return *this = (*this / other);
  }
  SIMD_INLINE vec_arm8& operator+=(const vec_arm8 other) {
    return *this = (*this + other);
  }
  SIMD_INLINE vec_arm8& operator-=(const vec_arm8 other) {
    return *this = (*this - other);
  }
  SIMD_INLINE vec_arm8& operator&=(const vec_arm8 other) {
    return *this = (*this & other);
  }
  SIMD_INLINE vec_arm8& operator|=(const vec_arm8 other) {
    return *this = (*this | other);
  }
  SIMD_INLINE vec_arm8& operator^=(const vec_arm8 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

template <typename T, size_t N>
struct VecT<T, N, ARM8> {
  using type = vec_arm8<T, N>;
};

using u8x16 = vec_arm8<uint8_t, 16>;
using u16x8 = vec_arm8<uint16_t, 8>;
using u32x4 = vec_arm8<uint32_t, 4>;
using u64x2 = vec_arm8<uint64_t, 2>;
using i8x16 = vec_arm8<int8_t, 16>;
using i16x8 = vec_arm8<int16_t, 8>;
using i32x4 = vec_arm8<int32_t, 4>;
using i64x2 = vec_arm8<int64_t, 2>;
using f32x4 = vec_arm8<float, 4>;
using f64x2 = vec_arm8<double, 2>;

using u8x8 = vec_arm8<uint8_t, 8>;
using u16x4 = vec_arm8<uint16_t, 4>;
using u32x2 = vec_arm8<uint32_t, 2>;
using u64x1 = vec_arm8<uint64_t, 1>;
using i8x8 = vec_arm8<int8_t, 8>;
using i16x4 = vec_arm8<int16_t, 4>;
using i32x2 = vec_arm8<int32_t, 2>;
using i64x1 = vec_arm8<int64_t, 1>;
using f32x2 = vec_arm8<float, 2>;
using f64x1 = vec_arm8<double, 1>;

using u8x4 = vec_arm8<uint8_t, 4>;
using u16x2 = vec_arm8<uint16_t, 2>;
using u32x1 = vec_arm8<uint32_t, 1>;
using i8x4 = vec_arm8<int8_t, 4>;
using i16x2 = vec_arm8<int16_t, 2>;
using i32x1 = vec_arm8<int32_t, 1>;
using f32x1 = vec_arm8<float, 1>;

// ------------------------------ Cast

// cast_to_u8
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<uint8_t, N> v) {
  return v;
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<uint16_t, N / 2> v) {
  return vec_arm8<uint8_t, N>(vreinterpretq_u8_u16(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<uint32_t, N / 4> v) {
  return vec_arm8<uint8_t, N>(vreinterpretq_u8_u32(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<uint64_t, N / 8> v) {
  return vec_arm8<uint8_t, N>(vreinterpretq_u8_u64(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<int8_t, N> v) {
  return vec_arm8<uint8_t, N>(vreinterpretq_u8_s8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<int16_t, N / 2> v) {
  return vec_arm8<uint8_t, N>(vreinterpretq_u8_s16(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<int32_t, N / 4> v) {
  return vec_arm8<uint8_t, N>(vreinterpretq_u8_s32(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<int64_t, N / 8> v) {
  return vec_arm8<uint8_t, N>(vreinterpretq_u8_s64(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<float, N / 4> v) {
  return vec_arm8<uint8_t, N>(vreinterpretq_u8_f32(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_to_u8(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<double, N / 8> v) {
  return vec_arm8<uint8_t, N>(vreinterpretq_u8_f64(v.raw));
}

// cast_u8_to
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> cast_u8_to(Desc<uint8_t, N, ARM8>,
                                            vec_arm8<uint8_t, N> v) {
  return v;
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> cast_u8_to(Desc<uint16_t, N, ARM8>,
                                             vec_arm8<uint8_t, N * 2> v) {
  return vec_arm8<uint16_t, N>(vreinterpretq_u16_u8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> cast_u8_to(Desc<uint32_t, N, ARM8>,
                                             vec_arm8<uint8_t, N * 4> v) {
  return vec_arm8<uint32_t, N>(vreinterpretq_u32_u8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> cast_u8_to(Desc<uint64_t, N, ARM8>,
                                             vec_arm8<uint8_t, N * 8> v) {
  return vec_arm8<uint64_t, N>(vreinterpretq_u64_u8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> cast_u8_to(Desc<int8_t, N, ARM8>,
                                           vec_arm8<uint8_t, N> v) {
  return vec_arm8<int8_t, N>(vreinterpretq_s8_u8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> cast_u8_to(Desc<int16_t, N, ARM8>,
                                            vec_arm8<uint8_t, N * 2> v) {
  return vec_arm8<int16_t, N>(vreinterpretq_s16_u8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> cast_u8_to(Desc<int32_t, N, ARM8>,
                                            vec_arm8<uint8_t, N * 4> v) {
  return vec_arm8<int32_t, N>(vreinterpretq_s32_u8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> cast_u8_to(Desc<int64_t, N, ARM8>,
                                            vec_arm8<uint8_t, N * 8> v) {
  return vec_arm8<int64_t, N>(vreinterpretq_s64_u8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<float, N> cast_u8_to(Desc<float, N, ARM8>,
                                          vec_arm8<uint8_t, N * 4> v) {
  return vec_arm8<float, N>(vreinterpretq_f32_u8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> cast_u8_to(Desc<double, N, ARM8>,
                                           vec_arm8<uint8_t, N * 8> v) {
  return vec_arm8<double, N>(vreinterpretq_f64_u8(v.raw));
}

// cast_to
template <typename T, size_t N, typename FromT>
SIMD_INLINE vec_arm8<T, N> cast_to(
    Desc<T, N, ARM8> d, vec_arm8<FromT, N * sizeof(T) / sizeof(FromT)> v) {
  const auto u8 = cast_to_u8(Desc<uint8_t, N * sizeof(T), ARM8>(), v);
  return cast_u8_to(d, u8);
}

// ------------------------------ Set

// Returns a vector with all lanes set to "t".
SIMD_INLINE vec_arm8<uint8_t> set1(Full<uint8_t, ARM8>, const uint8_t t) {
  return vec_arm8<uint8_t>(vdupq_n_u8(t));
}
SIMD_INLINE vec_arm8<uint16_t> set1(Full<uint16_t, ARM8>, const uint16_t t) {
  return vec_arm8<uint16_t>(vdupq_n_u16(t));
}
SIMD_INLINE vec_arm8<uint32_t> set1(Full<uint32_t, ARM8>, const uint32_t t) {
  return vec_arm8<uint32_t>(vdupq_n_u32(t));
}
SIMD_INLINE vec_arm8<uint64_t> set1(Full<uint64_t, ARM8>, const uint64_t t) {
  return vec_arm8<uint64_t>(vdupq_n_u64(t));
}
SIMD_INLINE vec_arm8<int8_t> set1(Full<int8_t, ARM8>, const int8_t t) {
  return vec_arm8<int8_t>(vdupq_n_s8(t));
}
SIMD_INLINE vec_arm8<int16_t> set1(Full<int16_t, ARM8>, const int16_t t) {
  return vec_arm8<int16_t>(vdupq_n_s16(t));
}
SIMD_INLINE vec_arm8<int32_t> set1(Full<int32_t, ARM8>, const int32_t t) {
  return vec_arm8<int32_t>(vdupq_n_s32(t));
}
SIMD_INLINE vec_arm8<int64_t> set1(Full<int64_t, ARM8>, const int64_t t) {
  return vec_arm8<int64_t>(vdupq_n_s64(t));
}
SIMD_INLINE vec_arm8<float> set1(Full<float, ARM8>, const float t) {
  return vec_arm8<float>(vdupq_n_f32(t));
}
SIMD_INLINE vec_arm8<double> set1(Full<double, ARM8>, const double t) {
  return vec_arm8<double>(vdupq_n_f64(t));
}

// 64
SIMD_INLINE vec_arm8<uint8_t, 8> set1(Desc<uint8_t, 8, ARM8>, const uint8_t t) {
  return vec_arm8<uint8_t, 8>(vdup_n_u8(t));
}
SIMD_INLINE vec_arm8<uint16_t, 4> set1(Desc<uint16_t, 4, ARM8>,
                                       const uint16_t t) {
  return vec_arm8<uint16_t, 4>(vdup_n_u16(t));
}
SIMD_INLINE vec_arm8<uint32_t, 2> set1(Desc<uint32_t, 2, ARM8>,
                                       const uint32_t t) {
  return vec_arm8<uint32_t, 2>(vdup_n_u32(t));
}
SIMD_INLINE vec_arm8<uint64_t, 1> set1(Desc<uint64_t, 1, ARM8>,
                                       const uint64_t t) {
  return vec_arm8<uint64_t, 1>(vdup_n_u64(t));
}
SIMD_INLINE vec_arm8<int8_t, 8> set1(Desc<int8_t, 8, ARM8>, const int8_t t) {
  return vec_arm8<int8_t, 8>(vdup_n_s8(t));
}
SIMD_INLINE vec_arm8<int16_t, 4> set1(Desc<int16_t, 4, ARM8>, const int16_t t) {
  return vec_arm8<int16_t, 4>(vdup_n_s16(t));
}
SIMD_INLINE vec_arm8<int32_t, 2> set1(Desc<int32_t, 2, ARM8>, const int32_t t) {
  return vec_arm8<int32_t, 2>(vdup_n_s32(t));
}
SIMD_INLINE vec_arm8<int64_t, 1> set1(Desc<int64_t, 1, ARM8>, const int64_t t) {
  return vec_arm8<int64_t, 1>(vdup_n_s64(t));
}
SIMD_INLINE vec_arm8<float, 2> set1(Desc<float, 2, ARM8>, const float t) {
  return vec_arm8<float, 2>(vdup_n_f32(t));
}
SIMD_INLINE vec_arm8<double, 1> set1(Desc<double, 1, ARM8>, const double t) {
  return vec_arm8<double, 1>(vdup_n_f64(t));
}

// 32
SIMD_INLINE vec_arm8<uint8_t, 4> set1(Desc<uint8_t, 4, ARM8>, const uint8_t t) {
  return vec_arm8<uint8_t, 4>(vdup_n_u8(t));
}
SIMD_INLINE vec_arm8<uint16_t, 2> set1(Desc<uint16_t, 2, ARM8>,
                                       const uint16_t t) {
  return vec_arm8<uint16_t, 2>(vdup_n_u16(t));
}
SIMD_INLINE vec_arm8<uint32_t, 1> set1(Desc<uint32_t, 1, ARM8>,
                                       const uint32_t t) {
  return vec_arm8<uint32_t, 1>(vdup_n_u32(t));
}
SIMD_INLINE vec_arm8<int8_t, 4> set1(Desc<int8_t, 4, ARM8>, const int8_t t) {
  return vec_arm8<int8_t, 4>(vdup_n_s8(t));
}
SIMD_INLINE vec_arm8<int16_t, 2> set1(Desc<int16_t, 2, ARM8>, const int16_t t) {
  return vec_arm8<int16_t, 2>(vdup_n_s16(t));
}
SIMD_INLINE vec_arm8<int32_t, 1> set1(Desc<int32_t, 1, ARM8>, const int32_t t) {
  return vec_arm8<int32_t, 1>(vdup_n_s32(t));
}
SIMD_INLINE vec_arm8<float, 1> set1(Desc<float, 1, ARM8>, const float t) {
  return vec_arm8<float, 1>(vdup_n_f32(t));
}

// Returns an all-zero vector.
template <typename T, size_t N>
SIMD_INLINE vec_arm8<T, N> setzero(Desc<T, N, ARM8> d) {
  return set1(d, 0);
}

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <typename T, size_t N, typename T2>
SIMD_INLINE vec_arm8<T, N> iota(Desc<T, N, ARM8> d, const T2 first) {
  SIMD_ALIGN T lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

// Returns a vector with uninitialized elements.
template <typename T, size_t N>
SIMD_INLINE vec_arm8<T, N> undefined(Desc<T, N, ARM8> d) {
  SIMD_DIAGNOSTICS(push)
  SIMD_DIAGNOSTICS_OFF(disable : 4701, ignored "-Wuninitialized")
  typename raw_arm8<T, N>::type a;
  return vec_arm8<T, N>(a);
  SIMD_DIAGNOSTICS(pop)
}

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> operator+(const vec_arm8<uint8_t, N> a,
                                           const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vaddq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> operator+(const vec_arm8<uint16_t, N> a,
                                            const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vaddq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> operator+(const vec_arm8<uint32_t, N> a,
                                            const vec_arm8<uint32_t, N> b) {
  return vec_arm8<uint32_t, N>(vaddq_u32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> operator+(const vec_arm8<uint64_t, N> a,
                                            const vec_arm8<uint64_t, N> b) {
  return vec_arm8<uint64_t, N>(vaddq_u64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> operator+(const vec_arm8<int8_t, N> a,
                                          const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vaddq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> operator+(const vec_arm8<int16_t, N> a,
                                           const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vaddq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator+(const vec_arm8<int32_t, N> a,
                                           const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vaddq_s32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> operator+(const vec_arm8<int64_t, N> a,
                                           const vec_arm8<int64_t, N> b) {
  return vec_arm8<int64_t, N>(vaddq_s64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator+(const vec_arm8<float, N> a,
                                         const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vaddq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator+(const vec_arm8<double, N> a,
                                          const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vaddq_f64(a.raw, b.raw));
}

// ------------------------------ Subtraction

// Unsigned
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> operator-(const vec_arm8<uint8_t, N> a,
                                           const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vsubq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> operator-(const vec_arm8<uint16_t, N> a,
                                            const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vsubq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> operator-(const vec_arm8<uint32_t, N> a,
                                            const vec_arm8<uint32_t, N> b) {
  return vec_arm8<uint32_t, N>(vsubq_u32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> operator-(const vec_arm8<uint64_t, N> a,
                                            const vec_arm8<uint64_t, N> b) {
  return vec_arm8<uint64_t, N>(vsubq_u64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> operator-(const vec_arm8<int8_t, N> a,
                                          const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vsubq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> operator-(const vec_arm8<int16_t, N> a,
                                           const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vsubq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator-(const vec_arm8<int32_t, N> a,
                                           const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vsubq_s32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> operator-(const vec_arm8<int64_t, N> a,
                                           const vec_arm8<int64_t, N> b) {
  return vec_arm8<int64_t, N>(vsubq_s64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator-(const vec_arm8<float, N> a,
                                         const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vsubq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator-(const vec_arm8<double, N> a,
                                          const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vsubq_f64(a.raw, b.raw));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> saturated_add(const vec_arm8<uint8_t, N> a,
                                               const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vqaddq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> saturated_add(const vec_arm8<uint16_t, N> a,
                                                const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vqaddq_u16(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> saturated_add(const vec_arm8<int8_t, N> a,
                                              const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vqaddq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> saturated_add(const vec_arm8<int16_t, N> a,
                                               const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vqaddq_s16(a.raw, b.raw));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> saturated_subtract(
    const vec_arm8<uint8_t, N> a, const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vqsubq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> saturated_subtract(
    const vec_arm8<uint16_t, N> a, const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vqsubq_u16(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> saturated_subtract(
    const vec_arm8<int8_t, N> a, const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vqsubq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> saturated_subtract(
    const vec_arm8<int16_t, N> a, const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vqsubq_s16(a.raw, b.raw));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> average_round(const vec_arm8<uint8_t, N> a,
                                               const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vrhaddq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> average_round(const vec_arm8<uint16_t, N> a,
                                                const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vrhaddq_u16(a.raw, b.raw));
}

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> abs(const vec_arm8<int8_t, N> v) {
  return vec_arm8<int8_t, N>(vabsq_s8(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> abs(const vec_arm8<int16_t, N> v) {
  return vec_arm8<int16_t, N>(vabsq_s16(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> abs(const vec_arm8<int32_t, N> v) {
  return vec_arm8<int32_t, N>(vabsq_s32(v.raw));
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> shift_left(const vec_arm8<uint16_t, N> v) {
  return vec_arm8<uint16_t, N>(vshlq_n_u16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> shift_right(const vec_arm8<uint16_t, N> v) {
  return vec_arm8<uint16_t, N>(vshrq_n_u16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> shift_left(const vec_arm8<uint32_t, N> v) {
  return vec_arm8<uint32_t, N>(vshlq_n_u32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> shift_right(const vec_arm8<uint32_t, N> v) {
  return vec_arm8<uint32_t, N>(vshrq_n_u32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> shift_left(const vec_arm8<uint64_t, N> v) {
  return vec_arm8<uint64_t, N>(vshlq_n_u64(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> shift_right(const vec_arm8<uint64_t, N> v) {
  return vec_arm8<uint64_t, N>(vshrq_n_u64(v.raw, kBits));
}

// Signed (no i64 shr)
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<int16_t, N> shift_left(const vec_arm8<int16_t, N> v) {
  return vec_arm8<int16_t, N>(vshlq_n_s16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<int16_t, N> shift_right(const vec_arm8<int16_t, N> v) {
  return vec_arm8<int16_t, N>(vshrq_n_s16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<int32_t, N> shift_left(const vec_arm8<int32_t, N> v) {
  return vec_arm8<int32_t, N>(vshlq_n_s32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<int32_t, N> shift_right(const vec_arm8<int32_t, N> v) {
  return vec_arm8<int32_t, N>(vshrq_n_s32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_INLINE vec_arm8<int64_t, N> shift_left(const vec_arm8<int64_t, N> v) {
  return vec_arm8<int64_t, N>(vshlq_n_s64(v.raw, kBits));
}

// ------------------------------ Shift lanes by same variable #bits

// Extra overhead, use _var instead unless SSE4 support is required.

template <typename T, size_t N>
struct shift_left_count {
  vec_arm8<T> v;
};

template <typename T, size_t N>
struct shift_right_count {
  vec_arm8<T> v;
};

template <typename T, size_t N>
SIMD_INLINE shift_left_count<T, N> set_shift_left_count(Desc<T, N, ARM8> d,
                                                        const int bits) {
  return shift_left_count<T, N>{set1(d, bits)};
}

template <typename T, size_t N>
SIMD_INLINE shift_right_count<T, N> set_shift_right_count(Desc<T, N, ARM8> d,
                                                          const int bits) {
  return shift_right_count<T, N>{set1(d, -bits)};
}

// Unsigned (no u8)
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> shift_left_same(
    const vec_arm8<uint16_t, N> v, const shift_left_count<uint16_t, N> bits) {
  return vec_arm8<uint16_t, N>(vshlq_u16(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> shift_right_same(
    const vec_arm8<uint16_t, N> v, const shift_right_count<uint16_t, N> bits) {
  return vec_arm8<uint16_t, N>(vshlq_u16(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> shift_left_same(
    const vec_arm8<uint32_t, N> v, const shift_left_count<uint32_t, N> bits) {
  return vec_arm8<uint32_t, N>(vshlq_u32(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> shift_right_same(
    const vec_arm8<uint32_t, N> v, const shift_right_count<uint32_t, N> bits) {
  return vec_arm8<uint32_t, N>(vshlq_u32(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> shift_left_same(
    const vec_arm8<uint64_t, N> v, const shift_left_count<uint64_t, N> bits) {
  return vec_arm8<uint64_t, N>(vshlq_u64(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> shift_right_same(
    const vec_arm8<uint64_t, N> v, const shift_right_count<uint64_t, N> bits) {
  return vec_arm8<uint64_t, N>(vshlq_u64(v.raw, bits.v.raw));
}

// Signed (no i8,i64)
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> shift_left_same(
    const vec_arm8<int16_t, N> v, const shift_left_count<int16_t, N> bits) {
  return vec_arm8<int16_t, N>(vshlq_s16(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> shift_right_same(
    const vec_arm8<int16_t, N> v, const shift_right_count<int16_t, N> bits) {
  return vec_arm8<int16_t, N>(vshlq_s16(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> shift_left_same(
    const vec_arm8<int32_t, N> v, const shift_left_count<int32_t, N> bits) {
  return vec_arm8<int32_t, N>(vshlq_s32(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> shift_right_same(
    const vec_arm8<int32_t, N> v, const shift_right_count<int32_t, N> bits) {
  return vec_arm8<int32_t, N>(vshlq_s32(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> shift_left_same(
    const vec_arm8<int64_t, N> v, const shift_left_count<int64_t, N> bits) {
  return vec_arm8<int64_t, N>(vshlq_s64(v.raw, bits.v.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> operator<<(const vec_arm8<uint32_t, N> v,
                                             const vec_arm8<uint32_t, N> bits) {
  return vec_arm8<uint32_t, N>(vshlq_u32(v.raw, bits.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> operator>>(const vec_arm8<uint32_t, N> v,
                                             const vec_arm8<uint32_t, N> bits) {
  return vec_arm8<uint32_t, N>(
      vshlq_u32(v.raw, vnegq_s32(vreinterpretq_s32_u32(bits.raw))));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> operator<<(const vec_arm8<uint64_t, N> v,
                                             const vec_arm8<uint64_t, N> bits) {
  return vec_arm8<uint64_t, N>(vshlq_u64(v.raw, bits.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> operator>>(const vec_arm8<uint64_t, N> v,
                                             const vec_arm8<uint64_t, N> bits) {
  return vec_arm8<uint64_t, N>(
      vshlq_u64(v.raw, vnegq_s64(vreinterpretq_s64_u64(bits.raw))));
}

// Signed (no i8,i16)
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator<<(const vec_arm8<int32_t, N> v,
                                            const vec_arm8<int32_t, N> bits) {
  return vec_arm8<int32_t, N>(vshlq_s32(v.raw, bits.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator>>(const vec_arm8<int32_t, N> v,
                                            const vec_arm8<int32_t, N> bits) {
  return vec_arm8<int32_t, N>(vshlq_s32(v.raw, vnegq_s32(bits.raw)));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> operator<<(const vec_arm8<int64_t, N> v,
                                            const vec_arm8<int64_t, N> bits) {
  return vec_arm8<int64_t, N>(vshlq_s64(v.raw, bits.raw));
}

// ------------------------------ Minimum

// Unsigned (no u64)
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> min(const vec_arm8<uint8_t, N> a,
                                     const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vminq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> min(const vec_arm8<uint16_t, N> a,
                                      const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vminq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> min(const vec_arm8<uint32_t, N> a,
                                      const vec_arm8<uint32_t, N> b) {
  return vec_arm8<uint32_t, N>(vminq_u32(a.raw, b.raw));
}

// Signed (no i64)
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> min(const vec_arm8<int8_t, N> a,
                                    const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vminq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> min(const vec_arm8<int16_t, N> a,
                                     const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vminq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> min(const vec_arm8<int32_t, N> a,
                                     const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vminq_s32(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_INLINE vec_arm8<float, N> min(const vec_arm8<float, N> a,
                                   const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vminq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> min(const vec_arm8<double, N> a,
                                    const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vminq_f64(a.raw, b.raw));
}

// ------------------------------ Maximum

// Unsigned (no u64)
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> max(const vec_arm8<uint8_t, N> a,
                                     const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vmaxq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> max(const vec_arm8<uint16_t, N> a,
                                      const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vmaxq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> max(const vec_arm8<uint32_t, N> a,
                                      const vec_arm8<uint32_t, N> b) {
  return vec_arm8<uint32_t, N>(vmaxq_u32(a.raw, b.raw));
}

// Signed (no i64)
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> max(const vec_arm8<int8_t, N> a,
                                    const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vmaxq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> max(const vec_arm8<int16_t, N> a,
                                     const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vmaxq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> max(const vec_arm8<int32_t, N> a,
                                     const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vmaxq_s32(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_INLINE vec_arm8<float, N> max(const vec_arm8<float, N> a,
                                   const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vmaxq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> max(const vec_arm8<double, N> a,
                                    const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vmaxq_f64(a.raw, b.raw));
}

// ------------------------------ Integer multiplication

// Unsigned
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> operator*(const vec_arm8<uint16_t, N> a,
                                            const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vmulq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> operator*(const vec_arm8<uint32_t, N> a,
                                            const vec_arm8<uint32_t, N> b) {
  return vec_arm8<uint32_t, N>(vmulq_u32(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> operator*(const vec_arm8<int16_t, N> a,
                                           const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vmulq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator*(const vec_arm8<int32_t, N> a,
                                           const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vmulq_s32(a.raw, b.raw));
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> mul_high(const vec_arm8<int16_t, N> a,
                                          const vec_arm8<int16_t, N> b) {
  int32x4_t rlo = vmull_s16(vget_low_s16(a.raw), vget_low_s16(b.raw));
  int32x4_t rhi = vmull_high_s16(a.raw, b.raw);
  return vec_arm8<int16_t, N>(
      vuzp2q_s16(vreinterpretq_s16_s32(rlo), vreinterpretq_s16_s32(rhi)));
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_INLINE vec_arm8<int64_t> mul_even(const vec_arm8<int32_t> a,
                                       const vec_arm8<int32_t> b) {
  int32x4_t a_packed = vuzp1q_s32(a.raw, a.raw);
  int32x4_t b_packed = vuzp1q_s32(b.raw, b.raw);
  return vec_arm8<int64_t>(
      vmull_s32(vget_low_s32(a_packed), vget_low_s32(b_packed)));
}
SIMD_INLINE vec_arm8<uint64_t> mul_even(const vec_arm8<uint32_t> a,
                                        const vec_arm8<uint32_t> b) {
  uint32x4_t a_packed = vuzp1q_u32(a.raw, a.raw);
  uint32x4_t b_packed = vuzp1q_u32(b.raw, b.raw);
  return vec_arm8<uint64_t>(
      vmull_u32(vget_low_u32(a_packed), vget_low_u32(b_packed)));
}

// ------------------------------ Floating-point negate

template <size_t N>
SIMD_INLINE vec_arm8<float, N> neg(const vec_arm8<float, N> v) {
  return vec_arm8<float, N>(vnegq_f32(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> neg(const vec_arm8<double, N> v) {
  return vec_arm8<double, N>(vnegq_f64(v.raw));
}

// ------------------------------ Floating-point mul / div

template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator*(const vec_arm8<float, N> a,
                                         const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vmulq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator*(const vec_arm8<double, N> a,
                                          const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vmulq_f64(a.raw, b.raw));
}

template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator/(const vec_arm8<float, N> a,
                                         const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vdivq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator/(const vec_arm8<double, N> a,
                                          const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vdivq_f64(a.raw, b.raw));
}

// Approximate reciprocal
template <size_t N>
SIMD_INLINE vec_arm8<float, N> approximate_reciprocal(
    const vec_arm8<float, N> v) {
  return vec_arm8<float, N>(vrecpeq_f32(v.raw));
}

// ------------------------------ Floating-point multiply-add variants

// Returns add + mul * x
template <size_t N>
SIMD_INLINE vec_arm8<float, N> mul_add(const vec_arm8<float, N> mul,
                                       const vec_arm8<float, N> x,
                                       const vec_arm8<float, N> add) {
  return vec_arm8<float, N>(vfmaq_f32(add.raw, mul.raw, x.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> mul_add(const vec_arm8<double, N> mul,
                                        const vec_arm8<double, N> x,
                                        const vec_arm8<double, N> add) {
  return vec_arm8<double, N>(vfmaq_f64(add.raw, mul.raw, x.raw));
}

// Returns add - mul * x
template <size_t N>
SIMD_INLINE vec_arm8<float, N> nmul_add(const vec_arm8<float, N> mul,
                                        const vec_arm8<float, N> x,
                                        const vec_arm8<float, N> add) {
  return vec_arm8<float, N>(vfmsq_f32(add.raw, mul.raw, x.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> nmul_add(const vec_arm8<double, N> mul,
                                         const vec_arm8<double, N> x,
                                         const vec_arm8<double, N> add) {
  return vec_arm8<double, N>(vfmsq_f64(add.raw, mul.raw, x.raw));
}

// Slightly more expensive (extra negate)
namespace ext {

// Returns mul * x - sub
template <size_t N>
SIMD_INLINE vec_arm8<float, N> mul_subtract(const vec_arm8<float, N> mul,
                                            const vec_arm8<float, N> x,
                                            const vec_arm8<float, N> sub) {
  return neg(nmul_add(mul, x, sub));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> mul_subtract(const vec_arm8<double, N> mul,
                                             const vec_arm8<double, N> x,
                                             const vec_arm8<double, N> sub) {
  return neg(nmul_add(mul, x, sub));
}

// Returns -mul * x - sub
template <size_t N>
SIMD_INLINE vec_arm8<float, N> nmul_subtract(const vec_arm8<float, N> mul,
                                             const vec_arm8<float, N> x,
                                             const vec_arm8<float, N> sub) {
  return neg(mul_add(mul, x, sub));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> nmul_subtract(const vec_arm8<double, N> mul,
                                              const vec_arm8<double, N> x,
                                              const vec_arm8<double, N> sub) {
  return neg(mul_add(mul, x, sub));
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Full precision square root
template <size_t N>
SIMD_INLINE vec_arm8<float, N> sqrt(const vec_arm8<float, N> v) {
  return vec_arm8<float, N>(vsqrtq_f32(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> sqrt(const vec_arm8<double, N> v) {
  return vec_arm8<double, N>(vsqrtq_f64(v.raw));
}

// Approximate reciprocal square root
template <size_t N>
SIMD_INLINE vec_arm8<float, N> approximate_reciprocal_sqrt(
    const vec_arm8<float, N> v) {
  return vec_arm8<float, N>(vrsqrteq_f32(v.raw));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer
template <size_t N>
SIMD_INLINE vec_arm8<float, N> round(const vec_arm8<float, N> v) {
  return vec_arm8<float, N>(vrndnq_f32(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> round(const vec_arm8<double, N> v) {
  return vec_arm8<double, N>(vrndnq_f64(v.raw));
}

// Toward +infinity, aka ceiling
template <size_t N>
SIMD_INLINE vec_arm8<float, N> ceil(const vec_arm8<float, N> v) {
  return vec_arm8<float, N>(vrndpq_f32(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> ceil(const vec_arm8<double, N> v) {
  return vec_arm8<double, N>(vrndpq_f64(v.raw));
}

// Toward -infinity, aka floor
template <size_t N>
SIMD_INLINE vec_arm8<float, N> floor(const vec_arm8<float, N> v) {
  return vec_arm8<float, N>(vrndmq_f32(v.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> floor(const vec_arm8<double, N> v) {
  return vec_arm8<double, N>(vrndmq_f64(v.raw));
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> operator==(const vec_arm8<uint8_t, N> a,
                                            const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vceqq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> operator==(const vec_arm8<uint16_t, N> a,
                                             const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vceqq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> operator==(const vec_arm8<uint32_t, N> a,
                                             const vec_arm8<uint32_t, N> b) {
  return vec_arm8<uint32_t, N>(vceqq_u32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> operator==(const vec_arm8<uint64_t, N> a,
                                             const vec_arm8<uint64_t, N> b) {
  return vec_arm8<uint64_t, N>(vceqq_u64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> operator==(const vec_arm8<int8_t, N> a,
                                           const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vceqq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> operator==(const vec_arm8<int16_t, N> a,
                                            const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vceqq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator==(const vec_arm8<int32_t, N> a,
                                            const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vceqq_s32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> operator==(const vec_arm8<int64_t, N> a,
                                            const vec_arm8<int64_t, N> b) {
  return vec_arm8<int64_t, N>(vceqq_s64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator==(const vec_arm8<float, N> a,
                                          const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vceqq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator==(const vec_arm8<double, N> a,
                                           const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vceqq_f64(a.raw, b.raw));
}

// ------------------------------ Strict inequality

// Signed/float <
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> operator<(const vec_arm8<int8_t, N> a,
                                          const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vcltq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> operator<(const vec_arm8<int16_t, N> a,
                                           const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vcltq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator<(const vec_arm8<int32_t, N> a,
                                           const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vcltq_s32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> operator<(const vec_arm8<int64_t, N> a,
                                           const vec_arm8<int64_t, N> b) {
  return vec_arm8<int64_t, N>(vcltq_s64(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator<(const vec_arm8<float, N> a,
                                         const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vcltq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator<(const vec_arm8<double, N> a,
                                          const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vcltq_f64(a.raw, b.raw));
}

// Signed/float >
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> operator>(const vec_arm8<int8_t, N> a,
                                          const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vcgtq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> operator>(const vec_arm8<int16_t, N> a,
                                           const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vcgtq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator>(const vec_arm8<int32_t, N> a,
                                           const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vcgtq_s32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> operator>(const vec_arm8<int64_t, N> a,
                                           const vec_arm8<int64_t, N> b) {
  return vec_arm8<int64_t, N>(vcgtq_s64(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator>(const vec_arm8<float, N> a,
                                         const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vcgtq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator>(const vec_arm8<double, N> a,
                                          const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vcgtq_f64(a.raw, b.raw));
}

// ------------------------------ Weak inequality

// Float <= >=
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator<=(const vec_arm8<float, N> a,
                                          const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vcleq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator<=(const vec_arm8<double, N> a,
                                           const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vcleq_f64(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator>=(const vec_arm8<float, N> a,
                                          const vec_arm8<float, N> b) {
  return vec_arm8<float, N>(vcgeq_f32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator>=(const vec_arm8<double, N> a,
                                           const vec_arm8<double, N> b) {
  return vec_arm8<double, N>(vcgeq_f64(a.raw, b.raw));
}

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> operator&(const vec_arm8<int8_t, N> a,
                                          const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vandq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> operator&(const vec_arm8<uint8_t, N> a,
                                           const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vandq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> operator&(const vec_arm8<int16_t, N> a,
                                           const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vandq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> operator&(const vec_arm8<uint16_t, N> a,
                                            const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vandq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator&(const vec_arm8<int32_t, N> a,
                                           const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vandq_s32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> operator&(const vec_arm8<uint32_t, N> a,
                                            const vec_arm8<uint32_t, N> b) {
  return vec_arm8<uint32_t, N>(vandq_u32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> operator&(const vec_arm8<int64_t, N> a,
                                           const vec_arm8<int64_t, N> b) {
  return vec_arm8<int64_t, N>(vandq_s64(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> operator&(const vec_arm8<uint64_t, N> a,
                                            const vec_arm8<uint64_t, N> b) {
  return vec_arm8<uint64_t, N>(vandq_u64(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator&(const vec_arm8<float, N> a,
                                         const vec_arm8<float, N> b) {
  const Full<uint32_t, ARM8> d;
  return cast_to(Full<float, ARM8>(), cast_to(d, a) & cast_to(d, b));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator&(const vec_arm8<double, N> a,
                                          const vec_arm8<double, N> b) {
  const Full<uint64_t, ARM8> d;
  return cast_to(Full<double, ARM8>(), cast_to(d, a) & cast_to(d, b));
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> andnot(const vec_arm8<int8_t, N> not_mask,
                                       const vec_arm8<int8_t, N> mask) {
  return vec_arm8<int8_t, N>(vbicq_s8(mask.raw, not_mask.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> andnot(const vec_arm8<uint8_t, N> not_mask,
                                        const vec_arm8<uint8_t, N> mask) {
  return vec_arm8<uint8_t, N>(vbicq_u8(mask.raw, not_mask.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> andnot(const vec_arm8<int16_t, N> not_mask,
                                        const vec_arm8<int16_t, N> mask) {
  return vec_arm8<int16_t, N>(vbicq_s16(mask.raw, not_mask.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> andnot(const vec_arm8<uint16_t, N> not_mask,
                                         const vec_arm8<uint16_t, N> mask) {
  return vec_arm8<uint16_t, N>(vbicq_u16(mask.raw, not_mask.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> andnot(const vec_arm8<int32_t, N> not_mask,
                                        const vec_arm8<int32_t, N> mask) {
  return vec_arm8<int32_t, N>(vbicq_s32(mask.raw, not_mask.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> andnot(const vec_arm8<uint32_t, N> not_mask,
                                         const vec_arm8<uint32_t, N> mask) {
  return vec_arm8<uint32_t, N>(vbicq_u32(mask.raw, not_mask.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> andnot(const vec_arm8<int64_t, N> not_mask,
                                        const vec_arm8<int64_t, N> mask) {
  return vec_arm8<int64_t, N>(vbicq_s64(mask.raw, not_mask.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> andnot(const vec_arm8<uint64_t, N> not_mask,
                                         const vec_arm8<uint64_t, N> mask) {
  return vec_arm8<uint64_t, N>(vbicq_u64(mask.raw, not_mask.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<float, N> andnot(const vec_arm8<float, N> not_mask,
                                      const vec_arm8<float, N> mask) {
  const Desc<uint32_t, N, ARM8> du;
  uint32x4_t ret = vbicq_u32(cast_to(du, mask).raw, cast_to(du, not_mask).raw);
  return vec_arm8<float, N>(vreinterpretq_f32_u32(ret));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> andnot(const vec_arm8<double, N> not_mask,
                                       const vec_arm8<double, N> mask) {
  const Desc<uint64_t, N, ARM8> du;
  uint64x2_t ret = vbicq_u64(cast_to(du, mask).raw, cast_to(du, not_mask).raw);
  return vec_arm8<double, N>(vreinterpretq_f64_u64(ret));
}

// ------------------------------ Bitwise OR

template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> operator|(const vec_arm8<int8_t, N> a,
                                          const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(vorrq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> operator|(const vec_arm8<uint8_t, N> a,
                                           const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(vorrq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> operator|(const vec_arm8<int16_t, N> a,
                                           const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(vorrq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> operator|(const vec_arm8<uint16_t, N> a,
                                            const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(vorrq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator|(const vec_arm8<int32_t, N> a,
                                           const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(vorrq_s32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> operator|(const vec_arm8<uint32_t, N> a,
                                            const vec_arm8<uint32_t, N> b) {
  return vec_arm8<uint32_t, N>(vorrq_u32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> operator|(const vec_arm8<int64_t, N> a,
                                           const vec_arm8<int64_t, N> b) {
  return vec_arm8<int64_t, N>(vorrq_s64(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> operator|(const vec_arm8<uint64_t, N> a,
                                            const vec_arm8<uint64_t, N> b) {
  return vec_arm8<uint64_t, N>(vorrq_u64(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator|(const vec_arm8<float, N> a,
                                         const vec_arm8<float, N> b) {
  const Full<uint32_t, ARM8> d;
  return cast_to(Full<float, ARM8>(), cast_to(d, a) | cast_to(d, b));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator|(const vec_arm8<double, N> a,
                                          const vec_arm8<double, N> b) {
  const Full<uint64_t, ARM8> d;
  return cast_to(Full<double, ARM8>(), cast_to(d, a) & cast_to(d, b));
}

// ------------------------------ Bitwise XOR

template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> operator^(const vec_arm8<int8_t, N> a,
                                          const vec_arm8<int8_t, N> b) {
  return vec_arm8<int8_t, N>(veorq_s8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> operator^(const vec_arm8<uint8_t, N> a,
                                           const vec_arm8<uint8_t, N> b) {
  return vec_arm8<uint8_t, N>(veorq_u8(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> operator^(const vec_arm8<int16_t, N> a,
                                           const vec_arm8<int16_t, N> b) {
  return vec_arm8<int16_t, N>(veorq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> operator^(const vec_arm8<uint16_t, N> a,
                                            const vec_arm8<uint16_t, N> b) {
  return vec_arm8<uint16_t, N>(veorq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> operator^(const vec_arm8<int32_t, N> a,
                                           const vec_arm8<int32_t, N> b) {
  return vec_arm8<int32_t, N>(veorq_s32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> operator^(const vec_arm8<uint32_t, N> a,
                                            const vec_arm8<uint32_t, N> b) {
  return vec_arm8<uint32_t, N>(veorq_u32(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> operator^(const vec_arm8<int64_t, N> a,
                                           const vec_arm8<int64_t, N> b) {
  return vec_arm8<int64_t, N>(veorq_s64(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> operator^(const vec_arm8<uint64_t, N> a,
                                            const vec_arm8<uint64_t, N> b) {
  return vec_arm8<uint64_t, N>(veorq_u64(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<float, N> operator^(const vec_arm8<float, N> a,
                                         const vec_arm8<float, N> b) {
  const Full<uint32_t, ARM8> d;
  return cast_to(Full<float, ARM8>(), cast_to(d, a) ^ cast_to(d, b));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> operator^(const vec_arm8<double, N> a,
                                          const vec_arm8<double, N> b) {
  const Full<uint64_t, ARM8> d;
  return cast_to(Full<double, ARM8>(), cast_to(d, a) ^ cast_to(d, b));
}

// ------------------------------ Select/blend

// Returns a mask for use by select().
// blendv_ps/pd only check the sign bit, so this is a no-op on x86.
template <size_t N>
SIMD_INLINE vec_arm8<float, N> condition_from_sign(const vec_arm8<float, N> v) {
  const Part<float, N> df;
  const Part<int32_t, N> di;
  return cast_to(df, shift_right<31>(cast_to(di, v)));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> condition_from_sign(
    const vec_arm8<double, N> v) {
  const Part<double, N> df;
  const Part<int64_t, N> di;
  return cast_to(df, shift_right<63>(cast_to(di, v)));
}

// Returns mask ? b : a. "mask" must either have been returned by
// selector_from_mask, or callers must ensure its lanes are T(0) or ~T(0).
template <size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> select(const vec_arm8<uint8_t, N> a,
                                        const vec_arm8<uint8_t, N> b,
                                        const vec_arm8<uint8_t, N> mask) {
  return vec_arm8<uint8_t, N>(vbslq_u8(mask.raw, b.raw, a.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int8_t, N> select(const vec_arm8<int8_t, N> a,
                                       const vec_arm8<int8_t, N> b,
                                       const vec_arm8<int8_t, N> mask) {
  return vec_arm8<int8_t, N>(vbslq_s8(mask.raw, b.raw, a.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> select(const vec_arm8<uint16_t, N> a,
                                         const vec_arm8<uint16_t, N> b,
                                         const vec_arm8<uint16_t, N> mask) {
  return vec_arm8<uint16_t, N>(vbslq_u16(mask.raw, b.raw, a.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int16_t, N> select(const vec_arm8<int16_t, N> a,
                                        const vec_arm8<int16_t, N> b,
                                        const vec_arm8<int16_t, N> mask) {
  return vec_arm8<int16_t, N>(vbslq_s16(mask.raw, b.raw, a.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint32_t, N> select(const vec_arm8<uint32_t, N> a,
                                         const vec_arm8<uint32_t, N> b,
                                         const vec_arm8<uint32_t, N> mask) {
  return vec_arm8<uint32_t, N>(vbslq_u32(mask.raw, b.raw, a.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> select(const vec_arm8<int32_t, N> a,
                                        const vec_arm8<int32_t, N> b,
                                        const vec_arm8<int32_t, N> mask) {
  return vec_arm8<int32_t, N>(vbslq_s32(mask.raw, b.raw, a.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<uint64_t, N> select(const vec_arm8<uint64_t, N> a,
                                         const vec_arm8<uint64_t, N> b,
                                         const vec_arm8<uint64_t, N> mask) {
  return vec_arm8<uint64_t, N>(vbslq_u64(mask.raw, b.raw, a.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<int64_t, N> select(const vec_arm8<int64_t, N> a,
                                        const vec_arm8<int64_t, N> b,
                                        const vec_arm8<int64_t, N> mask) {
  return vec_arm8<int64_t, N>(vbslq_s64(mask.raw, b.raw, a.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<float, N> select(const vec_arm8<float, N> a,
                                      const vec_arm8<float, N> b,
                                      const vec_arm8<float, N> mask) {
  return vec_arm8<float, N>(vbslq_f32(mask.raw, b.raw, a.raw));
}
template <size_t N>
SIMD_INLINE vec_arm8<double, N> select(const vec_arm8<double, N> a,
                                       const vec_arm8<double, N> b,
                                       const vec_arm8<double, N> mask) {
  return vec_arm8<double, N>(vbslq_f64(mask.raw, b.raw, a.raw));
}

// ================================================== MEMORY

// ------------------------------ Load 128

SIMD_INLINE vec_arm8<uint8_t> load_unaligned(
    Full<uint8_t, ARM8>, const uint8_t* SIMD_RESTRICT aligned) {
  return vec_arm8<uint8_t>(vld1q_u8(aligned));
}
SIMD_INLINE vec_arm8<uint16_t> load_unaligned(
    Full<uint16_t, ARM8>, const uint16_t* SIMD_RESTRICT aligned) {
  return vec_arm8<uint16_t>(vld1q_u16(aligned));
}
SIMD_INLINE vec_arm8<uint32_t> load_unaligned(
    Full<uint32_t, ARM8>, const uint32_t* SIMD_RESTRICT aligned) {
  return vec_arm8<uint32_t>(vld1q_u32(aligned));
}
SIMD_INLINE vec_arm8<uint64_t> load_unaligned(
    Full<uint64_t, ARM8>, const uint64_t* SIMD_RESTRICT aligned) {
  return vec_arm8<uint64_t>(vld1q_u64(aligned));
}
SIMD_INLINE vec_arm8<int8_t> load_unaligned(
    Full<int8_t, ARM8>, const int8_t* SIMD_RESTRICT aligned) {
  return vec_arm8<int8_t>(vld1q_s8(aligned));
}
SIMD_INLINE vec_arm8<int16_t> load_unaligned(
    Full<int16_t, ARM8>, const int16_t* SIMD_RESTRICT aligned) {
  return vec_arm8<int16_t>(vld1q_s16(aligned));
}
SIMD_INLINE vec_arm8<int32_t> load_unaligned(
    Full<int32_t, ARM8>, const int32_t* SIMD_RESTRICT aligned) {
  return vec_arm8<int32_t>(vld1q_s32(aligned));
}
SIMD_INLINE vec_arm8<int64_t> load_unaligned(
    Full<int64_t, ARM8>, const int64_t* SIMD_RESTRICT aligned) {
  return vec_arm8<int64_t>(vld1q_s64(aligned));
}
SIMD_INLINE vec_arm8<float> load_unaligned(Full<float, ARM8>,
                                           const float* SIMD_RESTRICT aligned) {
  return vec_arm8<float>(vld1q_f32(aligned));
}
SIMD_INLINE vec_arm8<double> load_unaligned(
    Full<double, ARM8>, const double* SIMD_RESTRICT aligned) {
  return vec_arm8<double>(vld1q_f64(aligned));
}

template <typename T>
SIMD_INLINE vec_arm8<T> load(Full<T, ARM8> d, const T* SIMD_RESTRICT p) {
  return load_unaligned(d, p);
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T>
SIMD_INLINE vec_arm8<T> load_dup128(Full<T, ARM8> d,
                                    const T* const SIMD_RESTRICT p) {
  return load_unaligned(d, p);
}

// ------------------------------ Load 64

SIMD_INLINE vec_arm8<uint8_t, 8> load(Desc<uint8_t, 8, ARM8>,
                                      const uint8_t* SIMD_RESTRICT p) {
  return vec_arm8<uint8_t, 8>(vld1_u8(p));
}
SIMD_INLINE vec_arm8<uint16_t, 4> load(Desc<uint16_t, 4, ARM8>,
                                       const uint16_t* SIMD_RESTRICT p) {
  return vec_arm8<uint16_t, 4>(vld1_u16(p));
}
SIMD_INLINE vec_arm8<uint32_t, 2> load(Desc<uint32_t, 2, ARM8>,
                                       const uint32_t* SIMD_RESTRICT p) {
  return vec_arm8<uint32_t, 2>(vld1_u32(p));
}
SIMD_INLINE vec_arm8<uint64_t, 1> load(Desc<uint64_t, 1, ARM8>,
                                       const uint64_t* SIMD_RESTRICT p) {
  return vec_arm8<uint64_t, 1>(vld1_u64(p));
}
SIMD_INLINE vec_arm8<int8_t, 8> load(Desc<int8_t, 8, ARM8>,
                                     const int8_t* SIMD_RESTRICT p) {
  return vec_arm8<int8_t, 8>(vld1_s8(p));
}
SIMD_INLINE vec_arm8<int16_t, 4> load(Desc<int16_t, 4, ARM8>,
                                      const int16_t* SIMD_RESTRICT p) {
  return vec_arm8<int16_t, 4>(vld1_s16(p));
}
SIMD_INLINE vec_arm8<int32_t, 2> load(Desc<int32_t, 2, ARM8>,
                                      const int32_t* SIMD_RESTRICT p) {
  return vec_arm8<int32_t, 2>(vld1_s32(p));
}
SIMD_INLINE vec_arm8<int64_t, 1> load(Desc<int64_t, 1, ARM8>,
                                      const int64_t* SIMD_RESTRICT p) {
  return vec_arm8<int64_t, 1>(vld1_s64(p));
}
SIMD_INLINE vec_arm8<float, 2> load(Desc<float, 2, ARM8>,
                                    const float* SIMD_RESTRICT p) {
  return vec_arm8<float, 2>(vld1_f32(p));
}
SIMD_INLINE vec_arm8<double, 1> load(Desc<double, 1, ARM8>,
                                     const double* SIMD_RESTRICT p) {
  return vec_arm8<double, 1>(vld1_f64(p));
}

// ------------------------------ Load 32

// In the following load functions, |a| is purposely undefined.
// It is a required parameter to the intrinsic, however
// we don't actually care what is in it, and we don't want
// to introduce extra overhead by initializing it to something.

SIMD_INLINE vec_arm8<uint8_t, 4> load(Desc<uint8_t, 4, ARM8> d,
                                      const uint8_t* SIMD_RESTRICT p) {
  uint32x2_t a = undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return vec_arm8<uint8_t, 4>(vreinterpret_u8_u32(b));
}
SIMD_INLINE vec_arm8<uint16_t, 2> load(Desc<uint16_t, 2, ARM8> d,
                                       const uint16_t* SIMD_RESTRICT p) {
  uint32x2_t a = undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return vec_arm8<uint16_t, 2>(vreinterpret_u16_u32(b));
}
SIMD_INLINE vec_arm8<uint32_t, 1> load(Desc<uint32_t, 1, ARM8> d,
                                       const uint32_t* SIMD_RESTRICT p) {
  uint32x2_t a = undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(p, a, 0);
  return vec_arm8<uint32_t, 1>(b);
}
SIMD_INLINE vec_arm8<int8_t, 4> load(Desc<int8_t, 4, ARM8> d,
                                     const int8_t* SIMD_RESTRICT p) {
  int32x2_t a = undefined(d).raw;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return vec_arm8<int8_t, 4>(vreinterpret_s8_s32(b));
}
SIMD_INLINE vec_arm8<int16_t, 2> load(Desc<int16_t, 2, ARM8> d,
                                      const int16_t* SIMD_RESTRICT p) {
  int32x2_t a = undefined(d).raw;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return vec_arm8<int16_t, 2>(vreinterpret_s16_s32(b));
}
SIMD_INLINE vec_arm8<int32_t, 1> load(Desc<int32_t, 1, ARM8> d,
                                      const int32_t* SIMD_RESTRICT p) {
  int32x2_t a = undefined(d).raw;
  int32x2_t b = vld1_lane_s32(p, a, 0);
  return vec_arm8<int32_t, 1>(b);
}
SIMD_INLINE vec_arm8<float, 1> load(Desc<float, 1, ARM8> d,
                                    const float* SIMD_RESTRICT p) {
  float32x2_t a = undefined(d).raw;
  float32x2_t b = vld1_lane_f32(p, a, 0);
  return vec_arm8<float, 1>(b);
}

// ------------------------------ Store 128

SIMD_INLINE void store_unaligned(const vec_arm8<uint8_t> v, Full<uint8_t, ARM8>,
                                 uint8_t* SIMD_RESTRICT aligned) {
  vst1q_u8(aligned, v.raw);
}
SIMD_INLINE void store_unaligned(const vec_arm8<uint16_t> v,
                                 Full<uint16_t, ARM8>,
                                 uint16_t* SIMD_RESTRICT aligned) {
  vst1q_u16(aligned, v.raw);
}
SIMD_INLINE void store_unaligned(const vec_arm8<uint32_t> v,
                                 Full<uint32_t, ARM8>,
                                 uint32_t* SIMD_RESTRICT aligned) {
  vst1q_u32(aligned, v.raw);
}
SIMD_INLINE void store_unaligned(const vec_arm8<uint64_t> v,
                                 Full<uint64_t, ARM8>,
                                 uint64_t* SIMD_RESTRICT aligned) {
  vst1q_u64(aligned, v.raw);
}
SIMD_INLINE void store_unaligned(const vec_arm8<int8_t> v, Full<int8_t, ARM8>,
                                 int8_t* SIMD_RESTRICT aligned) {
  vst1q_s8(aligned, v.raw);
}
SIMD_INLINE void store_unaligned(const vec_arm8<int16_t> v, Full<int16_t, ARM8>,
                                 int16_t* SIMD_RESTRICT aligned) {
  vst1q_s16(aligned, v.raw);
}
SIMD_INLINE void store_unaligned(const vec_arm8<int32_t> v, Full<int32_t, ARM8>,
                                 int32_t* SIMD_RESTRICT aligned) {
  vst1q_s32(aligned, v.raw);
}
SIMD_INLINE void store_unaligned(const vec_arm8<int64_t> v, Full<int64_t, ARM8>,
                                 int64_t* SIMD_RESTRICT aligned) {
  vst1q_s64(aligned, v.raw);
}
SIMD_INLINE void store_unaligned(const vec_arm8<float> v, Full<float, ARM8>,
                                 float* SIMD_RESTRICT aligned) {
  vst1q_f32(aligned, v.raw);
}
SIMD_INLINE void store_unaligned(const vec_arm8<double> v, Full<double, ARM8>,
                                 double* SIMD_RESTRICT aligned) {
  vst1q_f64(aligned, v.raw);
}

template <typename T, size_t N>
SIMD_INLINE void store(vec_arm8<T, N> v, Desc<T, N, ARM8> d,
                       T* SIMD_RESTRICT p) {
  store_unaligned(v, d, p);
}

// ------------------------------ Store 64

SIMD_INLINE void store(const vec_arm8<uint8_t, 8> v, Desc<uint8_t, 8, ARM8>,
                       uint8_t* SIMD_RESTRICT p) {
  vst1_u8(p, v.raw);
}
SIMD_INLINE void store(const vec_arm8<uint16_t, 4> v, Desc<uint16_t, 4, ARM8>,
                       uint16_t* SIMD_RESTRICT p) {
  vst1_u16(p, v.raw);
}
SIMD_INLINE void store(const vec_arm8<uint32_t, 2> v, Desc<uint32_t, 2, ARM8>,
                       uint32_t* SIMD_RESTRICT p) {
  vst1_u32(p, v.raw);
}
SIMD_INLINE void store(const vec_arm8<uint64_t, 1> v, Desc<uint64_t, 1, ARM8>,
                       uint64_t* SIMD_RESTRICT p) {
  vst1_u64(p, v.raw);
}
SIMD_INLINE void store(const vec_arm8<int8_t, 8> v, Desc<int8_t, 8, ARM8>,
                       int8_t* SIMD_RESTRICT p) {
  vst1_s8(p, v.raw);
}
SIMD_INLINE void store(const vec_arm8<int16_t, 4> v, Desc<int16_t, 4, ARM8>,
                       int16_t* SIMD_RESTRICT p) {
  vst1_s16(p, v.raw);
}
SIMD_INLINE void store(const vec_arm8<int32_t, 2> v, Desc<int32_t, 2, ARM8>,
                       int32_t* SIMD_RESTRICT p) {
  vst1_s32(p, v.raw);
}
SIMD_INLINE void store(const vec_arm8<int64_t, 1> v, Desc<int64_t, 1, ARM8>,
                       int64_t* SIMD_RESTRICT p) {
  vst1_s64(p, v.raw);
}
SIMD_INLINE void store(const vec_arm8<float, 2> v, Desc<float, 2, ARM8>,
                       float* SIMD_RESTRICT p) {
  vst1_f32(p, v.raw);
}
SIMD_INLINE void store(const vec_arm8<double, 1> v, Desc<double, 1, ARM8>,
                       double* SIMD_RESTRICT p) {
  vst1_f64(p, v.raw);
}

// ------------------------------ Store 32

SIMD_INLINE void store(const vec_arm8<uint8_t, 4> v, Desc<uint8_t, 4, ARM8>,
                       uint8_t* SIMD_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u8(v.raw);
  vst1_lane_u32(p, a, 0);
}
SIMD_INLINE void store(const vec_arm8<uint16_t, 2> v, Desc<uint16_t, 2, ARM8>,
                       uint16_t* SIMD_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u16(v.raw);
  vst1_lane_u32(p, a, 0);
}
SIMD_INLINE void store(const vec_arm8<uint32_t, 1> v, Desc<uint32_t, 1, ARM8>,
                       uint32_t* SIMD_RESTRICT p) {
  vst1_lane_u32(p, v.raw, 0);
}
SIMD_INLINE void store(const vec_arm8<int8_t, 4> v, Desc<int8_t, 4, ARM8>,
                       int8_t* SIMD_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s8(v.raw);
  vst1_lane_s32(p, a, 0);
}
SIMD_INLINE void store(const vec_arm8<int16_t, 2> v, Desc<int16_t, 2, ARM8>,
                       int16_t* SIMD_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s16(v.raw);
  vst1_lane_s32(p, a, 0);
}
SIMD_INLINE void store(const vec_arm8<int32_t, 1> v, Desc<int32_t, 1, ARM8>,
                       int32_t* SIMD_RESTRICT p) {
  vst1_lane_s32(p, v.raw, 0);
}
SIMD_INLINE void store(const vec_arm8<float, 1> v, Desc<float, 1, ARM8>,
                       float* SIMD_RESTRICT p) {
  vst1_lane_f32(p, v.raw, 0);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
SIMD_INLINE void stream(const vec_arm8<T> v, Full<T, ARM8> d,
                        T* SIMD_RESTRICT aligned) {
  store(v, d, aligned);
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned: zero-extend.
SIMD_INLINE vec_arm8<uint16_t> convert_to(Full<uint16_t, ARM8>,
                                          const vec_arm8<uint8_t, 8> v) {
  return vec_arm8<uint16_t>(vmovl_u8(v.raw));
}
SIMD_INLINE vec_arm8<uint32_t> convert_to(Full<uint32_t, ARM8>,
                                          const vec_arm8<uint8_t, 4> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  return vec_arm8<uint32_t>(vmovl_u16(vget_low_u16(a)));
}
SIMD_INLINE vec_arm8<uint32_t> convert_to(Full<uint32_t, ARM8>,
                                          const vec_arm8<uint16_t, 4> v) {
  return vec_arm8<uint32_t>(vmovl_u16(v.raw));
}
SIMD_INLINE vec_arm8<uint64_t> convert_to(Full<uint64_t, ARM8>,
                                          const vec_arm8<uint32_t, 2> v) {
  return vec_arm8<uint64_t>(vmovl_u32(v.raw));
}
SIMD_INLINE vec_arm8<int16_t> convert_to(Full<int16_t, ARM8>,
                                         const vec_arm8<uint8_t, 8> v) {
  return vec_arm8<int16_t>(vmovl_u8(v.raw));
}
SIMD_INLINE vec_arm8<int32_t> convert_to(Full<int32_t, ARM8>,
                                         const vec_arm8<uint8_t, 4> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  return vec_arm8<int32_t>(vreinterpretq_s32_u16(vmovl_u16(vget_low_u16(a))));
}
SIMD_INLINE vec_arm8<int32_t> convert_to(Full<int32_t, ARM8>,
                                         const vec_arm8<uint16_t, 4> v) {
  return vec_arm8<int32_t>(vmovl_u16(v.raw));
}

SIMD_INLINE vec_arm8<uint32_t> u32_from_u8(const vec_arm8<uint8_t> v) {
  return convert_to(Full<uint32_t, ARM8>(), v);
}

// Signed: replicate sign bit.
SIMD_INLINE vec_arm8<int16_t> convert_to(Full<int16_t, ARM8>,
                                         const vec_arm8<int8_t, 8> v) {
  return vec_arm8<int16_t>(vmovl_s8(v.raw));
}
SIMD_INLINE vec_arm8<int32_t> convert_to(Full<int32_t, ARM8>,
                                         const vec_arm8<int8_t, 4> v) {
  int16x8_t a = vmovl_s8(v.raw);
  return vec_arm8<int32_t>(vmovl_s16(vget_low_s16(a)));
}
SIMD_INLINE vec_arm8<int32_t> convert_to(Full<int32_t, ARM8>,
                                         const vec_arm8<int16_t, 4> v) {
  return vec_arm8<int32_t>(vmovl_s16(v.raw));
}
SIMD_INLINE vec_arm8<int64_t> convert_to(Full<int64_t, ARM8>,
                                         const vec_arm8<int32_t, 2> v) {
  return vec_arm8<int64_t>(vmovl_s32(v.raw));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

template<size_t N>
SIMD_INLINE vec_arm8<uint16_t, N> convert_to(Part<uint16_t, N, ARM8>,
                                             const vec_arm8<int32_t> v) {
  return vec_arm8<uint16_t, N>(vqmovun_s32(v.raw));
}
template<size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> convert_to(Part<uint8_t, N, ARM8>,
                                            const vec_arm8<uint16_t> v) {
  return vec_arm8<uint8_t, N>(vqmovn_u16(v.raw));
}

template<size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> convert_to(Part<uint8_t, N, ARM8>,
                                            const vec_arm8<int16_t> v) {
  return vec_arm8<uint8_t, N>(vqmovun_s16(v.raw));
}

template<size_t N>
SIMD_INLINE vec_arm8<int16_t, N> convert_to(Part<int16_t, N, ARM8>,
                                            const vec_arm8<int32_t> v) {
  return vec_arm8<int16_t, N>(vqmovn_s32(v.raw));
}
template<size_t N>
SIMD_INLINE vec_arm8<int8_t, N> convert_to(Part<int8_t, N, ARM8>,
                                           const vec_arm8<int16_t> v) {
  return vec_arm8<int8_t, N>(vqmovn_s16(v.raw));
}

// In the following convert_to functions, |b| is purposely undefined.
// The value a needs to be extended to 128 bits so that vqmovn can be
// used and |b| is undefined so that no extra overhead is introduced.
SIMD_DIAGNOSTICS(push)
SIMD_DIAGNOSTICS_OFF(disable : 4701, ignored "-Wuninitialized")

template<size_t N>
SIMD_INLINE vec_arm8<uint8_t, N> convert_to(Part<uint8_t, N, ARM8>,
                                            const vec_arm8<int32_t> v) {
  vec_arm8<uint16_t, N> a = convert_to(Desc<uint16_t, N, ARM8>(), v);
  vec_arm8<uint16_t, N> b;
  uint16x8_t c = vcombine_u16(a.raw, b.raw);
  return vec_arm8<uint8_t, N>(vqmovn_u16(c));
}

template<size_t N>
SIMD_INLINE vec_arm8<int8_t, N> convert_to(Part<int8_t, N, ARM8>,
                                           const vec_arm8<int32_t> v) {
  vec_arm8<int16_t, N> a = convert_to(Desc<int16_t, N, ARM8>(), v);
  vec_arm8<int16_t, N> b;
  uint16x8_t c = vcombine_s16(a.raw, b.raw);
  return vec_arm8<int8_t, N>(vqmovn_s16(c));
}

SIMD_DIAGNOSTICS(pop)

// ------------------------------ Convert i32 <=> f32

template <size_t N>
SIMD_INLINE vec_arm8<float, N> convert_to(Part<float, N, ARM8>,
                                          const vec_arm8<int32_t, N> v) {
  return vec_arm8<float, N>(vcvtq_f32_s32(v.raw));
}
// Truncates (rounds toward zero).
template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> convert_to(Part<int32_t, N, ARM8>,
                                            const vec_arm8<float, N> v) {
  return vec_arm8<int32_t, N>(vcvtq_s32_f32(v.raw));
}

template <size_t N>
SIMD_INLINE vec_arm8<int32_t, N> nearest_int(const vec_arm8<float, N> v) {
  return vec_arm8<int32_t, N>(vcvtnq_s32_f32(v.raw));
}

// ================================================== SWIZZLE

// ------------------------------ 'Extract' other half (see any_part)

// These copy hi into lo
SIMD_INLINE vec_arm8<uint8_t, 8> other_half(const vec_arm8<uint8_t> v) {
  return vec_arm8<uint8_t, 8>(vget_high_u8(v.raw));
}
SIMD_INLINE vec_arm8<int8_t, 8> other_half(const vec_arm8<int8_t> v) {
  return vec_arm8<int8_t, 8>(vget_high_s8(v.raw));
}
SIMD_INLINE vec_arm8<uint16_t, 4> other_half(const vec_arm8<uint16_t> v) {
  return vec_arm8<uint16_t, 4>(vget_high_u16(v.raw));
}
SIMD_INLINE vec_arm8<int16_t, 4> other_half(const vec_arm8<int16_t> v) {
  return vec_arm8<int16_t, 4>(vget_high_s16(v.raw));
}
SIMD_INLINE vec_arm8<uint32_t, 2> other_half(const vec_arm8<uint32_t> v) {
  return vec_arm8<uint32_t, 2>(vget_high_u32(v.raw));
}
SIMD_INLINE vec_arm8<int32_t, 2> other_half(const vec_arm8<int32_t> v) {
  return vec_arm8<int32_t, 2>(vget_high_s32(v.raw));
}
SIMD_INLINE vec_arm8<uint64_t, 1> other_half(const vec_arm8<uint64_t> v) {
  return vec_arm8<uint64_t, 1>(vget_high_u64(v.raw));
}
SIMD_INLINE vec_arm8<int64_t, 1> other_half(const vec_arm8<int64_t> v) {
  return vec_arm8<int64_t, 1>(vget_high_s64(v.raw));
}
SIMD_INLINE vec_arm8<float, 2> other_half(const vec_arm8<float> v) {
  return vec_arm8<float, 2>(vget_high_f32(v.raw));
}
SIMD_INLINE vec_arm8<double, 1> other_half(const vec_arm8<double> v) {
  return vec_arm8<double, 1>(vget_high_f64(v.raw));
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_INLINE vec_arm8<T> combine_shift_right_bytes(const vec_arm8<T> hi,
                                                  const vec_arm8<T> lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  const Full<uint8_t, ARM8> d8;
  return cast_to(Full<T, ARM8>(),
                 vec_arm8<uint8_t>(vextq_u8(cast_to(d8, lo).raw,
                                            cast_to(d8, hi).raw, kBytes)));
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T, size_t N>
SIMD_INLINE vec_arm8<T, N> shift_left_bytes(const vec_arm8<T, N> v) {
  return combine_shift_right_bytes<16 - kBytes>(v, setzero(Full<T, ARM8>()));
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T, size_t N>
SIMD_INLINE vec_arm8<T, N> shift_right_bytes(const vec_arm8<T, N> v) {
  return combine_shift_right_bytes<kBytes>(setzero(Full<T, ARM8>()), v);
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_INLINE vec_arm8<uint16_t> broadcast(const vec_arm8<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return vec_arm8<uint16_t>(vdupq_laneq_u16(v.raw, kLane));
}
template <int kLane>
SIMD_INLINE vec_arm8<uint32_t> broadcast(const vec_arm8<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return vec_arm8<uint32_t>(vdupq_laneq_u32(v.raw, kLane));
}
template <int kLane>
SIMD_INLINE vec_arm8<uint64_t> broadcast(const vec_arm8<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return vec_arm8<uint64_t>(vdupq_laneq_u64(v.raw, kLane));
}

// Signed
template <int kLane>
SIMD_INLINE vec_arm8<int16_t> broadcast(const vec_arm8<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return vec_arm8<int16_t>(vdupq_laneq_s16(v.raw, kLane));
}
template <int kLane>
SIMD_INLINE vec_arm8<int32_t> broadcast(const vec_arm8<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return vec_arm8<int32_t>(vdupq_laneq_s32(v.raw, kLane));
}
template <int kLane>
SIMD_INLINE vec_arm8<int64_t> broadcast(const vec_arm8<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return vec_arm8<int64_t>(vdupq_laneq_s64(v.raw, kLane));
}

// Float
template <int kLane>
SIMD_INLINE vec_arm8<float> broadcast(const vec_arm8<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return vec_arm8<float>(vdupq_laneq_f32(v.raw, kLane));
}
template <int kLane>
SIMD_INLINE vec_arm8<double> broadcast(const vec_arm8<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return vec_arm8<double>(vdupq_laneq_f64(v.raw, kLane));
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
SIMD_INLINE vec_arm8<T> table_lookup_bytes(const vec_arm8<T> bytes,
                                           const vec_arm8<TI> from) {
  const Full<uint8_t, ARM8> d8;
  return cast_to(Full<T, ARM8>(),
                 vec_arm8<uint8_t>(vqtbl1q_u8(cast_to(d8, bytes).raw,
                                              cast_to(d8, from).raw)));
}

// ------------------------------ Hard-coded shuffles

// Notation: let vec_arm8<int32_t> have lanes 3,2,1,0 (0 is least-significant).
// shuffle_0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// combine_shift_right_bytes but the shuffle_abcd notation is more convenient.

// Swap 64-bit halves
template <typename T>
SIMD_INLINE vec_arm8<T> shuffle_1032(const vec_arm8<T> v) {
  return combine_shift_right_bytes<8>(v, v);
}
template <typename T>
SIMD_INLINE vec_arm8<T> shuffle_01(const vec_arm8<T> v) {
  return combine_shift_right_bytes<8>(v, v);
}

// Rotate right 32 bits
template <typename T>
SIMD_INLINE vec_arm8<T> shuffle_0321(const vec_arm8<T> v) {
  return combine_shift_right_bytes<4>(v, v);
}

// Rotate left 32 bits
template <typename T>
SIMD_INLINE vec_arm8<T> shuffle_2103(const vec_arm8<T> v) {
  return combine_shift_right_bytes<12>(v, v);
}

// Reverse
template <typename T>
SIMD_INLINE vec_arm8<T> shuffle_0123(const vec_arm8<T> v) {
  // TODO(janwas): more efficient implementation?
  static constexpr uint8_t bytes[16] = {15, 14, 13, 12, 11, 10, 9, 8,
                                        7,  6,  5,  4,  3,  2,  1, 0};
  return table_lookup_bytes(v, load(Full<uint8_t, ARM8>(), bytes));
}

// ------------------------------ Permute (runtime variable)

template <typename T>
SIMD_INLINE permute_arm8<T> set_table_indices(const Full<T, ARM8> d,
                                        const int32_t* idx) {
  const Full<uint8_t, ARM8> d8;
  SIMD_ALIGN uint8_t control[d8.N];
  for (size_t idx_byte = 0; idx_byte < d8.N; ++idx_byte) {
    const size_t idx_lane = idx_byte / sizeof(T);
    const size_t mod = idx_byte % sizeof(T);
    control[idx_byte] = idx[idx_lane] * sizeof(T) + mod;
  }
  return permute_arm8<T>{load(d8, control).raw};
}

SIMD_INLINE vec_arm8<uint32_t> table_lookup_lanes(const vec_arm8<uint32_t> v,
                                             const permute_arm8<uint32_t> idx) {
  return table_lookup_bytes(v, vec_arm8<uint8_t>(idx.raw));
}
SIMD_INLINE vec_arm8<int32_t> table_lookup_lanes(const vec_arm8<int32_t> v,
                                            const permute_arm8<int32_t> idx) {
  return table_lookup_bytes(v, vec_arm8<uint8_t>(idx.raw));
}
SIMD_INLINE vec_arm8<float> table_lookup_lanes(const vec_arm8<float> v,
                                          const permute_arm8<float> idx) {
  const Full<int32_t, ARM8> di;
  const Full<float, ARM8> df;
  return cast_to(
      df, table_lookup_bytes(cast_to(di, v), vec_arm8<uint8_t>(idx.raw)));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).

SIMD_INLINE vec_arm8<uint8_t> interleave_lo(const vec_arm8<uint8_t> a,
                                            const vec_arm8<uint8_t> b) {
  return vec_arm8<uint8_t>(vzip1q_u8(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint16_t> interleave_lo(const vec_arm8<uint16_t> a,
                                             const vec_arm8<uint16_t> b) {
  return vec_arm8<uint16_t>(vzip1q_u16(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint32_t> interleave_lo(const vec_arm8<uint32_t> a,
                                             const vec_arm8<uint32_t> b) {
  return vec_arm8<uint32_t>(vzip1q_u32(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint64_t> interleave_lo(const vec_arm8<uint64_t> a,
                                             const vec_arm8<uint64_t> b) {
  return vec_arm8<uint64_t>(vzip1q_u64(a.raw, b.raw));
}

SIMD_INLINE vec_arm8<int8_t> interleave_lo(const vec_arm8<int8_t> a,
                                           const vec_arm8<int8_t> b) {
  return vec_arm8<int8_t>(vzip1q_s8(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int16_t> interleave_lo(const vec_arm8<int16_t> a,
                                            const vec_arm8<int16_t> b) {
  return vec_arm8<int16_t>(vzip1q_s16(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int32_t> interleave_lo(const vec_arm8<int32_t> a,
                                            const vec_arm8<int32_t> b) {
  return vec_arm8<int32_t>(vzip1q_s32(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int64_t> interleave_lo(const vec_arm8<int64_t> a,
                                            const vec_arm8<int64_t> b) {
  return vec_arm8<int64_t>(vzip1q_s64(a.raw, b.raw));
}

SIMD_INLINE vec_arm8<float> interleave_lo(const vec_arm8<float> a,
                                          const vec_arm8<float> b) {
  return vec_arm8<float>(vzip1q_f32(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<double> interleave_lo(const vec_arm8<double> a,
                                           const vec_arm8<double> b) {
  return vec_arm8<double>(vzip1q_f64(a.raw, b.raw));
}

SIMD_INLINE vec_arm8<uint8_t> interleave_hi(const vec_arm8<uint8_t> a,
                                            const vec_arm8<uint8_t> b) {
  return vec_arm8<uint8_t>(vzip2q_u8(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint16_t> interleave_hi(const vec_arm8<uint16_t> a,
                                             const vec_arm8<uint16_t> b) {
  return vec_arm8<uint16_t>(vzip2q_u16(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint32_t> interleave_hi(const vec_arm8<uint32_t> a,
                                             const vec_arm8<uint32_t> b) {
  return vec_arm8<uint32_t>(vzip2q_u32(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint64_t> interleave_hi(const vec_arm8<uint64_t> a,
                                             const vec_arm8<uint64_t> b) {
  return vec_arm8<uint64_t>(vzip2q_u64(a.raw, b.raw));
}

SIMD_INLINE vec_arm8<int8_t> interleave_hi(const vec_arm8<int8_t> a,
                                           const vec_arm8<int8_t> b) {
  return vec_arm8<int8_t>(vzip2q_s8(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int16_t> interleave_hi(const vec_arm8<int16_t> a,
                                            const vec_arm8<int16_t> b) {
  return vec_arm8<int16_t>(vzip2q_s16(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int32_t> interleave_hi(const vec_arm8<int32_t> a,
                                            const vec_arm8<int32_t> b) {
  return vec_arm8<int32_t>(vzip2q_s32(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int64_t> interleave_hi(const vec_arm8<int64_t> a,
                                            const vec_arm8<int64_t> b) {
  return vec_arm8<int64_t>(vzip2q_s64(a.raw, b.raw));
}

SIMD_INLINE vec_arm8<float> interleave_hi(const vec_arm8<float> a,
                                          const vec_arm8<float> b) {
  return vec_arm8<float>(vzip2q_f32(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<double> interleave_hi(const vec_arm8<double> a,
                                           const vec_arm8<double> b) {
  return vec_arm8<double>(vzip2q_s64(a.raw, b.raw));
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

SIMD_INLINE vec_arm8<uint16_t> zip_lo(const vec_arm8<uint8_t> a,
                                      const vec_arm8<uint8_t> b) {
  return vec_arm8<uint16_t>(vzip1q_u8(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint32_t> zip_lo(const vec_arm8<uint16_t> a,
                                      const vec_arm8<uint16_t> b) {
  return vec_arm8<uint32_t>(vzip1q_u16(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint64_t> zip_lo(const vec_arm8<uint32_t> a,
                                      const vec_arm8<uint32_t> b) {
  return vec_arm8<uint64_t>(vzip1q_u32(a.raw, b.raw));
}

SIMD_INLINE vec_arm8<int16_t> zip_lo(const vec_arm8<int8_t> a,
                                     const vec_arm8<int8_t> b) {
  return vec_arm8<int16_t>(vzip1q_s8(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int32_t> zip_lo(const vec_arm8<int16_t> a,
                                     const vec_arm8<int16_t> b) {
  return vec_arm8<int32_t>(vzip1q_s16(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int64_t> zip_lo(const vec_arm8<int32_t> a,
                                     const vec_arm8<int32_t> b) {
  return vec_arm8<int64_t>(vzip1q_s32(a.raw, b.raw));
}

SIMD_INLINE vec_arm8<uint16_t> zip_hi(const vec_arm8<uint8_t> a,
                                      const vec_arm8<uint8_t> b) {
  return vec_arm8<uint16_t>(vzip2q_u8(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint32_t> zip_hi(const vec_arm8<uint16_t> a,
                                      const vec_arm8<uint16_t> b) {
  return vec_arm8<uint32_t>(vzip2q_u16(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<uint64_t> zip_hi(const vec_arm8<uint32_t> a,
                                      const vec_arm8<uint32_t> b) {
  return vec_arm8<uint64_t>(vzip2q_u32(a.raw, b.raw));
}

SIMD_INLINE vec_arm8<int16_t> zip_hi(const vec_arm8<int8_t> a,
                                     const vec_arm8<int8_t> b) {
  return vec_arm8<int16_t>(vzip2q_s8(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int32_t> zip_hi(const vec_arm8<int16_t> a,
                                     const vec_arm8<int16_t> b) {
  return vec_arm8<int32_t>(vzip2q_s16(a.raw, b.raw));
}
SIMD_INLINE vec_arm8<int64_t> zip_hi(const vec_arm8<int32_t> a,
                                     const vec_arm8<int32_t> b) {
  return vec_arm8<int64_t>(vzip2q_s32(a.raw, b.raw));
}

// ------------------------------ Parts

// Returns a part with value "t".
template <typename T>
SIMD_INLINE vec_arm8<T, 1> set_part(Desc<T, 1, ARM8> d, const T t) {
  return set1(d, t);
}

// Gets the single value stored in a vector/part.
template <typename T, size_t N>
SIMD_INLINE T get_part(Desc<T, 1, ARM8> d, const vec_arm8<T, N> v) {
  // TODO(janwas): more efficient implementation?
  SIMD_ALIGN T ret[N];
  store(v, Desc<T, N, ARM8>(), &ret);
  return ret[0];
}

// Returns part of a vector (unspecified whether upper or lower).
SIMD_INLINE vec_arm8<uint8_t, 8> any_part(Desc<uint8_t, 8, ARM8>,
                                          const vec_arm8<uint8_t> v) {
  return vec_arm8<uint8_t, 8>(vget_low_u8(v.raw));
}
SIMD_INLINE vec_arm8<uint16_t, 4> any_part(Desc<uint16_t, 4, ARM8>,
                                           const vec_arm8<uint16_t> v) {
  return vec_arm8<uint16_t, 4>(vget_low_u16(v.raw));
}
SIMD_INLINE vec_arm8<uint32_t, 2> any_part(Desc<uint32_t, 2, ARM8>,
                                           const vec_arm8<uint32_t> v) {
  return vec_arm8<uint32_t, 2>(vget_low_u32(v.raw));
}
SIMD_INLINE vec_arm8<uint64_t, 1> any_part(Desc<uint64_t, 1, ARM8>,
                                           const vec_arm8<uint64_t> v) {
  return vec_arm8<uint64_t, 1>(vget_low_u64(v.raw));
}
SIMD_INLINE vec_arm8<int8_t, 8> any_part(Desc<int8_t, 8, ARM8>,
                                         const vec_arm8<int8_t> v) {
  return vec_arm8<int8_t, 8>(vget_low_s8(v.raw));
}
SIMD_INLINE vec_arm8<int16_t, 4> any_part(Desc<int16_t, 4, ARM8>,
                                          const vec_arm8<int16_t> v) {
  return vec_arm8<int16_t, 4>(vget_low_s16(v.raw));
}
SIMD_INLINE vec_arm8<int32_t, 2> any_part(Desc<int32_t, 2, ARM8>,
                                          const vec_arm8<int32_t> v) {
  return vec_arm8<int32_t, 2>(vget_low_s32(v.raw));
}
SIMD_INLINE vec_arm8<int64_t, 1> any_part(Desc<int64_t, 1, ARM8>,
                                          const vec_arm8<int64_t> v) {
  return vec_arm8<int64_t, 1>(vget_low_s64(v.raw));
}
SIMD_INLINE vec_arm8<float, 2> any_part(Desc<float, 2, ARM8>,
                                        const vec_arm8<float> v) {
  return vec_arm8<float, 2>(vget_low_f32(v.raw));
}
SIMD_INLINE vec_arm8<double, 1> any_part(Desc<double, 1, ARM8>,
                                         const vec_arm8<double> v) {
  return vec_arm8<double, 1>(vget_low_f64(v.raw));
}

SIMD_INLINE vec_arm8<uint8_t, 4> any_part(Desc<uint8_t, 4, ARM8>,
                                          const vec_arm8<uint8_t> v) {
  return vec_arm8<uint8_t, 4>(vget_low_u8(v.raw));
}
SIMD_INLINE vec_arm8<uint16_t, 2> any_part(Desc<uint16_t, 2, ARM8>,
                                           const vec_arm8<uint16_t> v) {
  return vec_arm8<uint16_t, 2>(vget_low_u16(v.raw));
}
SIMD_INLINE vec_arm8<uint32_t, 1> any_part(Desc<uint32_t, 1, ARM8>,
                                           const vec_arm8<uint32_t> v) {
  return vec_arm8<uint32_t, 1>(vget_low_u32(v.raw));
}
SIMD_INLINE vec_arm8<int8_t, 4> any_part(Desc<int8_t, 4, ARM8>,
                                         const vec_arm8<int8_t> v) {
  return vec_arm8<int8_t, 4>(vget_low_s8(v.raw));
}
SIMD_INLINE vec_arm8<int16_t, 2> any_part(Desc<int16_t, 2, ARM8>,
                                          const vec_arm8<int16_t> v) {
  return vec_arm8<int16_t, 2>(vget_low_s16(v.raw));
}
SIMD_INLINE vec_arm8<int32_t, 1> any_part(Desc<int32_t, 1, ARM8>,
                                          const vec_arm8<int32_t> v) {
  return vec_arm8<int32_t, 1>(vget_low_s32(v.raw));
}
SIMD_INLINE vec_arm8<float, 1> any_part(Desc<float, 1, ARM8>,
                                        const vec_arm8<float> v) {
  return vec_arm8<float, 1>(vget_low_f32(v.raw));
}

// Returns full vector with the given part's lane broadcasted. Note that
// callers cannot use broadcast directly because part lane order is undefined.
template <int kLane, typename T, size_t N>
SIMD_INLINE vec_arm8<T> broadcast_part(Full<T, ARM8>, const vec_arm8<T, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return broadcast<kLane>(vec_arm8<T>(v.raw));
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
SIMD_INLINE vec_arm8<T> concat_lo_lo(const vec_arm8<T> hi,
                                     const vec_arm8<T> lo) {
  const Full<uint64_t, ARM8> d64;
  return cast_to(Full<T, ARM8>(),
                 interleave_lo(cast_to(d64, lo), cast_to(d64, hi)));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
SIMD_INLINE vec_arm8<T> concat_hi_hi(const vec_arm8<T> hi,
                                     const vec_arm8<T> lo) {
  const Full<uint64_t, ARM8> d64;
  return cast_to(Full<T, ARM8>(),
                 interleave_hi(cast_to(d64, lo), cast_to(d64, hi)));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves)
template <typename T>
SIMD_INLINE vec_arm8<T> concat_lo_hi(const vec_arm8<T> hi,
                                     const vec_arm8<T> lo) {
  return combine_shift_right_bytes<8>(hi, lo);
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
SIMD_INLINE vec_arm8<T> concat_hi_lo(const vec_arm8<T> hi,
                                     const vec_arm8<T> lo) {
  // TODO(janwas): more efficient implementation?
  SIMD_ALIGN const uint8_t mask[16] = {
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0};
  return select(hi, lo,
                cast_to(Full<T, ARM8>(), load(Full<uint8_t, ARM8>(), mask)));
}

// ------------------------------ Odd/even lanes

template<typename T>
SIMD_INLINE vec_arm8<T> odd_even(
    const vec_arm8<T> a, const vec_arm8<T> b) {
  const Full<uint8_t, ARM8> d8;
  SIMD_ALIGN constexpr uint8_t mask[16] = {
    ((0 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((1 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((2 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((3 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((4 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((5 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((6 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((7 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((8 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((9 / sizeof(T)) & 1)  ? 0 : 0xFF,
    ((10 / sizeof(T)) & 1) ? 0 : 0xFF,
    ((11 / sizeof(T)) & 1) ? 0 : 0xFF,
    ((12 / sizeof(T)) & 1) ? 0 : 0xFF,
    ((13 / sizeof(T)) & 1) ? 0 : 0xFF,
    ((14 / sizeof(T)) & 1) ? 0 : 0xFF,
    ((15 / sizeof(T)) & 1) ? 0 : 0xFF,
  };
  return select(a, b, load(d8, mask));
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ movemask

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..15 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_INLINE uint32_t movemask(const vec_arm8<uint8_t> v) {
  static constexpr uint8x16_t kCollapseMask = {
      1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80, 1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80,
  };
  int8x16_t signed_v = vreinterpretq_s8_u8(v.raw);
  int8x16_t signed_mask = vshrq_n_s8(signed_v, 7);
  uint8x16_t values = vreinterpretq_u8_s8(signed_mask) & kCollapseMask;

  uint8x8_t c0 = vget_low_u8(vpaddq_u8(values, values));
  uint8x8_t c1 = vpadd_u8(c0, c0);
  uint8x8_t c2 = vpadd_u8(c1, c1);

  return vreinterpret_u16_u8(c2)[0];
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_INLINE uint32_t movemask(const vec_arm8<float> v) {
  static constexpr uint32x4_t kCollapseMask = {1, 2, 4, 8};
  int32x4_t signed_v = vreinterpretq_s32_f32(v.raw);
  int32x4_t signed_mask = vshrq_n_s32(signed_v, 31);
  uint32x4_t values = vreinterpretq_u32_s32(signed_mask) & kCollapseMask;
  return vaddvq_u32(values);
}
SIMD_INLINE uint32_t movemask(const vec_arm8<double> v) {
  static constexpr uint64x2_t kCollapseMask = {1, 2};
  int64x2_t signed_v = vreinterpretq_s64_f64(v.raw);
  int64x2_t signed_mask = vshrq_n_s64(signed_v, 63);
  uint64x2_t values = vreinterpretq_u64_s64(signed_mask) & kCollapseMask;
  return (uint32_t)vaddvq_u64(values);
}

// ------------------------------ all_zero

// Returns whether all lanes are equal to zero.
template <typename T>
SIMD_INLINE bool all_zero(const vec_arm8<T> v) {
  const auto v64 = cast_to(Full<uint64_t, ARM8>(), v);
  uint32x2_t a = vqmovn_u64(v64.raw);
  return vreinterpret_u64_u32(a)[0] == 0;
}

// ------------------------------ Horizontal sum (reduction)

// Returns 64-bit sums of 8-byte groups.
SIMD_INLINE vec_arm8<uint64_t> sums_of_u8x8(
    const vec_arm8<uint8_t> v) {
  uint16x8_t a = vpaddlq_u8(v.raw);
  uint32x4_t b = vpaddlq_u16(a);
  return vec_arm8<uint64_t>(vpaddlq_u32(b));
}

// Supported for 32b and 64b vector types. Returns the sum in each lane.
SIMD_INLINE vec_arm8<uint32_t> sum_of_lanes(const vec_arm8<uint32_t> v) {
  return vec_arm8<uint32_t>(vdupq_n_u32(vaddvq_u32(v.raw)));
}
SIMD_INLINE vec_arm8<int32_t> sum_of_lanes(const vec_arm8<int32_t> v) {
  return vec_arm8<int32_t>(vdupq_n_s32(vaddvq_s32(v.raw)));
}
SIMD_INLINE vec_arm8<float> sum_of_lanes(const vec_arm8<float> v) {
  return vec_arm8<float>(vdupq_n_f32(vaddvq_f32(v.raw)));
}
SIMD_INLINE vec_arm8<uint64_t> sum_of_lanes(const vec_arm8<uint64_t> v) {
  return vec_arm8<uint64_t>(vdupq_n_u64(vaddvq_u64(v.raw)));
}
SIMD_INLINE vec_arm8<int64_t> sum_of_lanes(const vec_arm8<int64_t> v) {
  return vec_arm8<int64_t>(vdupq_n_s64(vaddvq_s64(v.raw)));
}
SIMD_INLINE vec_arm8<double> sum_of_lanes(const vec_arm8<double> v) {
  return vec_arm8<double>(vdupq_n_f64(vaddvq_f64(v.raw)));
}

}  // namespace ext

// TODO(user): wrappers for all intrinsics (in neon namespace).
}  // namespace pik

#endif  // SIMD_ENABLE & SIMD_ARM8
#endif  // PIK_SIMD_ARM64_NEON_H_
