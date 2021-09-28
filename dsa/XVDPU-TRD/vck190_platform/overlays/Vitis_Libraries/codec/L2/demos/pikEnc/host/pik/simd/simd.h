// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SIMD_SIMD_H_
#define PIK_SIMD_SIMD_H_

// Performance-portable SIMD API for SSE4/AVX2/ARMv8, later AVX-512 and POWER8.
// Each operation is efficient on all platforms.

#include <stddef.h>  // size_t
#include <stdint.h>
#include "pik/simd/arch.h"
#include "pik/simd/util.h"  // CopyBytes

#include "pik/simd/arm64_neon.h"
#include "pik/simd/scalar.h"
#include "pik/simd/x86_avx2.h"
#include "pik/simd/x86_sse4.h"

#if SIMD_ARCH == SIMD_ARCH_X86 && SIMD_ENABLE == SIMD_NONE
// No targets enabled, but we still need this for functions below.
#include <emmintrin.h>
#endif

// Use SIMD_TARGET to derive other macros. NOTE: SIMD_TARGET is only evaluated
// when these macros are expanded.
#define SIMD_CONCAT_IMPL(a, b) a##b
#define SIMD_CONCAT(a, b) SIMD_CONCAT_IMPL(a, b)

// Attributes; must precede every function declaration.
#define SIMD_ATTR SIMD_CONCAT(SIMD_ATTR_, SIMD_TARGET)

// Target-specific namespace, required when using foreach_target.h.
#define SIMD_NAMESPACE SIMD_CONCAT(N_, SIMD_TARGET)

// Which target is active, e.g. #if SIMD_TARGET_VALUE == SIMD_AVX2
#define SIMD_TARGET_VALUE SIMD_CONCAT(SIMD_, SIMD_TARGET)

// Functions common to multiple targets:
namespace pik {

// One Newton-Raphson iteration.
template <class V>
static SIMD_ATTR SIMD_INLINE V ReciprocalNR(const V x) {
  const auto rcp = approximate_reciprocal(x);
  const auto sum = rcp + rcp;
  const auto x_rcp = x * rcp;
  return nmul_add(x_rcp, rcp, sum);
}

// Primary template: default to actual division.
template <typename T, class V>
struct FastDivision {
  SIMD_ATTR V operator()(const V n, const V d) const { return n / d; }
};
// Partial specialization for float vectors.
template <class V>
struct FastDivision<float, V> {
  SIMD_ATTR V operator()(const V n, const V d) const {
    return n * ReciprocalNR(d);
  }
};

// Returns a name for the vector/part/scalar. The type prefix is u/i/f for
// unsigned/signed/floating point, followed by the number of bits per lane;
// then 'x' followed by the number of lanes. Example: u8x16. This is useful for
// understanding which instantiation of a generic test failed.
template <class D>
inline const char* vec_name() {
  using T = typename D::T;
  constexpr size_t N = D::N;
  constexpr int kTarget = D::Target::value;

  // Avoids depending on <type_traits>.
  const bool is_float = T(2.25) != T(2);
  const bool is_signed = T(-1) < T(0);
  constexpr char prefix = is_float ? 'f' : (is_signed ? 'i' : 'u');

  constexpr size_t bits = sizeof(T) * 8;
  constexpr char bits10 = '0' + (bits / 10);
  constexpr char bits1 = '0' + (bits % 10);

  // Scalars: omit the xN suffix.
  if (kTarget == SIMD_NONE) {
    static constexpr char name1[8] = {prefix, bits1};
    static constexpr char name2[8] = {prefix, bits10, bits1};
    return sizeof(T) == 1 ? name1 : name2;
  }

  constexpr char N1 = (N < 10) ? '\0' : '0' + (N % 10);
  constexpr char N10 = (N < 10) ? '0' + (N % 10) : '0' + (N / 10);

  static constexpr char name1[8] = {prefix, bits1, 'x', N10, N1};
  static constexpr char name2[8] = {prefix, bits10, bits1, 'x', N10, N1};
  return sizeof(T) == 1 ? name1 : name2;
}

// Cache control

SIMD_INLINE void stream(const uint32_t t, uint32_t* SIMD_RESTRICT aligned) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_stream_si32(reinterpret_cast<int*>(aligned), t);
#else
  CopyBytes<4>(&t, aligned);
#endif
}

SIMD_INLINE void stream(const uint64_t t, uint64_t* SIMD_RESTRICT aligned) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_stream_si64(reinterpret_cast<long long*>(aligned), t);
#else
  CopyBytes<8>(&t, aligned);
#endif
}

// Delays subsequent loads until prior loads are visible. On Intel CPUs, also
// serves as a full fence (waits for all prior instructions to complete).
// No effect on non-x86.
SIMD_INLINE void load_fence() {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_lfence();
#endif
}

// Ensures previous weakly-ordered stores are visible. No effect on non-x86.
SIMD_INLINE void store_fence() {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_sfence();
#endif
}

// Begins loading the cache line containing "p".
template <typename T>
SIMD_INLINE void prefetch(const T* p) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_prefetch(p, _MM_HINT_T0);
#elif SIMD_ARCH == SIMD_ARCH_ARM
  __pld(p);
#endif
}

// Invalidates and flushes the cache line containing "p". No effect on non-x86.
SIMD_INLINE void flush_cacheline(const void* p) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_clflush(p);
#endif
}

// Call during spin loops to potentially reduce contention/power consumption.
SIMD_INLINE void pause() {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_pause();
#endif
}

}  // namespace pik

#endif  // PIK_SIMD_SIMD_H_
