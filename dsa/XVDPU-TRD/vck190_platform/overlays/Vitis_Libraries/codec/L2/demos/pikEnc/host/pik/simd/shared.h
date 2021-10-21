// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SIMD_SHARED_H_
#define PIK_SIMD_SHARED_H_

// Definitions needed by multiple platform-specific headers.

#include <stddef.h>
#include <atomic>

// Ensures an array is aligned and suitable for load()/store() functions.
// Example: SIMD_ALIGN T lanes[d.N];
#define SIMD_ALIGN alignas(32)

// 4 instances of a given literal value, useful as input to load_dup128.
#define SIMD_REP4(literal) literal, literal, literal, literal
#define SIMD_REP8(literal) SIMD_REP4(literal), SIMD_REP4(literal)

// Alternative for asm volatile("" : : : "memory"), which has no effect.
#define SIMD_FENCE std::atomic_thread_fence(std::memory_order_acq_rel)

namespace pik {

// SIMD operations are implemented as overloaded functions selected using a
// "descriptor" D := Desc<T, N[, Target]>. For example: `D::V setzero(D)`.
// T is the lane type, N the number of lanes, Target is an instruction set
// (e.g. SSE4). The return type D::V is either a full vector of at least
// 128 bits, an N-lane (=2^j) part, or a scalar.

// Specialized in platform-specific headers. Only for use by PartTarget.
// Default: no change to Target. kBlocks = ceil(size / 16).
template <size_t kBlocks, class Target>
struct PartTargetT {
  using type = Target;
};

template <typename T, size_t N, class Target>
using PartTarget =
    typename PartTargetT<(N * sizeof(T) + 15) / 16, Target>::type;

// Specialized in platform-specific headers. Only for use by Desc and VT.
template <typename T, size_t N, class Target>
struct VecT {};

// Shorthand for function arg/return types. Overrides Target with the narrowest
// possible for the given N.
template <typename T, size_t N, class Target>
using VT = typename VecT<T, N, PartTarget<T, N, Target>>::type;

// Descriptor: properties that uniquely identify a vector/part/scalar. Used to
// select overloaded functions; see Full/Part/Scalar aliases below.
template <typename LaneT, size_t kLanes, class TargetT>
struct Desc {
  constexpr Desc() {}

  using T = LaneT;
  static constexpr size_t N = kLanes;
  using Target = TargetT;

  // Alias for the actual vector data, e.g. scalar<float> for <float, 1, NONE>,
  // returned by initializers such as setzero(). Parts and full vectors are
  // distinct types on x86 to avoid inadvertent conversions. By contrast, PPC
  // parts are merely aliases for full vectors to avoid wrapper overhead.
  using V = typename VecT<T, N, Target>::type;

  static_assert((N & (N - 1)) == 0, "N must be a power of two");
  static_assert(N <= Target::template NumLanes<T>(), "N too large");
};

// Shorthand for a full vector.
template <typename T, class Target>
using Full = Desc<T, Target::template NumLanes<T>(), Target>;

// Shorthand for a part (or full) vector. N=2^j. Note that PartTarget selects
// a 128-bit Target when T and N are small enough (avoids additional AVX2
// versions of SSE4 initializers/loads).
template <typename T, size_t N, class Target>
using Part = Desc<T, N, PartTarget<T, N, Target>>;

// Shorthand for Part/Full. NOTE: uses SIMD_TARGET at the moment of expansion,
// not its current (possibly undefined) value.
#define SIMD_FULL(T) Full<T, SIMD_TARGET>
#define SIMD_PART(T, N) Part<T, N, SIMD_TARGET>

// Type tags for get_half(Upper(), v) etc.
struct Upper {};
struct Lower {};
#define SIMD_HALF Lower()

// Unfortunately the GCC/Clang intrinsics do not accept int64_t*.
using GatherIndex64 = long long int;
static_assert(sizeof(GatherIndex64) == 8, "Must be 64-bit type");

}  // namespace pik

#endif  // PIK_SIMD_SHARED_H_
