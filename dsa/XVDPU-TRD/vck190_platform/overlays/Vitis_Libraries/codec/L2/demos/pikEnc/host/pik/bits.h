// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_BITS_H_
#define PIK_BITS_H_

// Specialized instructions for processing register-sized bit arrays.

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <stdint.h>
#include "pik/compiler_specific.h"

static PIK_INLINE int PopCount(const uint32_t x) {
#ifdef _MSC_VER
  return _mm_popcnt_u32(x);
#else
  return __builtin_popcount(x);
#endif
}

// Undefined results for x == 0.
static PIK_INLINE int NumZeroBitsAboveMSBNonzero(const uint32_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanReverse(&index, x);
  return index;
#else
  return __builtin_clz(x);
#endif
}
static PIK_INLINE int NumZeroBitsAboveMSBNonzero(const uint64_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanReverse64(&index, x);
  return index;
#else
  return __builtin_clzl(x);
#endif
}
static PIK_INLINE int NumZeroBitsBelowLSBNonzero(const uint32_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward(&index, x);
  return index;
#else
  return __builtin_ctz(x);
#endif
}
static PIK_INLINE int NumZeroBitsBelowLSBNonzero(const uint64_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward64(&index, x);
  return index;
#else
  return __builtin_ctzl(x);
#endif
}

// Returns bit width for x == 0.
static PIK_INLINE int NumZeroBitsAboveMSB(const uint32_t x) {
  return (x == 0) ? 32 : NumZeroBitsAboveMSBNonzero(x);
}
static PIK_INLINE int NumZeroBitsAboveMSB(const uint64_t x) {
  return (x == 0) ? 64 : NumZeroBitsAboveMSBNonzero(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB(const uint32_t x) {
  return (x == 0) ? 32 : NumZeroBitsBelowLSBNonzero(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB(const uint64_t x) {
  return (x == 0) ? 64 : NumZeroBitsBelowLSBNonzero(x);
}

// Returns base-2 logarithm, rounded down.
static PIK_INLINE int FloorLog2Nonzero(const uint32_t x) {
  return 31 ^ NumZeroBitsAboveMSBNonzero(x);
}
static PIK_INLINE int FloorLog2Nonzero(const uint64_t x) {
  return 63 ^ NumZeroBitsAboveMSBNonzero(x);
}

// Returns base-2 logarithm, rounded up.
static PIK_INLINE int CeilLog2Nonzero(const uint32_t x) {
  const int floor_log2 = FloorLog2Nonzero(x);
  if ((x & (x - 1)) == 0) return floor_log2;  // power of two
  return floor_log2 + 1;
}

static PIK_INLINE int CeilLog2Nonzero(const uint64_t x) {
  const int floor_log2 = FloorLog2Nonzero(x);
  if ((x & (x - 1)) == 0) return floor_log2;  // power of two
  return floor_log2 + 1;
}

#endif  // PIK_BITS_H_
