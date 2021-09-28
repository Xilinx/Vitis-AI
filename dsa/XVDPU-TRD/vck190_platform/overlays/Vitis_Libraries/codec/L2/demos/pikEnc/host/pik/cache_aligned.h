// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_CACHE_ALIGNED_H_
#define PIK_CACHE_ALIGNED_H_

// Memory allocator with support for alignment + misalignment.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>  // memcpy
#include <atomic>
#include <memory>

#include "pik/arch_specific.h"
#include "pik/compiler_specific.h"
#include "pik/simd/simd.h"
#include "pik/status.h"

namespace pik {

// Functions that depend on the cache line size.
class CacheAligned {
 public:
  static void PrintStats();

  static constexpr size_t kPointerSize = sizeof(void*);
  static constexpr size_t kCacheLineSize = 64;
  // To avoid RFOs, match L2 fill size (pairs of lines).
  static constexpr size_t kAlignment = 2 * kCacheLineSize;
  // Minimum multiple for which cache set conflicts and/or loads blocked by
  // preceding stores can occur.
  static constexpr size_t kAlias = 2048;

  // Returns a 'random' (cyclical) offset suitable for Allocate.
  static size_t NextOffset();

  // Returns null or memory whose address is congruent to `offset` (mod kAlias).
  // This reduces cache conflicts and load/store stalls, especially with large
  // allocations that would otherwise have similar alignments. At least
  // `payload_size` (which can be zero) bytes will be accessible.
  static void* Allocate(const size_t payload_size, size_t offset);

  static void* Allocate(const size_t payload_size) {
    return Allocate(payload_size, NextOffset());
  }

  static void Free(const void* aligned_pointer);

  // Overwrites `to` without loading it into cache (read-for-ownership).
  // Copies kCacheLineSize bytes from/to naturally aligned addresses.
  template <typename T>
  static SIMD_ATTR void StreamCacheLine(const T* PIK_RESTRICT from,
                                        T* PIK_RESTRICT to) {
    static_assert(16 % sizeof(T) == 0, "T must fit in a lane");
#if SIMD_TARGET_VALUE != SIMD_NONE
    constexpr size_t kLanes = 16 / sizeof(T);
    const SIMD_PART(T, kLanes) d;
    PIK_COMPILER_FENCE;
    const auto v0 = load(d, from + 0 * kLanes);
    const auto v1 = load(d, from + 1 * kLanes);
    const auto v2 = load(d, from + 2 * kLanes);
    const auto v3 = load(d, from + 3 * kLanes);
    static_assert(sizeof(v0) * 4 == kCacheLineSize, "Wrong #vectors");
    // Fences prevent the compiler from reordering loads/stores, which may
    // interfere with write-combining.
    PIK_COMPILER_FENCE;
    stream(v0, d, to + 0 * kLanes);
    stream(v1, d, to + 1 * kLanes);
    stream(v2, d, to + 2 * kLanes);
    stream(v3, d, to + 3 * kLanes);
    PIK_COMPILER_FENCE;
#else
    memcpy(to, from, kCacheLineSize);
#endif
  }
};

// Avoids the need for a function pointer (deleter) in CacheAlignedUniquePtr.
struct CacheAlignedDeleter {
  void operator()(uint8_t* aligned_pointer) const {
    return CacheAligned::Free(aligned_pointer);
  }
};

using CacheAlignedUniquePtr = std::unique_ptr<uint8_t[], CacheAlignedDeleter>;

// Does not invoke constructors.
static inline CacheAlignedUniquePtr AllocateArray(const size_t bytes) {
  return CacheAlignedUniquePtr(
      static_cast<uint8_t*>(CacheAligned::Allocate(bytes)),
      CacheAlignedDeleter());
}

static inline CacheAlignedUniquePtr AllocateArray(const size_t bytes,
                                                  const size_t offset) {
  return CacheAlignedUniquePtr(
      static_cast<uint8_t*>(CacheAligned::Allocate(bytes, offset)),
      CacheAlignedDeleter());
}

}  // namespace pik

#endif  // PIK_CACHE_ALIGNED_H_
