// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SIMD_TARGETS_H_
#define PIK_SIMD_TARGETS_H_

// Definitions of the supported targets (= instruction sets).

#include <stddef.h>
#include <utility>  // std::forward
#include "pik/simd/arch.h"
#include "pik/simd/compiler_specific.h"  // SIMD_TARGET_ATTR

namespace pik {

// The SIMD_ENABLE macro expands to a bitfield of one or more targets:
#define SIMD_NONE 0
#define SIMD_AVX2 2
#define SIMD_SSE4 4
#define SIMD_AVX512 16
#define SIMD_PPC8 1  // v2.07 or 3
#define SIMD_ARM8 8

#define SIMD_ENABLE SIMD_NONE

// "Enabling" a target leads to:
// - including a header that defines its vector type and specializes VecT;
// - an additional SIMD_TARGET redefinition and another #include of a
//   target-independent source file from foreach_target, to generate a
//   specialization of Functor::operator()<SIMD_TARGET>;
// - a conditional call to operator()<SIMD_TARGET> in Dispatch().
//
// To entirely disable an instruction set (e.g. if not supported by the
// compiler), comment it out below. Specifying this in source code simplifies
// the build system and avoids needing custom compiler options.
#ifndef SIMD_ENABLE
#if SIMD_ARCH == SIMD_ARCH_X86
#define SIMD_ENABLE (SIMD_SSE4 | SIMD_AVX2)
#elif SIMD_ARCH == SIMD_ARCH_PPC
#define SIMD_ENABLE SIMD_PPC8
#elif SIMD_ARCH == SIMD_ARCH_ARM
#define SIMD_ENABLE SIMD_ARM8
#error "Unsupported platform"
#endif  // #if SIMD_ARCH
#endif  // #ifndef SIMD_ENABLE

// Sets SIMD_TARGET to the 'best' target in SIMD_ENABLE. This is only useful for
// single-target code; for runtime dispatch, use foreach_target.h to generate
// specializations for all enabled targets.
#if SIMD_ENABLE & SIMD_AVX2
#define SIMD_TARGET AVX2
#elif SIMD_ENABLE & SIMD_SSE4
#define SIMD_TARGET SSE4
#elif SIMD_ENABLE & SIMD_PPC8
#define SIMD_TARGET PPC8
#elif SIMD_ENABLE & SIMD_ARM8
#define SIMD_TARGET ARM8
#else
#define SIMD_TARGET NONE
#endif

// SIMD_TARGET serves two purposes: specializing functors and selecting the
// definition of other macros (e.g. SIMD_ATTR). For the former, we use structs
// instead of SIMD_SSE4=4 so that the mangled names are easier to understand.
// The latter requires that struct names match the macro name without the SIMD_
// prefix, e.g. SIMD_TARGET=SSE4.
struct NONE {
  static constexpr int value = SIMD_NONE;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 1;
  }
};
#define SIMD_ATTR_NONE

#if SIMD_ENABLE & SIMD_SSE4
struct SSE4 {
  static constexpr int value = SIMD_SSE4;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 16 / sizeof(T);
  }
};
#define SIMD_ATTR_SSE4 SIMD_TARGET_ATTR("sse4.1")
#endif

#if SIMD_ENABLE & SIMD_AVX2
struct AVX2 {
  static constexpr int value = SIMD_AVX2;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 32 / sizeof(T);
  }
};
#define SIMD_ATTR_AVX2 SIMD_TARGET_ATTR("avx,avx2,fma")
#endif

#if SIMD_ENABLE & SIMD_AVX512
struct AVX512 {
  static constexpr int value = SIMD_AVX512;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 64 / sizeof(T);
  }
};
#endif

#if SIMD_ENABLE & SIMD_PPC8
struct PPC8 {
  static constexpr int value = SIMD_PPC8;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 16 / sizeof(T);
  }
};
#endif

#if SIMD_ENABLE & SIMD_ARM8
struct ARM8 {
  static constexpr int value = SIMD_ARM8;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 16 / sizeof(T);
  }
};
#define SIMD_ATTR_ARM8 SIMD_TARGET_ATTR("armv8-a+crypto")
#endif

// Strongly-typed enum ensures the argument to Dispatch is a single target, not
// a bitfield.
enum class Target {
#if SIMD_ENABLE & SIMD_AVX2
  kAVX2 = SIMD_AVX2,
#endif
#if SIMD_ENABLE & SIMD_SSE4
  kSSE4 = SIMD_SSE4,
#endif
#if SIMD_ENABLE & SIMD_PPC8
  kPPC8 = SIMD_PPC8,
#endif
#if SIMD_ENABLE & SIMD_ARM8
  kARM8 = SIMD_ARM8,
#endif
  kNONE = SIMD_NONE
};

// Returns func.operator()<Target>(args), where Target::value == target. Calling
// a member function template instead of a class template allows stateful
// functors. Dispatch overhead is low but prefer to call this infrequently by
// hoisting this call to higher levels.
template <class Func, typename... Args>
SIMD_INLINE auto Dispatch(const Target target, Func&& func, Args&&... args)
    -> decltype(std::forward<Func>(func).template operator()<NONE>(
        std::forward<Args>(args)...)) {
  switch (target) {
#if SIMD_ENABLE & SIMD_AVX2
    case Target::kAVX2:
      return std::forward<Func>(func).template operator()<AVX2>(
          std::forward<Args>(args)...);
#endif
#if SIMD_ENABLE & SIMD_SSE4
    case Target::kSSE4:
      return std::forward<Func>(func).template operator()<SSE4>(
          std::forward<Args>(args)...);
#endif
#if SIMD_ENABLE & SIMD_PPC8
    case Target::kPPC8:
      return std::forward<Func>(func).template operator()<PPC8>(
          std::forward<Args>(args)...);
#endif
#if SIMD_ENABLE & SIMD_ARM8
    case Target::kARM8:
      return std::forward<Func>(func).template operator()<ARM8>(
          std::forward<Args>(args)...);
#endif

    case Target::kNONE:
      return std::forward<Func>(func).template operator()<NONE>(
          std::forward<Args>(args)...);
  }
}

// All targets supported by the current CPU. Cheap to construct.
class TargetBitfield {
 public:
  TargetBitfield();

  int Bits() const { return bits_; }
  bool Any() const { return bits_ != 0; }

  // Returns 'best' (widest/most recent) target amongst those supported.
  Target Best() const {
#if SIMD_ENABLE & SIMD_AVX2
    if (bits_ & SIMD_AVX2) return Target::kAVX2;
#endif
#if SIMD_ENABLE & SIMD_SSE4
    if (bits_ & SIMD_SSE4) return Target::kSSE4;
#endif
#if SIMD_ENABLE & SIMD_PPC8
    if (bits_ & SIMD_PPC8) return Target::kPPC8;
#endif
#if SIMD_ENABLE & SIMD_ARM8
    if (bits_ & SIMD_ARM8) return Target::kARM8;
#endif
    return Target::kNONE;
  }

  void Clear(Target target) { bits_ &= ~static_cast<int>(target); }

  // Calls func.operator()<Target>(args) for all targets.
  template <class Func, typename... Args>
  SIMD_INLINE void Foreach(Func&& func, Args&&... args) const {
#if SIMD_ENABLE & SIMD_SSE4
    if (bits_ & SIMD_SSE4) {
      std::forward<Func>(func).template operator()<SSE4>(
          std::forward<Args>(args)...);
    }
#endif
#if SIMD_ENABLE & SIMD_AVX2
    if (bits_ & SIMD_AVX2) {
      std::forward<Func>(func).template operator()<AVX2>(
          std::forward<Args>(args)...);
    }
#endif
#if SIMD_ENABLE & SIMD_PPC8
    if (bits_ & SIMD_PPC8) {
      std::forward<Func>(func).template operator()<PPC8>(
          std::forward<Args>(args)...);
    }
#endif
#if SIMD_ENABLE & SIMD_ARM8
    if (bits_ & SIMD_ARM8) {
      std::forward<Func>(func).template operator()<ARM8>(
          std::forward<Args>(args)...);
    }
#endif

    std::forward<Func>(func).template operator()<NONE>(
        std::forward<Args>(args)...);
  }

 private:
  int bits_;
};

}  // namespace pik

#endif  // PIK_SIMD_TARGETS_H_
