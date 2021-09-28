// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SIMD_COMPILER_SPECIFIC_H_
#define PIK_SIMD_COMPILER_SPECIFIC_H_

// Compiler-specific includes and definitions.

// SIMD_COMPILER expands to one of the following:
#define SIMD_COMPILER_CLANG 1
#define SIMD_COMPILER_GCC 2
#define SIMD_COMPILER_MSVC 3

#ifdef _MSC_VER
#define SIMD_COMPILER SIMD_COMPILER_MSVC
#elif defined(__clang__)
#define SIMD_COMPILER SIMD_COMPILER_CLANG
#elif defined(__GNUC__)
#define SIMD_COMPILER SIMD_COMPILER_GCC
#else
#error "Unsupported compiler"
#endif

#if SIMD_COMPILER == SIMD_COMPILER_MSVC
#include <intrin.h>

#define SIMD_RESTRICT __restrict
#define SIMD_INLINE __forceinline
#define SIMD_NOINLINE __declspec(noinline)
#define SIMD_LIKELY(expr) expr
#define SIMD_TRAP __debugbreak
#define SIMD_TARGET_ATTR(feature_str)
#define SIMD_DIAGNOSTICS(tokens) __pragma(warning(tokens))
#define SIMD_DIAGNOSTICS_OFF(msc, gcc) SIMD_DIAGNOSTICS(msc)

#else

#define SIMD_RESTRICT __restrict__
#define SIMD_INLINE \
  inline __attribute__((always_inline)) __attribute__((flatten))
#define SIMD_NOINLINE inline __attribute__((noinline))
#define SIMD_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define SIMD_TRAP __builtin_trap
#define SIMD_TARGET_ATTR(feature_str) __attribute__((target(feature_str)))
#define SIMD_PRAGMA(tokens) _Pragma(#tokens)
#define SIMD_DIAGNOSTICS(tokens) SIMD_PRAGMA(GCC diagnostic tokens)
#define SIMD_DIAGNOSTICS_OFF(msc, gcc) SIMD_DIAGNOSTICS(gcc)

#endif

#endif  // PIK_SIMD_COMPILER_SPECIFIC_H_
