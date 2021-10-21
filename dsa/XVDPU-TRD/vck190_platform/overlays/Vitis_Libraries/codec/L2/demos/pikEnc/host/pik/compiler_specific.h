// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COMPILER_SPECIFIC_H_
#define PIK_COMPILER_SPECIFIC_H_

// Macros for compiler version + nonstandard keywords, e.g. __builtin_expect.

#include <stdint.h>

// #if is shorter and safer than #ifdef. *_VERSION are zero if not detected,
// otherwise 100 * major + minor version. Note that other packages check for
// #ifdef COMPILER_MSVC, so we cannot use that same name.

#ifdef _MSC_VER
#define PIK_COMPILER_MSVC _MSC_VER
#else
#define PIK_COMPILER_MSVC 0
#endif

#ifdef __GNUC__
#define PIK_COMPILER_GCC (__GNUC__ * 100 + __GNUC_MINOR__)
#else
#define PIK_COMPILER_GCC 0
#endif

#ifdef __clang__
// For reasons unknown, Forge currently explicitly defines these to 0.0.
#define PIK_COMPILER_CLANG 1  // (__clang_major__ * 100 + __clang_minor__)
// Clang pretends to be GCC for compatibility.
#undef PIK_COMPILER_GCC
#define PIK_COMPILER_GCC 0
#else
#define PIK_COMPILER_CLANG 0
#endif

#if PIK_COMPILER_MSVC
#define PIK_RESTRICT __restrict
#elif PIK_COMPILER_GCC || PIK_COMPILER_CLANG
#define PIK_RESTRICT __restrict__
#else
#define PIK_RESTRICT
#endif

#if PIK_COMPILER_MSVC
#define PIK_INLINE __forceinline
#define PIK_NOINLINE __declspec(noinline)
#else
#define PIK_INLINE inline __attribute__((always_inline))
#define PIK_NOINLINE __attribute__((noinline))
#endif

#if PIK_COMPILER_MSVC
#define PIK_NORETURN __declspec(noreturn)
#elif PIK_COMPILER_GCC || PIK_COMPILER_CLANG
#define PIK_NORETURN __attribute__((noreturn))
#endif

#if PIK_COMPILER_MSVC
#define PIK_UNREACHABLE __assume(false)
#elif PIK_COMPILER_CLANG || PIK_COMPILER_GCC >= 405
#define PIK_UNREACHABLE __builtin_unreachable()
#else
#define PIK_UNREACHABLE
#endif

#if PIK_COMPILER_MSVC
// Unsupported, __assume is not the same.
#define PIK_LIKELY(expr) expr
#define PIK_UNLIKELY(expr) expr
#else
#define PIK_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define PIK_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#endif

#if PIK_COMPILER_MSVC
#include <intrin.h>

#pragma intrinsic(_ReadWriteBarrier)
#define PIK_COMPILER_FENCE _ReadWriteBarrier()
#elif PIK_COMPILER_GCC || PIK_COMPILER_CLANG
#define PIK_COMPILER_FENCE asm volatile("" : : : "memory")
#else
#define PIK_COMPILER_FENCE
#endif

// Returns a void* pointer which the compiler then assumes is N-byte aligned.
// Example: float* PIK_RESTRICT aligned = (float*)PIK_ASSUME_ALIGNED(in, 32);
//
// The assignment semantics are required by GCC/Clang. ICC provides an in-place
// __assume_aligned, whereas MSVC's __assume appears unsuitable.
#if PIK_COMPILER_CLANG
// Early versions of Clang did not support __builtin_assume_aligned.
#define PIK_HAS_ASSUME_ALIGNED __has_builtin(__builtin_assume_aligned)
#elif PIK_COMPILER_GCC
#define PIK_HAS_ASSUME_ALIGNED 1
#else
#define PIK_HAS_ASSUME_ALIGNED 0
#endif

#if PIK_HAS_ASSUME_ALIGNED
#define PIK_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
#else
#define PIK_ASSUME_ALIGNED(ptr, align) (ptr) /* not supported */
#endif

#ifdef __has_attribute
#define PIK_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define PIK_HAVE_ATTRIBUTE(x) 0
#endif

// Raises warnings if the function return value is unused. Should appear as the
// first part of a function definition/declaration.
#if PIK_HAVE_ATTRIBUTE(nodiscard)
#define PIK_MUST_USE_RESULT [[nodiscard]]
#elif PIK_COMPILER_CLANG && PIK_HAVE_ATTRIBUTE(warn_unused_result)
#define PIK_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define PIK_MUST_USE_RESULT
#endif

#if PIK_HAVE_ATTRIBUTE(__format__)
#define PIK_FORMAT(idx_fmt, idx_arg) \
  __attribute__((__format__(__printf__, idx_fmt, idx_arg)))
#else
#define PIK_FORMAT(idx_fmt, idx_arg)
#endif

#endif  // PIK_COMPILER_SPECIFIC_H_
