// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SIMD_ARCH_H_
#define PIK_SIMD_ARCH_H_

// Sets SIMD_ARCH to one of the following based on predefined macros:

#define SIMD_ARCH_X86 8
#define SIMD_ARCH_PPC 9
#define SIMD_ARCH_ARM 0xA

#if defined(__x86_64__) || defined(_M_X64)
#define SIMD_ARCH SIMD_ARCH_X86

#elif defined(__powerpc64__) || defined(_M_PPC)
#define SIMD_ARCH SIMD_ARCH_PPC

#elif defined(__aarch64__)
#define SIMD_ARCH SIMD_ARCH_ARM

#else
#error "Unsupported platform"
#endif

#endif  // PIK_SIMD_ARCH_H_
