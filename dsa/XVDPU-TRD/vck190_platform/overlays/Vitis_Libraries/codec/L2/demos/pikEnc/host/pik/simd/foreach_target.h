// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// Includes a specified file for every enabled SIMD_TARGET. This is used to
// generate template instantiations to be called via runtime dispatche.

#ifndef SIMD_ATTR_IMPL
#error "Must set SIMD_ATTR_IMPL to name of include file"
#endif

#if SIMD_ENABLE & SIMD_AVX2
#undef SIMD_TARGET
#define SIMD_TARGET AVX2
#include SIMD_ATTR_IMPL
#endif

#if SIMD_ENABLE & SIMD_SSE4
#undef SIMD_TARGET
#define SIMD_TARGET SSE4
#include SIMD_ATTR_IMPL
#endif

#if SIMD_ENABLE & SIMD_PPC8
#undef SIMD_TARGET
#define SIMD_TARGET PPC8
#include SIMD_ATTR_IMPL
#endif

#if SIMD_ENABLE & SIMD_ARM8
#undef SIMD_TARGET
#define SIMD_TARGET ARM8
#include SIMD_ATTR_IMPL
#endif

#undef SIMD_TARGET
#define SIMD_TARGET NONE
#include SIMD_ATTR_IMPL

#undef SIMD_ATTR_IMPL
