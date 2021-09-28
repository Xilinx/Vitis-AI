// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/epf.h"

// Edge-preserving smoothing: 7x8 weighted average based on L1 patch similarity.

#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <numeric>  // std::accumulate

#ifndef EPF_DUMP_SIGMA
#define EPF_DUMP_SIGMA 0
#endif
#ifndef EPF_ENABLE_STATS
#define EPF_ENABLE_STATS 0
#endif

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/ac_strategy.h"
#include "pik/common.h"
#include "pik/descriptive_statistics.h"
#include "pik/fields.h"
#include "pik/profiler.h"
#include "pik/simd/simd.h"
#include "pik/status.h"
#if EPF_DUMP_SIGMA
#include "pik/image_io.h"
#endif

#if 1
#define EPF_ASSERT(condition)                           \
  while (!(condition)) {                                \
    printf("EPF assert failed at line %d\n", __LINE__); \
    exit(1);                                            \
  }

#else
#define EPF_ASSERT(condition)
#endif

namespace pik {

EpfParams::EpfParams() { Bundle::Init(this); }

}  // namespace pik

// Must include "normally" so the build system understands the dependency.
#include "pik/epf_target.cc"

#define SIMD_ATTR_IMPL "pik/epf_target.cc"
#include "pik/simd/foreach_target.h"
