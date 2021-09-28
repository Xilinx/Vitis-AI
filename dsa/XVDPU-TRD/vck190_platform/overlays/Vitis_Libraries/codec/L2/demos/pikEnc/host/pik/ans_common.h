// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_ANS_COMMON_H_
#define PIK_ANS_COMMON_H_

#include <vector>

#include "pik/compiler_specific.h"

namespace pik {

// Returns the precision (number of bits) that should be used to store
// a histogram count such that Log2Floor(count) == logcount.
PIK_INLINE int GetPopulationCountPrecision(int logcount) {
  return (logcount + 1) >> 1;
}

// Returns a histogram where the counts are positive, differ by at most 1,
// and add up to total_count. The bigger counts (if any) are at the beginning
// of the histogram.
std::vector<int> CreateFlatHistogram(int length, int total_count);

}  // namespace pik

#endif  // PIK_ANS_COMMON_H_
