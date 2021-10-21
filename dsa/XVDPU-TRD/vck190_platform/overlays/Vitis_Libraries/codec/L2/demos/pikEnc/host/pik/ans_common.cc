// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/ans_common.h"

#include "pik/status.h"

namespace pik {

std::vector<int> CreateFlatHistogram(int length, int total_count) {
  PIK_ASSERT(length > 0);
  PIK_ASSERT(length <= total_count);
  const int count = total_count / length;
  std::vector<int> result(length, count);
  const int rem_counts = total_count % length;
  for (int i = 0; i < rem_counts; ++i) {
    ++result[i];
  }
  return result;
}

}  // namespace pik
