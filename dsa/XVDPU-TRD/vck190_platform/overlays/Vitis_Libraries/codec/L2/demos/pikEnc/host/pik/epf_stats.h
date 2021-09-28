// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_EPF_STATS_H_
#define PIK_EPF_STATS_H_

// Statistics/debug info for epf.h.

#include <stdio.h>
#include <stdlib.h>

#include "pik/descriptive_statistics.h"

namespace pik {

// Per-thread.
struct EpfStats {
  void Assimilate(const EpfStats& other) {
    total += other.total;
    skipped += other.skipped;
    less += other.less;
    greater += other.greater;

    for (int c = 0; c < 3; ++c) {
      s_ranges[c].Assimilate(other.s_ranges[c]);
    }
    s_quant.Assimilate(other.s_quant);
    s_sigma.Assimilate(other.s_sigma);
  }

  void Print() const {
    const int stats = Stats::kNoSkewKurt + Stats::kNoGeomean;
    printf(
        "EPF total blocks: %zu; skipped: %zu (%f%%); outside %zu|%zu (%f%%)\n"
        "ranges: %s\n        %s\n        %s\nquant: %s\nsigma: %s\n",
        total, skipped, 100.0 * skipped / total, less, greater,
        100.0 * (less + greater) / total, s_ranges[0].ToString(stats).c_str(),
        s_ranges[1].ToString(stats).c_str(),
        s_ranges[2].ToString(stats).c_str(), s_quant.ToString(stats).c_str(),
        s_sigma.ToString(stats).c_str());
  }

  // # blocks
  size_t total = 0;
  size_t skipped = 0;  // sigma == 0 => no filter
  // Outside LUT range:
  size_t less = 0;
  size_t greater = 0;

  Stats s_ranges[3];
  Stats s_quant;
  Stats s_sigma;
};

}  // namespace pik

#endif  // PIK_EPF_STATS_H_
