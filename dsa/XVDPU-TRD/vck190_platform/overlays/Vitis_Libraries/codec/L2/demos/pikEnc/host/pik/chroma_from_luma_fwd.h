// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_CHROMA_FROM_LUMA_FWD_H_
#define PIK_CHROMA_FROM_LUMA_FWD_H_

// Chroma-from-luma statistics (separate header avoids circular dependencies)

#include "pik/descriptive_statistics.h"
#include "pik/image.h"

namespace pik {

struct CFL_Stats {
  CFL_Stats() {
    for (size_t k = 0; k < 64; ++k) {
      sum_abs_residual_x[k] = sum_abs_residual_b[k] = sum_abs_restored_x[k] =
          sum_abs_restored_b[k] = 0.0;
      smaller_x[k] = smaller_b[k] = 0;
      total[k] = 0;
    }
  }

  void Update(const Image3F& residuals, const Image3F& restored,
              const Rect& rect, const size_t bx_mul) {
    PIK_ASSERT(&residuals != &restored);

    for (size_t rect_by = 0; rect_by < rect.ysize(); ++rect_by) {
      const size_t by = rect.y0() + rect_by;
      const float* row_restored_x = restored.ConstPlaneRow(0, by);
      const float* row_restored_b = restored.ConstPlaneRow(2, by);
      const float* row_residual_x = residuals.ConstPlaneRow(0, by);
      const float* row_residual_b = residuals.ConstPlaneRow(2, by);

      for (size_t rect_bx = 0; rect_bx < rect.xsize(); ++rect_bx) {
        const size_t bx = rect.x0() + rect_bx;
        for (size_t k = 0; k < bx_mul; ++k) {
          const size_t x = bx * bx_mul + k;
          total[k] += 1;

          const float abs_residual_x = std::abs(row_residual_x[x]);
          const float abs_restored_x = std::abs(row_restored_x[x]);
          sum_abs_residual_x[k] += abs_residual_x;
          sum_abs_restored_x[k] += abs_restored_x;
          smaller_x[k] += abs_residual_x <= abs_restored_x;
          if (abs_restored_x > 1E-6f) {
            const float ratio = abs_residual_x / abs_restored_x;
            if (ratio > 1E6) {
              printf("ratio %E restored %.3f res %.5f at bx %zu(%zu) k %zu\n",
                     ratio, abs_restored_x, abs_residual_x, bx, bx_mul, k);
            }
            ratio_x.Notify(ratio);
          }

          const float abs_residual_b = std::abs(row_residual_b[x]);
          const float abs_restored_b = std::abs(row_restored_b[x]);
          sum_abs_residual_b[k] += abs_residual_b;
          sum_abs_restored_b[k] += abs_restored_b;
          smaller_b[k] += abs_residual_b <= abs_restored_b;
          if (abs_restored_b > 1E-6f) {
            const float ratio = abs_residual_b / abs_restored_b;
            ratio_b.Notify(ratio);
          }
        }
      }
    }
  }

  void Assimilate(const CFL_Stats& other) {
    rx.Assimilate(other.rx);
    rb.Assimilate(other.rb);

    ratio_x.Assimilate(other.ratio_x);
    ratio_b.Assimilate(other.ratio_b);

    for (size_t k = 0; k < 64; ++k) {
      sum_abs_restored_x[k] += other.sum_abs_restored_x[k];
      sum_abs_restored_b[k] += other.sum_abs_restored_b[k];
      sum_abs_residual_x[k] += other.sum_abs_residual_x[k];
      sum_abs_residual_b[k] += other.sum_abs_residual_b[k];
      smaller_x[k] += other.smaller_x[k];
      smaller_b[k] += other.smaller_b[k];
      total[k] += other.total[k];
    }
  }

  void Print() const {
    for (size_t k = 0; k < 64; ++k) {
      if (sum_abs_restored_x[k] == 0.0 && sum_abs_restored_b[k] == 0) {
        continue;
      }
      printf(
          " %2zu: residual %.3E %.3E restored %.3E %.3E  smaller %6.2f %6.2f\n",
          k, sum_abs_residual_x[k], sum_abs_residual_b[k],
          sum_abs_restored_x[k], sum_abs_restored_b[k],
          100.0 * smaller_x[k] / total[k], 100.0 * smaller_b[k] / total[k]);
    }
    printf("%s\n%s\n", ratio_x.ToString().c_str(), ratio_b.ToString().c_str());

    const int flags = Stats::kNoSkewKurt + Stats::kNoGeomean;
    printf("Corr %s %s\n", rx.ToString(flags).c_str(),
           rb.ToString(flags).c_str());
  }

  // Correlation coefficients
  Stats rx;
  Stats rb;

  // residual/restored
  Stats ratio_x;
  Stats ratio_b;

  double sum_abs_restored_x[64];
  double sum_abs_restored_b[64];
  double sum_abs_residual_x[64];
  double sum_abs_residual_b[64];
  size_t smaller_x[64];
  size_t smaller_b[64];
  size_t total[64];
};

}  // namespace pik

#endif  // PIK_CHROMA_FROM_LUMA_FWD_H_
