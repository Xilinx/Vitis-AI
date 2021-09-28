// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/pik_info.h"

namespace pik {

void PikInfo::DumpCoeffImage(const char* label,
                             const Image3S& coeff_image) const {
  PIK_ASSERT(coeff_image.xsize() % 64 == 0);
  Image3S reshuffled(coeff_image.xsize() / 8, coeff_image.ysize() * 8);
  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < coeff_image.ysize(); y++) {
      for (int x = 0; x < coeff_image.xsize(); x += 64) {
        for (int i = 0; i < 64; i++) {
          reshuffled.PlaneRow(c, 8 * y + i / 8)[x / 8 + i % 8] =
              coeff_image.PlaneRow(c, y)[x + i];
        }
      }
    }
  }
  DumpImage(label, reshuffled);
}

}  // namespace pik
