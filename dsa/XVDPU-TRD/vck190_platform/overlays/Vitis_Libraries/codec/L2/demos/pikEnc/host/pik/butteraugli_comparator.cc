// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/butteraugli_comparator.h"

#include "pik/opsin_inverse.h"

namespace pik {

namespace {

Image3F LinearFromOpsin(const Image3F& opsin) {
  Image3F linear(opsin.xsize(), opsin.ysize());
  OpsinToLinear(opsin, Rect(opsin), &linear);
  return linear;
}

}  // namespace

ButteraugliComparator::ButteraugliComparator(const Image3F& opsin,
                                             float hf_asymmetry,
                                             float multiplier)
    : xsize_(opsin.xsize()),
      ysize_(opsin.ysize()),
      comparator_(ScaleImage(multiplier, LinearFromOpsin(opsin)), hf_asymmetry),
      distance_(0.0),
      multiplier_(multiplier),
      distmap_(xsize_, ysize_) {
  ZeroFillImage(&distmap_);
}

void ButteraugliComparator::Compare(const Image3F& linear_rgb) {
  PIK_CHECK(SameSize(distmap_, linear_rgb));
  if (multiplier_ == 1) {
    comparator_.Diffmap(linear_rgb, distmap_);
  } else {
    comparator_.Diffmap(ScaleImage(multiplier_, linear_rgb), distmap_);
  }
  distance_ = butteraugli::ButteraugliScoreFromDiffmap(distmap_);
}

void ButteraugliComparator::Mask(Image3F* mask, Image3F* mask_dc) {
  comparator_.Mask(mask, mask_dc);
}

}  // namespace pik
