// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_IMAGE_TEST_UTILS_H_
#define PIK_IMAGE_TEST_UTILS_H_

#include <math.h>
#include <stddef.h>

#include "gtest/gtest.h"
#include "pik/compiler_specific.h"
#include "pik/image.h"

namespace pik {

template <typename T>
void VerifyEqual(const Image<T>& expected, const Image<T>& actual) {
  PIK_CHECK(SameSize(expected, actual));
  for (size_t y = 0; y < expected.ysize(); ++y) {
    const T* const PIK_RESTRICT row_expected = expected.Row(y);
    const T* const PIK_RESTRICT row_actual = actual.Row(y);
    for (size_t x = 0; x < expected.xsize(); ++x) {
      ASSERT_EQ(row_expected[x], row_actual[x]) << x << " " << y;
    }
  }
}

template <typename T>
void VerifyEqual(const Image3<T>& expected, const Image3<T>& actual) {
  for (int c = 0; c < 3; ++c) {
    VerifyEqual(expected.Plane(c), actual.Plane(c));
  }
}

}  // namespace pik
#endif  // PIK_IMAGE_TEST_UTILS_H_
