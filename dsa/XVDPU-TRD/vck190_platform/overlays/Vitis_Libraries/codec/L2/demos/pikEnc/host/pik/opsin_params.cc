// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/opsin_params.h"

#include "pik/linalg.h"

namespace pik {

static const float* ComputeOpsinAbsorbanceInverseMatrix() {
  float* inverse = new float[9];
  for (int i = 0; i < 9; i++) {
    inverse[i] = kOpsinAbsorbanceMatrix[i];
  }
  Inv3x3Matrix(inverse);
  return inverse;
}

const float* GetOpsinAbsorbanceInverseMatrix() {
  static const float* kOpsinInverse = ComputeOpsinAbsorbanceInverseMatrix();
  return kOpsinInverse;
}

}  // namespace pik
