// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/deconvolve.h"
#include <vector>
#include "pik/optimize.h"
#include "pik/status.h"

namespace pik {

namespace {

void Convolve(const float* inp1, int n1, const float* inp2, int n2,
              float* out) {
  for (int i = 0; i < n1 + n2 - 1; i++) {
    out[i] = 0.0;
  }
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      out[i + j] += inp1[i] * inp2[j];
    }
  }
}

void ConvolveReversed(const float* inp1, int n1, const float* reverse_inp2,
                      int n2, float* out) {
  for (int i = 0; i < n1 + n2 - 1; i++) {
    out[i] = 0.0;
  }
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      out[i + j] += inp1[i] * reverse_inp2[n2 - j - 1];
    }
  }
}

struct LossFunction {
  float Compute(const std::vector<float>& w, std::vector<float>* df) const {
    // Size of w is guaranteed to be odd.
    std::vector<float> result(filter_length + w.size() - 1, 0.0);
    Convolve(filter, filter_length, &w[0], w.size(), &result[0]);
    result[result.size() / 2] -= 1.0;
    float sumsq = 0.0;
    for (int i = 0; i < result.size(); i++) {
      sumsq += result[i] * result[i];
    }
    // TODO(robryk): This is not operator norm, nor an upper bound: it's a
    // lower bound on operator norm and an upper bound for operator norm * C(n),
    // where n is sum of filter sizes. We should actually optimize for operator
    // norm here.
    std::vector<float> derivs(result.size() + filter_length - 1, 0.0);
    ConvolveReversed(&result[0], result.size(), filter, filter_length,
                     &derivs[0]);
    for (int i = 0; i < w.size(); i++) {
      (*df)[i] = -2 * derivs[i + filter_length - 1];
    }
    {
      // Outside of midpoint, regularize the sharpening kernel towards zero.
      static const double kRegularizationWeight = 0.00001;
      for (int i = 0; i < w.size(); ++i) {
        if (i == w.size() / 2) {
          continue;
        }
        sumsq += kRegularizationWeight * w[i] * w[i];
        (*df)[i] += -2 * kRegularizationWeight * w[i];
      }
    }
    return sumsq;
  }
  const float* filter;
  int filter_length;  // Guaranteed to be odd.
};

}  // namespace

float InvertConvolution(const float* filter, int filter_length,
                        float* inverse_filter, int inverse_filter_length) {
  PIK_CHECK(filter_length % 2 == 1);
  PIK_CHECK(inverse_filter_length % 2 == 1);
  LossFunction loss;
  loss.filter = filter;
  loss.filter_length = filter_length;
  constexpr int kMaxIter = 1000;
  constexpr float kGradNormThreshold = 1e-8;
  std::vector<float> inverse =
      optimize::OptimizeWithScaledConjugateGradientMethod(
          loss, std::vector<float>(inverse_filter_length, 0.0),
          kGradNormThreshold, kMaxIter);
  for (int i = 0; i < inverse_filter_length; i++) {
    inverse_filter[i] = inverse[i];
  }
  std::vector<float> dummy(inverse_filter_length);
  return loss.Compute(inverse, &dummy);
}

}  // namespace pik
