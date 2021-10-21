// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/optimize.h"

#include "gtest/gtest.h"

namespace pik {
namespace optimize {
namespace {

// F(w) = (w - w_min)^2.
struct SimpleQuadraticFunction {
  explicit SimpleQuadraticFunction(const std::vector<double>& w0) : w_min(w0) {}

  double Compute(const std::vector<double>& w, std::vector<double>* df) const {
    std::vector<double> dw = w - w_min;
    *df = -2.0 * dw;
    return dw * dw;
  }

  std::vector<double> w_min;
};

// F(alpha, beta, gamma| x,y) = \sum_i(y_i - (alpha x_i ^ gamma + beta))^2.
struct PowerFunction {
  explicit PowerFunction(const std::vector<double>& x0,
                         const std::vector<double>& y0)
      : x(x0), y(y0) {}

  double Compute(const std::vector<double>& w, std::vector<double>* df) const {
    double loss_function = 0;
    (*df)[0] = 0;
    (*df)[1] = 0;
    (*df)[2] = 0;
    for (int ind = 0; ind < y.size(); ++ind) {
      if (x[ind] != 0) {
        double l_f = y[ind] - (w[0] * pow(x[ind], w[1]) + w[2]);
        (*df)[0] += 2.0 * l_f * pow(x[ind], w[1]);
        (*df)[1] += 2.0 * l_f * w[0] * pow(x[ind], w[1]) * log(x[ind]);
        (*df)[2] += 2.0 * l_f * 1;
        loss_function += l_f * l_f;
      }
    }
    return loss_function;
  }

  std::vector<double> x;
  std::vector<double> y;
};

TEST(OptimizeTest, SimpleQuadraticFunction) {
  std::vector<double> w_min(2);
  w_min[0] = 1.0;
  w_min[1] = 2.0;
  SimpleQuadraticFunction f(w_min);
  std::vector<double> w(2);
  static const double kPrecision = 1e-8;
  w = optimize::OptimizeWithScaledConjugateGradientMethod(f, w, kPrecision, 0);
  EXPECT_NEAR(w[0], 1.0, kPrecision);
  EXPECT_NEAR(w[1], 2.0, kPrecision);
}

TEST(OptimizeTest, PowerFunction) {
  std::vector<double> x(10);
  std::vector<double> y(10);
  for (int ind = 0; ind < 10; ++ind) {
    x[ind] = 1. * ind;
    y[ind] = 2. * pow(x[ind], 3) + 5.;
  }
  PowerFunction f(x, y);
  std::vector<double> w(3);

  static const double kPrecision = 0.01;
  w = optimize::OptimizeWithScaledConjugateGradientMethod(f, w, kPrecision, 0);
  EXPECT_NEAR(w[0], 2.0, kPrecision);
  EXPECT_NEAR(w[1], 3.0, kPrecision);
  EXPECT_NEAR(w[2], 5.0, kPrecision);
}

}  // namespace
}  // namespace optimize
}  // namespace pik
