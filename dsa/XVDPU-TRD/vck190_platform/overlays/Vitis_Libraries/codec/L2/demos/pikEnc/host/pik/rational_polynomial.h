// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_RATIONAL_POLYNOMIAL_H_
#define PIK_RATIONAL_POLYNOMIAL_H_

// Fast SIMD evaluation of rational polynomials for approximating functions.

#include "pik/compiler_specific.h"
#include "pik/simd/simd.h"

namespace pik {

// Approximates smooth functions via rational polynomials (i.e. dividing two
// polynomials). Supports V = SIMD or Scalar<T> inputs.

// Evaluates the polynomial using Horner's method, which is faster than
// Clenshaw recurrence for Chebyshev polynomials.
//
// "kDeg" is the degree of the numerator and denominator polynomials;
// kDegP == kDegQ + 1 = 3 or 4 is usually a good choice.
template <class D, int kDegP, int kDegQ>
class RationalPolynomial {
  using T = typename D::T;
  using V = typename D::V;
  static_assert(kDegP <= 7, "Unroll more iterations");
  static_assert(kDegQ <= 7, "Unroll more iterations");

 public:
  template <typename U>
  SIMD_ATTR void SetCoefficients(const U (&p)[kDegP + 1],
                                 const U (&q)[kDegQ + 1]) {
    for (int i = 0; i <= kDegP; ++i) {
      p_[i] = set1(D(), static_cast<T>(p[i]));
    }
    for (int i = 0; i <= kDegQ; ++i) {
      q_[i] = set1(D(), static_cast<T>(q[i]));
    }
  }

  SIMD_ATTR void GetCoefficients(T (*p)[kDegP + 1], T (*q)[kDegQ + 1]) const {
    const SIMD_PART(T, 1) d;
    for (int i = 0; i <= kDegP; ++i) {
      store(any_part(d, p_[i]), d, (*p) + i);
    }
    for (int i = 0; i <= kDegQ; ++i) {
      store(any_part(d, q_[i]), d, (*q) + i);
    }
  }

  template <typename U>
  SIMD_ATTR RationalPolynomial(const U (&p)[kDegP + 1],
                               const U (&q)[kDegQ + 1]) {
    SetCoefficients(p, q);
  }

  // Evaluates the polynomial at x.
  SIMD_ATTR PIK_INLINE V operator()(const V x) const {
    V yp = p_[kDegP];
    V yq = q_[kDegQ];
    PIK_COMPILER_FENCE;
    if (kDegP >= 1) yp = mul_add(yp, x, p_[kDegP - 1]);
    if (kDegQ >= 1) yq = mul_add(yq, x, q_[kDegQ - 1]);
    PIK_COMPILER_FENCE;
    if (kDegP >= 2) yp = mul_add(yp, x, p_[kDegP - 2]);
    if (kDegQ >= 2) yq = mul_add(yq, x, q_[kDegQ - 2]);
    PIK_COMPILER_FENCE;
    if (kDegP >= 3) yp = mul_add(yp, x, p_[kDegP - 3]);
    if (kDegQ >= 3) yq = mul_add(yq, x, q_[kDegQ - 3]);
    PIK_COMPILER_FENCE;
    if (kDegP >= 4) yp = mul_add(yp, x, p_[kDegP - 4]);
    if (kDegQ >= 4) yq = mul_add(yq, x, q_[kDegQ - 4]);
    PIK_COMPILER_FENCE;
    if (kDegP >= 5) yp = mul_add(yp, x, p_[kDegP - 5]);
    if (kDegQ >= 5) yq = mul_add(yq, x, q_[kDegQ - 5]);
    PIK_COMPILER_FENCE;
    if (kDegP >= 6) yp = mul_add(yp, x, p_[kDegP - 6]);
    if (kDegQ >= 6) yq = mul_add(yq, x, q_[kDegQ - 6]);
    PIK_COMPILER_FENCE;
    if (kDegP >= 7) yp = mul_add(yp, x, p_[kDegP - 7]);
    if (kDegQ >= 7) yq = mul_add(yq, x, q_[kDegQ - 7]);

    // Division is faster for a single evaluation but the Triple below are
    // much faster with NR, and we use the same approach to here so that we
    // compute the same max error as reached below.
    return FastDivision<T, V>()(yp, yq);
  }

 private:
  // Horner coefficients in ascending order.
  V p_[kDegP + 1];
  V q_[kDegQ + 1];
};

// Evaluates a rational polynomial via Horner's scheme. Equivalent to
// RationalPolynomial poly(p, q); return poly(x). This can be more efficient
// because the coefficients are loaded directly from memory, whereas set1
// can result in copying them from RIP+x to stack frame. load_dup128 allows us
// to specify constants (replicated 4x) independently of the lane count.
template <int NP, int NQ, class V, typename T>
SIMD_ATTR PIK_INLINE V EvalRationalPolynomial(const V x, const T (&p)[NP],
                                              const T (&q)[NQ]) {
  const SIMD_FULL(T) d;
  constexpr int kDegP = NP / 4 - 1;
  constexpr int kDegQ = NQ / 4 - 1;
  auto yp = load_dup128(d, &p[kDegP * 4]);
  auto yq = load_dup128(d, &q[kDegQ * 4]);
  PIK_COMPILER_FENCE;
  if (kDegP >= 1) yp = mul_add(yp, x, load_dup128(d, &p[(kDegP - 1) * 4]));
  if (kDegQ >= 1) yq = mul_add(yq, x, load_dup128(d, &q[(kDegQ - 1) * 4]));
  PIK_COMPILER_FENCE;
  if (kDegP >= 2) yp = mul_add(yp, x, load_dup128(d, &p[(kDegP - 2) * 4]));
  if (kDegQ >= 2) yq = mul_add(yq, x, load_dup128(d, &q[(kDegQ - 2) * 4]));
  PIK_COMPILER_FENCE;
  if (kDegP >= 3) yp = mul_add(yp, x, load_dup128(d, &p[(kDegP - 3) * 4]));
  if (kDegQ >= 3) yq = mul_add(yq, x, load_dup128(d, &q[(kDegQ - 3) * 4]));
  PIK_COMPILER_FENCE;
  if (kDegP >= 4) yp = mul_add(yp, x, load_dup128(d, &p[(kDegP - 4) * 4]));
  if (kDegQ >= 4) yq = mul_add(yq, x, load_dup128(d, &q[(kDegQ - 4) * 4]));
  PIK_COMPILER_FENCE;
  if (kDegP >= 5) yp = mul_add(yp, x, load_dup128(d, &p[(kDegP - 5) * 4]));
  if (kDegQ >= 5) yq = mul_add(yq, x, load_dup128(d, &q[(kDegQ - 5) * 4]));
  PIK_COMPILER_FENCE;
  if (kDegP >= 6) yp = mul_add(yp, x, load_dup128(d, &p[(kDegP - 6) * 4]));
  if (kDegQ >= 6) yq = mul_add(yq, x, load_dup128(d, &q[(kDegQ - 6) * 4]));
  PIK_COMPILER_FENCE;
  if (kDegP >= 7) yp = mul_add(yp, x, load_dup128(d, &p[(kDegP - 7) * 4]));
  if (kDegQ >= 7) yq = mul_add(yq, x, load_dup128(d, &q[(kDegQ - 7) * 4]));

  return FastDivision<T, V>()(yp, yq);
}

// Evaluates three at once for better FMA utilization and fewer loads.
template <int NP, int NQ, class V, typename T>
SIMD_ATTR void EvalRationalPolynomialTriple(const V x0, const V x1, const V x2,
                                            const T (&p)[NP], const T (&q)[NQ],
                                            V* PIK_RESTRICT y0,
                                            V* PIK_RESTRICT y1,
                                            V* PIK_RESTRICT y2) {
  // Computing both polynomials in parallel is slightly faster than sequential
  // (better utilization of FMA slots despite higher register pressure).
  const SIMD_FULL(T) d;
  constexpr int kDegP = NP / 4 - 1;
  constexpr int kDegQ = NQ / 4 - 1;
  V yp0 = load_dup128(d, &p[kDegP * 4]);
  V yq0 = load_dup128(d, &q[kDegQ * 4]);
  V yp1 = yp0;
  V yq1 = yq0;
  V yp2 = yp0;
  V yq2 = yq0;
  V c;
  if (kDegP >= 1) {
    c = load_dup128(d, &p[(kDegP - 1) * 4]);
    yp0 = mul_add(yp0, x0, c);
    yp1 = mul_add(yp1, x1, c);
    yp2 = mul_add(yp2, x2, c);
  }
  if (kDegQ >= 1) {
    c = load_dup128(d, &q[(kDegQ - 1) * 4]);
    yq0 = mul_add(yq0, x0, c);
    yq1 = mul_add(yq1, x1, c);
    yq2 = mul_add(yq2, x2, c);
  }
  if (kDegP >= 2) {
    c = load_dup128(d, &p[(kDegP - 2) * 4]);
    yp0 = mul_add(yp0, x0, c);
    yp1 = mul_add(yp1, x1, c);
    yp2 = mul_add(yp2, x2, c);
  }
  if (kDegQ >= 2) {
    c = load_dup128(d, &q[(kDegQ - 2) * 4]);
    yq0 = mul_add(yq0, x0, c);
    yq1 = mul_add(yq1, x1, c);
    yq2 = mul_add(yq2, x2, c);
  }
  if (kDegP >= 3) {
    c = load_dup128(d, &p[(kDegP - 3) * 4]);
    yp0 = mul_add(yp0, x0, c);
    yp1 = mul_add(yp1, x1, c);
    yp2 = mul_add(yp2, x2, c);
  }
  if (kDegQ >= 3) {
    c = load_dup128(d, &q[(kDegQ - 3) * 4]);
    yq0 = mul_add(yq0, x0, c);
    yq1 = mul_add(yq1, x1, c);
    yq2 = mul_add(yq2, x2, c);
  }
  if (kDegP >= 4) {
    c = load_dup128(d, &p[(kDegP - 4) * 4]);
    yp0 = mul_add(yp0, x0, c);
    yp1 = mul_add(yp1, x1, c);
    yp2 = mul_add(yp2, x2, c);
  }
  if (kDegQ >= 4) {
    c = load_dup128(d, &q[(kDegQ - 4) * 4]);
    yq0 = mul_add(yq0, x0, c);
    yq1 = mul_add(yq1, x1, c);
    yq2 = mul_add(yq2, x2, c);
  }
  if (kDegP >= 5) {
    c = load_dup128(d, &p[(kDegP - 5) * 4]);
    yp0 = mul_add(yp0, x0, c);
    yp1 = mul_add(yp1, x1, c);
    yp2 = mul_add(yp2, x2, c);
  }
  if (kDegQ >= 5) {
    c = load_dup128(d, &q[(kDegQ - 5) * 4]);
    yq0 = mul_add(yq0, x0, c);
    yq1 = mul_add(yq1, x1, c);
    yq2 = mul_add(yq2, x2, c);
  }
  if (kDegP >= 6) {
    c = load_dup128(d, &p[(kDegP - 6) * 4]);
    yp0 = mul_add(yp0, x0, c);
    yp1 = mul_add(yp1, x1, c);
    yp2 = mul_add(yp2, x2, c);
  }
  if (kDegQ >= 6) {
    c = load_dup128(d, &q[(kDegQ - 6) * 4]);
    yq0 = mul_add(yq0, x0, c);
    yq1 = mul_add(yq1, x1, c);
    yq2 = mul_add(yq2, x2, c);
  }
  if (kDegP >= 7) {
    c = load_dup128(d, &p[(kDegP - 7) * 4]);
    yp0 = mul_add(yp0, x0, c);
    yp1 = mul_add(yp1, x1, c);
    yp2 = mul_add(yp2, x2, c);
  }
  if (kDegQ >= 7) {
    c = load_dup128(d, &q[(kDegQ - 7) * 4]);
    yq0 = mul_add(yq0, x0, c);
    yq1 = mul_add(yq1, x1, c);
    yq2 = mul_add(yq2, x2, c);
  }

  // Much faster than division when computing three at once.
  *y0 = FastDivision<T, V>()(yp0, yq0);
  *y1 = FastDivision<T, V>()(yp1, yq1);
  *y2 = FastDivision<T, V>()(yp2, yq2);
}

}  // namespace pik

#endif  // PIK_RATIONAL_POLYNOMIAL_H_
