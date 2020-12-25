/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/tests/exhaustive_op_test_utils.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {

using Eigen::half;

template <typename T, size_t N>
T EvaluatePolynomial(T x, const std::array<T, N>& coeffs) {
  T result = 0;
  for (T c : coeffs) {
    result = result * x + c;
  }
  return result;
}

// There's no std::erfinv, so we have to implement it ourselves.  This follows
// Wichura 1998, https://www.jstor.org/stable/2347330 which, notably, is a
// different implementation from that in math.cc.
float HostErfInv(float x) {
  std::array<double, 8> kPolyA = {
      8.8709406962545514830200e2, 1.1819493347062294404278e4,
      2.3782041382114385731252e4, 1.6235862515167575384252e4,
      4.8548868893843886794648e3, 6.9706266534389598238465e2,
      4.7072688112383978012285e1, 1.1975323115670912564578e0,
  };
  std::array<double, 8> kPolyB = {
      5.2264952788528545610e3, 2.8729085735721942674e4, 3.9307895800092710610e4,
      2.1213794301586595867e4, 5.3941960214247511077e3, 6.8718700749205790830e2,
      4.2313330701600911252e1, 1.0000000000000000000e0,
  };
  std::array<double, 8> kPolyC = {
      7.74545014278341407640e-4, 2.27238449892691845833e-2,
      2.41780725177450611770e-1, 1.27045825245236838258e0,
      3.64784832476320460504e0,  5.76949722146069140550e0,
      4.63033784615654529590e0,  1.42343711074968357734e0,
  };
  std::array<double, 8> kPolyD = {
      1.4859850019840355905497876e-9, 7.7441459065157709165577218e-4,
      2.1494160384252876777097297e-2, 2.0945065210512749128288442e-1,
      9.7547832001787427186894837e-1, 2.3707661626024532365971225e0,
      2.9036514445419946173133295e0,  1.4142135623730950488016887e0,
  };
  std::array<double, 8> kPolyE = {
      2.01033439929228813265e-7, 2.71155556874348757815e-5,
      1.24266094738807843860e-3, 2.65321895265761230930e-2,
      2.96560571828504891230e-1, 1.78482653991729133580e0,
      5.46378491116411436990e0,  6.65790464350110377720e0,
  };
  std::array<double, 8> kPolyF = {
      2.891024605872965461538222e-15, 2.010321207683943062279931e-7,
      2.611088405080593625138020e-5,  1.112800997078859844711555e-3,
      2.103693768272068968719679e-2,  1.936480946950659106176712e-1,
      8.482908416595164588112026e-1,  1.414213562373095048801689e0,
  };

  if (std::abs(x) > 1 || std::isnan(x)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (std::abs(x) == 1) {
    return std::copysign(std::numeric_limits<float>::infinity(), x);
  }

  float unsigned_result = [&] {
    float y = std::abs(x);
    if (y <= 0.85) {
      double r = 0.180625 - 0.25 * y * y;
      return (y * EvaluatePolynomial(r, kPolyA)) /
             EvaluatePolynomial(r, kPolyB);
    } else {
      double r = std::sqrt(std::log(2.0) - std::log1p(-y));
      if (r <= 5.0) {
        r -= 1.6;
        return EvaluatePolynomial(r, kPolyC) / EvaluatePolynomial(r, kPolyD);
      } else {
        r -= 5;
        return EvaluatePolynomial(r, kPolyE) / EvaluatePolynomial(r, kPolyF);
      }
    }
  }();
  return std::copysign(unsigned_result, x);
}

// Digamma implementation using a polynomial from Cephes.  Notably this is a
// different implementation from the one in math.cc.
float HostDigamma(float x) {
  // Euler-Mascheroni constant
  float kGamma = 0.57721566490153286061;
  float kPi = M_PI;

  std::array<float, 4> kPoly = {
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  float reflection = 0;
  if (x <= 0) {
    float floor = std::floor(x);
    if (x == floor) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    // Compute reflection term, pi * cot(pi * x).
    reflection = x - floor;
    if (reflection == 0.5) {
      reflection = 0;
    } else {
      if (reflection > 0.5) {
        reflection = x - (floor + 1.0f);
      }
      reflection = kPi / std::tan(kPi * reflection);
    }
    x = 1 - x;
  }

  float result = 0;
  if (x <= 10 && x == std::floor(x)) {
    // Special case for integers <= 10.
    for (int i = 1; i < x; ++i) {
      result += 1.0f / i;
    }
    result -= kGamma;
  } else {
    float w = 0;
    for (; x < 10; ++x) {
      w += 1.0f / x;
    }
    if (x < 1e8) {
      float z = 1.0f / (x * x);
      result = z * EvaluatePolynomial(z, kPoly);
    }
    result = std::log(x) - 0.5f / x - result - w;
  }

  // Compute the final, reflected value.
  return result - reflection;
}

template <PrimitiveType T>
using ExhaustiveUnaryTest = ExhaustiveOpTestBase<T, 1>;

// Exhaustive test for unary operations for <= 32bit floating point types.
//
// Test parameter is a tuple containing
//   - primitive type under test,
//   - (begin, end) range under test, as zero-extended int64s bitcast to the
//     primtive type under test.
template <PrimitiveType T>
class Exhaustive32BitOrLessUnaryTest
    : public ExhaustiveUnaryTest<T>,
      public ::testing::WithParamInterface<std::pair<int64, int64>> {
 public:
  // Sets error parameters appropriately for testing sin/cos/tan.
  void SetParamsForSinCosTan();

 protected:
  using typename ExhaustiveUnaryTest<T>::NativeT;

 private:
  int64 GetInputSize() override {
    int64 begin, end;
    std::tie(begin, end) = GetParam();
    VLOG(2) << "Checking range [" << begin << ", " << end << ")";
    return end - begin;
  }

  // Generates all the input values for the test. The the range of the bit
  // representation of the input values is described by the test parameter as
  // a pair of int64 representing the starting bit pattern and the ending
  // pattern. Each bit representation is first truncated to the integral type of
  // the same bit as the type being tested, if needed, and then bitcasted to the
  // type being tested.
  void FillInput(std::array<Literal, 1>* input_literal) override {
    using IntegralT =
        typename ExhaustiveOpTestBase<T, 1>::ComponentIntegralNativeT;
    int64 input_size = (*input_literal)[0].element_count();
    int64 begin, end;
    std::tie(begin, end) = GetParam();
    VLOG(2) << "Checking range [" << begin << ", " << end << ")";
    CHECK_EQ(input_size, end - begin);

    absl::Span<NativeT> input_arr = (*input_literal)[0].data<NativeT>();
    for (int64 i = 0; i < input_size; i++) {
      IntegralT input_val = i + begin;
      input_arr[i] =
          this->ConvertAndReplaceKnownIncorrectValueWith(input_val, 0);
    }
  }
};

typedef Exhaustive32BitOrLessUnaryTest<F32> ExhaustiveF32UnaryTest;
typedef Exhaustive32BitOrLessUnaryTest<F16> ExhaustiveF16UnaryTest;
typedef Exhaustive32BitOrLessUnaryTest<BF16> ExhaustiveBF16UnaryTest;

#define XLA_TEST_FLOAT_32_BITS_OR_LESS(test_name, ...) \
  XLA_TEST_P(ExhaustiveF32UnaryTest, test_name)        \
  __VA_ARGS__                                          \
  XLA_TEST_P(ExhaustiveF16UnaryTest, test_name)        \
  __VA_ARGS__                                          \
  XLA_TEST_P(ExhaustiveBF16UnaryTest, test_name)       \
  __VA_ARGS__

XLA_TEST_FLOAT_32_BITS_OR_LESS(Log, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    error_spec_gen = +[](NativeT x) { return ErrorSpec{0.001, 0.001}; };
  }
  Run(Log, std::log, error_spec_gen);
})

XLA_TEST_FLOAT_32_BITS_OR_LESS(Log1p, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    error_spec_gen = +[](NativeT x) { return ErrorSpec{0.001, 0.001}; };
  }
  Run(Log1p, std::log1p, error_spec_gen);
})

XLA_TEST_FLOAT_32_BITS_OR_LESS(Exp, {
  // When x < -105, the true value of exp(x) is smaller than the smallest F32,
  // so exp(x) should return exactly 0. We want our implementation of exp to
  // return exactly 0 as well, as not doing so implies either that our
  // implementation of exp is not following the asymptotic behavior that exp(x)
  // approaches 0 as x approaches -inf, or that our implementation is not
  // approaching 0 fast enough.
  ErrorSpecGen error_spec_gen = +[](NativeT x) {
    if (x < static_cast<NativeT>(-105)) {
      return ErrorSpec{0, 0};
    }
    return GetDefaultSpecGenerator()(x);
  };

  // Our CPU implementation of exp returns one incorrect value: says
  // exp(88.7228394) = max-float, but the correct answer is inf.  We deem this
  // acceptable and check for it explicitly so that we can be aware if anything
  // changes.
  if (platform_ == "Host") {
    auto host_exp_with_overflow = +[](float f) {
      if (f == 88.7228394f) {
        return 3.40282347e+38f;
      }
      return std::exp(f);
    };
    Run(Exp, host_exp_with_overflow, error_spec_gen);
  } else {
    Run(Exp, std::exp, error_spec_gen);
  }
})

XLA_TEST_FLOAT_32_BITS_OR_LESS(Expm1, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (ty_ == F32) {
    error_spec_gen = +[](NativeT x) { return ErrorSpec{0, 0.00015}; };
  }

  // Our CPU implementation of expm1 returns one incorrect value: says
  // exp(88.7228394) = max-float, but the correct answer is inf.  We deem this
  // acceptable and check for it explicitly so that we can be aware if anything
  // changes.
  if (platform_ == "Host") {
    auto host_expm1_with_overflow = +[](float f) {
      if (f == 88.7228394f) {
        return 3.40282347e+38f;
      }
      return std::expm1(f);
    };
    Run(Expm1, host_expm1_with_overflow, error_spec_gen);
  } else {
    Run(Expm1, std::expm1, error_spec_gen);
  }
})

// It feels a little overkill to exhaustively test sqrt and pow(x, 0.5), but
// this *did* find a bug, namely that some backends were assuming sqrt(x) ==
// pow(x, 0.5), but this is not true for x == -inf.
XLA_TEST_FLOAT_32_BITS_OR_LESS(PowOneHalf, {
  EvaluateOp fn = +[](float x) { return std::pow(x, 0.5f); };
  // TODO(b/123837116): Enable the test for all values after fixing the bug.
  if (platform_ != "Host" && platform_ != "CUDA") {
    fn = +[](float x) {
      if (x == -std::numeric_limits<float>::infinity()) {
        return std::nanf("");
      }
      return std::pow(x, 0.5f);
    };
  }
  Run([](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); }, fn);
})

XLA_TEST_FLOAT_32_BITS_OR_LESS(Rsqrt, {
  Run(
      Rsqrt, +[](float x) { return 1 / std::sqrt(x); });
})

XLA_TEST_FLOAT_32_BITS_OR_LESS(Sqrt, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "Host" || platform_ == "CUDA") {
    error_spec_gen = +[](NativeT x) {
      auto spec = GetDefaultSpecGenerator()(x);
      spec.strict_signed_zeros = true;
      return spec;
    };
  }

  Run(Sqrt, std::sqrt, error_spec_gen);
})

// TODO(jlebar): Test trig functions over complex inputs.
XLA_TEST_P(ExhaustiveF32UnaryTest, Acosh) {
  // Error inherited from Log, which our implementation of Acosh uses.
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA") {
    error_spec_gen = +[](float x) { return ErrorSpec{0.001, 0.001}; };
  }

  Run(Acosh, std::acosh, error_spec_gen);
}
XLA_TEST_P(ExhaustiveF16UnaryTest, Acosh) { Run(Acosh, std::acosh); }
XLA_TEST_P(ExhaustiveBF16UnaryTest, Acosh) { Run(Acosh, std::acosh); }

// Tests for Asinh
XLA_TEST_P(ExhaustiveF32UnaryTest, Asinh) {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA") {
    error_spec_gen = +[](float x) { return ErrorSpec{0.001, 0.001}; };
  }

  Run(Asinh, std::asinh, error_spec_gen);
}
XLA_TEST_P(ExhaustiveF16UnaryTest, Asinh) { Run(Asinh, std::asinh); }
XLA_TEST_P(ExhaustiveBF16UnaryTest, Asinh) { Run(Asinh, std::asinh); }

XLA_TEST_FLOAT_32_BITS_OR_LESS(Atanh, { Run(Atanh, std::atanh); })
XLA_TEST_FLOAT_32_BITS_OR_LESS(Acos, { Run(Acos, std::acos); })
XLA_TEST_FLOAT_32_BITS_OR_LESS(Asin, { Run(Asin, std::asin); })

XLA_TEST_FLOAT_32_BITS_OR_LESS(Cosh, {
  // Our cosh implementation incorrectly overflows to inf for +/-89.4159851.
  // The correct answer of 3.40281961e+38 (0x7f7fffec) is very close to
  // max-float, so we deem this acceptable.
  //
  // This does not occur on CPU because we have an offsetting error in our
  // implementation of exp.
  float (*host_cosh)(float);
  if (platform_ == "Host") {
    host_cosh = &std::cosh;
  } else {
    host_cosh = +[](float x) {
      if (std::abs(x) == 89.4159851f) {
        return std::numeric_limits<float>::infinity();
      }
      return std::cosh(x);
    };
  }
  Run(Cosh, host_cosh);
})

XLA_TEST_FLOAT_32_BITS_OR_LESS(Sinh, {
  // Our sinh implementation incorrectly overflows to +/-inf for +/-89.4159851.
  // The correct answer of 3.40281961e+38 (0x7f7fffec) is very close to
  // max-float, so we deem this acceptable.
  //
  // This does not occur on CPU because we have an offsetting error in our
  // implementation of exp.
  float (*host_sinh)(float);
  if (platform_ == "Host") {
    host_sinh = &std::sinh;
  } else {
    host_sinh = +[](float x) {
      if (std::abs(x) == 89.4159851f) {
        return std::copysign(std::numeric_limits<float>::infinity(), x);
      }
      return std::sinh(x);
    };
  }
  Run(Sinh, host_sinh);
})

XLA_TEST_FLOAT_32_BITS_OR_LESS(Tanh, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "CUDA") {
    error_spec_gen = +[](NativeT x) {
      return x <= static_cast<NativeT>(-20.0) || x >= static_cast<NativeT>(20.0)
                 ? ErrorSpec{0, 0}
                 : GetDefaultSpecGenerator()(x);
    };
  }
  Run(Tanh, std::tanh, error_spec_gen);
})

template <PrimitiveType T>
void Exhaustive32BitOrLessUnaryTest<T>::SetParamsForSinCosTan() {
  if (this->platform_ == "Host" || this->platform_ == "CUDA") {
    return;
  }

  // Non CPU/GPU targets may have used the Cody-Waite range reduction technique
  // and will not provide meaningful results for sin/cos/tan if magnitudes
  // exceed 2**p.
  if (T == F32) {
    this->known_incorrect_fn_ = [](int64 v) {
      float f = BitCast<float>(static_cast<uint32>(v));
      return std::abs(f) > (1 << 13);
    };
  } else if (T == BF16) {
    this->known_incorrect_fn_ = [](int64 v) {
      float f = static_cast<float>(BitCast<bfloat16>(static_cast<uint16>(v)));
      return std::abs(f) > (1 << 13);
    };
  }
}

XLA_TEST_P(ExhaustiveF32UnaryTest, Cos) {
  SetParamsForSinCosTan();
  Run(
      Cos, std::cos, +[](NativeT) {
        return ErrorSpec{0.001, 0.001};
      });
}
XLA_TEST_P(ExhaustiveF16UnaryTest, Cos) {
  SetParamsForSinCosTan();
  Run(Cos, std::cos);
}
XLA_TEST_P(ExhaustiveBF16UnaryTest, Cos) {
  SetParamsForSinCosTan();
  Run(Cos, std::cos);
}

XLA_TEST_P(ExhaustiveF32UnaryTest, Sin) {
  SetParamsForSinCosTan();
  Run(
      Sin, std::sin, +[](NativeT) {
        return ErrorSpec{0.001, 0.001};
      });
}
XLA_TEST_P(ExhaustiveF16UnaryTest, Sin) {
  SetParamsForSinCosTan();
  Run(Sin, std::sin);
}
XLA_TEST_P(ExhaustiveBF16UnaryTest, Sin) {
  SetParamsForSinCosTan();
  Run(Sin, std::sin);
}

XLA_TEST_P(ExhaustiveF32UnaryTest, Tan) {
  SetParamsForSinCosTan();
  Run(
      Tan, std::tan, +[](NativeT) {
        return ErrorSpec{0.001, 0.001};
      });
}
XLA_TEST_P(ExhaustiveF16UnaryTest, Tan) {
  SetParamsForSinCosTan();
  Run(Tan, std::tan);
}
XLA_TEST_P(ExhaustiveBF16UnaryTest, Tan) {
  SetParamsForSinCosTan();
  Run(Tan, std::tan);
}

// TODO(jlebar): Enable these.
// XLA_TEST_FLOAT_32_BITS_OR_LESS(Atan) { Run(Atan, std::atan); }
// XLA_TEST_FLOAT_32_BITS_OR_LESS(Atan2) { Run(Atan2, std::atan2); }

XLA_TEST_FLOAT_32_BITS_OR_LESS(Erf, { Run(Erf, std::erf); })
XLA_TEST_FLOAT_32_BITS_OR_LESS(Erfc, { Run(Erfc, std::erfc); })
XLA_TEST_FLOAT_32_BITS_OR_LESS(ErfInv, { Run(ErfInv, HostErfInv); })
XLA_TEST_FLOAT_32_BITS_OR_LESS(Digamma, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA") {
    // TODO(b/123956399): This is a fairly high error, significantly higher than
    // we see on CPU/GPU.
    error_spec_gen = +[](NativeT) { return ErrorSpec{0.01, 0.01}; };
  }

  if (platform_ == "CUDA") {
    // On GPU we get a wrong answer for the denormal inputs +/-2.93873588e-39
    // (0x00200000 and 0x80200000).  These should return -/+inf (at least
    // according to our reference implementation!) but XLA:GPU returns
    // -/+3.40282326e+38 (0xff7ffffe and 0x7f7ffffe).
    //
    // I deem this an acceptable result, as XLA:GPU flushes denormals, and as
    // the results we get here are very close to MAX_FLOAT.  We just hardcode
    // these results, as this is better than ignoring these inputs altogether.
    auto host_digamma_with_gpu_ftz_errors = +[](float x) {
      if (BitCast<uint32>(x) == 0x00200000 ||
          BitCast<uint32>(x) == 0x80200000) {
        return std::copysign(std::numeric_limits<float>::max(), -x);
      }
      return HostDigamma(x);
    };
    Run(Digamma, host_digamma_with_gpu_ftz_errors, error_spec_gen);
  } else {
    Run(Digamma, HostDigamma, error_spec_gen);
  }
})

XLA_TEST_FLOAT_32_BITS_OR_LESS(Lgamma, {
  // Our implementation gets within 0.0001 rel error except for ~20 denormal
  // inputs on GPU.  Anyway 0.001 rel error should be good enough for lgamma.
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "CUDA" && (ty_ == F32 || ty_ == F16)) {
    error_spec_gen = +[](NativeT x) {
      auto spec = GetDefaultSpecGenerator()(x);
      spec.rel_err = 0.001;
      return spec;
    };
  }

  float (*host_lgamma)(float) = std::lgamma;
  if (platform_ != "Host" && platform_ != "CUDA") {
    // TODO(b/123956399): This is a fairly high error, significantly higher than
    // we see on CPU/GPU.
    error_spec_gen = +[](NativeT) { return ErrorSpec{0.01, 0.01}; };

    // Overflows to inf for input 4.08500343e+36 (0x7c44af8e).
    if (ty_ == F32) {
      host_lgamma = +[](float v) {
        if (BitCast<uint32>(v) == 0x7c44af8e) {
          return std::numeric_limits<float>::infinity();
        }
        return std::lgamma(v);
      };
    }
  }
  Run(Lgamma, host_lgamma, error_spec_gen);
})

XLA_TEST_FLOAT_32_BITS_OR_LESS(Round, { Run(Round, std::round); })

#if defined(UNARY_TEST_TARGET_F32_OR_SMALLER)

INSTANTIATE_TEST_SUITE_P(F32, ExhaustiveF32UnaryTest,
                         ::testing::ValuesIn(CreateExhaustiveF32Ranges()));

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
INSTANTIATE_TEST_SUITE_P(F16, ExhaustiveF16UnaryTest,
                         ::testing::Values(std::make_pair(0, 1 << 16)));
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
INSTANTIATE_TEST_SUITE_P(BF16, ExhaustiveBF16UnaryTest,
                         ::testing::Values(std::make_pair(0, 1 << 16)));
#endif

#endif

// Exhaustive test for unary operations for double.
//
// Test parameter is a tuple containing
//   - primitive type under test,
//   - FpValues representing a set of double values.

class ExhaustiveF64UnaryTest : public ExhaustiveUnaryTest<F64>,
                               public ::testing::WithParamInterface<FpValues> {
 private:
  int64 GetInputSize() override {
    FpValues values = GetParam();
    return values.GetTotalNumValues();
  }

  void FillInput(std::array<Literal, 1>* input_literal) override {
    FpValues fp_values = GetParam();
    int64 input_size = (*input_literal)[0].element_count();
    LOG(INFO) << "Checking fp values " << fp_values.ToString() << ", "
              << input_size;
    absl::Span<double> input_arr = (*input_literal)[0].data<double>();

    uint64 i = 0;
    for (auto bits : fp_values) {
      input_arr[i] = this->ConvertAndReplaceKnownIncorrectValueWith(bits, 1);
      ++i;
    }
    CHECK_EQ(i, input_size);
  }
};

XLA_TEST_P(ExhaustiveF64UnaryTest, Log) { Run(Log, std::log); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Log1p) { Run(Log1p, std::log1p); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Exp) { Run(Exp, std::exp); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Expm1) { Run(Expm1, std::expm1); }

// TODO(b/138385863): Turn on the test for GPU after fixing the bug.
XLA_TEST_P(ExhaustiveF64UnaryTest, DISABLED_ON_GPU(PowOneHalf)) {
  Run([](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); },
      +[](double x) { return std::pow(x, 0.5); });
}

XLA_TEST_P(ExhaustiveF64UnaryTest, Rsqrt) {
  Run(
      Rsqrt, +[](double x) { return 1 / std::sqrt(x); });
}

XLA_TEST_P(ExhaustiveF64UnaryTest, Sqrt) { Run(Sqrt, std::sqrt); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Acosh) { Run(Acosh, std::acosh); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Asinh) { Run(Asinh, std::asinh); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Atanh) { Run(Atanh, std::atanh); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Acos) { Run(Acos, std::acos); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Asin) { Run(Asin, std::asin); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Cosh) { Run(Cosh, std::cosh); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Sinh) { Run(Sinh, std::sinh); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Tanh) {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "CUDA") {
    error_spec_gen = +[](NativeT x) {
      return x <= static_cast<NativeT>(-20.0) || x >= static_cast<NativeT>(20.0)
                 ? ErrorSpec{0, 0}
                 : GetDefaultSpecGenerator()(x);
    };
  }
  Run(Tanh, std::tanh, error_spec_gen);
}

XLA_TEST_P(ExhaustiveF64UnaryTest, Cos) { Run(Cos, std::cos); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Sin) { Run(Sin, std::sin); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Tan) { Run(Tan, std::tan); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Round) { Run(Round, std::round); }

XLA_TEST_P(ExhaustiveF64UnaryTest, Erf) {
  Run(Erf, std::erf, [](NativeT x) { return ErrorSpec{1e-20, 1e-20}; });
}

XLA_TEST_P(ExhaustiveF64UnaryTest, Erfc) {
  Run(Erfc, std::erfc, [](NativeT x) { return ErrorSpec{1e-20, 1e-20}; });
}

#if defined(UNARY_TEST_TARGET_F64)
#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF64UnaryTest,
    ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()));

INSTANTIATE_TEST_SUITE_P(NormalValues, ExhaustiveF64UnaryTest,
                         ::testing::Values(GetNormals<double>(1000)));

// Tests a total of 4000000000 inputs, with 16000000 inputs in each sub-test, to
// keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnituedNormalValues, ExhaustiveF64UnaryTest,
    ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<double>(
        4000000000ull, 16000000)));
#endif
#endif

// T is the Primitive Type of the complex number
// Test parameter is a tuple containing
//   - primitive type under test,
//   - two FpValues representing the values for the real and imaginary
//     components. The complex numbers for the test input is the cartesian
//     product of the values represented by the two FpValues.
template <PrimitiveType T>
class ExhaustiveComplexUnaryTestBase
    : public ExhaustiveUnaryTest<T>,
      public ::testing::WithParamInterface<std::tuple<FpValues, FpValues>> {
 protected:
  using typename ExhaustiveUnaryTest<T>::NativeT;

  void SetParamsForTanh() {
    // TODO(b/138126045): Current libc++ implementation of the complex tanh
    //                    function returns (NaN, NaN) when the imaginary
    //                    component is more than half of the max value.
    // TODO(b/138750327): Current libc++ implementation of the complex tanh
    //                    function returns (1, 0) when the real component is
    //                    negative infinity, when it should return (-1, 0).
    // We only need to set the former as incorrect values for C128 because when
    // testing with C64, we first cast our input to a C128 value.
    this->known_incorrect_fn_ = [&](int64 v) {
      double f = this->ConvertValue(v);
      return (T == C128 &&
              std::abs(f) > std::numeric_limits<double>::max() / 2) ||
             f == -std::numeric_limits<double>::infinity();
    };
  }

 private:
  // Generates the input complex literal given the FpValues representation for
  // the real and imaginary components.
  void FillInput(std::array<Literal, 1>* input_literal) override {
    FpValues real_values = std::get<0>(GetParam());
    FpValues imag_values = std::get<1>(GetParam());

    VLOG(2) << " testing input total "
            << real_values.GetTotalNumValues() * imag_values.GetTotalNumValues()
            << ", range " << real_values.ToString() << " "
            << imag_values.ToString();

    absl::Span<NativeT> input_arr = (*input_literal)[0].data<NativeT>();

    uint64 i = 0;
    for (auto real : real_values) {
      for (auto imag : imag_values) {
        input_arr[i] =
            NativeT(this->ConvertAndReplaceKnownIncorrectValueWith(real, 1),
                    this->ConvertAndReplaceKnownIncorrectValueWith(imag, 1));

        ++i;
      }
    }
  }

  int64 GetInputSize() override {
    FpValues real_values = std::get<0>(GetParam());
    FpValues imag_values = std::get<1>(GetParam());
    return real_values.GetTotalNumValues() * imag_values.GetTotalNumValues();
  }
};

typedef ExhaustiveComplexUnaryTestBase<C64> ExhaustiveC64UnaryTest;
typedef ExhaustiveComplexUnaryTestBase<C128> ExhaustiveC128UnaryTest;

// TODO(b/138578594): Enable the test for the CPU backend after fixing the bug.
XLA_TEST_P(ExhaustiveC64UnaryTest, DISABLED_ON_CPU(Log)) {
  Run(Log, [](complex64 x) { return std::log<float>(x); });
}

XLA_TEST_P(ExhaustiveC64UnaryTest, Sqrt) {
  Run(Sqrt, [](complex64 x) {
    return static_cast<complex64>(
        std::sqrt<double>(static_cast<complex128>(x)));
  });
}

XLA_TEST_P(ExhaustiveC64UnaryTest, Rsqrt) {
  Run(Rsqrt, [](complex64 x) {
    return static_cast<complex64>(
        complex128(1, 0) / std::sqrt<double>(static_cast<complex128>(x)));
  });
}

// The current libc++ implementation of the complex tanh function provides
// less accurate results when the denomenator of a complex tanh is small, due
// to floating point precision loss. To avoid this issue for complex64 numbers,
// we cast it to and from a complex128 when computing tanh.
XLA_TEST_P(ExhaustiveC64UnaryTest, Tanh) {
  SetParamsForTanh();
  ErrorSpecGen error_spec_gen = +[](complex64 x) {
    // This implementation of Tanh becomes less accurate when the denominator
    // is small.
    if (std::cosh(2 * x.real()) + std::cos(2 * x.imag()) < 1e-4) {
      return ErrorSpec{5e-2, 5e-2};
    }

    return GetDefaultSpecGenerator()(x);
  };
  Run(
      Tanh,
      +[](complex64 x) {
        return static_cast<complex64>(std::tanh(static_cast<complex128>(x)));
      },
      error_spec_gen);
}

#if defined(UNARY_TEST_TARGET_COMPLEX)
INSTANTIATE_TEST_SUITE_P(
    F32SpecialValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    F32SpecialAndNormalValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::Values(GetNormals<float>(10000))));

INSTANTIATE_TEST_SUITE_P(
    F32NormalAndSpecialValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<float>(10000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    F32NormalAndNormalValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(::testing::Values(GetNormals<float>(10000)),
                       ::testing::Values(GetNormals<float>(10000))));

// Tests a total of 40000 ^ 2 inputs, with 4000 ^ 2 inputs in each sub-test, to
// keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    F32LargeAndSmallMagnituedNormalValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<float>(40000,
                                                                         4000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<float>(40000, 4000))));
#endif


XLA_TEST_P(ExhaustiveC128UnaryTest, Log) {
  // TODO(b/138578313): Enable the test for all values after fixing the bug.
  known_incorrect_fn_ = [&](int64 v) {
    double f = this->ConvertValue(v);
    return std::fpclassify(f) == FP_NAN || std::abs(f) > 1.0e+300 ||
           std::abs(f) < 1.0e-300;
  };
  Run(Log, [](complex128 x) { return std::log<double>(x); });
}

XLA_TEST_P(ExhaustiveC128UnaryTest, Sqrt) {
  // Similar to the Tanh bug.
  known_incorrect_fn_ = [&](int64 v) {
    double f = this->ConvertValue(v);
    return std::abs(f) > std::numeric_limits<double>::max() / 2;
  };
  Run(Sqrt, [](complex128 x) { return std::sqrt<double>(x); });
}

XLA_TEST_P(ExhaustiveC128UnaryTest, Rsqrt) {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "CUDA") {
    // Edge case on CUDA backend where the Log of a complex number made up of
    // the smallest denormals is more accurate than the interpreter backend.
    error_spec_gen = [](complex128 x) {
      constexpr double denorm_min = std::numeric_limits<double>::denorm_min();
      if (std::abs(x.real()) == denorm_min &&
          std::abs(x.imag()) == denorm_min) {
        return ErrorSpec(0.5, 0.5);
      }
      return GetDefaultSpecGenerator()(x);
    };
  }
  Run(
      Rsqrt,
      [](complex128 x) { return complex128(1, 0) / std::sqrt<double>(x); },
      error_spec_gen);
}

XLA_TEST_P(ExhaustiveC128UnaryTest, Tanh) {
  SetParamsForTanh();
  Run(
      Tanh, +[](complex128 x) { return std::tanh(x); });
}

#if defined(UNARY_TEST_TARGET_COMPLEX)
#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    SpecialAndNormalValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::Values(GetNormals<double>(10000))));

INSTANTIATE_TEST_SUITE_P(
    NormalAndSpecialValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<double>(10000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    F32NormalAndNormalValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(::testing::Values(GetNormals<double>(10000)),
                       ::testing::Values(GetNormals<double>(10000))));

// Tests a total of 40000 ^ 2 inputs, with 2000 ^ 2 inputs in each sub-test, to
// keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnituedNormalValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000))));
#endif
#endif

}  // namespace xla
