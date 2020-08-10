/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/math.h"

// This macro is required to make MSVC defines math constants in math.h
#define _USE_MATH_DEFINES
#include <math.h>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace {

// Evaluate the polynomial given `x` and coefficients in decreasing order.
template <typename FP>
XlaOp EvaluatePolynomial(XlaOp x, absl::Span<const FP> coefficients) {
  static_assert(std::is_floating_point<FP>::value,
                "Template-argument 'FP' must be a floating-point type");
  XlaOp poly = ScalarLike(x, 0.0);
  for (FP c : coefficients) {
    poly = poly * x + ScalarLike(x, c);
  }
  return poly;
}

}  // namespace

// Returns operation(operand), except if `operand` is one of the types in
// upcast_types, in which case first converts it to F32, and then converts the
// result down to the original type.
static XlaOp DoWithUpcastToF32(XlaOp operand,
                               absl::Span<const PrimitiveType> upcast_types,
                               const std::function<XlaOp(XlaOp)>& operation) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));
    PrimitiveType elem_ty = shape.element_type();
    bool needs_upcast = absl::c_linear_search(upcast_types, elem_ty);

    if (needs_upcast) {
      operand = ConvertElementType(operand, F32);
    }
    XlaOp result = operation(operand);
    if (needs_upcast) {
      result = ConvertElementType(result, elem_ty);
    }
    return result;
  });
}

// TODO(jlebar): Use this function in more places in this file to restrict the
// domain of other functions.
static Status EnsureOperandIsRealFp(absl::string_view op_name, XlaOp operand) {
  auto& b = *operand.builder();
  TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));
  auto elem_ty = shape.element_type();
  if (!primitive_util::IsFloatingPointType(elem_ty)) {
    return InvalidArgument(
        "Operands to %s must be real-valued floating-point, but got %s",
        op_name, PrimitiveType_Name(elem_ty));
  }
  return Status::OK();
}

XlaOp IsPosInf(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsPosInf", operand));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));
    // Note that this is only correct for floating-point types.  If we wanted it
    // to be correct for all types, we'd need to Gt(MaxFiniteValue).
    return Eq(operand, MaxValue(&b, shape.element_type()));
  });
}

XlaOp IsNegInf(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsNegInf", operand));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));
    // Note that this is only correct for floating-point types.  If we wanted it
    // to be correct for all types, we'd need to Lt(MinFiniteValue).
    return Eq(operand, MinValue(&b, shape.element_type()));
  });
}

XlaOp IsInf(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsInf", operand));
    return IsPosInf(Abs(operand));
  });
}

XlaOp IsNan(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsNan", operand));
    return Ne(operand, operand);
  });
}

XlaOp IsNegZero(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsNegZero", operand));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));

    // The bitwise representation of -0 in bfloat16 and IEEE 754 is 0x80...0
    // (sign bit on, all other bits off).
    switch (shape.element_type()) {
      case F64:
        return Eq(BitcastConvertType(operand, U64),
                  ConstantR0WithType(&b, U64, uint64{1} << 63));
      case F32:
        return Eq(BitcastConvertType(operand, U32),
                  ConstantR0WithType(&b, U32, uint32{1} << 31));
      case F16:
      case BF16:
        // Not all XLA backends handle U16 well, so we convert to F32/U32.
        // TODO(jlebar): It would be nice if we could stay in (B)F16/U16 for
        // backends that *do* support it.
        return Eq(BitcastConvertType(ConvertElementType(operand, F32), U32),
                  ConstantR0WithType(&b, U32, uint32{1} << 31));
      default:
        LOG(FATAL) << "Expected real fp type.";
    }
  });
}

XlaOp Square(XlaOp operand) { return operand * operand; }

XlaOp Reciprocal(XlaOp operand) { return ScalarLike(operand, 1.0) / operand; }

// Computes an approximation of the error function complement (1 - erf(x)).
//
// Precondition: abs(x) >= 1.  Otherwise, use ErfImpl.
//
// This follows Cephes's f32 implementation of erfc.
static XlaOp ErfcImpl32(XlaOp x) {
  // Coefficients for erfc(f32), from Cephes.
  const double kMaxlog = 88.72283905206835;
  // erfc(x) = exp(-x^2) P(1/x^2), 1 < x < 2
  static const std::array<float, 9> kErfcPCoefficient{
      +2.326819970068386E-2, -1.387039388740657E-1, +3.687424674597105E-1,
      -5.824733027278666E-1, +6.210004621745983E-1, -4.944515323274145E-1,
      +3.404879937665872E-1, -2.741127028184656E-1, +5.638259427386472E-1,
  };
  // erfc(x) = exp(-x^2) R(1/x^2), 2 <= x < kMaxlog
  static const std::array<float, 8> kErfcRCoefficient{
      -1.047766399936249E+1, +1.297719955372516E+1, -7.495518717768503E+0,
      +2.921019019210786E+0, -1.015265279202700E+0, +4.218463358204948E-1,
      -2.820767439740514E-1, +5.641895067754075E-1,
  };
  XlaOp abs_x = Abs(x);
  XlaOp z = Exp(-x * x);
  XlaOp q = ScalarLike(x, 1) / abs_x;
  XlaOp y = q * q;
  XlaOp p = Select(Lt(abs_x, ScalarLike(x, 2.0)),
                   EvaluatePolynomial<float>(y, kErfcPCoefficient),
                   EvaluatePolynomial<float>(y, kErfcRCoefficient));
  y = z * q * p;
  XlaOp y_clamp = Select(Lt(z, ScalarLike(x, -kMaxlog)), ScalarLike(x, 0), y);
  return Select(Lt(x, ScalarLike(x, 0)), ScalarLike(x, 2.0) - y_clamp, y_clamp);
}

// Compute a polynomial approximation of the error function.
//
// Precondition: abs(x) <= 1.  Otherwise, use ErfcImpl.
//
// This follows Cephes's f32 implementation of erf.
static XlaOp ErfImpl32(XlaOp x) {
  // Coefficients for by erf(f32), from Cephes.
  //
  // erf(x) = x P(x^2), 0 < x < 1
  static const std::array<float, 7> kErfTCoefficient{
      +7.853861353153693E-5, -8.010193625184903E-4, +5.188327685732524E-3,
      -2.685381193529856E-2, +1.128358514861418E-1, -3.761262582423300E-1,
      +1.128379165726710E+0,
  };
  return x * EvaluatePolynomial<float>(x * x, kErfTCoefficient);
}

static XlaOp ErfcImpl64(XlaOp x) {
  // Coefficients for erfc(f64), from Cephes.
  const double kMaxlog = 7.09782712893383996843E2;
  // erfc(x) = exp(-x^2) P(|x|) / Q(|x|), 1 < x < 8
  static const std::array<double, 9> kErfcPCoefficient{
      2.46196981473530512524E-10, 5.64189564831068821977E-1,
      7.46321056442269912687E0,   4.86371970985681366614E1,
      1.96520832956077098242E2,   5.26445194995477358631E2,
      9.34528527171957607540E2,   1.02755188689515710272E3,
      5.57535335369399327526E2};
  static const std::array<double, 9> kErfcQCoefficient{
      1.00000000000000000000E0, 1.32281951154744992508E1,
      8.67072140885989742329E1, 3.54937778887819891062E2,
      9.75708501743205489753E2, 1.82390916687909736289E3,
      2.24633760818710981792E3, 1.65666309194161350182E3,
      5.57535340817727675546E2};

  // erfc(x) = exp(-x^2) R(|x|) / S(|x|), 8 <= x < kMaxlog
  static const std::array<double, 6> kErfcRCoefficient{
      5.64189583547755073984E-1, 1.27536670759978104416E0,
      5.01905042251180477414E0,  6.16021097993053585195E0,
      7.40974269950448939160E0,  2.97886665372100240670E0};
  static const std::array<double, 7> kErfcSCoefficient{
      1.00000000000000000000E0, 2.26052863220117276590E0,
      9.39603524938001434673E0, 1.20489539808096656605E1,
      1.70814450747565897222E1, 9.60896809063285878198E0,
      3.36907645100081516050E0};

  XlaOp z = -x * x;
  XlaOp abs_x = Abs(x);
  XlaOp y =
      Select(Lt(abs_x, ScalarLike(x, 8.0)),
             Exp(z) * EvaluatePolynomial<double>(abs_x, kErfcPCoefficient) /
                 EvaluatePolynomial<double>(abs_x, kErfcQCoefficient),
             Exp(z) * EvaluatePolynomial<double>(abs_x, kErfcRCoefficient) /
                 EvaluatePolynomial<double>(abs_x, kErfcSCoefficient));
  XlaOp y_clamp = Select(Lt(z, ScalarLike(x, -kMaxlog)), ScalarLike(x, 0), y);
  return Select(Lt(x, ScalarLike(x, 0)), ScalarLike(x, 2.0) - y_clamp, y_clamp);
}

// Compute a polynomial approximation of the error function.
//
// Precondition: abs(x) <= 1.  Otherwise, use ErfcImpl.
static XlaOp ErfImpl64(XlaOp x) {
  // Coefficients for by erf(f64), from Cephes.
  //
  // erf(x) = x T(x^2) / U(x^2), 0 < x < 1
  static std::array<double, 5> kErfTCoefficient{
      9.60497373987051638749E0, 9.00260197203842689217E1,
      2.23200534594684319226E3, 7.00332514112805075473E3,
      5.55923013010394962768E4};
  static std::array<double, 6> kErfUCoefficient{
      1.00000000000000000000E0, 3.35617141647503099647E1,
      5.21357949780152679795E2, 4.59432382970980127987E3,
      2.26290000613890934246E4, 4.92673942608635921086E4};
  XlaOp z = x * x;
  return x * EvaluatePolynomial<double>(z, kErfTCoefficient) /
         EvaluatePolynomial<double>(z, kErfUCoefficient);
}

XlaOp Erfc(XlaOp x) {
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Erfc", x));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(x));
    // erfc(x) =
    //   erfc_impl(x)           if x > 1
    //   1 - erf_impl(x)        otherwise
    if (shape.element_type() == F64) {
      return Select(Gt(Abs(x), ScalarLike(x, 1)), ErfcImpl64(x),
                    ScalarLike(x, 1) - ErfImpl64(x));
    }
    // Erf(c)Impl don't have enough precision when run with bf16 intermediates
    // (not surprising!), so upcast to f32 in this case.
    return DoWithUpcastToF32(x, {BF16, F16}, [](XlaOp x) {
      return Select(Gt(Abs(x), ScalarLike(x, 1)), ErfcImpl32(x),
                    ScalarLike(x, 1) - ErfImpl32(x));
    });
  });
}

XlaOp Erf(XlaOp x) {
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Erf", x));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(x));
    // erf(x) =
    //   erf_impl(x)            if x < 1
    //   1 - erfc_impl(x)       otherwise
    if (shape.element_type() == F64) {
      return Select(Lt(Abs(x), ScalarLike(x, 1)), ErfImpl64(x),
                    ScalarLike(x, 1) - ErfcImpl64(x));
    }
    // Erf(c)Impl don't have enough precision when run with bf16 intermediates
    // (not surprising!), so upcast to f32 in this case.
    return DoWithUpcastToF32(x, {BF16, F16}, [](XlaOp x) {
      return Select(Lt(Abs(x), ScalarLike(x, 1)), ErfImpl32(x),
                    ScalarLike(x, 1) - ErfcImpl32(x));
    });
  });
}

// Approximation for the inverse error function from
//   Giles, M., "Approximating the erfinv function".
// The approximation has the form:
//   w = -log((1 - x) * (1 + x))
//   if ( w < 5 ) {
//     w = w - 2.5
//     p = sum_{i=1}^n lq[i]*w^i
//   } else {
//     w = sqrt(w) - 3
//     p = sum_{i=1}^n gq[i]*w^i
//   }
//   return p*x
XlaOp ErfInv(XlaOp x) {
  constexpr int kDegree = 9;
  constexpr std::array<float, 9> w_less_than_5_constants = {
      2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
      -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
      -0.00417768164f,  0.246640727f,    1.50140941f};
  constexpr std::array<float, 9> w_greater_than_5_constants = {
      -0.000200214257f, 0.000100950558f, 0.00134934322f,
      -0.00367342844f,  0.00573950773f,  -0.0076224613f,
      0.00943887047f,   1.00167406f,     2.83297682f};

  auto one = ScalarLike(x, 1.0);
  auto w = -Log((one - x) * (one + x));

  auto lt = Lt(w, ScalarLike(x, 5.0));
  auto coefficient = [&](int i) {
    return Select(lt, FullLike(x, w_less_than_5_constants[i]),
                  FullLike(x, w_greater_than_5_constants[i]));
  };
  w = Select(lt, w - ScalarLike(x, 2.5), Sqrt(w) - ScalarLike(x, 3.0));
  auto p = coefficient(0);
  for (int i = 1; i < kDegree; ++i) {
    p = coefficient(i) + p * w;
  }

  // Result modulo edge cases.
  XlaOp result = p * x;

  // Handle edge cases, namely erfinv(+/-1) = +/-inf.  (The above computation is
  // indeterminate, and can give nan or -/+inf.)
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, b.GetShape(x));
    return Select(Eq(Abs(x), ScalarLike(x, 1)),
                  x * MaxValue(&b, shape.element_type()), result);
  });
}

namespace {
// Coefficients for the Lanczos approximation of the gamma function. The
// coefficients are uniquely determined by the choice of g and n (kLanczosGamma
// and kLanczosCoefficients.size() + 1). The coefficients below correspond to
// [7, 9]. [5, 7], [7, 9], [9, 10], and [607/128.0, 15] were evaluated and [7,
// 9] seemed to be the least sensitive to the quality of the log function. In
// particular, [5, 7] is the only choice where -1.5e-5 <= lgamma(2) <= 1.5e-5
// for a particularly inaccurate log function.
static constexpr double kLanczosGamma = 7;  // aka g
static constexpr double kBaseLanczosCoeff = 0.99999999999980993227684700473478;
static constexpr std::array<double, 8> kLanczosCoefficients = {
    676.520368121885098567009190444019, -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,   -176.61502916214059906584551354,
    12.507343278686904814458936853,     -0.13857109526572011689554707,
    9.984369578019570859563e-6,         1.50563273514931155834e-7};
}  // namespace

// Compute the Lgamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
// lgamma(z + 1) = (log(2) + log(pi)) / 2 + (z + 1/2) * log(t(z)) - t(z) + A(z)
// t(z) = z + kLanczosGamma + 1/2
// A(z) = kBaseLanczosCoeff + sigma(k = 1, n, kLanczosCoefficients[i] / (z + k))
XlaOp Lgamma(XlaOp input) {
  auto do_it = [](XlaOp input) {
    XlaOp one_half = ScalarLike(input, 0.5);
    XlaOp one = ScalarLike(input, 1);

    XlaOp pi = ScalarLike(input, M_PI);
    XlaOp log_pi = ScalarLike(input, std::log(M_PI));
    XlaOp log_sqrt_two_pi =
        ScalarLike(input, (std::log(2) + std::log(M_PI)) / 2);

    XlaOp lanczos_gamma_plus_one_half = ScalarLike(input, kLanczosGamma + 0.5);
    XlaOp log_lanczos_gamma_plus_one_half =
        ScalarLike(input, std::log(kLanczosGamma + 0.5));

    XlaOp base_lanczos_coeff = ScalarLike(input, kBaseLanczosCoeff);

    // If the input is less than 0.5 use Euler's reflection formula:
    // gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
    XlaOp need_to_reflect = Lt(input, one_half);
    XlaOp z = Select(need_to_reflect, -input, input - one);

    XlaOp x = base_lanczos_coeff;
    for (int i = 0; i < kLanczosCoefficients.size(); ++i) {
      XlaOp lanczos_coefficient = ScalarLike(input, kLanczosCoefficients[i]);
      XlaOp index = ScalarLike(input, i);
      x = x + lanczos_coefficient / (z + index + one);
    }

    // To improve accuracy on platforms with less-precise log implementations,
    // compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
    // the device.
    // log(t) = log(kLanczosGamma + 0.5 + z)
    //        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
    XlaOp t = lanczos_gamma_plus_one_half + z;
    XlaOp log_t = log_lanczos_gamma_plus_one_half +
                  Log1p(z / lanczos_gamma_plus_one_half);

    // Compute the final result (modulo reflection).  t(z) may be large, and we
    // need to be careful not to overflow to infinity in the first term of
    //
    //   (z + 1/2) * log(t(z)) - t(z).
    //
    // Therefore we compute this as
    //
    //   (z + 1/2 - t(z) / log(t(z))) * log(t(z)).
    //
    XlaOp log_y = log_sqrt_two_pi + (z + one_half - t / log_t) * log_t + Log(x);

    // Compute the reflected value, used when x < 0.5:
    //
    //   lgamma(x) = log(pi) - lgamma(1-x) - log(abs(sin(pi * x))).
    //
    // (The abs is because lgamma is the log of the absolute value of the gamma
    // function.)
    //
    // We have to be careful when computing the final term above. gamma(x) goes
    // to +/-inf at every integer x < 0, and this is controlled by the
    // sin(pi * x) term.  The slope is large, so precision is particularly
    // important.
    //
    // Because abs(sin(pi * x)) has period 1, we can equivalently use
    // abs(sin(pi * frac(x))), where frac(x) is the fractional part of x.  This
    // is more numerically accurate: It doesn't overflow to inf like pi * x can,
    // and if x is an integer, it evaluates to 0 exactly, which is significant
    // because we then take the log of this value, and log(0) is inf.
    //
    // We don't have a frac(x) primitive in XLA and computing it is tricky, but
    // because abs(sin(pi * x)) = abs(sin(pi * abs(x))), it's good enough for
    // our purposes to use abs(frac(x)) = abs(x) - floor(abs(x)).
    //
    // Furthermore, pi * abs(frac(x)) loses precision when abs(frac(x)) is close
    // to 1.  To remedy this, we can use the fact that sin(pi * x) in the domain
    // [0, 1] is symmetric across the line Y=0.5.
    //
    XlaOp abs_input = Abs(input);
    XlaOp abs_frac_input = abs_input - Floor(abs_input);
    // Convert values of abs_frac_input > 0.5 to (1 - frac_input) to improve
    // precision of pi * abs_frac_input for values of abs_frac_input close to 1.
    XlaOp reduced_frac_input =
        Select(Gt(abs_frac_input, ScalarLike(abs_frac_input, 0.5)),
               ScalarLike(abs_frac_input, 1) - abs_frac_input, abs_frac_input);
    XlaOp reflection_denom = Log(Sin(pi * reduced_frac_input));

    // Avoid computing -inf - inf, which is nan.  If reflection_denom is +/-inf,
    // then it "wins" and the result is +/-inf.
    XlaOp reflection =
        Select(IsFinite(reflection_denom), log_pi - reflection_denom - log_y,
               -reflection_denom);
    XlaOp result = Select(need_to_reflect, reflection, log_y);

    // lgamma(+/-inf) = +inf.
    XlaOp inf_bcast = FullLike(input, std::numeric_limits<float>::infinity());
    return Select(IsInf(input), inf_bcast, result);
  };

  auto& b = *input.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Lgamma", input));
    // F16 and BF16 don't provide sufficient precision for intermediate results
    // here (although it's better than you might expect!), so do the
    // computations in F32.
    return DoWithUpcastToF32(input, {BF16, F16}, do_it);
  });
}

// Compute the Digamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
// digamma(z + 1) = log(t(z)) + A'(z) / A(z) - kLanczosGamma / t(z)
// t(z) = z + kLanczosGamma + 1/2
// A(z) = kBaseLanczosCoeff + sigma(k = 1, n, kLanczosCoefficients[i] / (z + k))
// A'(z) = sigma(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
XlaOp Digamma(XlaOp input) {
  auto do_it = [](XlaOp input) {
    XlaOp zero = ScalarLike(input, 0);
    XlaOp one_half = ScalarLike(input, 0.5);
    XlaOp one = ScalarLike(input, 1);

    XlaOp pi = ScalarLike(input, M_PI);

    XlaOp lanczos_gamma = ScalarLike(input, kLanczosGamma);
    XlaOp lanczos_gamma_plus_one_half = ScalarLike(input, kLanczosGamma + 0.5);
    XlaOp log_lanczos_gamma_plus_one_half =
        ScalarLike(input, std::log(kLanczosGamma + 0.5));

    XlaOp base_lanczos_coeff = ScalarLike(input, kBaseLanczosCoeff);

    // If the input is less than 0.5 use Euler's reflection formula:
    // digamma(x) = digamma(1 - x) - pi * cot(pi * x)
    XlaOp need_to_reflect = Lt(input, one_half);
    XlaOp z = Select(need_to_reflect, -input, input - one);

    XlaOp num = zero;
    XlaOp denom = base_lanczos_coeff;
    for (int i = 0; i < kLanczosCoefficients.size(); ++i) {
      XlaOp lanczos_coefficient = ScalarLike(input, kLanczosCoefficients[i]);
      XlaOp index = ScalarLike(input, i);
      num = num - lanczos_coefficient / ((z + index + one) * (z + index + one));
      denom = denom + lanczos_coefficient / (z + index + one);
    }

    // To improve accuracy on platforms with less-precise log implementations,
    // compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
    // the device.
    // log(t) = log(kLanczosGamma + 0.5 + z)
    //        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
    XlaOp t = lanczos_gamma_plus_one_half + z;
    XlaOp log_t = log_lanczos_gamma_plus_one_half +
                  Log1p(z / lanczos_gamma_plus_one_half);

    XlaOp y = log_t + num / denom - lanczos_gamma / t;

    // We need to be careful how we compute cot(pi * input) below: For
    // near-integral values of `input`, pi * input can lose precision.
    //
    // Input is already known to be less than 0.5 (otherwise we don't have to
    // reflect).  We shift values smaller than -0.5 into the range [-.5, .5] to
    // increase precision of pi * input and the resulting cotangent.
    XlaOp reduced_input = input + Abs(Floor(input + ScalarLike(input, 0.5)));
    XlaOp reflection =
        y - pi * Cos(pi * reduced_input) / Sin(pi * reduced_input);
    XlaOp real_result = Select(need_to_reflect, reflection, y);

    // Digamma has poles at negative integers and zero; return nan for those.
    return Select(And(Le(input, zero), Eq(input, Floor(input))),
                  FullLike(input, std::numeric_limits<float>::quiet_NaN()),
                  real_result);
  };

  auto& b = *input.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Digamma", input));
    return DoWithUpcastToF32(input, {BF16, F16}, do_it);
  });
}

// Implements Banker's rounding: numbers that are equidistant between two
// integers are rounded towards even.
XlaOp RoundToEven(XlaOp x) {
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // Reject non-real non-fp inputs (What does it even mean to round a complex
    // number?  Do you round each component equally?  In that case, you should
    // just ask for that explicitly.)
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("RoundToEven", x));

    auto half = ScalarLike(x, 0.5);
    auto one = ScalarLike(x, 1.0);
    auto two = ScalarLike(x, 2.0);

    auto round_val = Floor(x);
    auto fraction = x - round_val;
    auto nearest_even_int = round_val - two * Floor(half * x);
    auto is_odd = Eq(nearest_even_int, one);
    return Select(Or(Gt(fraction, half), And(Eq(fraction, half), is_odd)),
                  round_val + one, round_val);
  });
}

// Trigonometric functions.

// acos(x) = 2 * atan(sqrt(1 - x^2) / (1 + x)) if x != -1
//           pi                                if x == -1
XlaOp Acos(XlaOp x) {
  return Select(Ne(x, FullLike(x, -1)),
                ScalarLike(x, 2.0) * Atan2(Sqrt(ScalarLike(x, 1.0) - x * x),
                                           ScalarLike(x, 1.0) + x),
                FullLike(x, M_PI));
}

// asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
XlaOp Asin(XlaOp x) {
  return ScalarLike(x, 2.0) *
         Atan2(x, ScalarLike(x, 1.0) + Sqrt(ScalarLike(x, 1.0) - x * x));
}

XlaOp Atan(XlaOp x) { return Atan2(x, ScalarLike(x, 1.0)); }

XlaOp Tan(XlaOp x) {
  return DoWithUpcastToF32(x, {F16}, [](XlaOp x) { return Sin(x) / Cos(x); });
}

// Hyperbolic trigonometric functions.

// acosh(x) = log(x + sqrt(x^2 - 1))      if x >= -1
//          = log(x + sqrt((x+1)*(x-1)))
// acosh(x) = nan                         if x < -1
//
// If x^2 will overflow, we approximate sqrt(x^2 - 1) == x and compute as
// log(2*x) = log(2) + log(x).  (Note this works because negative x never
// overflows; x < -1 simply yields nan.  This is quite different than asinh!)
XlaOp Acosh(XlaOp x) {
  XlaBuilder* b = x.builder();
  return b->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(x));

    auto one = ScalarLike(x, 1);
    auto neg_one = ScalarLike(x, -1);
    auto nan = FullLike(x, std::numeric_limits<float>::quiet_NaN());

    // return
    //
    //   nan                        if x < -1
    //   log(x) + log(2)            if x >= sqrt_max_value
    //   log(x + sqrt((x+1)*(x-1))) otherwise
    //
    // TODO(jlebar): For now, we ignore the question of overflow if x is a
    // complex type, because we don't yet have exhaustive tests for complex trig
    // functions.
    auto naive_result = Log(x + Sqrt((x + one) * (x - one)));
    if (primitive_util::IsComplexType(shape.element_type())) {
      return naive_result;
    }
    auto overflow_result = Log(x) + Log(ScalarLike(x, 2));

    auto sqrt_max_value = Sqrt(MaxFiniteValue(b, shape.element_type()));
    return Select(Lt(x, neg_one), nan,
                  Select(Ge(x, sqrt_max_value), overflow_result, naive_result));
  });
}

// asinh(x) = log(x + sqrt(x^2 + 1))
//
// If x^2 will overflow and x is positive, we can approximate x + sqrt(x^2 + 1)
// as 2*x and return log(2) + log(x).
//
// If x is negative, the above would give us some trouble; we can't approximate
// the result as x + abs(x) = 0!  But we're saved by the fact that asinh(-x) =
// -asinh(x).
XlaOp Asinh(XlaOp x) {
  XlaBuilder* b = x.builder();
  auto do_it = [&](XlaOp x) -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(x));
    auto one = ScalarLike(x, 1);

    // Let a = abs(x).  Compute
    //
    //   y = log(a + sqrt(a*a + 1))  if a < sqrt_max_value, or
    //   y = log(a) + log(2)         otherwise
    //
    // and then return
    //
    //   y * sign(x).
    //
    // TODO(jlebar): For now, we ignore the question of overflow if x is a
    // complex type, because we don't yet have exhaustive tests for complex trig
    // functions.
    if (primitive_util::IsComplexType(shape.element_type())) {
      return Log(x + Sqrt(x * x + one));
    }
    auto a = Abs(x);
    auto naive_result = Log(a + Sqrt(a * a + one));
    auto overflow_result = Log(Abs(a)) + Log(ScalarLike(a, 2));
    auto sqrt_max_value = Sqrt(MaxFiniteValue(b, shape.element_type()));
    return Sign(x) *
           Select(Ge(a, sqrt_max_value), overflow_result, naive_result);
  };
  // These upcasts are not strictly necessary on all platforms to get within our
  // error tolerances, so we could relax this if it ever mattered.
  return DoWithUpcastToF32(x, {BF16, F16}, [&](XlaOp x) {
    return b->ReportErrorOrReturn(do_it(x));
  });
}

// atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) <= 1
// atanh(x) = nan                          otherwise
XlaOp Atanh(XlaOp x) {
  XlaBuilder* b = x.builder();
  auto do_it = [&](XlaOp x) -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(x));
    auto naive_result =
        Log((ScalarLike(x, 1.0) + x) / (ScalarLike(x, 1.0) - x)) *
        ScalarLike(x, 0.5);

    // TODO(jlebar): For now, we ignore the nan edge case for complex inputs,
    // because we don't yet have exhaustive tests for complex trig functions.
    if (primitive_util::IsComplexType(shape.element_type())) {
      return naive_result;
    }

    auto nan = FullLike(x, std::numeric_limits<float>::quiet_NaN());
    return Select(Gt(Abs(x), ScalarLike(x, 1)), nan, naive_result);
  };
  return DoWithUpcastToF32(x, {BF16}, [&](XlaOp x) {  //
    return b->ReportErrorOrReturn(do_it(x));
  });
}

// Cosh(x) = (e^x + e^-x) / 2
//         = e^(x + log(1/2)) + e^(-x + log(1/2)).
//
// The second formulation avoids overflowing when e^x = inf but (e^x)/2 is not
// inf.
//
// This incorrectly overflows to inf for two f32 input values, namely
// +/-89.4159851, due to rounding error when computing x +/- log(1/2).  The
// correct answer of 3.40281961e+38 (0x7f7fffec) is very close to max-float, so
// we deem this acceptable.
XlaOp Cosh(XlaOp x) {
  return DoWithUpcastToF32(x, {BF16, F16}, [](XlaOp x) {
    auto log_one_half = Log(ScalarLike(x, 0.5));
    return Exp(x + log_one_half) + Exp(-x + log_one_half);
  });
}

// Sinh(x) = (e^x - e^-x) / 2
//         = e^(x + log(1/2)) - e^(-x + log(1/2)).
//
// The second formulation avoids overflowing when e^x = inf but (e^x)/2 is not
// inf.
//
// This incorrectly overflows to +/-inf for two f32 input values, namely
// +/-89.4159851, due to rounding error when computing x +/- log(1/2).  The
// correct answer of 3.40281961e+38 (0x7f7fffec) is very close to max-float, so
// we deem this acceptable.
XlaOp Sinh(XlaOp x) {
  return DoWithUpcastToF32(x, {BF16, F16}, [](XlaOp x) {
    auto log_one_half = Log(ScalarLike(x, 0.5));
    return Exp(x + log_one_half) - Exp(-x + log_one_half);
  });
}

XlaOp MaybeConjugate(XlaOp x, bool conjugate) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    auto perform_conj =
        primitive_util::IsComplexType(shape.element_type()) && conjugate;
    return perform_conj ? Conj(x) : x;
  });
}

XlaOp NextAfter(XlaOp from, XlaOp to) {
  auto builder = from.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, builder->GetShape(from));
    int bitwidth = primitive_util::BitWidth(shape.element_type());
    auto int_type = primitive_util::UnsignedIntegralTypeForBitWidth(bitwidth);
    auto from_as_int = BitcastConvertType(from, int_type);
    auto to_as_int = BitcastConvertType(to, int_type);

    // The result is NaN if either "from" or "to" are NaN.
    auto from_is_nan = Ne(from, from);
    auto to_is_nan = Ne(to, to);
    auto nan_input = Or(from_is_nan, to_is_nan);
    auto result_for_nan =
        Broadcast(ScalarLike(from, std::numeric_limits<double>::quiet_NaN()),
                  shape.dimensions());
    result_for_nan = BitcastConvertType(result_for_nan, int_type);

    // The sign bit is the MSB.
    const int64 sign_mask = int64{1} << (bitwidth - 1);
    // Discard the sign bit to make the result non-negative.
    auto from_abs = And(from_as_int, ScalarLike(from_as_int, ~sign_mask));
    auto to_abs = And(to_as_int, ScalarLike(to_as_int, ~sign_mask));

    // When both "from" and "to" are equal, the result is "to".
    // N.B. It would not make a difference if we chose the result to be "from".
    auto from_and_to_are_equal = Eq(from_as_int, to_as_int);
    auto result_for_equal = to_as_int;

    // When both "from" and "to" are both 0, the result is "to". This ensures we
    // get a zero signed like "to".
    auto from_is_zero = Eq(from_abs, ZerosLike(from_abs));
    auto to_is_zero = Eq(to_abs, ZerosLike(to_abs));
    auto result_for_both_zero = to_as_int;

    auto from_sign = And(from_as_int, ScalarLike(from_as_int, sign_mask));
    auto to_sign = And(to_as_int, ScalarLike(to_as_int, sign_mask));

    // If from == 0 && to != 0, we need to return the smallest subnormal number
    // signed like "to".
    auto result_for_from_zero_to_non_zero =
        Or(to_sign, ScalarLike(from_as_int, 1));

    // If the sign of "from" and "to" disagree:
    // - we need to make the magnitude of "from" smaller so that it is closer to
    //   zero.
    //
    // Otherwise the signs agree:
    // - "from" with a magnitude larger than "to" means we need to make the
    //   magnitude smaller.
    // - "from" with a magnitude smaller than "to" means we need to make the
    //   magnitude larger.
    // - "from" with the same magnitude and sign as "to" has already been
    //   handled.
    auto signs_disagree = Ne(from_sign, to_sign);
    auto from_magnitude_larger_than_to = Gt(from_abs, to_abs);
    auto result_has_smaller_magnitude =
        Or(from_magnitude_larger_than_to, signs_disagree);
    auto magnitude_adjustment =
        Select(result_has_smaller_magnitude,
               Broadcast(ScalarLike(from_as_int, -1), shape.dimensions()),
               Broadcast(ScalarLike(from_as_int, 1), shape.dimensions()));
    auto result = Add(from_as_int, magnitude_adjustment);
    // Handle from == ±0.
    result = Select(from_is_zero,
                    Select(to_is_zero, result_for_both_zero,
                           result_for_from_zero_to_non_zero),
                    result);
    // Handle from == to.
    result = Select(from_and_to_are_equal, result_for_equal, result);
    // Handle isnan(from) || isnan(to).
    result = Select(nan_input, result_for_nan, result);

    // Cast back to the original type.
    return BitcastConvertType(result, shape.element_type());
  });
}

XlaOp Logistic(XlaOp x) {
  auto half = xla::ScalarLike(x, 0.5);
  return half + half * xla::Tanh(half * x);
}

}  // namespace xla
