/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

static std::array<bool, 2> use_bf16_params{true, false};

// Test fixture for the HloEvaluator.
//
// In bf16 mode, all f32 shapes are converted to bf16 before running.
class HloEvaluatorTest : public HloTestBase {
 public:
  HloEvaluatorTest() : use_bfloat16_(false) { InitializeFftData(); }

  StatusOr<Literal> Evaluate(
      absl::Span<const Literal* const> arg_literals = {}) {
    if (use_bfloat16_) {
      HloElementTypeConverter(F32, BF16).Run(m_.get()).ValueOrDie();
    }
    return evaluator_.Evaluate(*m_->entry_computation(), arg_literals);
  }

  // Evaluate function that takes in a local module instead of using m_
  // that is in HloTestBase. Once m_ in HloTestBase is
  // removed, this should be the default Evaluate function.
  Literal EvaluateWithModule(
      HloModule* module, absl::Span<const Literal* const> arg_literals = {}) {
    if (use_bfloat16_) {
      HloElementTypeConverter(F32, BF16).Run(m_.get()).ValueOrDie();
    }
    return evaluator_.Evaluate(*module->entry_computation(), arg_literals)
        .ConsumeValueOrDie();
  }

  void TestUnaryOp(HloOpcode opcode, Literal expected, Literal input,
                   float aabs = 0) {
    HloComputation::Builder b(TestName());
    auto c1 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));
    b.AddInstruction(HloInstruction::CreateUnary(expected.shape(), opcode, c1));
    m_->AddEntryComputation(b.Build());

    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

    auto element_type = expected.shape().element_type();
    if (element_type == F32 || element_type == F64) {
      ErrorSpec error(aabs);
      EXPECT_TRUE(LiteralTestUtil::Near(expected, result, error));
    } else {
      EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
    }
  }

  void TestBinaryOp(HloOpcode opcode, Literal expected, Literal lhs,
                    Literal rhs) {
    HloComputation::Builder b(TestName());
    auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs)));
    auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs)));
    b.AddInstruction(
        HloInstruction::CreateBinary(expected.shape(), opcode, c1, c2));
    m_->AddEntryComputation(b.Build());

    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestTernaryOp(HloOpcode opcode, Literal expected, Literal src0,
                     Literal src1, Literal src2) {
    HloComputation::Builder b(TestName());
    auto operand0 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(src0)));
    auto operand1 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(src1)));
    auto operand2 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(src2)));
    b.AddInstruction(HloInstruction::CreateTernary(
        expected.shape(), opcode, operand0, operand1, operand2));
    m_->AddEntryComputation(b.Build());

    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  std::unique_ptr<HloComputation> MaxComputationScalarF32() {
    HloComputation::Builder max_computation("max");
    Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    auto param_lhs = max_computation.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
    auto param_rhs = max_computation.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
    max_computation.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kMaximum, param_lhs, param_rhs));
    return max_computation.Build();
  }

  void ReduceWindowMaxIotaTest(int window_size, int padding, int stride,
                               int window_dilation, int base_dilation,
                               const Literal& expected) {
    HloComputation::Builder b(TestName());

    // arg:
    // f32[4,4] {
    //  {  0,  1,  2,  3 },
    //  {  4,  5,  6,  7 },
    //  {  8,  9, 10, 11 },
    //  { 12, 13, 14, 15 }
    // }
    auto arg_array = absl::make_unique<Array2D<float>>(4, 4);
    arg_array->FillIota(0);
    auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

    HloInstruction* arg_instruction = b.AddInstruction(
        HloInstruction::CreateConstant(std::move(arg_literal)));
    auto init_value = b.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));
    auto max_func = m_->AddEmbeddedComputation(MaxComputationScalarF32());

    Window window;
    WindowDimension dim;
    dim.set_size(window_size);
    dim.set_stride(stride);
    dim.set_padding_low(padding);
    dim.set_padding_high(padding);
    dim.set_window_dilation(window_dilation);
    dim.set_base_dilation(base_dilation);
    *window.add_dimensions() = dim;
    *window.add_dimensions() = dim;

    int dim0 = expected.shape().dimensions(0);
    int dim1 = expected.shape().dimensions(1);
    Shape shape = ShapeUtil::MakeShape(F32, {dim0, dim1});
    b.AddInstruction(HloInstruction::CreateReduceWindow(
        shape, arg_instruction, init_value, window, max_func));

    m_->AddEntryComputation(b.Build());
    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

 protected:
  explicit HloEvaluatorTest(bool use_bfloat16) : use_bfloat16_(use_bfloat16) {
    InitializeFftData();
  }

  // Initializes data sets used in FFT tests below.
  void InitializeFftData();

  HloEvaluator evaluator_;

  const bool use_bfloat16_;
  std::unique_ptr<HloModule> m_ = CreateNewVerifiedModule();

  // Data sets used in FFT tests below.
  ErrorSpec fft_error_ = ErrorSpec(1e-4, 1e-5);
  Literal fft_c64x2x4x8_;
  Literal fft_c64x2x4x8_1d_;
  Literal fft_c64x2x4x8_2d_;
  Literal fft_c64x2x4x8_3d_;
};

// Lets you write TEST_Ps that run twice, once with and once without bf16.
class HloEvaluatorBf16Test : public ::testing::WithParamInterface<bool>,
                             public HloEvaluatorTest {
 protected:
  HloEvaluatorBf16Test() : HloEvaluatorTest(/*use_bfloat16=*/GetParam()) {}
};

INSTANTIATE_TEST_SUITE_P(HloEvaluatorTest_Instantiation, HloEvaluatorBf16Test,
                         ::testing::ValuesIn(use_bf16_params));

// Verifies that HloEvaluator evaluates a HLO instruction that performs clamp
// with 3 operands.
TEST_P(HloEvaluatorBf16Test, DoesClamp) {
  auto low = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});
  auto value = LiteralUtil::CreateR2<float>({{0.f, 5.f}, {0.f, 4.f}});
  auto high = LiteralUtil::CreateR2<float>({{2.f, 4.f}, {4.f, 4.f}});

  Shape shape = low.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({{0, 4}, {2, 4}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that clamping of int64 does not cause loss of precision
TEST_P(HloEvaluatorBf16Test, DoesClampInt64) {
  auto ones = [](int bits) { return (int64{1} << bits) - 1; };

  auto low =
      LiteralUtil::CreateR2<int64>({{0, ones(54)}, {ones(54), ones(58)}});
  auto value = LiteralUtil::CreateR2<int64>({{0, ones(56)}, {0, ones(58)}});
  auto high = LiteralUtil::CreateR2<int64>(
      {{ones(54), ones(55)}, {ones(56), ones(58)}});

  Shape shape = low.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected =
      LiteralUtil::CreateR2<int64>({{0, ones(55)}, {ones(54), ones(58)}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DISABLED_DoesClampSpecialBroadcast) {
  auto low = LiteralUtil::CreateR0<float>(0.f);
  auto value = LiteralUtil::CreateR2<float>({{-1.f, 0.f}, {1.f, 2.f}});
  auto high = LiteralUtil::CreateR0<float>(1.f);

  Shape shape = value.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({{0, 0}, {1, 1}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs select
// with 3 operands.
TEST_P(HloEvaluatorBf16Test, DoesSelect) {
  auto pred = LiteralUtil::CreateR2<bool>({{true, false}, {false, true}});
  auto on_true = LiteralUtil::CreateR2<float>({{2.f, 4.f}, {4.f, 4.f}});
  auto on_false = LiteralUtil::CreateR2<float>({{0.f, 5.f}, {0.f, 4.f}});

  Shape shape = on_true.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(pred)));
  auto c2 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(on_true)));
  auto c3 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(on_false)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kSelect, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  auto expected = LiteralUtil::CreateR2<float>({{2, 5}, {0, 4}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise addition with 2 operands.
TEST_F(HloEvaluatorTest, DoesAdd) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{3, 4}, {-96, 8}});
  TestBinaryOp(HloOpcode::kAdd, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise and with 2 operands.
TEST_P(HloEvaluatorBf16Test, DoesAnd) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{0, 0}, {4, 4}});
  TestBinaryOp(HloOpcode::kAnd, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise or with 2 operands.
TEST_F(HloEvaluatorTest, DoesOr) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{3, 4}, {-100, 4}});
  TestBinaryOp(HloOpcode::kOr, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise or with 2 operands.
TEST_F(HloEvaluatorTest, DoesXor) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{3, 4}, {-104, 0}});
  TestBinaryOp(HloOpcode::kXor, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise multiply with 2 operands.
TEST_F(HloEvaluatorTest, DoesMultiply) {
  auto lhs = LiteralUtil::CreateR2<int32>({{-1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int32>(
      {{std::numeric_limits<int32>::min(), 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int32>(
      {{std::numeric_limits<int32>::min(), 0}, {-400, 16}});
  TestBinaryOp(HloOpcode::kMultiply, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise divide with 2 operands.
TEST_F(HloEvaluatorTest, DoesDivideInt64) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{0, 0}, {-25, 1}});
  TestBinaryOp(HloOpcode::kDivide, std::move(expected), std::move(lhs),
               std::move(rhs));
}

TEST_F(HloEvaluatorTest, DoesClampS64) {
  auto low = LiteralUtil::CreateR1<int64>(
      {-8616761059752331528LL, 6780561065411491190LL, -8616761059752331528LL});
  auto value = LiteralUtil::CreateR1<int64>(
      {-6780561065411491190LL, 6780561065411491180LL, 4241131823772864090LL});
  auto high = LiteralUtil::CreateR1<int64>(
      {-6780561065411491180LL, 8616761059752331528LL, 3832151243857508051LL});
  auto expected = LiteralUtil::CreateR1<int64>(
      {-6780561065411491190LL, 6780561065411491190LL, 3832151243857508051LL});
  TestTernaryOp(HloOpcode::kClamp, std::move(expected), std::move(low),
                std::move(value), std::move(high));
}

TEST_P(HloEvaluatorBf16Test, DoesDivideDouble) {
  auto lhs = LiteralUtil::CreateR2<double>({{1.0, 0.0}, {-100.0, 4.0}});
  auto rhs = LiteralUtil::CreateR2<double>({{2.2, 4.0}, {4.0, 4.0}});
  auto expected =
      LiteralUtil::CreateR2<double>({{0.45454545454545453, 0}, {-25, 1}});
  TestBinaryOp(HloOpcode::kDivide, std::move(expected), std::move(lhs),
               std::move(rhs));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise abs op with 1 operand.
TEST_F(HloEvaluatorTest, DoesAbsR2) {
  auto operand = LiteralUtil::CreateR2<int64>({{1, -20}, {-100, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{1, 20}, {100, 4}});
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorBf16Test, DoesAbsR0) {
  auto operand = LiteralUtil::CreateR0<float>(-1.0f);
  auto expected = LiteralUtil::CreateR0<float>(1.0f);
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorBf16Test, DoesAbsR1WithZeroSize) {
  auto operand = LiteralUtil::CreateR1<float>({});
  auto expected = LiteralUtil::CreateR1<float>({});
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}

TEST_F(HloEvaluatorTest, DoesAbsC128) {
  auto x = LiteralUtil::CreateR0<complex128>({1, 2});
  auto expected_real = LiteralUtil::CreateR0<double>(2.23607);
  TestUnaryOp(HloOpcode::kAbs, std::move(expected_real), std::move(x), 3e-06);
}

TEST_F(HloEvaluatorTest, DoesNegateR2) {
  auto operand = LiteralUtil::CreateR2<int32>(
      {{0, std::numeric_limits<int32>::min()}, {-1, 4}});
  auto expected = LiteralUtil::CreateR2<int32>(
      {{0, std::numeric_limits<int>::min()}, {1, -4}});
  TestUnaryOp(HloOpcode::kNegate, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorBf16Test, DoesCosR2) {
  auto operand = LiteralUtil::CreateR2<float>({{0, M_PI}, {-M_PI, 2 * M_PI}});
  auto expected = LiteralUtil::CreateR2<float>({{1, -1}, {-1, 1}});
  TestUnaryOp(HloOpcode::kCos, std::move(expected), std::move(operand),
              use_bfloat16_ ? 0.031250 : 9.5367431640625E-7);
}
TEST_P(HloEvaluatorBf16Test, DoesSinR2) {
  auto operand = LiteralUtil::CreateR2<float>({{0, M_PI}, {-M_PI, 2 * M_PI}});
  auto expected = LiteralUtil::CreateR2<float>({{0, 0}, {0, 0}});
  TestUnaryOp(HloOpcode::kSin, std::move(expected), std::move(operand),
              use_bfloat16_ ? 0.031250 : 9.5367431640625E-7);
}
TEST_F(HloEvaluatorTest, DoesNotR2) {
  auto operand =
      LiteralUtil::CreateR2<int32>({{0, std::numeric_limits<int>::min()},
                                    {-1, std::numeric_limits<int>::max()}});
  auto expected =
      LiteralUtil::CreateR2<int32>({{-1, std::numeric_limits<int>::max()},
                                    {0, std::numeric_limits<int>::min()}});
  TestUnaryOp(HloOpcode::kNot, std::move(expected), std::move(operand));
}

TEST_F(HloEvaluatorTest, DoesRealC128) {
  auto x = LiteralUtil::CreateR1<complex128>({{1, 0}, {-100, 4}});
  auto expected_real = LiteralUtil::CreateR1<double>({1, -100});
  TestUnaryOp(HloOpcode::kReal, std::move(expected_real), std::move(x));
}

TEST_F(HloEvaluatorTest, DoesImagC128) {
  auto x = LiteralUtil::CreateR1<complex128>({{1, 0}, {-100, 4}});
  auto expected_imag = LiteralUtil::CreateR1<double>({0, 4});
  TestUnaryOp(HloOpcode::kImag, std::move(expected_imag), std::move(x));
}

// Verifies that HloEvaluator evaluates a HLO Computation with non-parameter nor
// constant operands.
TEST_F(HloEvaluatorTest, DoesTraverseInstructions) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto rhs2 = LiteralUtil::CreateR2<int64>({{1, -20}, {-100, 4}});
  std::vector<const Literal*> args = {&lhs, &rhs, &rhs2};

  Shape shape = ShapeUtil::MakeShape(S64, {2, 2});

  HloComputation::Builder b(TestName());
  auto param_lhs =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));
  auto param_rhs =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));
  auto lhs_instruction = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, param_lhs, param_rhs));

  auto param_rhs2 =
      b.AddInstruction(HloInstruction::CreateParameter(2, shape, "rhs2"));
  b.AddInstruction(HloInstruction::CreateBinary(shape, HloOpcode::kAdd,
                                                lhs_instruction, param_rhs2));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate(args));

  auto expected = LiteralUtil::CreateR2<int64>({{4, -16}, {-196, 12}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies Reshape operation is correctly evaluated.
TEST_F(HloEvaluatorTest, DoesReshape) {
  HloComputation::Builder b(TestName());
  const int64 dimensions[] = {11, 8, 7, 5, 9};
  TF_ASSERT_OK_AND_ASSIGN(auto literal,
                          LiteralUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, dimensions), 0.0, 1.0));
  auto literal_clone = literal.Clone();
  HloInstruction* literal_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {8, 7, 11, 9, 5});
  const int64 permutation[] = {1, 2, 0, 4, 3};
  b.AddInstruction(
      HloInstruction::CreateTranspose(shape, literal_instruction, permutation));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  using NativeT = typename primitive_util::PrimitiveTypeToNative<F32>::type;
  result.EachCell<NativeT>([&](absl::Span<const int64> indices, NativeT value) {
    std::vector<int64> rindexes = Permute(permutation, indices);
    EXPECT_NEAR(value, literal_clone.Get<NativeT>(rindexes), 0.031250);
  });
}

// Verifies Broadcast operation is correctly evaluated.
TEST_F(HloEvaluatorTest, DoesBroadcast) {
  HloComputation::Builder b(TestName());
  auto input_literal = LiteralUtil::CreateR2<int32>({{1, 2}, {3, 4}, {5, 6}});
  auto output_literal = LiteralUtil::CreateR3<int32>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{1, 2}, {3, 4}, {5, 6}}});
  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal.shape(), literal_instruction, {1, 2}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  EXPECT_TRUE(LiteralTestUtil::Equal(result, output_literal));
}

TEST_F(HloEvaluatorTest, DoesBroadcastScalar) {
  HloComputation::Builder b(TestName());
  auto input_literal = LiteralUtil::CreateR0<int32>(111);
  auto output_literal = LiteralUtil::CreateR2<int32>(
      {{111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}});

  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  // Broadcast dimension should be empty in the case of scalars.
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal.shape(), literal_instruction,
      /*broadcast_dimensions=*/{}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  EXPECT_TRUE(LiteralTestUtil::Equal(result, output_literal));
}

TEST_F(HloEvaluatorTest, DoesConcatenateSimple) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int64>({{-1, -2}, {100, 200}})));
  HloInstruction* operand2 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int64>({{-2, -3}, {-100, -200}})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {4, 2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<int64>(
      {{-1, -2}, {100, 200}, {-2, -3}, {-100, -200}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, ConcatenateHandlesShapeWithZeroElement) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64>({100, 200})));
  HloInstruction* operand2 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64>({})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR1<int64>({100, 200});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ConvertWithSameLayout) {
  HloComputation::Builder b(TestName());

  auto input_literal = LiteralUtil::CreateR2<int32>({{1, 2}, {3, 4}, {5, 6}});
  auto expected =
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  ASSERT_TRUE(LayoutUtil::LayoutsInShapesEqual(input_literal.shape(),
                                               expected.shape()));

  HloInstruction* constant = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateConvert(expected.shape(), constant));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

TEST_P(HloEvaluatorBf16Test, ConvertWithDifferentLayout) {
  HloComputation::Builder b(TestName());

  auto input_literal = LiteralUtil::CreateR2WithLayout<int32>(
      {{1, 2}, {3, 4}, {5, 6}}, LayoutUtil::MakeLayout({0, 1}));
  auto expected = LiteralUtil::CreateR2WithLayout<float>(
      {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, LayoutUtil::MakeLayout({1, 0}));
  ASSERT_FALSE(LayoutUtil::LayoutsInShapesEqual(input_literal.shape(),
                                                expected.shape()));

  HloInstruction* constant = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateConvert(expected.shape(), constant));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

PaddingConfig CreatePaddingConfig(
    std::initializer_list<std::array<int64, 3>> padding_dimensions) {
  PaddingConfig padding_config;

  for (auto& paddings_per_dim : padding_dimensions) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(paddings_per_dim[0]);
    dimension->set_edge_padding_high(paddings_per_dim[1]);
    dimension->set_interior_padding(paddings_per_dim[2]);
  }
  return padding_config;
}

TEST_F(HloEvaluatorTest, Pad2DIntegerArrayWithZeroDimension) {
  auto operand = LiteralUtil::CreateR2<int32>({{}, {}});
  HloComputation::Builder b(TestName());
  auto operand_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(operand)));

  constexpr int32 kPadValue = 10;
  auto pad_value = LiteralUtil::CreateR0<int32>(kPadValue);
  auto padding_value_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(pad_value)));

  auto padding_config = CreatePaddingConfig({{{1, 0, 2}}, {{0, 2, 1}}});
  Shape shape = ShapeUtil::MakeShape(S32, {5, 2});
  b.AddInstruction(HloInstruction::CreatePad(
      shape, operand_instruction, padding_value_instruction, padding_config));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<int32>(
      {{10, 10}, {10, 10}, {10, 10}, {10, 10}, {10, 10}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Pad4DFloatArrayWithInteriorPadding) {
  HloComputation::Builder b(TestName());

  Array4D<float> input_array(3, 2, 1, 1, {1, 2, 3, 4, 5, 6});
  auto input = LiteralUtil::CreateR4FromArray4D<float>(input_array);
  HloInstruction* input_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));
  constexpr float kPadValue = 1.5;
  auto pad_value = LiteralUtil::CreateR0<float>(kPadValue);
  HloInstruction* pad_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(pad_value)));

  Shape shape = ShapeUtil::MakeShape(F32, {8, 5, 1, 1});
  auto r4_padding_on_dim0_dim1 =
      CreatePaddingConfig({{{1, 0, 2}}, {{0, 2, 1}}, {{0, 0, 0}}, {{0, 0, 0}}});
  b.AddInstruction(HloInstruction::CreatePad(
      shape, input_instruction, pad_instruction, r4_padding_on_dim0_dim1));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected_array = absl::make_unique<Array4D<float>>(8, 5, 1, 1);
  expected_array->Fill(kPadValue);
  (*expected_array)(1, 0, 0, 0) = 1.0f;
  (*expected_array)(1, 2, 0, 0) = 2.0f;
  (*expected_array)(4, 0, 0, 0) = 3.0f;
  (*expected_array)(4, 2, 0, 0) = 4.0f;
  (*expected_array)(7, 0, 0, 0) = 5.0f;
  (*expected_array)(7, 2, 0, 0) = 6.0f;

  auto expected = LiteralUtil::CreateR4FromArray4D<float>(*expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, NegativePadding2D) {
  HloComputation::Builder b(TestName());

  // input_array:
  // f32[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto input_array = absl::make_unique<Array2D<float>>(4, 3);
  input_array->FillUnique(1.0f);
  auto input = LiteralUtil::CreateR2FromArray2D<float>(*input_array);
  HloInstruction* input_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));

  auto pad_value_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.718f)));

  auto r2_padding_on_dim0_dim1 =
      CreatePaddingConfig({{{-1, -2, 0}}, {{-2, 4, 0}}});
  Shape shape = ShapeUtil::MakeShape(F32, {1, 5});
  b.AddInstruction(HloInstruction::CreatePad(shape, input_instruction,
                                             pad_value_instruction,
                                             r2_padding_on_dim0_dim1));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  // f32[1,5] { 7.0, 2.718, 2.718, 2.718, 2.718 }
  auto expected_array = absl::make_unique<Array2D<float>>(1, 5);
  (*expected_array)(0, 0) = 7.0f;
  (*expected_array)(0, 1) = 2.718f;
  (*expected_array)(0, 2) = 2.718f;
  (*expected_array)(0, 3) = 2.718f;
  (*expected_array)(0, 4) = 2.718f;
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(*expected_array);

  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, ErrorSpec(0.031250)));
}

TEST_P(HloEvaluatorBf16Test, NegativeAndInteriorPadding2D) {
  HloComputation::Builder b(TestName());

  // f32[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto input_array = absl::make_unique<Array2D<float>>(4, 3);
  input_array->FillUnique(1.0f);
  auto input = LiteralUtil::CreateR2FromArray2D<float>(*input_array);
  HloInstruction* input_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));

  auto pad_value_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.718f)));

  PaddingConfig padding_config = MakeNoPaddingConfig(2);

  // Negative padding that results in zero dimensions.
  auto r2_padding_on_dim0_dim1 =
      CreatePaddingConfig({{{-2, -5, 1}}, {{-2, 4, 2}}});

  Shape shape = ShapeUtil::MakeShape(F32, {0, 9});
  b.AddInstruction(HloInstruction::CreatePad(shape, input_instruction,
                                             pad_value_instruction,
                                             r2_padding_on_dim0_dim1));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected_array = absl::make_unique<Array2D<float>>(0, 9);
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(*expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DotRank2AndRank1) {
  HloComputation::Builder b(TestName());

  // lhs:
  // f32[4,1] {
  //  { 1 },
  //  { 2 },
  //  { 3 },
  //  { 4 },
  // }
  auto lhs_array = absl::make_unique<Array2D<float>>(4, 1);
  lhs_array->FillUnique(1.0f);
  auto lhs_literal = LiteralUtil::CreateR2FromArray2D<float>(*lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  // rhs:
  // f32[2] { 1, 2 },
  auto rhs_literal = LiteralUtil::CreateR2<float>({{1, 2}});
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {4, 2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums,
                                             DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  // clang-format off
  auto expected_array = Array2D<float>({
      {1.f, 2.f},
      {2.f, 4.f},
      {3.f, 6.f},
      {4.f, 8.f},
  });
  // clang-format on
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DotRank1AndRank2) {
  HloComputation::Builder b(TestName());

  // lhs:
  // f32[3]
  //  { 1, 2, 3 },
  auto lhs_literal = LiteralUtil::CreateR1<float>({1, 2, 3});
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  // rhs:
  // f32[3,2] {
  //  { 1, 2 },
  //  { 3, 4 },
  //  { 5, 6 },
  // }
  auto rhs_array = absl::make_unique<Array2D<float>>(3, 2);
  rhs_array->FillUnique(1.0f);
  auto rhs_literal = LiteralUtil::CreateR2FromArray2D<float>(*rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums,
                                             DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR1<float>({22.f, 28.f});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DotRank2AndRank2) {
  HloComputation::Builder b(TestName());

  // lhs:
  // f32[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto lhs_array = absl::make_unique<Array2D<float>>(4, 3);
  lhs_array->FillUnique(1.0f);
  auto lhs_literal = LiteralUtil::CreateR2FromArray2D<float>(*lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  // rhs:
  // f32[3,2] {
  //  { 1, 2 },
  //  { 3, 4 },
  //  { 5, 6 },
  // }
  auto rhs_array = absl::make_unique<Array2D<float>>(3, 2);
  rhs_array->FillUnique(1.0f);
  auto rhs_literal = LiteralUtil::CreateR2FromArray2D<float>(*rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {4, 2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums,
                                             DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected_array = Array2D<float>({
      {22.f, 28.f},
      {58.f, 76.f},
      {94.f, 124.f},
      {130.f, 172.f},
  });
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DotRank4AndRank4) {
  HloComputation::Builder b(TestName());

  auto lhs_array = absl::make_unique<Array4D<float>>(2, 2, 3, 1);
  lhs_array->FillIota(1.0f);
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(*lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  auto rhs_array = absl::make_unique<Array4D<float>>(2, 2, 3, 1);
  rhs_array->FillIota(2.0f);
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(*rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 1, 1});
  DotDimensionNumbers dot_dnums;

  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(2);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums,
                                             DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  float expected_1 = 0;
  for (float i = 1.0f; i < 7.0f; ++i) {
    expected_1 += i * i + i;
  }
  float expected_2 = 0;
  for (float i = 7.0f; i < 13.0f; ++i) {
    expected_2 += i * i + i;
  }
  auto expected_array = Array3D<float>({{{expected_1}}, {{expected_2}}});
  auto expected = LiteralUtil::CreateR3FromArray3D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, SimpleConv1D) {
  HloComputation::Builder b(TestName());

  Array3D<float> lhs_array = {{{1, 2, 3}}};
  auto lhs_literal = LiteralUtil::CreateR3FromArray3D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array3D<float> rhs_array = {{{3.f, 4.f}}};
  auto rhs_literal = LiteralUtil::CreateR3FromArray3D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_output_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);

  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 3});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  Array3D<float> expected_array = {{{11.f, 18.f, 9.f}}};
  auto expected = LiteralUtil::CreateR3FromArray3D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Simple4x4Conv2DWith2x2Kernel) {
  HloComputation::Builder b(TestName());

  Array4D<float> lhs_array(1, 1, 4, 4);
  // clang-format off
  lhs_array.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 2);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums =
      XlaBuilder::CreateDefaultConvDimensionNumbers(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 4, 4});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  Array4D<float> expected_array(1, 1, 4, 4);
  // clang-format off
  expected_array.FillWithYX(Array2D<float>({
    {100, 126, 152,  76},
    {204, 230, 256, 124},
    {308, 334, 360, 172},
    {149, 160, 171,  80},
  }));
  // clang-format on
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Conv2DGeneralDimensionsReversed) {
  HloComputation::Builder b(TestName());

  // clang-format off
  // Input dimensions: [feature=2, height=3, batch=1, width=4]
  Array4D<float> input({
    {{{1, 2, 3, 4}},
     {{5, 6, 7, 8}},
     {{9, 10, 11, 12}}},
    {{{13, 14, 15, 16}},
     {{17, 18, 19, 20}},
     {{21, 22, 23, 24}}}
  });
  // Weight dimensions:
  // [kernel_output_feature=1, width=3, kernel_input_feature=2, height=3]
  Array4D<float> weight({{
    {{1, 7, 13},
     {4, 10, 16}},
    {{2, 8, 14},
     {5, 11, 17}},
    {{3, 9, 15},
     {6, 12, 18}}
  }});
  // clang-format on

  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(input);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(weight);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));
  rhs_instruction = b.AddInstruction(HloInstruction::CreateReverse(
      rhs_instruction->shape(), rhs_instruction, {3, 1}));

  Window window;
  WindowDimension dim;
  dim.set_size(3);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  dim.set_window_reversal(true);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(2);
  dnums.set_output_batch_dimension(2);
  dnums.set_input_feature_dimension(0);
  dnums.set_output_feature_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(1);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  // clang-format off
  // Result dimensions: [feature=1, height=1, batch=1, width=2]
  Array4D<float> expected_array({{{{2514, 2685}}}});
  Array4D<float> expected_array_bf16({{{{2512, 2688}}}});
  // clang-format on
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(
      use_bfloat16_ ? expected_array_bf16 : expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Conv2DGeneralDimensions) {
  HloComputation::Builder b(TestName());

  // clang-format off
  // Input dimensions: [feature=2, height=3, batch=1, width=4]
  Array4D<float> input({
    {{{1, 2, 3, 4}},
     {{5, 6, 7, 8}},
     {{9, 10, 11, 12}}},
    {{{13, 14, 15, 16}},
     {{17, 18, 19, 20}},
     {{21, 22, 23, 24}}}
  });
  // Weight dimensions:
  // [kernel_output_feature=1, width=3, kernel_input_feature=2, height=3]
  Array4D<float> weight({{
    {{1, 7, 13},
     {4, 10, 16}},
    {{2, 8, 14},
     {5, 11, 17}},
    {{3, 9, 15},
     {6, 12, 18}}
  }});
  // clang-format on

  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(input);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(weight);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(3);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(2);
  dnums.set_output_batch_dimension(2);
  dnums.set_input_feature_dimension(0);
  dnums.set_output_feature_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(1);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  // clang-format off
  // Result dimensions: [feature=1, height=1, batch=1, width=2]
  Array4D<float> expected_array({{{{2514, 2685}}}});
  Array4D<float> expected_array_bf16({{{{2512, 2688}}}});
  // clang-format on
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(
      use_bfloat16_ ? expected_array_bf16 : expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DilatedBaseConv2DWithHighPadding) {
  HloComputation::Builder b(TestName());

  Array4D<float> lhs_array(1, 1, 4, 4);
  // clang-format off
  lhs_array.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 2);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(2);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums =
      XlaBuilder::CreateDefaultConvDimensionNumbers(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 7, 7});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  Array4D<float> expected_array(1, 1, 7, 7);
  expected_array.FillWithYX(Array2D<float>({
      {5, 12, 10, 18, 15, 24, 20},
      {35, 48, 42, 56, 49, 64, 56},
      {25, 36, 30, 42, 35, 48, 40},
      {63, 80, 70, 88, 77, 96, 84},
      {45, 60, 50, 66, 55, 72, 60},
      {91, 112, 98, 120, 105, 128, 112},
      {65, 84, 70, 90, 75, 96, 80},
  }));
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DilatedBaseConv2DWithLowAndHighPadding) {
  HloComputation::Builder b(TestName());

  Array4D<float> lhs_array(1, 1, 4, 4);
  // clang-format off
  lhs_array.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 2);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(1);
  dim.set_padding_high(1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(2);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums =
      XlaBuilder::CreateDefaultConvDimensionNumbers(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 8, 8});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  Array4D<float> expected_array(1, 1, 8, 8);
  expected_array.FillWithYX(Array2D<float>({
      {8, 7, 16, 14, 24, 21, 32, 28},
      {6, 5, 12, 10, 18, 15, 24, 20},
      {40, 35, 48, 42, 56, 49, 64, 56},
      {30, 25, 36, 30, 42, 35, 48, 40},
      {72, 63, 80, 70, 88, 77, 96, 84},
      {54, 45, 60, 50, 66, 55, 72, 60},
      {104, 91, 112, 98, 120, 105, 128, 112},
      {78, 65, 84, 70, 90, 75, 96, 80},
  }));
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test,
       DilatedWindowAndBaseConv2DWithDifferentLowAndHighPaddingAndStrides) {
  HloComputation::Builder b(TestName());

  Array4D<float> lhs_array(1, 1, 4, 4);
  // clang-format off
  lhs_array.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 3);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6, 7},
    {8, 9, 10},
  }));
  // clang-format on
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(2);
  dim.set_padding_high(2);
  dim.set_window_dilation(2);
  dim.set_base_dilation(2);
  *window.add_dimensions() = dim;
  dim.set_size(3);
  dim.set_stride(3);
  dim.set_padding_low(2);
  dim.set_padding_high(-1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(3);
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums =
      XlaBuilder::CreateDefaultConvDimensionNumbers(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 9, 3});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  Array4D<float> expected_array(1, 1, 9, 3);
  expected_array.FillWithYX(Array2D<float>({
      {10, 20, 30},
      {0, 0, 0},
      {57, 74, 91},
      {0, 0, 0},
      {125, 142, 159},
      {0, 0, 0},
      {193, 210, 227},
      {0, 0, 0},
      {91, 98, 105},
  }));
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Conv2DGroupedConvolution) {
  HloComputation::Builder b(TestName());
  std::vector<int64> input_dims = {1, 2, 2, 4};
  std::vector<int64> filter_dims = {2, 2, 2, 8};
  Shape input_shape = ShapeUtil::MakeShapeWithType<float>(input_dims);
  Shape filter_shape = ShapeUtil::MakeShapeWithType<float>(filter_dims);
  // Tensorflow dimension numbers for 2D convolution.
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  std::vector<float> input_elems(ShapeUtil::ElementsIn(input_shape));
  std::iota(input_elems.begin(), input_elems.end(), -7);
  auto input_r1 = LiteralUtil::CreateR1<float>(input_elems);
  auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input_r4)));

  std::vector<float> filter_elems(ShapeUtil::ElementsIn(filter_shape));
  std::iota(filter_elems.begin(), filter_elems.end(), -31);
  auto filter_r1 = LiteralUtil::CreateR1<float>(filter_elems);
  auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(filter_r4)));

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 8});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction,
      /*feature_group_count=*/2, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  Array4D<float> expected_array(1, 1, 1, 8);
  expected_array.FillWithYX(
      Array2D<float>({{668, 664, 660, 656, 668, 680, 692, 704}}));
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Initialization of data sets for FFT tests:

void HloEvaluatorTest::InitializeFftData() {
  // clang-format off
  fft_c64x2x4x8_ = LiteralUtil::CreateR3<complex64>({
    {{{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0},
      {4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0}, {7.0, 0.0}},
     {{0.0, 0.0}, {0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0},
      {0.0, 4.0}, {0.0, 5.0}, {0.0, 6.0}, {0.0, 7.0}},
     {{0.0, 7.0}, {1.0, 6.0}, {2.0, 5.0}, {3.0, 4.0},
      {4.0, 3.0}, {5.0, 2.0}, {6.0, 1.0}, {7.0, 0.0}},
     {{7.0, 0.0}, {6.0, 1.0}, {5.0, 2.0}, {4.0, 3.0},
      {3.0, 4.0}, {2.0, 5.0}, {1.0, 6.0}, {0.0, 7.0}}},
    {{{-4.0, 0.0}, {-3.0, 0.0}, {-2.0, 0.0}, {-1.0, 0.0},
      {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}},
     {{0.0, -4.0}, {0.0, -3.0}, {0.0, -2.0}, {0.0, -1.0},
      {0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}, {0.0, 4.0}},
     {{3.5, 3.5}, {-1.707107, -0.707107}, {-1.0, -0.0}, {-0.707107, 0.292893},
      {-0.5, 0.5}, {-0.292893, 0.707107}, {0.0, 1.0}, {0.707107, 1.707107}},
     {{3.5, 3.5}, {1.707107, 0.707107}, {1.0, 0.0}, {0.707107, -0.292893},
      {0.5, -0.5}, {0.292893, -0.707107}, {-0.0, -1.0}, {-0.707107, -1.707107}}}
  });
  fft_c64x2x4x8_1d_ = LiteralUtil::CreateR3<complex64>({
    {{{28.0, 0.0}, {-4.0, 9.656854}, {-4.0, 4.0}, {-4.0, 1.656854},
      {-4.0, 0.0}, {-4.0, -1.656854}, {-4.0, -4.0}, {-4.0, -9.656854}},
     {{0.0, 28.0}, {-9.656854, -4.0}, {-4.0, -4.0}, {-1.656854, -4.0},
      {0.0, -4.0}, {1.656854, -4.0}, {4.0, -4.0}, {9.656854, -4.0}},
     {{28.0, 28.0}, {5.656854, 13.656854}, {0.0, 8.0}, {-2.343146, 5.656854},
      {-4.0, 4.0}, {-5.656854, 2.343146}, {-8.0, -0.0}, {-13.656854, -5.656854}},  // NOLINT
     {{28.0, 28.0}, {-5.656854, -13.656854}, {-0.0, -8.0}, {2.343146, -5.656854},  // NOLINT
      {4.0, -4.0}, {5.656854, -2.343146}, {8.0, 0.0}, {13.656854, 5.656854}}},
    {{{0.0, 0.0}, {-5.0, 12.071068}, {-4.0, 4.0}, {-5.0, 2.071068},
      {-4.0, 0.0}, {-5.0, -2.071068}, {-4.0, -4.0}, {-5.0, -12.071068}},
     {{0.0, 0.0}, {-12.071068, -5.0}, {-4.0, -4.0}, {-2.071068, -5.0},
      {0.0, -4.0}, {2.071068, -5.0}, {4.0, -4.0}, {12.071068, -5.0}},
     {{0.0, 7.0}, {1.0, 6.0}, {2.0, 5.0}, {3.0, 4.0},
      {4.0, 3.0}, {5.0, 2.0}, {6.0, 1.0}, {7.0, 0.0}},
     {{7.0, 0.0}, {6.0, 1.0}, {5.0, 2.0}, {4.0, 3.0},
      {3.0, 4.0}, {2.0, 5.0}, {1.0, 6.0}, {0.0, 7.0}}}
  });
  fft_c64x2x4x8_2d_ = LiteralUtil::CreateR3<complex64>({
    {{{84.0, 84.0}, {-13.656854, 5.656854}, {-8.0, 0.0}, {-5.656854, -2.343146},
      {-4.0, -4.0}, {-2.343146, -5.656854}, {0.0, -8.0}, {5.656854, -13.656854}},  // NOLINT
     {{0.0, 0.0}, {0.0, -0.0}, {0.0, 0.0}, {0.0, 0.0},
      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
     {{28.0, -28.0}, {16.970562, 40.970562}, {0.0, 24.0}, {-7.029438, 16.970562},      // NOLINT
      {-12.0, 12.0}, {-16.970562, 7.029438}, {-24.0, 0.0}, {-40.970562, -16.970562}},  // NOLINT
     {{0.0, -56.0}, {-19.313708, -8.0}, {-8.0, -8.0}, {-3.313708, -8.0},
      {0.0, -8.0}, {3.313708, -8.0}, {8.0, -8.0}, {19.313708, -8.0}}},
    {{{7.0, 7.0}, {-10.071068, 14.071068}, {-1.0, 7.0}, {-0.071068, 4.071068},
      {3.0, 3.0}, {4.071068, -0.071068}, {7.0, -1.0}, {14.071068, -10.071068}},
     {{0.0, 0.0}, {-12.0, 24.142136}, {-12.0, 8.0}, {-16.0, 4.142136},
      {-16.0, 0.0}, {-20.0, -4.142136}, {-20.0, -8.0}, {-24.0, -24.142136}},
     {{-7.0, 7.0}, {2.071068, 22.071068}, {-3.0, 11.0}, {-3.928932, 8.071068},
      {-3.0, 3.0}, {-4.071068, -0.071068}, {-3.0, -5.0}, {-10.071068, -14.071068}},  // NOLINT
     {{0.0, -14.0}, {0.0, -12.0}, {0.0, -10.0}, {0.0, -8.0},
      {0.0, -6.0}, {0.0, -4.0}, {0.0, -2.0}, {0.0, 0.0}}}
  });
  fft_c64x2x4x8_3d_ = LiteralUtil::CreateR3<complex64>({
    {{{91.0, 91.0}, {-23.727922, 19.727922}, {-9.0, 7.0}, {-5.727922, 1.727922},
      {-1.0, -1.0}, {1.727922, -5.727922}, {7.0, -9}, {19.727922, -23.727922}},
     {{0.0, 0.0}, {-12.0, 24.142136}, {-12.0, 8.0}, {-16.0, 4.142136},
      {-16.0, 0.0}, {-20.0, -4.142136}, {-20.0, -8.0}, {-24.0, -24.142136}},
     {{21.0, -21.0}, {19.041630, 63.041630}, {-3.0, 35.0}, {-10.958370, 25.041630},     // NOLINT
      {-15.0, 15.0}, {-21.041630, 6.958370}, {-27.0, -5.0}, {-51.041630, -31.041630}},  // NOLINT
     {{0.0, -70.0}, {-19.313708, -20.0}, {-8.0, -18.0}, {-3.313708, -16.0},
      {0.0, -14.0}, {3.313708, -12.0}, {8.0, -10.0}, {19.313708, -8.0}}},
    {{{77.0, 77.0}, {-3.585786, -8.414214}, {-7.0, -7.0}, {-5.585786, -6.414214},   // NOLINT
      {-7.0, -7.0}, {-6.414214, -5.585786}, {-7.0, -7.0}, {-8.414214, -3.585786}},  // NOLINT
     {{0.0, 0.0}, {12.0, -24.142136}, {12.0, -8.0}, {16.0, -4.142136},
      {16.0, 0.0}, {20.0, 4.142136}, {20.0, 8.0}, {24.0, 24.142136}},
     {{35.0, -35.0}, {14.899494, 18.899494}, {3.0, 13.0}, {-3.100506, 8.899494},
      {-9.0, 9.0}, {-12.899494, 7.100506}, {-21.0, 5.0}, {-30.899494, -2.899494}},  // NOLINT
     {{0.0, -42.0}, {-19.313708, 4.0}, {-8.0, 2.0}, {-3.313708, 0.0},
      {0.0, -2.0}, {3.313708, -4.0}, {8.0, -6.0}, {19.313708, -8.0}}}
  });
  // clang-format on
}

// Simple FFT tests:

TEST_F(HloEvaluatorTest, 1D_FFT_4_on_c64x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[4] parameter(0)
  ROOT fft = c64[4] fft(operand), fft_type=FFT, fft_length={4}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>(
      {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}});
  auto expected = LiteralUtil::CreateR1<complex64>(
      {{10.0, 0.0}, {-2.0, 2.0}, {-2.0, 0.0}, {-2.0, -2.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IFFT_4_on_c64x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[4] parameter(0)
  ROOT ifft = c64[4] fft(operand), fft_type=IFFT, fft_length={4}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>(
      {{10.0, 0.0}, {-2.0, 2.0}, {-2.0, 0.0}, {-2.0, -2.0}});
  auto expected = LiteralUtil::CreateR1<complex64>(
      {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_RFFT_4_on_f32x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[4] parameter(0)
  ROOT rfft = c64[3] fft(operand), fft_type=RFFT, fft_length={4}
}
)";
  auto input = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  auto expected =
      LiteralUtil::CreateR1<complex64>({{10.0, 0.0}, {-2.0, 2.0}, {-2.0, 0.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IRFFT_4_on_c64x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3] parameter(0)
  ROOT irfft = f32[4] fft(operand), fft_type=IRFFT, fft_length={4}
}
)";
  auto input =
      LiteralUtil::CreateR1<complex64>({{10.0, 0.0}, {-2.0, 2.0}, {-2.0, 0.0}});
  auto expected = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// 1D FFT tests:

TEST_F(HloEvaluatorTest, 1D_FFT_8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT fft = c64[2, 4, 8] fft(operand), fft_type=FFT, fft_length={8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_1d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_1d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IFFT_8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT ifft = c64[2, 4, 8] fft(operand), fft_type=IFFT, fft_length={8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_1d_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_RFFT_8_on_f32x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[8] parameter(0)
  ROOT rfft = c64[5] fft(operand), fft_type=RFFT, fft_length={8}
}
)";
  auto input =
      LiteralUtil::CreateR1<float>({1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1});
  auto expected = LiteralUtil::CreateR1<complex64>({{39.6, 0.0},
                                                    {-3.6, 8.691169},
                                                    {-3.6, 3.6},
                                                    {-3.6, 1.491169},
                                                    {-3.6, 0.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IRFFT_8_on_c64x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[5] parameter(0)
  ROOT irfft = f32[8] fft(operand), fft_type=IRFFT, fft_length={8}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>({{39.6, 0.0},
                                                 {-3.6, 8.691169},
                                                 {-3.6, 3.6},
                                                 {-3.6, 1.491169},
                                                 {-3.6, 0.0}});
  auto expected =
      LiteralUtil::CreateR1<float>({1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_RFFT_9_on_f32x9) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[9] parameter(0)
  ROOT rfft = c64[5] fft(operand), fft_type=RFFT, fft_length={9}
}
)";
  auto input = LiteralUtil::CreateR1<float>(
      {1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.9});
  auto expected = LiteralUtil::CreateR1<complex64>({{49.5, 0.0},
                                                    {-3.360560, 11.705792},
                                                    {-3.893717, 5.712929},
                                                    {-4.5, 3.117691},
                                                    {-4.895723, 1.021942}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IRFFT_9_on_c64x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[5] parameter(0)
  ROOT irfft = f32[9] fft(operand), fft_type=IRFFT, fft_length={9}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>({{49.5, 0.0},
                                                 {-3.360560, 11.705792},
                                                 {-3.893717, 5.712929},
                                                 {-4.5, 3.117691},
                                                 {-4.895723, 1.021942}});
  auto expected = LiteralUtil::CreateR1<float>(
      {1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.9});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// 2D FFT tests:

TEST_F(HloEvaluatorTest, 2D_FFT_4x8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT fft = c64[2, 4, 8] fft(operand), fft_type=FFT, fft_length={4, 8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_2d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_2d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_IFFT_4x8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT ifft = c64[2, 4, 8] fft(operand), fft_type=IFFT, fft_length={4, 8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_2d_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_RFFT_3x8_on_f32x3x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 8] parameter(0)
  ROOT rfft = c64[3, 5] fft(operand), fft_type=RFFT, fft_length={3, 8}
}
)";
  auto input =
      LiteralUtil::CreateR2<float>({{1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1},
                                    {8.1, 7.2, 6.3, 5.4, 4.5, 3.6, 2.7, 1.8},
                                    {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8}});
  auto expected = LiteralUtil::CreateR2<complex64>({{{118.8, 0.0},
                                                     {-4.4, 10.622540},
                                                     {-4.4, 4.4},
                                                     {-4.4, 1.822540},
                                                     {-4.4, 0.0}},
                                                    {{0.0, 0.0},
                                                     {-19.926162, 0.797280},
                                                     {-10.128203, -3.728203},
                                                     {-6.069756, -5.602720},
                                                     {-3.2, -6.928203}},
                                                    {{0.0, 0.0},
                                                     {13.526162, 14.653687},
                                                     {3.728203, 10.128203},
                                                     {-0.330244, 8.253687},
                                                     {-3.2, 6.928203}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_IRFFT_3x8_on_c64x3x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 5] parameter(0)
  ROOT irfft = f32[3, 8] fft(operand), fft_type=IRFFT, fft_length={3, 8}
}
)";
  auto input = LiteralUtil::CreateR2<complex64>({{{118.8, 0.0},
                                                  {-4.4, 10.622540},
                                                  {-4.4, 4.4},
                                                  {-4.4, 1.822540},
                                                  {-4.4, 0.0}},
                                                 {{0.0, 0.0},
                                                  {-19.926162, 0.797280},
                                                  {-10.128203, -3.728203},
                                                  {-6.069756, -5.602720},
                                                  {-3.2, -6.928203}},
                                                 {{0.0, 0.0},
                                                  {13.526162, 14.653687},
                                                  {3.728203, 10.128203},
                                                  {-0.330244, 8.253687},
                                                  {-3.2, 6.928203}}});
  auto expected =
      LiteralUtil::CreateR2<float>({{1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1},
                                    {8.1, 7.2, 6.3, 5.4, 4.5, 3.6, 2.7, 1.8},
                                    {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_RFFT_3x9_on_f32x3x9) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 9] parameter(0)
  ROOT rfft = c64[3, 5] fft(operand), fft_type=RFFT, fft_length={3, 9}
}
)";
  auto input = LiteralUtil::CreateR2<float>(
      {{1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1},
       {9.1, 8.2, 7.3, 6.4, 5.5, 4.6, 3.7, 2.8, 1.9},
       {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9}});
  auto expected = LiteralUtil::CreateR2<complex64>({{{148.5, 0.0},
                                                     {-4.95, 13.600013},
                                                     {-4.95, 5.899180},
                                                     {-4.95, 2.857884},
                                                     {-4.95, 0.872819}},
                                                    {{0.0, 0.0},
                                                     {-25.014467, 2.096690},
                                                     {-12.888800, -3.503916},
                                                     {-8.1, -5.715768},
                                                     {-4.974333, -7.159452}},
                                                    {{0.0, 0.0},
                                                     {17.814467, 17.685147},
                                                     {5.688800, 12.084542},
                                                     {0.9, 9.872690},
                                                     {-2.225667, 8.429006}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_IRFFT_3x9_on_c64x3x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 5] parameter(0)
  ROOT irfft = f32[3, 9] fft(operand), fft_type=IRFFT, fft_length={3, 9}
}
)";
  auto input = LiteralUtil::CreateR2<complex64>({{{148.5, 0.0},
                                                  {-4.95, 13.600013},
                                                  {-4.95, 5.899180},
                                                  {-4.95, 2.857884},
                                                  {-4.95, 0.872819}},
                                                 {{0.0, 0.0},
                                                  {-25.014467, 2.096690},
                                                  {-12.888800, -3.503916},
                                                  {-8.1, -5.715768},
                                                  {-4.974333, -7.159452}},
                                                 {{0.0, 0.0},
                                                  {17.814467, 17.685147},
                                                  {5.688800, 12.084542},
                                                  {0.9, 9.872690},
                                                  {-2.225667, 8.429006}}});
  auto expected = LiteralUtil::CreateR2<float>(
      {{1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1},
       {9.1, 8.2, 7.3, 6.4, 5.5, 4.6, 3.7, 2.8, 1.9},
       {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// 3D FFT tests:

TEST_F(HloEvaluatorTest, 3D_FFT_2x4x8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT fft = c64[2, 4, 8] fft(operand), fft_type=FFT, fft_length={2, 4, 8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_3d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_3d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_IFFT_2x4x8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT ifft = c64[2, 4, 8] fft(operand), fft_type=IFFT, fft_length={2, 4, 8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_3d_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_RFFT_3x3x4_on_f32x3x3x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 3, 4] parameter(0)
  ROOT rfft = c64[3, 3, 3] fft(operand), fft_type=RFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<float>(
      {{{1.8, 2.7, 3.6, 4.5}, {8.1, 7.2, 6.3, 5.4}, {1.1, 2.2, 3.3, 4.4}},
       {{5.4, 6.3, 7.2, 8.1}, {4.5, 3.6, 2.7, 1.8}, {5.5, 6.6, 7.7, 8.8}},
       {{-1.8, -2.7, -3.6, -4.5},
        {-5.4, -6.3, -7.2, -8.1},
        {1.9, 2.9, 3.9, 4.9}}});
  auto expected = LiteralUtil::CreateR3<complex64>(
      {{{{92.8, 0.0}, {-2.8, 2.8}, {-2.8, 0.0}},
        {{-5.9, 35.160631}, {-11.519100, -8.919100}, {-1.3, -10.219100}},
        {{-5.9, -35.160631}, {8.919100, 11.519100}, {-1.3, 10.219100}}},
       {{{29.5, -81.579593}, {1.390897, 5.190897}, {-1.9, 3.290897}},
        {{-25.1, -49.017038}, {1.044486, 4.844486}, {-1.9, 2.944486}},
        {{11.8, 27.712813}, {1.517691, 4.717691}, {-1.6, 3.117691}}},
       {{{29.5, 81.579593}, {-5.190897, -1.390897}, {-1.9, -3.290897}},
        {{11.8, -27.712813}, {-4.717691, -1.517691}, {-1.6, -3.117691}},
        {{-25.1, 49.017038}, {-4.844486, -1.044486}, {-1.9, -2.944486}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_IRFFT_3x3x4_on_c64x3x3x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 3] parameter(0)
  ROOT irfft = f32[3, 3, 4] fft(operand), fft_type=IRFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{92.8, 0.0}, {-2.8, 2.8}, {-2.8, 0.0}},
        {{-5.9, 35.160631}, {-11.519100, -8.919100}, {-1.3, -10.219100}},
        {{-5.9, -35.160631}, {8.919100, 11.519100}, {-1.3, 10.219100}}},
       {{{29.5, -81.579593}, {1.390897, 5.190897}, {-1.9, 3.290897}},
        {{-25.1, -49.017038}, {1.044486, 4.844486}, {-1.9, 2.944486}},
        {{11.8, 27.712813}, {1.517691, 4.717691}, {-1.6, 3.117691}}},
       {{{29.5, 81.579593}, {-5.190897, -1.390897}, {-1.9, -3.290897}},
        {{11.8, -27.712813}, {-4.717691, -1.517691}, {-1.6, -3.117691}},
        {{-25.1, 49.017038}, {-4.844486, -1.044486}, {-1.9, -2.944486}}}});
  auto expected = LiteralUtil::CreateR3<float>(
      {{{1.8, 2.7, 3.6, 4.5}, {8.1, 7.2, 6.3, 5.4}, {1.1, 2.2, 3.3, 4.4}},
       {{5.4, 6.3, 7.2, 8.1}, {4.5, 3.6, 2.7, 1.8}, {5.5, 6.6, 7.7, 8.8}},
       {{-1.8, -2.7, -3.6, -4.5},
        {-5.4, -6.3, -7.2, -8.1},
        {1.9, 2.9, 3.9, 4.9}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_RFFT_3x3x5_on_f32x3x3x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 3, 5] parameter(0)
  ROOT rfft = c64[3, 3, 3] fft(operand), fft_type=RFFT, fft_length={3, 3, 5}
}
)";
  auto input = LiteralUtil::CreateR3<float>({{{1.8, 2.7, 3.6, 4.5, 5.4},
                                              {8.1, 7.2, 6.3, 5.4, 4.5},
                                              {1.1, 2.2, 3.3, 4.4, 5.5}},
                                             {{5.4, 6.3, 7.2, 8.1, 9.0},
                                              {4.5, 3.6, 2.7, 1.8, 0.9},
                                              {5.5, 6.6, 7.7, 8.8, 9.9}},
                                             {{-1.8, -2.7, -3.6, -4.5, -5.4},
                                              {-5.4, -6.3, -7.2, -8.1, -9.0},
                                              {1.9, 2.9, 3.9, 4.9, 5.9}}});
  auto expected = LiteralUtil::CreateR3<complex64>(
      {{{{119.5, 0.0}, {-3.5, 4.817337}, {-3.5, 1.137219}},
        {{-5.75, 56.724664}, {-19.206730, -10.537254}, {-5.775483, -12.245880}},
        {{-5.75, -56.724664}, {15.956730, 15.010495}, {2.525483, 13.301869}}},
       {{{39.25, -106.088112}, {3.286913, 7.382528}, {-1.038404, 4.885305}},
        {{-29.0, -64.951905}, {2.690922, 6.949515}, {-1.179098, 4.452292}},
        {{16.75, 30.743902}, {3.363918, 6.649878}, {-0.733751, 4.546954}}},
       {{{39.25, 106.088112}, {-8.036913, -0.844714}, {-3.711596, -3.341936}},
        {{16.75, -30.743902}, {-7.363918, -1.144350}, {-3.266249, -3.247275}},
        {{-29.0, 64.951905}, {-7.440922, -0.411701}, {-3.570902, -2.908924}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_IRFFT_3x3x5_on_c64x3x3x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 3] parameter(0)
  ROOT irfft = f32[3, 3, 5] fft(operand), fft_type=IRFFT, fft_length={3, 3, 5}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{119.5, 0.0}, {-3.5, 4.817337}, {-3.5, 1.137219}},
        {{-5.75, 56.724664}, {-19.206730, -10.537254}, {-5.775483, -12.245880}},
        {{-5.75, -56.724664}, {15.956730, 15.010495}, {2.525483, 13.301869}}},
       {{{39.25, -106.088112}, {3.286913, 7.382528}, {-1.038404, 4.885305}},
        {{-29.0, -64.951905}, {2.690922, 6.949515}, {-1.179098, 4.452292}},
        {{16.75, 30.743902}, {3.363918, 6.649878}, {-0.733751, 4.546954}}},
       {{{39.25, 106.088112}, {-8.036913, -0.844714}, {-3.711596, -3.341936}},
        {{16.75, -30.743902}, {-7.363918, -1.144350}, {-3.266249, -3.247275}},
        {{-29.0, 64.951905}, {-7.440922, -0.411701}, {-3.570902, -2.908924}}}});
  auto expected = LiteralUtil::CreateR3<float>({{{1.8, 2.7, 3.6, 4.5, 5.4},
                                                 {8.1, 7.2, 6.3, 5.4, 4.5},
                                                 {1.1, 2.2, 3.3, 4.4, 5.5}},
                                                {{5.4, 6.3, 7.2, 8.1, 9.0},
                                                 {4.5, 3.6, 2.7, 1.8, 0.9},
                                                 {5.5, 6.6, 7.7, 8.8, 9.9}},
                                                {{-1.8, -2.7, -3.6, -4.5, -5.4},
                                                 {-5.4, -6.3, -7.2, -8.1, -9.0},
                                                 {1.9, 2.9, 3.9, 4.9, 5.9}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// FFT tests with non-default data layout:

TEST_F(HloEvaluatorTest, 1D_FFT_8_on_c64x2x4x8_with_layout) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8]{0, 2, 1} parameter(0)
  ROOT fft = c64[2, 4, 8]{1, 2, 0} fft(operand), fft_type=FFT, fft_length={8}
}
)";
  auto input = fft_c64x2x4x8_.Relayout(LayoutUtil::MakeLayout({0, 2, 1}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_1d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_1d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_FFT_4x8_on_c64x2x4x8_with_layout) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8]{2, 0, 1} parameter(0)
  ROOT fft = c64[2, 4, 8]{1, 0, 2} fft(operand), fft_type=FFT, fft_length={4, 8}
}
)";
  auto input = fft_c64x2x4x8_.Relayout(LayoutUtil::MakeLayout({2, 0, 1}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_2d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_2d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_FFT_2x4x8_on_c64x2x4x8_with_layout) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8]{1, 2, 0} parameter(0)
  ROOT fft =
    c64[2, 4, 8]{0, 2, 1} fft(operand), fft_type=FFT, fft_length={2, 4, 8}
}
)";
  auto input = fft_c64x2x4x8_.Relayout(LayoutUtil::MakeLayout({1, 2, 0}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_3d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_3d_, result, fft_error_));
}

// FFT tests with unusual parameters:

// Zero-length transform.
TEST_F(HloEvaluatorTest, 1D_FFT_0_on_c64x1x1x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 1] parameter(0)
  ROOT fft = c64[1, 1, 1, 1] fft(operand), fft_type=FFT, fft_length={0}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}}}});
  auto expected = LiteralUtil::CreateR4<complex64>({{{{{0.0, 0.0}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Zero-length axis.
TEST_F(HloEvaluatorTest, 1D_FFT_1_on_c64x1x1x1x0) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 0] parameter(0)
  ROOT fft = c64[1, 1, 1, 0] fft(operand), fft_type=FFT, fft_length={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto input,
      LiteralUtil::CreateR4<complex64>({{{{}}}}).Reshape({1, 1, 1, 0}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// Some/all dimensions have length 1.
TEST_F(HloEvaluatorTest, 1D_FFT_1_on_c64x1x1x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 1] parameter(0)
  ROOT fft = c64[1, 1, 1, 1] fft(operand), fft_type=FFT, fft_length={1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// Zero-length transform.
TEST_F(HloEvaluatorTest, 3D_FFT_1x0x1_on_c64x1x1x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 1] parameter(0)
  ROOT fft = c64[1, 1, 1, 1] fft(operand), fft_type=FFT, fft_length={1, 0, 1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}}}});
  auto expected = LiteralUtil::CreateR4<complex64>({{{{{0.0, 0.0}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Zero-length axis.
TEST_F(HloEvaluatorTest, 3D_FFT_1x1x1_on_c64x0x1x0x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[0, 1, 0, 1] parameter(0)
  ROOT fft = c64[0, 1, 0, 1] fft(operand), fft_type=FFT, fft_length={1, 1, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto input,
      LiteralUtil::CreateR4<complex64>({{{{}}}}).Reshape({0, 1, 0, 1}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// Some/all dimensions have length 1.
TEST_F(HloEvaluatorTest, 3D_FFT_1x1x1_on_c64x1x1x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 1] parameter(0)
  ROOT fft = c64[1, 1, 1, 1] fft(operand), fft_type=FFT, fft_length={1, 1, 1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// Some/all dimensions have length 1.
TEST_F(HloEvaluatorTest, 3D_FFT_3x1x1_on_c64x1x3x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 3, 1, 1] parameter(0)
  ROOT fft = c64[1, 3, 1, 1] fft(operand), fft_type=FFT, fft_length={3, 1, 1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>(
      {{{{{42.24, 24.42}}}, {{{-42.24, 24.42}}}, {{{42.24, -24.42}}}}});
  auto expected =
      LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}},
                                         {{{84.5367, 97.5818}}},
                                         {{{-0.0566792, -48.7418}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Some/all dimensions have length 1.
TEST_F(HloEvaluatorTest, 3D_IFFT_3x1x1_on_c64x1x3x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 3, 1, 1] parameter(0)
  ROOT ifft = c64[1, 3, 1, 1] fft(operand), fft_type=IFFT, fft_length={3, 1, 1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}},
                                                  {{{84.5367, 97.5818}}},
                                                  {{{-0.0566792, -48.7418}}}}});
  auto expected = LiteralUtil::CreateR4<complex64>(
      {{{{{42.24, 24.42}}}, {{{-42.24, 24.42}}}, {{{42.24, -24.42}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Odd transform length.
TEST_F(HloEvaluatorTest, 1D_FFT_5_on_c64x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[5] parameter(0)
  ROOT fft = c64[5] fft(operand), fft_type=FFT, fft_length={5}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>(
      {{1.0, 5.0}, {2.0, 4.0}, {3.0, 3.0}, {4.0, 2.0}, {5.0, 1.0}});
  auto expected = LiteralUtil::CreateR1<complex64>({{15.0, 15.0},
                                                    {0.940955, 5.94095},
                                                    {-1.6877, 3.3123},
                                                    {-3.3123, 1.6877},
                                                    {-5.94095, -0.940955}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Odd transform length.
TEST_F(HloEvaluatorTest, 1D_IFFT_5_on_c64x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[5] parameter(0)
  ROOT ifft = c64[5] fft(operand), fft_type=IFFT, fft_length={5}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>({{15.0, 15.0},
                                                 {0.940955, 5.94095},
                                                 {-1.6877, 3.3123},
                                                 {-3.3123, 1.6877},
                                                 {-5.94095, -0.940955}});
  auto expected = LiteralUtil::CreateR1<complex64>(
      {{1.0, 5.0}, {2.0, 4.0}, {3.0, 3.0}, {4.0, 2.0}, {5.0, 1.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 1D_FFT_4_on_zero_c64x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[4] parameter(0)
  ROOT fft = c64[4] fft(operand), fft_type=FFT, fft_length={4}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>(
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 3D_FFT_3x3x4_on_zero_c64x3x3x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 4] parameter(0)
  ROOT fft = c64[3, 3, 4] fft(operand), fft_type=FFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 3D_IFFT_3x3x4_on_zero_c64x3x3x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 4] parameter(0)
  ROOT ifft = c64[3, 3, 4] fft(operand), fft_type=IFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 3D_RFFT_3x3x4_on_zero_f32x3x3x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 3, 4] parameter(0)
  ROOT rfft = c64[3, 3, 3] fft(operand), fft_type=RFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<float>(
      {{{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
       {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
       {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}}});
  auto expected = LiteralUtil::CreateR3<complex64>(
      {{{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 3D_IRFFT_3x3x4_on_zero_c64x3x3x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 3] parameter(0)
  ROOT irfft = f32[3, 3, 4] fft(operand), fft_type=IRFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}}});
  auto expected = LiteralUtil::CreateR3<float>(
      {{{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
       {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
       {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Input values, for which IRFFT discards non-zero imaginary parts.
TEST_F(HloEvaluatorTest, 2D_IRFFT_3x4_on_c64x3x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3] parameter(0)
  ROOT irfft = f32[3, 4] fft(operand), fft_type=IRFFT, fft_length={3, 4}
}
)";
  auto input =
      LiteralUtil::CreateR2<complex64>({{{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}},
                                        {{3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}},
                                        {{6.0, 0.0}, {7.0, 0.0}, {8.0, 0.0}}});
  auto expected =
      LiteralUtil::CreateR2<float>({{4.0, -0.5, 0.0, -0.5},
                                    {-1.5, 0.433013, 0.0, -0.433013},
                                    {-1.5, -0.433013, 0.0, 0.433013}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

class HloEvaluatorPreciseReduceTest : public HloTestBase {};

// Tests that Reduce doesn't lose precision when adding many numbers (because
// it accumulates its result in a double).
TEST_F(HloEvaluatorPreciseReduceTest, AddReductionPrecisionTest) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder b(TestName());

  constexpr int kNumElements = 1 << 25;  // float += 1 saturates at 1<<24
  std::vector<float> v(kNumElements, 1.0f);
  HloInstruction* arg_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(v)));
  HloInstruction* init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = m->AddEmbeddedComputation(add_computation.Build());

  HloInstruction* reduce_instruction = b.AddInstruction(
      HloInstruction::CreateReduce(scalar_shape, arg_instruction, init_value,
                                   /*dimensions_to_reduce=*/{0}, add_func));
  m->AddEntryComputation(b.Build());

  HloEvaluator hlo_eval;
  Literal result = hlo_eval.Evaluate(reduce_instruction).ConsumeValueOrDie();
  LiteralTestUtil::ExpectR0Equal<float>(kNumElements, result);
}

// Reducing many numbers should be fast because it doesn't create
// intermediate Literals; the microbenchmark should finish in < 1 msec.
void BM_ReducePrecisely(int num_iters) {
  tensorflow::testing::StopTiming();
  HloComputation::Builder b("BM_ReducePrecisely");
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsFromFlags());
  HloModule module("BM_ReducePrecisely", config);

  constexpr int kNumElements = 1 << 25;  // float += 1 saturates at 1<<24
  std::vector<float> v(kNumElements, 1.0f);
  HloInstruction* arg_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(v)));
  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = module.AddEmbeddedComputation(add_computation.Build());

  HloInstruction* reduce_instruction = b.AddInstruction(
      HloInstruction::CreateReduce(scalar_shape, arg_instruction, init_value,
                                   /*dimensions_to_reduce=*/{0}, add_func));
  module.AddEntryComputation(b.Build());

  HloEvaluator hlo_eval;
  tensorflow::testing::StartTiming();
  hlo_eval.Evaluate(reduce_instruction).ConsumeValueOrDie();
  tensorflow::testing::StopTiming();
}

BENCHMARK(BM_ReducePrecisely);

TEST_P(HloEvaluatorBf16Test, ReduceAdd) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = absl::make_unique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = m_->AddEmbeddedComputation(add_computation.Build());

  Shape shape = ShapeUtil::MakeShape(F32, {2});
  b.AddInstruction(
      HloInstruction::CreateReduce(shape, arg_instruction, init_value,
                                   /*dimensions_to_reduce=*/{1}, add_func));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR1<float>({6, 18});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMax) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = absl::make_unique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));
  auto max_func = m_->AddEmbeddedComputation(MaxComputationScalarF32());

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  Shape shape = ShapeUtil::MakeShape(F32, {1, 2});
  b.AddInstruction(HloInstruction::CreateReduceWindow(
      shape, arg_instruction, init_value, window, max_func));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({{6, 7}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaWindowDilation) {
  auto expected = LiteralUtil::CreateR2<float>({{10, 11}, {14, 15}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/1,
      /*window_dilation=*/2,
      /*base_dilation=*/1,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaStrideWindowDilation) {
  auto expected = LiteralUtil::CreateR2<float>({{10}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/2,
      /*window_dilation=*/2,
      /*base_dilation=*/1,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaBaseDilation) {
  auto expected = LiteralUtil::CreateR2<float>({{0, 1, 1, 2, 2, 3},
                                                {4, 5, 5, 6, 6, 7},
                                                {4, 5, 5, 6, 6, 7},
                                                {8, 9, 9, 10, 10, 11},
                                                {8, 9, 9, 10, 10, 11},
                                                {12, 13, 13, 14, 14, 15}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/1,
      /*window_dilation=*/1,
      /*base_dilation=*/2,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaStrideBaseDilation) {
  auto expected =
      LiteralUtil::CreateR2<float>({{0, 1, 2}, {4, 5, 6}, {8, 9, 10}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/2,
      /*window_dilation=*/1,
      /*base_dilation=*/2,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaStrideBothDilation) {
  auto expected =
      LiteralUtil::CreateR2<float>({{5, 6, 7}, {9, 10, 11}, {13, 14, 15}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/2,
      /*window_dilation=*/2,
      /*base_dilation=*/2,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaPaddingStrideBaseDilation) {
  // The base is dilated first, and then padding is applied, hence this result.
  auto expected =
      LiteralUtil::CreateR2<float>({{0, 2, 3}, {8, 10, 11}, {12, 14, 15}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/3,
      /*padding=*/1,
      /*stride=*/3,
      /*window_dilation=*/1,
      /*base_dilation=*/2,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowAdd) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = absl::make_unique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = m_->AddEmbeddedComputation(add_computation.Build());

  Window window;
  WindowDimension dim;
  dim.set_size(1);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(1);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(HloInstruction::CreateReduceWindow(
      shape, arg_instruction, init_value, window, add_func));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({{1, 3, 5}, {5, 11, 13}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowAdd6D) {
  HloComputation::Builder b(TestName());

  // arg: f32[4,4,4,4,4,4] full of ones. Using small dims to limit run-time.
  std::vector<int64> input_dims(6, 4);
  Literal arg_literal =
      LiteralUtil::CreateFullWithDescendingLayout<float>(input_dims, 1.0f);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = m_->AddEmbeddedComputation(add_computation.Build());

  Window window;

  WindowDimension trivial_dim;
  trivial_dim.set_size(1);
  trivial_dim.set_stride(1);
  trivial_dim.set_padding_low(0);
  trivial_dim.set_padding_high(0);
  trivial_dim.set_window_dilation(1);
  trivial_dim.set_base_dilation(1);

  WindowDimension active_dim;
  active_dim.set_size(2);
  active_dim.set_stride(1);
  active_dim.set_padding_low(0);
  active_dim.set_padding_high(0);
  active_dim.set_window_dilation(1);
  active_dim.set_base_dilation(1);

  *window.add_dimensions() = trivial_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = trivial_dim;
  *window.add_dimensions() = trivial_dim;

  Shape shape = ShapeUtil::MakeShape(F32, {4, 3, 3, 3, 4, 4});
  b.AddInstruction(HloInstruction::CreateReduceWindow(
      shape, arg_instruction, init_value, window, add_func));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  std::vector<int64> output_dims = {4, 3, 3, 3, 4, 4};
  Literal result_literal =
      LiteralUtil::CreateFullWithDescendingLayout<float>(output_dims, 8.0f);
  EXPECT_TRUE(LiteralTestUtil::Equal(result_literal, result));
}

TEST_P(HloEvaluatorBf16Test, StridedSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[3,5] {
  //  { 1, 2, 3, 4, 5 },
  //  { 9, 10, 11, 12, 13 },
  //  { 17, 18, 19, 20, 21 },
  // }
  auto operand_array = absl::make_unique<Array2D<float>>(3, 5);
  operand_array->FillUnique(1.0f);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 1});
  b.AddInstruction(HloInstruction::CreateSlice(shape, operand,
                                               /*start_indices=*/{0, 2},
                                               /*limit_indices=*/{3, 5},
                                               /*strides=*/{2, 3}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({
      {3},
      {19},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DynamicSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,4] {
  //  { 1, 2, 3, 4 },
  //  { 5, 6, 7, 8 },
  // }
  auto operand_array = absl::make_unique<Array2D<float>>(2, 4);
  operand_array->FillUnique(1.0f);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto zero = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto one = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(
      HloInstruction::CreateDynamicSlice(shape, operand, {zero, one}, {2, 3}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({
      {2, 3, 4},
      {6, 7, 8},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that the HloEvaluator's implementation goes along with existing
// backends' behavior, although this is not required by the spec.
TEST_P(HloEvaluatorBf16Test, DynamicSliceModSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,4] {
  //  { 1, 2, 3, 4 },
  //  { 5, 6, 7, 8 },
  // }
  auto operand_array = absl::make_unique<Array2D<float>>(2, 4);
  operand_array->FillUnique(1.0f);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto two = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(2)));
  auto one = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(
      HloInstruction::CreateDynamicSlice(shape, operand, {two, one}, {2, 3}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({
      {2, 3, 4},
      {6, 7, 8},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DynamicSliceUpdate) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = absl::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto zero = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto one = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));

  auto update = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<double>({{-2.0, -3.0}, {-6.0, -7.0}})));

  Shape shape = ShapeUtil::MakeShape(F64, {2, 3});
  b.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      shape, operand, update, {zero, one}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<double>({
      {1, -2, -3},
      {5, -6, -7},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, SetAndGetTuples) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = absl::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);
  auto operand_literal2 =
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array);

  HloInstruction* operand2 = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal2)));
  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64>({0, 1})));

  auto tuple =
      b.AddInstruction(HloInstruction::CreateTuple({operand1, operand2}));

  Shape shape = ShapeUtil::MakeShape(F64, {2, 3});
  b.AddInstruction(HloInstruction::CreateGetTupleElement(shape, tuple, 1));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<double>({
      {1, 2, 3},
      {5, 6, 7},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, SetAndGetNestedTuples) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = absl::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);

  HloInstruction* operand2 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array)));
  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64>({0, 1})));

  auto tuple1 =
      b.AddInstruction(HloInstruction::CreateTuple({operand1, operand2}));
  auto tuple2 =
      b.AddInstruction(HloInstruction::CreateTuple({operand2, operand2}));

  auto outer_tuple =
      b.AddInstruction(HloInstruction::CreateTuple({tuple1, tuple2}));

  b.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple2->shape(), outer_tuple, 1));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto result_inner_literal =
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array);
  auto expected =
      LiteralUtil::MakeTuple({&result_inner_literal, &result_inner_literal});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Reverse) {
  HloComputation::Builder b(TestName());

  // Input shape is float[4x3x2x1].
  // clang-format off
  Array4D<float> input({
    {{{1.0f}, {2.0f}},
     {{3.0f}, {4.0f}},
     {{5.0f}, {6.0f}}},
    {{{7.0f}, {8.0f}},
     {{9.0f}, {10.0f}},
     {{11.0f}, {12.0f}}},
    {{{13.0f}, {14.0f}},
     {{15.0f}, {16.0f}},
     {{17.0f}, {18.0f}}},
    {{{19.0f}, {20.0f}},
     {{21.0f}, {22.0f}},
     {{23.0f}, {24.0f}}},
  });
  // clang-format on
  auto operand_literal = LiteralUtil::CreateR4FromArray4D<float>(input);
  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  const Shape shape = ShapeUtil::MakeShape(F32, {4, 3, 2, 1});
  b.AddInstruction(HloInstruction::CreateReverse(shape, operand, {0, 1}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  // clang-format off
  auto expected = LiteralUtil::CreateR4FromArray4D<float>({
    {{{23.0f}, {24.0f}},
     {{21.0f}, {22.0f}},
     {{19.0f}, {20.0f}}},

    {{{17.0f}, {18.0f}},
     {{15.0f}, {16.0f}},
     {{13.0f}, {14.0f}}},

    {{{11.0f}, {12.0f}},
     {{9.0f}, {10.0f}},
     {{7.0f}, {8.0f}}},

    {{{5.0f}, {6.0f}},
     {{3.0f}, {4.0f}},
     {{1.0f}, {2.0f}}},
  });
  // clang-format on

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, EvaluateWithSubstitutions) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  HloInstruction* param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* square = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, param0, param0));
  HloInstruction* add = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, square));

  // Evaluate add with param0 = {1, 2, 3, 4}, square = {10, 20, 30, 40}.
  HloEvaluator evaluator;
  Literal param0_literal = LiteralUtil::CreateR1<float>({1, 2, 3, 4});
  Literal square_literal = LiteralUtil::CreateR1<float>({10, 20, 30, 40});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.EvaluateWithSubstitutions(
          add, {{param0, &param0_literal}, {square, &square_literal}}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({11, 22, 33, 44}), result));
}

// Check that EvaluateWithSubstitutions works if one of the operands to the op
// we're evaluating is a constant.
TEST_P(HloEvaluatorBf16Test, EvaluateWithSubstitutionsWithConstantOperand) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  HloInstruction* param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* square = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, param0, param0));
  HloInstruction* constant = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 2, 3, 4})));
  HloInstruction* add = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, constant, square));

  // Evaluate add with square = {10, 20, 30, 40}.
  HloEvaluator evaluator;
  Literal square_literal = LiteralUtil::CreateR1<float>({10, 20, 30, 40});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.EvaluateWithSubstitutions(add, {{square, &square_literal}}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({11, 22, 33, 44}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherV1) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 3}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {7, 8, 9}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherV2) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 3}, {4, 6}, {7, 9}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherMultipleBatchDims) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,3,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 2}, {2, 1}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR3<int32>(
          {{{1, 3}, {4, 6}, {7, 9}}, {{3, 2}, {6, 5}, {9, 8}}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherNd) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1,2}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{-1, 1}, {-4, 4}}), result));
}

TEST_F(HloEvaluatorTest,
       EvaluateGather_TensorFlowGatherNdNonDefaultIndexVectorDim) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1,2}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{-2, 2}, {-1, 1}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_DynamicSlice) {
  const char* hlo_text = R"(
HloModule DynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[1,1] gather(operand, indices),
      offset_dims={0,1},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({1, 1});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32>({{5}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_BatchDynamicSlice) {
  const char* hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,1,1] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{2, 1}, {1, 1}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR3<int32>({{{8}}, {{5}}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,0] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand = LiteralUtil::CreateR2<int32>({{}, {}, {}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32>({{}, {}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_NoOutputWindowDims) {
  const string hlo_text = R"(
HloModule GatherXd

ENTRY main {
  operand = s32[3] parameter(0)
  indices = s32[2,2,1] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal operand = LiteralUtil::CreateR1<int32>({0, 1, 2});
  Literal start_indices =
      LiteralUtil::CreateR3<int32>({{{0}, {1}}, {{2}, {1}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{0, 1}, {2, 1}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatterV1_Update) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{10, 20, 30}, {4, 5, 6}, {70, 80, 90}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatterV2_Update) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV2

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={0},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32>({{10, 30}, {40, 60}, {70, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{10, 2, 30}, {40, 5, 60}, {70, 8, 90}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_Add) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{11, 22, 33}, {4, 5, 6}, {77, 88, 99}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_Mul) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

mul_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT mul = s32[] multiply(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=mul_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{10, 40, 90}, {4, 5, 6}, {490, 640, 810}}),
      result));
}

TEST_P(HloEvaluatorBf16Test, EvaluateScatter_TensorFlowScatter_F32) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(f32[] lhs, f32[] rhs)
}

ENTRY main {
  operand = f32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f32[2,3] parameter(2)
  ROOT scatter = f32[3,3] scatter(operand, indices, updates),
      to_apply=add_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand = LiteralUtil::CreateR2<float>(
      {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({2, 1});
  Literal updates =
      LiteralUtil::CreateR2<float>({{0.4, 1.1, 0.7}, {2.3, 3.1, 1.6}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Near(
      LiteralUtil::CreateR2<float>(
          {{1.1, 2.2, 3.3}, {6.7, 8.6, 8.2}, {8.1, 9.9, 10.6}}),
      result, ErrorSpec{0.1, 0.01}));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_RepeatedIndices) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({1, 1});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {84, 105, 126}, {7, 8, 9}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_MultipleBatchDims) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterMultipleBatchDims

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=2
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 2}, {2, 1}});
  Literal updates = LiteralUtil::CreateR3<int32>(
      {{{10, 30}, {40, 60}, {70, 90}}, {{5, 5}, {5, 5}, {5, 5}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{11, 7, 38}, {44, 10, 71}, {77, 13, 104}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatterNd) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNd

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32>({{-10, 10}, {-40, 40}});
  Literal expected =
      LiteralUtil::CreateR3<int32>({{{-10, 10}, {-2, 2}, {-3, 3}},  //
                                    {{-40, 40}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest,
       EvaluateScatter_TensorFlowScatterNd_NonDefaultIndexVectorDim) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNdNonDefaultIndexVectorDim

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32>({{-10, 10}, {-20, 20}});
  Literal expected =
      LiteralUtil::CreateR3<int32>({{{-20, 20}, {-10, 10}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},      //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_DynamicUpdateSlice) {
  const char* hlo_text = R"(
HloModule DynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[1,1] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={0,1},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({1, 1});
  Literal updates = LiteralUtil::CreateR2<int32>({{10}});
  Literal expected =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 10, 6}, {7, 8, 9}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_BatchDynamicUpdateSlice) {
  const char* hlo_text = R"(
HloModule BatchDynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,1,1] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{2, 1}, {1, 1}});
  Literal updates = LiteralUtil::CreateR3<int32>({{{10}}, {{20}}});
  Literal expected =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 20, 6}, {7, 10, 9}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter_ZeroDimBounds

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,0] parameter(2)
  ROOT scatter = s32[3,0] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand = LiteralUtil::CreateR2<int32>({{}, {}, {}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{}, {}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(operand, result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_NoUpdateWindowDims) {
  const string hlo_text = R"(
HloModule Scatter_NoUpdateWindowDims

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3] parameter(0)
  indices = s32[2,2,1] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=2
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal operand = LiteralUtil::CreateR1<int32>({0, 1, 2});
  Literal scatter_indices =
      LiteralUtil::CreateR3<int32>({{{0}, {1}}, {{2}, {1}}});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20}, {30, 40}});
  Literal expected = LiteralUtil::CreateR1<int32>({10, 61, 32});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_NegativeIndices) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter_NegativeIndices

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // No updates should happen for the negative indices.
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({-1, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {77, 88, 99}}),
      EvaluateWithModule(module.get(),
                         {&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_OobIndices) {
  const string hlo_text = R"(
HloModule BatchDynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[3,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // No updates should happen for the OOB indices.
  Literal scatter_indices = LiteralUtil::CreateR2<int32>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483647, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 30, 60}, {7, 20, 9}}),
      EvaluateWithModule(module.get(),
                         {&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_OobUpdateWindow) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNd_OobUpdateWindow

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[1,2] parameter(1)
  updates = s32[1,2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 2}});
  Literal updates = LiteralUtil::CreateR3<int32>({{{-10, 10}, {-40, 40}}});
  // Given the update window size of 2,2 and the index of 0,2, the update window
  // will be OOB. So, nothing should be updated.
  Literal expected = operand.Clone();
  EXPECT_TRUE(LiteralTestUtil::Equal(
      expected, EvaluateWithModule(module.get(),
                                   {&operand, &scatter_indices, &updates})));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise comparison with 2 bfloat16 operands.
TEST_F(HloEvaluatorTest, DoesCompareBF16) {
  // lhs >= rhs
  auto lhs = LiteralUtil::CreateR2<bfloat16>(
      {{bfloat16(0.25), bfloat16(0.35), bfloat16(0.125)},
       {bfloat16(-0.25), bfloat16(-0.35), bfloat16(-0.125)}});
  auto rhs = LiteralUtil::CreateR2<bfloat16>(
      {{bfloat16(0.5), bfloat16(0.125), bfloat16(0.125)},
       {bfloat16(0.25), bfloat16(-0.375), bfloat16(-0.127)}});
  auto expected =
      LiteralUtil::CreateR2<bool>({{false, true, true}, {false, true, true}});

  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs)));
  b.AddInstruction(HloInstruction::CreateCompare(expected.shape(), c1, c2,
                                                 ComparisonDirection::kGe));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Bf16Reduction) {
  const string hlo_text = R"(
HloModule Bf16Reduction

add_bf16 (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(bf16[] lhs, bf16[] rhs)
}

ENTRY main {
  arg0 = bf16[4]{0} parameter(0)
  init = bf16[] constant(0)
  ROOT %reduce = bf16[] reduce(arg0, init), dimensions={0}, to_apply=add_bf16
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal arg = LiteralUtil::CreateR1<bfloat16>(
      {bfloat16(1.0f), bfloat16(3.0f), bfloat16(-2.0f), bfloat16(42.0f)});
  Literal expected = LiteralUtil::CreateR0<bfloat16>(bfloat16(44.0f));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&arg}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, MixedPrecisionReduction) {
  const string hlo_text = R"(
HloModule MixedPrecisionReduction

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  arg0 = f32[4]{0} parameter(0)
  init = f32[] constant(0)
  ROOT %reduce = bf16[] reduce(arg0, init), dimensions={0}, to_apply=add_f32
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal arg = LiteralUtil::CreateR1<float>({1.0f, 3.0f, -2.0f, 42.0f});
  Literal expected = LiteralUtil::CreateR0<bfloat16>(bfloat16(44.0f));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&arg}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, DontFailOnCallUnimplementedOps) {
  // Infeed triggers unimplemented error within HandleCall, and we verify that
  // the Evaluator does fail in such case.
  const string hlo_text = R"(
HloModule DontFailOnCall

call {
  token0 = token[] after-all()
  ROOT infeed = ((u32[3]{0}, pred[]), token[]) infeed(token0)
}

ENTRY main {
  ROOT result = ((u32[3]{0}, pred[]), token[]) call(), to_apply=call
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto statusor = Evaluate();
  EXPECT_FALSE(statusor.status().ok());
}

TEST_F(HloEvaluatorTest, DontFailOnFusionWithUnimplementedOps) {
  // Infeed triggers unimplemented error within HandleFusion, and we verify that
  // the Evaluator does fail in such case.
  const string hlo_text = R"(
HloModule DontFailOnFusion

fused_computation {
  token0 = token[] after-all()
  ROOT infeed = ((u32[3]{0}, pred[]), token[]) infeed(token0)
}

ENTRY main {
  ROOT result = ((u32[3]{0}, pred[]), token[]) fusion(), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto statusor = Evaluate();
  EXPECT_FALSE(statusor.status().ok());
}

TEST_P(HloEvaluatorBf16Test, SliceWithDifferentLayout) {
  // Regression test for b/114735354.
  const string hlo_text = R"(
HloModule SliceWithDifferentLayout

ENTRY main {
  arg = f32[2,2,2]{0,1,2} parameter(0)
  ROOT %slice = f32[2,2,2]{1,0,2} slice(f32[2,2,2]{0,1,2} %arg), slice={[0:2], [0:2], [0:2]}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal arg = LiteralUtil::CreateR3WithLayout<float>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
      LayoutUtil::MakeLayout({0, 1, 2}));
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&arg}));
  EXPECT_TRUE(LiteralTestUtil::Equal(arg, actual));
}

TEST_P(HloEvaluatorBf16Test, Bitcast) {
  // Regression test for b/114735354.
  constexpr absl::string_view hlo_text_base = R"(
HloModule Bitcast

ENTRY main {
  param = %s[32,121]{1,0} parameter(0)
  ROOT bitcast = %s[121,32,1]{0,1,2} bitcast(%s[32,121]{1,0} param)
}
)";
  string hlo_text;
  if (use_bfloat16_) {
    hlo_text = absl::StrFormat(hlo_text_base, "bf16", "bf16", "bf16");
  } else {
    hlo_text = absl::StrFormat(hlo_text_base, "f32", "f32", "f32");
  }
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&args[0]}));
  if (use_bfloat16_) {
    EXPECT_TRUE(
        absl::c_equal(args[0].data<bfloat16>(), actual.data<bfloat16>()));
  } else {
    EXPECT_TRUE(absl::c_equal(args[0].data<float>(), actual.data<float>()));
  }
}

// Check that s32 under/overflow doesn't trigger a ubsan failure.
TEST_F(HloEvaluatorTest, Int32Overflow) {
  constexpr absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  c1 = s32[] constant(1073741824)  // 2^30
  sum = s32[] add(c1, c1)  // 2^31, i.e. INT_MIN

  c2 = s32[] constant(-2147483648)  // -2^31
  sub = s32[] subtract(c2, c1)  // -2^31 - 2^30, underflows

  mul = s32[] multiply(c1, c1)
  ROOT tuple = (s32[], s32[], s32[]) tuple(sum, sub, mul)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(auto literal, Evaluate({}));
  std::vector<Literal> actual = literal.DecomposeTuple();
  ASSERT_EQ(actual.size(), 3);

  uint32 pow30 = uint32{1} << 30;
  uint32 pow31 = uint32{1} << 31;
  EXPECT_EQ(actual[0].GetFirstElement<int32>(), static_cast<int32>(pow31));
  EXPECT_EQ(actual[1].GetFirstElement<int32>(),
            static_cast<int32>(-(pow31 + pow30)));
  EXPECT_EQ(actual[2].GetFirstElement<int32>(),
            static_cast<int32>(pow31 * pow31));
}

TEST_F(HloEvaluatorTest, GetDimensionSize) {
  constexpr absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  size = u32[] parameter(0)

  data = s32[4] parameter(1)

  sum = s32[4] add(data, data)

  ROOT dynamic_size = u32[] get-dimension-size(sum), dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  // Set up dynamic parameter binding.
  TF_CHECK_OK(m_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{0, {}},
      DynamicParameterBinding::DynamicDimension{1, {}, 0}));

  TF_ASSERT_OK_AND_ASSIGN(DynamicDimensionInference dynamic_dimension_inference,
                          DynamicDimensionInference::Run(m_.get()));

  evaluator_.set_dynamic_dimension_inference(&dynamic_dimension_inference);
  Literal size_arg = LiteralUtil::CreateR0<uint32>(3);
  Literal data_arg = LiteralUtil::CreateR1<int32>({1, 2, 3, 4});

  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&size_arg, &data_arg}));

  EXPECT_EQ(actual.GetFirstElement<uint32>(), static_cast<uint32>(3));
}

// Check that we get a useful error if we pass inputs of the wrong shape.
TEST_F(HloEvaluatorTest, EvaluateWithWrongInputShapes) {
  constexpr absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  p0 = s32[1] parameter(0)
  ROOT sum = s32[1] add(p0, p0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal input_wrong_shape = LiteralUtil::CreateR1<int32>({0, 1});

  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_, {&input_wrong_shape})
                .status()
                .error_message(),
            "Shape mismatch at parameter 0. Computation expected s32[1]{0}, "
            "but arg was s32[2]{0}.");
  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_->entry_computation(), {&input_wrong_shape})
                .status()
                .error_message(),
            "Shape mismatch at parameter 0. Computation expected s32[1]{0}, "
            "but arg was s32[2]{0}.");
}

// Check that we get a useful error if we pass too many or too few inputs.
TEST_F(HloEvaluatorTest, EvaluateWithWrongNumberOfInputs) {
  constexpr absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  p0 = s32[1] parameter(0)
  ROOT sum = s32[1] add(p0, p0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal input = LiteralUtil::CreateR1<int32>({0});

  EXPECT_EQ(
      HloEvaluator().Evaluate(*m_, {&input, &input}).status().error_message(),
      "Expected 1 argument, but got 2.");
  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_->entry_computation(), {&input, &input})
                .status()
                .error_message(),
            "Expected 1 argument, but got 2.");
}

TEST_F(HloEvaluatorTest, PreserveFusionInputLayout) {
  constexpr absl::string_view hlo_text = R"(
    HloModule FusionInputLayout

    fused_computation {
      param_0 = f32[20,20]{0,1} parameter(0)
      ROOT bitcast = f32[20,20]{1,0} bitcast(param_0)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[20,20]{0,1} parameter(0)
      ROOT fusion = f32[20,20]{1,0} fusion(parameter.0),
        kind=kLoop, calls=fused_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();

  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&args[0]}));
  EXPECT_TRUE(absl::c_equal(args[0].data<float>(), actual.data<float>()));
}

TEST_F(HloEvaluatorTest, PreserveFusionOutputLayout) {
  constexpr absl::string_view hlo_text = R"(
    HloModule FusionOutputLayout

    fused_computation {
      param_0 = f32[20,20]{1,0} parameter(0)
      ROOT bitcast = f32[20,20]{0,1} bitcast(param_0)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[20,20]{1,0} parameter(0)
      ROOT fusion = f32[20,20]{0,1} fusion(parameter.0),
        kind=kLoop, calls=fused_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&args[0]}));
  EXPECT_TRUE(absl::c_equal(args[0].data<float>(), actual.data<float>()));
}

TEST_F(HloEvaluatorTest, PreserveMOFusionOutputLayout) {
  constexpr absl::string_view hlo_text = R"(
    HloModule MOFusionOutputLayout

    fused_computation {
      param_0 = f32[20,20]{1,0} parameter(0)
      bitcast = f32[20,20]{0,1} bitcast(param_0)
      ROOT tuple = (f32[20,20]{0,1}) tuple(bitcast)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[20,20]{1,0} parameter(0)
      ROOT fusion = (f32[20,20]{0,1}) fusion(parameter.0),
        kind=kLoop, calls=fused_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(Literal actual_tuple, Evaluate({&args[0]}));
  std::vector<Literal> actual_literals = actual_tuple.DecomposeTuple();
  EXPECT_TRUE(
      absl::c_equal(args[0].data<float>(), actual_literals[0].data<float>()));
}

// Tests that custom_calls fail to evaluate when no handler is specified.
TEST_F(HloEvaluatorTest, EvaluateCustomCall_NoHandler) {
  constexpr absl::string_view hlo_text = R"(
    HloModule EvaluateCustomCall_NoHandler
    ENTRY kernel_entry {
      parameter.0 = u32[2,2]{1,0} parameter(0)
      ROOT test_root = (u32[2,2]{1,0}) custom-call(parameter.0),
          custom_call_target="_my_custom_call"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  EXPECT_EQ(HloEvaluator().Evaluate(*m_, {&args[0]}).status().code(),
            ::tensorflow::error::UNIMPLEMENTED);
}

// Tests when a custom_call handler returns an error.
TEST_F(HloEvaluatorTest, EvaluateCustomCall_HandlerError) {
  constexpr absl::string_view hlo_text = R"(
    HloModule EvaluateCustomCall_HandlerError
    ENTRY kernel_entry {
      parameter.0 = u32[2,2]{1,0} parameter(0)
      ROOT test_root = (u32[2,2]{1,0}) custom-call(parameter.0),
          custom_call_target="_my_custom_call"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  HloEvaluator evaluator;
  evaluator.set_custom_call_handler(
      [](HloInstruction* custom_call, absl::Span<const Literal*> operands) {
        return InternalError("Test error");
      });
  EXPECT_EQ(evaluator.Evaluate(*m_, {&args[0]}).status().code(),
            ::tensorflow::error::INTERNAL);
}

// Tests the custom_call handler on calls with many inputs.
// We sum the operands so that we can verify the operand and output literals
// are properly mapped for access.
TEST_F(HloEvaluatorTest, EvaluateCustomCall_ManyInputs) {
  constexpr absl::string_view hlo_text = R"(
    HloModule EvaluateCustomCall_ManyInputs
    ENTRY kernel_entry {
      parameter.0 = u32[1]{0} parameter(0)
      parameter.1 = u32[1]{0} parameter(1)
      ROOT test_root = u32[1]{0} custom-call(parameter.0, parameter.1),
          custom_call_target="_my_custom_call"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  HloEvaluator evaluator;
  evaluator.set_custom_call_handler(
      [](HloInstruction* custom_call, absl::Span<const Literal*> operands) {
        EXPECT_EQ(HloOpcode::kCustomCall, custom_call->opcode());
        EXPECT_EQ("_my_custom_call", custom_call->custom_call_target());
        EXPECT_EQ(2, custom_call->operand_count());
        EXPECT_EQ(2, operands.size());
        auto output = Literal::CreateFromShape(custom_call->shape());
        auto operand0_data = operands[0]->data<uint32>();
        auto operand1_data = operands[1]->data<uint32>();
        auto output_data = output.data<uint32>();
        output_data[0] = operand0_data[0] + operand1_data[0];
        return output;
      });
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      evaluator.Evaluate(*m_->entry_computation(), {&args[0], &args[1]}));
  auto arg0_data = args[0].data<uint32>();
  auto arg1_data = args[1].data<uint32>();
  std::vector<uint32> expected_data = {arg0_data[0] + arg1_data[0]};
  EXPECT_TRUE(absl::c_equal(expected_data, actual_literal.data<uint32>()));
}

TEST_F(HloEvaluatorTest, IsFiniteF16) {
  constexpr absl::string_view hlo_text = R"(
  HloModule test

  ENTRY IsFiniteTest {
    c = f16[6] constant({nan, 7, nan, -1, inf, -inf})
    ROOT is-finite = pred[6] is-finite(c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_THAT(actual_literal.data<bool>(),
              ::testing::ElementsAre(false, true, false, true, false, false));
}

TEST_F(HloEvaluatorTest, IsFiniteBf16) {
  constexpr absl::string_view hlo_text = R"(
  HloModule test

  ENTRY IsFiniteTest {
    c = bf16[6] constant({nan, 7, nan, -1, inf, -inf})
    ROOT is-finite = pred[6] is-finite(c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_THAT(actual_literal.data<bool>(),
              ::testing::ElementsAre(false, true, false, true, false, false));
}

// Check that evaluating `f32[<huge>, 0] iota` doesn't oom (it's an empty
// array!).
TEST_F(HloEvaluatorTest, ZeroSizedIotaWithHugeDimension) {
  constexpr absl::string_view hlo_text = R"(
  HloModule test
  ENTRY t {
    ROOT i = f32[1000000000000, 0] iota(), iota_dimension=0
  })";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_THAT(actual_literal.data<float>(), ::testing::IsEmpty());
}

}  // namespace
}  // namespace xla
