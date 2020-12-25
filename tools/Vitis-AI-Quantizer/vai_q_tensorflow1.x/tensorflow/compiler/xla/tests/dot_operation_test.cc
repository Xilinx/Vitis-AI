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

#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class DotOperationTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001, 1e-5};
};

#if defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16) && \
    defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
using TypesF16F32 = ::testing::Types<float>;
using TypesF16F32F64 = ::testing::Types<float>;
using TypesF16F32F64CF64 = ::testing::Types<float>;
#elif !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16) && \
    !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
using TypesF16F32 = ::testing::Types<Eigen::half, float>;
using TypesF16F32F64 = ::testing::Types<Eigen::half, float, double>;
using TypesF16F32F64CF64 =
    ::testing::Types<Eigen::half, float, double, complex64>;
#elif !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16) && \
    defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64) &&    \
    defined(XLA_BACKEND_DOES_NOT_SUPPORT_COMPLEX)
using TypesF16F32 = ::testing::Types<Eigen::half, float>;
using TypesF16F32F64 = ::testing::Types<Eigen::half, float>;
using TypesF16F32F64CF64 = ::testing::Types<Eigen::half, float>;
#else
#error "Situation not handled yet"
#endif

// Check that we can safely pass an input tuple's elements to a dot operation.
XLA_TEST_F(DotOperationTest, DotOfInputTupleElem) {
  XlaBuilder builder(TestName());

  XlaOp param;
  auto param_data = CreateParameterAndTransferLiteral(
      0,
      LiteralUtil::MakeTupleFromSlices(
          {LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}),
           LiteralUtil::CreateR2<float>({{5, 6}, {7, 8}})}),
      "arg0", &builder, &param);
  auto lhs = GetTupleElement(param, 0);
  auto rhs = GetTupleElement(param, 1);
  Dot(lhs, rhs);

  ComputeAndCompareLiteral(&builder,
                           LiteralUtil::CreateR2<float>({{19, 22}, {43, 50}}),
                           {param_data.get()});
}

template <typename T>
class DotOperationTest_F16F32F64CF64 : public DotOperationTest {};
TYPED_TEST_CASE(DotOperationTest_F16F32F64CF64, TypesF16F32F64CF64);

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, ZeroElementVectorDot) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());

  auto lhs = ConstantR1<T>(&builder, {});
  auto rhs = ConstantR1<T>(&builder, {});
  Dot(lhs, rhs);

  this->template ComputeAndCompareR0<T>(&builder, static_cast<T>(0.0), {},
                                        this->error_spec_);
}

template <typename T>
class DotOperationTest_F16F32F64 : public DotOperationTest {};
TYPED_TEST_CASE(DotOperationTest_F16F32F64, TypesF16F32F64);

XLA_TYPED_TEST(DotOperationTest_F16F32F64, TrivialMatrixVectorDot) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());
  auto lhs = ConstantR2FromArray2D<T>(&builder, {{3.0f, 4.0f}});
  auto rhs = ConstantFromArray<T>(&builder, {3.0f, 4.0f});
  Dot(lhs, rhs);

  this->template ComputeAndCompareR1<T>(&builder, {static_cast<T>(25.0f)}, {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, OneElementVectorDot) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());
  auto lhs = ConstantR1<T>(&builder, {static_cast<T>(2.0f)});
  auto rhs = ConstantR1<T>(&builder, {static_cast<T>(3.0f)});
  Dot(lhs, rhs);

  this->template ComputeAndCompareR0<T>(&builder, static_cast<T>(6.0f), {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, VectorDot) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());
  auto lhs = ConstantFromArray<T>(&builder, {1.0f, 2.5f, 42.0f});
  auto rhs = ConstantFromArray<T>(&builder, {11.0f, -1.0f, 0.5f});
  Dot(lhs, rhs);

  this->template ComputeAndCompareR0<T>(&builder, static_cast<T>(29.5f), {},
                                        this->error_spec_);
}

std::vector<int64> MinorToMajorForIsRowMajor(bool row_major) {
  return {row_major ? 1 : 0, row_major ? 0 : 1};
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, Dot_0x2_2x0) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());
  auto lhs = ConstantR2FromArray2D<T>(&builder, Array2D<T>(0, 2));
  auto rhs = ConstantR2FromArray2D<T>(&builder, Array2D<T>(2, 0));
  Dot(lhs, rhs);

  this->template ComputeAndCompareR2<T>(&builder, Array2D<T>(0, 0), {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, Dot_0x2_2x3) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());
  auto lhs = ConstantR2FromArray2D<T>(&builder, Array2D<T>(0, 2));
  auto rhs = ConstantR2FromArray2D<T>(
      &builder, {{7.0f, 8.0f, 9.0f}, {42.0f, 77.0f, 101.0f}});
  Dot(lhs, rhs);

  this->template ComputeAndCompareR2<T>(&builder, Array2D<T>(0, 3), {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, Dot_3x2_2x0) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());
  auto lhs = ConstantR2FromArray2D<T>(
      &builder, {{7.0f, 8.0f}, {9.0f, 42.0f}, {77.0f, 101.0f}});
  auto rhs = ConstantR2FromArray2D<T>(&builder, Array2D<T>(2, 0));
  Dot(lhs, rhs);

  this->template ComputeAndCompareR2<T>(&builder, Array2D<T>(3, 0), {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, Dot_2x0_0x2) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());
  auto lhs = ConstantR2FromArray2D<T>(&builder, Array2D<T>(2, 0));
  auto rhs = ConstantR2FromArray2D<T>(&builder, Array2D<T>(0, 2));
  Dot(lhs, rhs);

  this->template ComputeAndCompareR2<T>(
      &builder, Array2D<T>(2, 2, static_cast<T>(0.0f)), {}, this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, FusedDot) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());
  auto param0 =
      Parameter(&builder, 0, ShapeUtil::MakeShapeWithType<T>({2, 4}), "arg0");
  auto param1 =
      Parameter(&builder, 1, ShapeUtil::MakeShapeWithType<T>({4, 1}), "arg1");
  auto exp0 = Exp(param0);
  Dot(exp0, param1);

  auto lhs_handle =
      this->client_
          ->TransferToServer(LiteralUtil::CreateR2FromArray2D<T>(
              {{1.0f, 2.0f, 3.0f, 4.0f}, {-1.0f, -2.0f, -3.0f, -4.0f}}))
          .ConsumeValueOrDie();
  auto rhs_handle = this->client_
                        ->TransferToServer(LiteralUtil::CreateR2FromArray2D<T>(
                            {{1.0f}, {2.0f}, {3.0f}, {4.0f}}))
                        .ConsumeValueOrDie();

  if (std::is_same<Eigen::half, T>::value) {
    this->error_spec_ = ErrorSpec{0.0001, 1e-3};
  }

  this->template ComputeAndCompareR2<T>(
      &builder, Array2D<T>({{296.14560492846033f}, {0.8611737683031964f}}),
      {lhs_handle.get(), rhs_handle.get()}, this->error_spec_);
}

template <typename T>
class SquareMatrixDot : public DotOperationTest {
 public:
  void TestImpl(bool lhs_row_major, bool rhs_row_major) {
    auto lhs_handle =
        client_
            ->TransferToServer(LiteralUtil::CreateFromArrayWithLayout<T>(
                {{1.0f, 2.0f}, {3.0f, -4.0f}},
                LayoutUtil::MakeLayout(
                    MinorToMajorForIsRowMajor(lhs_row_major))))
            .ConsumeValueOrDie();
    auto rhs_handle =
        client_
            ->TransferToServer(LiteralUtil::CreateFromArrayWithLayout<T>(
                {{1.0f, 6.0f}, {7.0f, -4.0f}},
                LayoutUtil::MakeLayout(
                    MinorToMajorForIsRowMajor(rhs_row_major))))
            .ConsumeValueOrDie();
    XlaBuilder builder(TestName());
    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    Dot(Parameter(&builder, 0, ShapeUtil::MakeShape(prim_type, {2, 2}), "lhs"),
        Parameter(&builder, 1, ShapeUtil::MakeShape(prim_type, {2, 2}), "rhs"));

    Array2D<T> expected({{15.0f, -2.0f}, {-25.0f, 34.0f}});
    ComputeAndCompareR2<T>(&builder, expected,
                           {lhs_handle.get(), rhs_handle.get()}, error_spec_);
  }
};

TYPED_TEST_CASE(SquareMatrixDot, TypesF16F32F64CF64);
XLA_TYPED_TEST(SquareMatrixDot, TypesFF) { this->TestImpl(false, false); }
XLA_TYPED_TEST(SquareMatrixDot, TypesFT) { this->TestImpl(false, true); }
XLA_TYPED_TEST(SquareMatrixDot, TypesTF) { this->TestImpl(true, false); }
XLA_TYPED_TEST(SquareMatrixDot, TypesTT) { this->TestImpl(true, true); }

struct DotTestParam {
  int m;
  int k;
  int n;
  bool dot_lhs_row_major;
  bool dot_rhs_row_major;
  bool has_addend;
  bool addend_row_major;
};

string PrintDotTestParam(
    const ::testing::TestParamInfo<DotTestParam>& test_param) {
  const DotTestParam& param = test_param.param;
  if (param.has_addend) {
    return absl::StrCat(param.m, "x", param.k, "x", param.n, "_MajorToMinor",
                        param.dot_lhs_row_major ? "T" : "F",
                        param.dot_rhs_row_major ? "T" : "F",
                        param.addend_row_major ? "T" : "F");
  } else {
    return absl::StrCat(param.m, "x", param.k, "x", param.n, "_MajorToMinor",
                        param.dot_lhs_row_major ? "T" : "F",
                        param.dot_rhs_row_major ? "T" : "F");
  }
}

class ParametricDotTest : public DotOperationTest,
                          public ::testing::WithParamInterface<DotTestParam> {
 protected:
  template <typename NativeT>
  void TestImpl();
};

template <typename NativeT>
void ParametricDotTest::TestImpl() {
  DotTestParam param = GetParam();

  std::unique_ptr<Array2D<NativeT>> dot_lhs_data =
      MakeLinspaceArray2D<NativeT>(0.0, 1.0, param.m, param.k);
  Literal dot_lhs_lit = LiteralUtil::CreateR2FromArray2DWithLayout(
      *dot_lhs_data, LayoutUtil::MakeLayout(
                         MinorToMajorForIsRowMajor(param.dot_lhs_row_major)));
  std::unique_ptr<GlobalData> dot_lhs_handle =
      client_->TransferToServer(dot_lhs_lit).ConsumeValueOrDie();

  std::unique_ptr<Array2D<NativeT>> dot_rhs_data =
      MakeLinspaceArray2D<NativeT>(0.0, 1.0, param.k, param.n);
  Layout rhs_layout = LayoutUtil::MakeLayout(
      MinorToMajorForIsRowMajor(param.dot_rhs_row_major));
  Literal dot_rhs_lit =
      LiteralUtil::CreateR2FromArray2DWithLayout(*dot_rhs_data, rhs_layout);
  std::unique_ptr<GlobalData> dot_rhs_handle =
      client_->TransferToServer(dot_rhs_lit).ConsumeValueOrDie();

  std::unique_ptr<Array2D<NativeT>> addend_data;
  Literal addend_lit;
  std::unique_ptr<GlobalData> addend_handle;

  if (param.has_addend) {
    addend_data = MakeLinspaceArray2D<NativeT>(0.0, 1.0, param.m, param.n);
    addend_lit = LiteralUtil::CreateR2FromArray2DWithLayout(
        *addend_data, LayoutUtil::MakeLayout(
                          MinorToMajorForIsRowMajor(param.addend_row_major)));
    addend_handle = client_->TransferToServer(addend_lit).ConsumeValueOrDie();
  }

  XlaBuilder builder(TestName());
  auto prim_type = primitive_util::NativeToPrimitiveType<NativeT>();
  auto result =
      Dot(Parameter(&builder, 0,
                    ShapeUtil::MakeShapeWithLayout(
                        prim_type, {param.m, param.k},
                        MinorToMajorForIsRowMajor(param.dot_lhs_row_major)),
                    "dot_lhs"),
          Parameter(&builder, 1,
                    ShapeUtil::MakeShapeWithLayout(
                        prim_type, {param.k, param.n},
                        MinorToMajorForIsRowMajor(param.dot_rhs_row_major)),
                    "dot_rhs"));

  if (param.has_addend) {
    result =
        Add(result,
            Parameter(&builder, 2,
                      ShapeUtil::MakeShapeWithLayout(
                          prim_type, {param.m, param.n},
                          MinorToMajorForIsRowMajor(param.addend_row_major)),
                      "addend"));
  }

  std::unique_ptr<Array2D<NativeT>> expected;
  if (param.has_addend) {
    expected = ReferenceUtil::ApplyElementwise2D(
        std::plus<NativeT>(),
        *ReferenceUtil::MatmulArray2D(*dot_lhs_data, *dot_rhs_data),
        *addend_data);
  } else {
    expected = ReferenceUtil::MatmulArray2D(*dot_lhs_data, *dot_rhs_data);
  }

  std::vector<GlobalData*> args = {dot_lhs_handle.get(), dot_rhs_handle.get()};
  if (param.has_addend) {
    args.push_back(addend_handle.get());
  }
  ErrorSpec error_spec(0.3, 3e-3);
  if (std::is_same<Eigen::half, NativeT>::value) {
    error_spec = ErrorSpec(0.3, 5e-3);
  }
  ComputeAndCompareR2<NativeT>(&builder, *expected, args, error_spec);
}

std::vector<DotTestParam> CreateDotTestParameters() {
  std::vector<DotTestParam> params;

  auto add_matrix_matrix_dot_test = [&](int m, int k, int n) {
    for (bool lhs_row_major : {true, false}) {
      for (bool rhs_row_major : {true, false}) {
        params.push_back({/*m=*/m, /*k=*/k, /*n=*/n,
                          /*dot_lhs_row_major=*/lhs_row_major,
                          /*dot_rhs_row_major=*/rhs_row_major,
                          /*has_addend=*/false, /*addend_row_major=*/true});
      }
    }
  };

  add_matrix_matrix_dot_test(/*m=*/1, /*k=*/23, /*n=*/42);
  add_matrix_matrix_dot_test(/*m=*/23, /*k=*/1, /*n=*/42);
  add_matrix_matrix_dot_test(/*m=*/23, /*k=*/42, /*n=*/1);
  add_matrix_matrix_dot_test(/*m=*/1, /*k=*/23, /*n=*/1);
  add_matrix_matrix_dot_test(/*m=*/1, /*k=*/1, /*n=*/1);
  add_matrix_matrix_dot_test(/*m=*/12, /*k=*/117, /*n=*/7);
  add_matrix_matrix_dot_test(/*m=*/270, /*k=*/270, /*n=*/520);
  add_matrix_matrix_dot_test(/*m=*/260, /*k=*/3, /*n=*/520);

  return params;
}

#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
XLA_TEST_P(ParametricDotTest, TestF16) { TestImpl<Eigen::half>(); }
#endif
XLA_TEST_P(ParametricDotTest, TestF32) { TestImpl<float>(); }
XLA_TEST_P(ParametricDotTest, TestF64) { TestImpl<double>(); }

INSTANTIATE_TEST_CASE_P(DotTests, ParametricDotTest,
                        ::testing::ValuesIn(CreateDotTestParameters()),
                        PrintDotTestParam);

class ParametricDotTestWithoutLayoutAssignment : public ParametricDotTest {
 public:
  ParametricDotTestWithoutLayoutAssignment() {
    execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
        "layout-assignment");
    execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
        "tiling-assignment");
    // Disable algebraic simplification because the pass may replace a dot
    // instruction with a layout-changing multiplication instruction.
    execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
        "algsimp");
  }
};

std::vector<DotTestParam> CreateNoLayoutAssignmentDotTestParameters() {
  std::vector<DotTestParam> params;

  auto add_matrix_vector_dot_test = [&](int k, int n) {
    for (bool lhs_row_major : {true, false}) {
      for (bool rhs_row_major : {true, false}) {
        for (bool has_addend : {true, false}) {
          // The addend needs to be row major to match the result of the dot.
          params.push_back({/*m=*/1, /*k=*/k, /*n=*/n,
                            /*dot_lhs_row_major=*/lhs_row_major,
                            /*dot_rhs_row_major=*/rhs_row_major,
                            /*has_addend=*/has_addend,
                            /*addend_row_major=*/true});
          if (n != 1) {
            params.push_back({/*m=*/n, /*k=*/k, /*n=*/1,
                              /*dot_lhs_row_major=*/lhs_row_major,
                              /*dot_rhs_row_major=*/rhs_row_major,
                              /*has_addend=*/has_addend,
                              /*addend_row_major=*/true});
          }
        }
      }
    }
  };

  add_matrix_vector_dot_test(/*k=*/8, /*n=*/8);
  add_matrix_vector_dot_test(/*k=*/130, /*n=*/8);
  add_matrix_vector_dot_test(/*k=*/8, /*n=*/130);
  add_matrix_vector_dot_test(/*k=*/290, /*n=*/130);
  add_matrix_vector_dot_test(/*k=*/1, /*n=*/1);
  add_matrix_vector_dot_test(/*k=*/1, /*n=*/16);
  add_matrix_vector_dot_test(/*k=*/1, /*n=*/4);
  add_matrix_vector_dot_test(/*k=*/1, /*n=*/3);
  add_matrix_vector_dot_test(/*k=*/3, /*n=*/16);
  add_matrix_vector_dot_test(/*k=*/3, /*n=*/3);
  add_matrix_vector_dot_test(/*k=*/29, /*n=*/29);
  add_matrix_vector_dot_test(/*k=*/8, /*n=*/2);
  add_matrix_vector_dot_test(/*k=*/2, /*n=*/8);
  add_matrix_vector_dot_test(/*k=*/259, /*n=*/258);

  return params;
}

#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
XLA_TEST_P(ParametricDotTestWithoutLayoutAssignment, TestF16) {
  TestImpl<Eigen::half>();
}
#endif
XLA_TEST_P(ParametricDotTestWithoutLayoutAssignment, TestF32) {
  TestImpl<float>();
}
XLA_TEST_P(ParametricDotTestWithoutLayoutAssignment, TestF64) {
  TestImpl<double>();
}

INSTANTIATE_TEST_CASE_P(
    DotTests, ParametricDotTestWithoutLayoutAssignment,
    ::testing::ValuesIn(CreateNoLayoutAssignmentDotTestParameters()),
    PrintDotTestParam);

template <typename T>
class NonsquareMatrixDot : public DotOperationTest {
 public:
  void TestImpl(bool lhs_row_major, bool rhs_row_major) {
    auto lhs_handle =
        client_
            ->TransferToServer(LiteralUtil::CreateFromArrayWithLayout<T>(
                {{1.0f, 2.0f, 3.0f}, {3.0f, -4.0f, -1.0f}},
                LayoutUtil::MakeLayout(
                    MinorToMajorForIsRowMajor(lhs_row_major))))
            .ConsumeValueOrDie();
    auto rhs_handle =
        client_
            ->TransferToServer(LiteralUtil::CreateFromArrayWithLayout<T>(
                {{1.0f, 6.0f}, {2.0f, 3.0f}, {7.0f, -4.0f}},
                LayoutUtil::MakeLayout(
                    MinorToMajorForIsRowMajor(rhs_row_major))))
            .ConsumeValueOrDie();

    XlaBuilder builder(TestName());
    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    Dot(Parameter(&builder, 0, ShapeUtil::MakeShape(prim_type, {2, 3}), "lhs"),
        Parameter(&builder, 1, ShapeUtil::MakeShape(prim_type, {3, 2}), "rhs"));

    Array2D<T> expected({{26.0f, 0.0f}, {-12.0f, 10.0f}});

    ComputeAndCompareR2<T>(&builder, expected,
                           {lhs_handle.get(), rhs_handle.get()}, error_spec_);
  }
};

TYPED_TEST_CASE(NonsquareMatrixDot, TypesF16F32F64CF64);
XLA_TYPED_TEST(NonsquareMatrixDot, TestFF) { this->TestImpl(false, false); }
XLA_TYPED_TEST(NonsquareMatrixDot, TestFT) { this->TestImpl(false, true); }
XLA_TYPED_TEST(NonsquareMatrixDot, TestTF) { this->TestImpl(true, false); }
XLA_TYPED_TEST(NonsquareMatrixDot, TestTT) { this->TestImpl(true, true); }

XLA_TEST_F(DotOperationTest, MatrixVectorC64) {
  auto lhs_handle =
      client_
          ->TransferToServer(LiteralUtil::CreateR2WithLayout<complex64>(
              {{1.0, 2.0, 3.0, -4.0}}, LayoutUtil::MakeLayout({1, 0})))
          .ConsumeValueOrDie();
  auto rhs_handle =
      client_
          ->TransferToServer(LiteralUtil::CreateR2WithLayout<complex64>(
              {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {-4.0, 4.0}},
              LayoutUtil::MakeLayout({1, 0})))
          .ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  auto prim_type = primitive_util::NativeToPrimitiveType<complex64>();
  Dot(Parameter(&builder, 0, ShapeUtil::MakeShape(prim_type, {1, 4}), "lhs"),
      Parameter(&builder, 1, ShapeUtil::MakeShape(prim_type, {4, 2}), "rhs"));

  Array2D<complex64> expected({{30.0, -2.0}});

  ComputeAndCompareR2<complex64>(
      &builder, expected, {lhs_handle.get(), rhs_handle.get()}, error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, ConcurrentMatMult) {
  using T = TypeParam;

  XlaBuilder builder(this->TestName());
  auto matrix1 =
      ConstantR2FromArray2D<T>(&builder, {{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto matrix2 =
      ConstantR2FromArray2D<T>(&builder, {{5.0f, 6.0f}, {7.0f, 8.0f}});
  auto matrix12 = Dot(matrix1, matrix2);
  auto matrix21 = Dot(matrix2, matrix1);
  Add(matrix12, matrix21);

  Array2D<T> expected({{42.0f, 56.0f}, {74.0f, 96.0f}});
  this->template ComputeAndCompareR2<T>(&builder, expected, {},
                                        this->error_spec_);
}

template <typename T>
class DotOperationTestForBatchMatMul : public DotOperationTest {};
TYPED_TEST_CASE(DotOperationTestForBatchMatMul, TypesF16F32F64);

// Regression test for b/32055648. The root of the graph is a kFusion of 4
// bitcasts. Although bitcasts don't map to thunks, the root should still be
// sync-dependent on bitcasts' operands.
XLA_TYPED_TEST(DotOperationTestForBatchMatMul, Types) {
  using T = TypeParam;
  XlaBuilder builder(this->TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShapeWithType<T>({2, 2, 2, 2}),
                     "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShapeWithType<T>({2, 2, 2, 2}),
                     "y");

  auto x_flat = Reshape(x, {0, 1, 2, 3}, {4, 2, 2});
  auto y_flat = Reshape(y, {0, 1, 2, 3}, {4, 2, 2});

  // Slice batches into individual matrices and multiply them.
  std::vector<XlaOp> out_slices;
  for (int i = 0; i < 4; ++i) {
    // Slice off individual matrices and reshape to 2D tensors.
    auto x_slice = Slice(x_flat, {i, 0, 0}, {i + 1, 2, 2}, {1, 1, 1});
    x_slice = Reshape(x_slice, {0, 1, 2}, {2, 2});
    auto y_slice = Slice(y_flat, {i, 0, 0}, {i + 1, 2, 2}, {1, 1, 1});
    y_slice = Reshape(y_slice, {0, 1, 2}, {2, 2});

    auto out = Dot(x_slice, y_slice);
    out = Reshape(out, {0, 1}, {1, 2, 2});
    out_slices.push_back(out);
  }
  auto out_flat = ConcatInDim(&builder, out_slices, 0);
  Reshape(out_flat, {0, 1, 2}, {2, 2, 2, 2});

  auto x_data = this->client_
                    ->TransferToServer(LiteralUtil::CreateR4FromArray4D<T>(
                        {{{{1000.0f, 100.0f}, {10.0f, 1.0f}},
                          {{2000.0f, 200.0f}, {20.0f, 2.0f}}},
                         {{{3000.0f, 300.0f}, {30.0f, 3.0f}},
                          {{4000.0f, 400.0f}, {40.0f, 4.0f}}}}))
                    .ConsumeValueOrDie();
  auto y_data =
      this->client_
          ->TransferToServer(LiteralUtil::CreateR4FromArray4D<T>(
              {{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
               {{{11.0f, 22.0f}, {33.0f, 44.0f}},
                {{55.0f, 66.0f}, {77.0f, 88.0f}}}}))
          .ConsumeValueOrDie();

  if (std::is_same<Eigen::half, T>::value) {
    this->error_spec_ = ErrorSpec{0.0001, 1e-3};
  }
  this->template ComputeAndCompareR4<T>(
      &builder,
      /*expected=*/
      {{{{1300.0f, 2400.0f}, {13.0f, 24.0f}},
        {{11400.0f, 13600.0f}, {114.0f, 136.0f}}},
       {{{42900.0f, 79200.0f}, {429.0f, 792.0f}},
        {{250800.0f, 299200.0f}, {2508.0f, 2992.0f}}}},
      {x_data.get(), y_data.get()}, this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, GeneralMatMul) {
  using T = TypeParam;

  XlaBuilder builder(this->TestName());
  auto x =
      Parameter(&builder, 0, ShapeUtil::MakeShapeWithType<T>({2, 2, 2}), "x");
  auto y =
      Parameter(&builder, 1, ShapeUtil::MakeShapeWithType<T>({2, 2, 2}), "y");

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);

  DotGeneral(x, y, dnums);

  auto x_data =
      this->client_
          ->TransferToServer(LiteralUtil::CreateR3FromArray3D<T>(
              {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}}))
          .ConsumeValueOrDie();

  auto y_data =
      this->client_
          ->TransferToServer(LiteralUtil::CreateR3FromArray3D<T>(
              {{{1.0f, 0.0f}, {0.0f, 1.0f}}, {{1.0f, 0.0f}, {0.0f, 1.0f}}}))
          .ConsumeValueOrDie();

  this->template ComputeAndCompareR3<T>(
      &builder,
      /*expected=*/
      {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
      {x_data.get(), y_data.get()}, this->error_spec_);
}

#ifndef XLA_TEST_BACKEND_CPU
// TODO(b/74459949): failed on CPU on 2018-10-29.
XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, GeneralMatMulR3LhsR2Rhs) {
  using T = TypeParam;

  XlaBuilder builder(this->TestName());
  auto x =
      Parameter(&builder, 0, ShapeUtil::MakeShapeWithType<T>({2, 2, 2}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShapeWithType<T>({2, 2}), "y");

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);

  DotGeneral(x, y, dnums);

  auto x_data =
      this->client_
          ->TransferToServer(LiteralUtil::CreateR3FromArray3D<T>(
              {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}}))
          .ConsumeValueOrDie();

  auto y_data = this->client_
                    ->TransferToServer(LiteralUtil::CreateR2FromArray2D<T>(
                        {{1.0f, 0.0f}, {0.0f, 1.0f}}))
                    .ConsumeValueOrDie();

  this->template ComputeAndCompareR2<T>(
      &builder,
      /*expected=*/{{1.0f, 2.0f}, {7.0f, 8.0f}}, {x_data.get(), y_data.get()},
      this->error_spec_);
}

// TODO(b/74459949): failed on CPU on 2018-10-29.
XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, GeneralMatMulR2LhsR3Rhs) {
  using T = TypeParam;

  XlaBuilder builder(this->TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShapeWithType<T>({2, 2}), "x");
  auto y =
      Parameter(&builder, 1, ShapeUtil::MakeShapeWithType<T>({2, 2, 2}), "y");

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);

  DotGeneral(x, y, dnums);

  auto x_data = this->client_
                    ->TransferToServer(LiteralUtil::CreateR2FromArray2D<T>(
                        {{1.0f, 0.0f}, {0.0f, 1.0f}}))
                    .ConsumeValueOrDie();

  auto y_data =
      this->client_
          ->TransferToServer(LiteralUtil::CreateR3FromArray3D<T>(
              {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}}))
          .ConsumeValueOrDie();

  this->template ComputeAndCompareR2<T>(
      &builder,
      /*expected=*/{{1.0f, 2.0f}, {7.0f, 8.0f}}, {x_data.get(), y_data.get()},
      this->error_spec_);
}
#endif  // XLA_TEST_BACKEND_CPU

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, GeneralMatMulMultipleBatch) {
  using T = TypeParam;

  XlaBuilder builder(this->TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShapeWithType<T>({2, 2, 2, 2}),
                     "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShapeWithType<T>({2, 2, 2, 2}),
                     "y");

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(3);
  dnums.add_rhs_contracting_dimensions(2);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_lhs_batch_dimensions(1);
  dnums.add_rhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(1);

  DotGeneral(x, y, dnums);

  auto x_data =
      this->client_
          ->TransferToServer(LiteralUtil::CreateR4FromArray4D<T>(
              {{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
               {{{9.0f, 10.0f}, {11.0f, 12.0f}},
                {{13.0f, 14.0f}, {15.0f, 16.0f}}}}))
          .ConsumeValueOrDie();

  auto y_data =
      this->client_
          ->TransferToServer(LiteralUtil::CreateR4FromArray4D<T>(
              {{{{1.0f, 0.0f}, {0.0f, 1.0f}}, {{1.0f, 0.0f}, {0.0f, 1.0f}}},
               {{{0.0f, 1.0f}, {1.0f, 0.0f}}, {{0.0f, 1.0f}, {1.0f, 0.0f}}}}))
          .ConsumeValueOrDie();

  this->template ComputeAndCompareR4<T>(
      &builder,
      /*expected=*/
      {{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
       {{{10.0f, 9.0f}, {12.0f, 11.0f}}, {{14.0f, 13.0f}, {16.0f, 15.0f}}}},
      {x_data.get(), y_data.get()}, this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, TransposeFolding) {
  using T = TypeParam;
  for (bool transpose_lhs : {false, true}) {
    for (bool transpose_rhs : {false, true}) {
      for (bool row_major : {false, true}) {
        std::unique_ptr<Array2D<T>> lhs(
            new Array2D<T>({{1.0f, 2.0f, 3.0f}, {3.0f, -4.0f, -1.0f}}));
        std::unique_ptr<Array2D<T>> rhs(
            new Array2D<T>({{1.0f, 6.0f}, {2.0f, 3.0f}, {7.0f, -4.0f}}));

        if (transpose_lhs) {
          lhs = ReferenceUtil::TransposeArray2D(*lhs);
        }
        if (transpose_rhs) {
          rhs = ReferenceUtil::TransposeArray2D(*rhs);
        }
        auto lhs_handle =
            this->client_
                ->TransferToServer(
                    LiteralUtil::CreateR2FromArray2DWithLayout<T>(
                        *lhs, LayoutUtil::MakeLayout(
                                  MinorToMajorForIsRowMajor(row_major))))
                .ConsumeValueOrDie();
        auto rhs_handle =
            this->client_
                ->TransferToServer(
                    LiteralUtil::CreateR2FromArray2DWithLayout<T>(
                        *rhs, LayoutUtil::MakeLayout(
                                  MinorToMajorForIsRowMajor(row_major))))
                .ConsumeValueOrDie();

        XlaBuilder builder(this->TestName());
        auto prim_type = primitive_util::NativeToPrimitiveType<T>();
        auto lhs_arg = Parameter(
            &builder, 0,
            ShapeUtil::MakeShape(prim_type, {lhs->height(), lhs->width()}),
            "lhs");
        auto rhs_arg = Parameter(
            &builder, 1,
            ShapeUtil::MakeShape(prim_type, {rhs->height(), rhs->width()}),
            "rhs");
        if (transpose_lhs) {
          lhs_arg = Transpose(lhs_arg, {1, 0});
        }
        if (transpose_rhs) {
          rhs_arg = Transpose(rhs_arg, {1, 0});
        }
        Dot(lhs_arg, rhs_arg);

        Array2D<T> expected({{26.0f, 0.0f}, {-12.0f, 10.0f}});
        VLOG(1) << "TestTransposeFolding " << transpose_lhs << " "
                << transpose_rhs << " " << row_major;
        this->template ComputeAndCompareR2<T>(
            &builder, expected, {lhs_handle.get(), rhs_handle.get()},
            this->error_spec_);
      }
    }
  }
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64,
               DotOfConcatOptimizationWithConstLHS) {
  using T = TypeParam;
  auto prim_type = primitive_util::NativeToPrimitiveType<T>();

  std::unique_ptr<Array2D<T>> constant_lhs_array(
      new Array2D<T>({{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                      {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}}));

  XlaBuilder builder(this->TestName());
  auto lhs_constant = ConstantR2FromArray2D(&builder, *constant_lhs_array);
  auto rhs_arg_0 = Parameter(
      &builder, 0, ShapeUtil::MakeShape(prim_type, {2, 2}), "rhs_arg_0");
  auto rhs_arg_1 = Parameter(
      &builder, 1, ShapeUtil::MakeShape(prim_type, {3, 2}), "rhs_arg_1");
  auto rhs_arg_2 = Parameter(
      &builder, 2, ShapeUtil::MakeShape(prim_type, {1, 2}), "rhs_arg_2");
  Dot(lhs_constant,
      ConcatInDim(&builder, {rhs_arg_0, rhs_arg_1, rhs_arg_2}, 0));

  std::unique_ptr<Array2D<T>> arg_0_value_array(
      new Array2D<T>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  std::unique_ptr<Array2D<T>> arg_1_value_array(
      new Array2D<T>({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}));
  std::unique_ptr<Array2D<T>> arg_2_value_array(new Array2D<T>({{1.0f, 2.0f}}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_0_value,
      this->client_->TransferToServer(
          LiteralUtil::CreateR2FromArray2D<T>(*arg_0_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_1_value,
      this->client_->TransferToServer(
          LiteralUtil::CreateR2FromArray2D<T>(*arg_1_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_2_value,
      this->client_->TransferToServer(
          LiteralUtil::CreateR2FromArray2D<T>(*arg_2_value_array)));

  Array2D<T> expected({{53.0f, 74.0f}, {45.0f, 66.0f}});
  this->template ComputeAndCompareR2<T>(
      &builder, expected,
      {arg_0_value.get(), arg_1_value.get(), arg_2_value.get()},
      this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64,
               DotOfConcatOptimizationWithConstRHS) {
  using T = TypeParam;
  std::unique_ptr<Array2D<T>> constant_rhs_array(
      new Array2D<T>({{1.0f, 2.0f},
                      {3.0f, 4.0f},
                      {5.0f, 6.0f},
                      {6.0f, 5.0f},
                      {4.0f, 3.0f},
                      {2.0f, 1.0f}}));

  XlaBuilder builder(this->TestName());
  auto rhs_constant = ConstantR2FromArray2D(&builder, *constant_rhs_array);
  auto lhs_arg_0 = Parameter(
      &builder, 0, ShapeUtil::MakeShapeWithType<T>({2, 2}), "lhs_arg_0");
  auto lhs_arg_1 = Parameter(
      &builder, 1, ShapeUtil::MakeShapeWithType<T>({2, 3}), "lhs_arg_1");
  auto lhs_arg_2 = Parameter(
      &builder, 2, ShapeUtil::MakeShapeWithType<T>({2, 1}), "lhs_arg_2");
  Dot(ConcatInDim(&builder, {lhs_arg_0, lhs_arg_1, lhs_arg_2}, 1),
      rhs_constant);

  std::unique_ptr<Array2D<T>> arg_0_value_array(
      new Array2D<T>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  std::unique_ptr<Array2D<T>> arg_1_value_array(
      new Array2D<T>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}));
  std::unique_ptr<Array2D<T>> arg_2_value_array(
      new Array2D<T>({{1.0f}, {2.0f}}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_0_value,
      this->client_->TransferToServer(
          LiteralUtil::CreateR2FromArray2D<T>(*arg_0_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_1_value,
      this->client_->TransferToServer(
          LiteralUtil::CreateR2FromArray2D<T>(*arg_1_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_2_value,
      this->client_->TransferToServer(
          LiteralUtil::CreateR2FromArray2D<T>(*arg_2_value_array)));

  Array2D<T> expected({{38.0f, 36.0f}, {93.0f, 91.0f}});
  this->template ComputeAndCompareR2<T>(
      &builder, expected,
      {arg_0_value.get(), arg_1_value.get(), arg_2_value.get()},
      this->error_spec_);
}

XLA_TEST_F(DotOperationTest, DotOfGatherOptimizationWithConstRHSClassicMM) {
  std::unique_ptr<Array2D<float>> constant_lhs_array(new Array2D<float>(
      {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}));
  std::unique_ptr<Array2D<float>> constant_rhs_array(
      new Array2D<float>({{1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0},
                          {7.0, 8.0, 9.0},
                          {9.0, 8.0, 7.0},
                          {6.0, 5.0, 4.0},
                          {3.0, 2.0, 1.0}}));
  // Dot result to slice from: {{114, 105, 96}, {96, 105, 114}}

  XlaBuilder builder(TestName());
  auto lhs_constant = ConstantR2FromArray2D(&builder, *constant_lhs_array);
  auto rhs_constant = ConstantR2FromArray2D(&builder, *constant_rhs_array);
  auto one = ConstantR0<int32>(&builder, 1);
  auto zero = ConstantR0<int32>(&builder, 0);
  auto dynamic_slice = DynamicSlice(lhs_constant, {one, zero}, {1, 6});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  DotGeneral(dynamic_slice, rhs_constant, dot_dnums);

  Array2D<float> expected({{96.0, 105.0, 114.0}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, DotOfGatherOptimizationWithConstLHSClassicMM) {
  std::unique_ptr<Array2D<float>> constant_lhs_array(new Array2D<float>(
      {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}));
  std::unique_ptr<Array2D<float>> constant_rhs_array(
      new Array2D<float>({{1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0},
                          {7.0, 8.0, 9.0},
                          {9.0, 8.0, 7.0},
                          {6.0, 5.0, 4.0},
                          {3.0, 2.0, 1.0}}));
  // Dot result to slice from: {{114, 105, 96}, {96, 105, 114}}

  XlaBuilder builder(TestName());
  auto lhs_constant = ConstantR2FromArray2D(&builder, *constant_lhs_array);
  auto rhs_constant = ConstantR2FromArray2D(&builder, *constant_rhs_array);
  auto zero = ConstantR0<int32>(&builder, 0);
  auto one = ConstantR0<int32>(&builder, 1);
  auto dynamic_slice = DynamicSlice(rhs_constant, {zero, one}, {6, 1});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  DotGeneral(lhs_constant, dynamic_slice, dot_dnums);

  Array2D<float> expected({{105.0}, {105.0}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest,

           DotOfGatherOptimizationWithConstRHSReverseMM) {
  std::unique_ptr<Array2D<float>> constant_lhs_array(
      new Array2D<float>({{1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0},
                          {7.0, 8.0, 9.0},
                          {9.0, 8.0, 7.0},
                          {6.0, 5.0, 4.0},
                          {3.0, 2.0, 1.0}}));
  std::unique_ptr<Array2D<float>> constant_rhs_array(new Array2D<float>(
      {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}));
  // Dot result to slice from: {{114, 96}, {105, 105}, {96, 114}}

  XlaBuilder builder(TestName());
  auto lhs_constant = ConstantR2FromArray2D(&builder, *constant_lhs_array);
  auto rhs_constant = ConstantR2FromArray2D(&builder, *constant_rhs_array);
  auto zero = ConstantR0<int32>(&builder, 0);
  auto one = ConstantR0<int32>(&builder, 1);
  auto dynamic_slice = DynamicSlice(lhs_constant, {zero, one}, {6, 1});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(1);
  DotGeneral(dynamic_slice, rhs_constant, dot_dnums);

  Array2D<float> expected({{105.0, 105.0}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, DotOfGatherOptimizationWithConstLHSReverseMM) {
  std::unique_ptr<Array2D<float>> constant_lhs_array(
      new Array2D<float>({{1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0},
                          {7.0, 8.0, 9.0},
                          {9.0, 8.0, 7.0},
                          {6.0, 5.0, 4.0},
                          {3.0, 2.0, 1.0}}));
  std::unique_ptr<Array2D<float>> constant_rhs_array(new Array2D<float>(
      {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}));
  // Dot result to slice from: {{114, 96}, {105, 105}, {96, 114}}

  XlaBuilder builder(TestName());
  auto lhs_constant = ConstantR2FromArray2D(&builder, *constant_lhs_array);
  auto rhs_constant = ConstantR2FromArray2D(&builder, *constant_rhs_array);
  auto zero = ConstantR0<int32>(&builder, 0);
  auto one = ConstantR0<int32>(&builder, 1);
  auto dynamic_slice = DynamicSlice(rhs_constant, {one, zero}, {1, 6});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(1);
  DotGeneral(lhs_constant, dynamic_slice, dot_dnums);

  Array2D<float> expected({{96.0}, {105.0}, {114.0}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, DotOfGatherOptimizationWithConstRHSRows) {
  std::unique_ptr<Array2D<float>> constant_lhs_array(
      new Array2D<float>({{1.0, 2.0},
                          {3.0, 4.0},
                          {5.0, 6.0},
                          {6.0, 5.0},
                          {4.0, 3.0},
                          {2.0, 1.0}}));
  std::unique_ptr<Array2D<float>> constant_rhs_array(
      new Array2D<float>({{1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0},
                          {7.0, 8.0, 9.0},
                          {9.0, 8.0, 7.0},
                          {6.0, 5.0, 4.0},
                          {3.0, 2.0, 1.0}}));
  // Dot result to slice from: {{132, 129, 126}, {126, 129, 132}}

  XlaBuilder builder(TestName());
  auto lhs_constant = ConstantR2FromArray2D(&builder, *constant_lhs_array);
  auto rhs_constant = ConstantR2FromArray2D(&builder, *constant_rhs_array);
  auto zero = ConstantR0<int32>(&builder, 0);
  auto one = ConstantR0<int32>(&builder, 1);
  auto dynamic_slice = DynamicSlice(lhs_constant, {zero, one}, {6, 1});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  DotGeneral(dynamic_slice, rhs_constant, dot_dnums);

  Array2D<float> expected({{126.0, 129.0, 132.0}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, DotOfGatherOptimizationWithConstLHSRows) {
  std::unique_ptr<Array2D<float>> constant_lhs_array(
      new Array2D<float>({{1.0, 2.0},
                          {3.0, 4.0},
                          {5.0, 6.0},
                          {6.0, 5.0},
                          {4.0, 3.0},
                          {2.0, 1.0}}));
  std::unique_ptr<Array2D<float>> constant_rhs_array(
      new Array2D<float>({{1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0},
                          {7.0, 8.0, 9.0},
                          {9.0, 8.0, 7.0},
                          {6.0, 5.0, 4.0},
                          {3.0, 2.0, 1.0}}));
  // Dot result to slice from: {{132, 129, 126}, {126, 129, 132}}

  XlaBuilder builder(TestName());
  auto lhs_constant = ConstantR2FromArray2D(&builder, *constant_lhs_array);
  auto rhs_constant = ConstantR2FromArray2D(&builder, *constant_rhs_array);
  auto zero = ConstantR0<int32>(&builder, 0);
  auto one = ConstantR0<int32>(&builder, 1);
  auto dynamic_slice = DynamicSlice(rhs_constant, {zero, one}, {6, 1});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  DotGeneral(lhs_constant, dynamic_slice, dot_dnums);

  Array2D<float> expected({{129.0}, {129.0}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, DotOfGatherOptimizationWithConstRHSCols) {
  std::unique_ptr<Array2D<float>> constant_lhs_array(new Array2D<float>(
      {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}));
  std::unique_ptr<Array2D<float>> constant_rhs_array(
      new Array2D<float>({{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                          {7.0, 8.0, 9.0, 9.0, 8.0, 7.0},
                          {6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}));
  // Dot result to slice from: {{91, 168, 56}, {56, 168, 91}}

  XlaBuilder builder(TestName());
  auto lhs_constant = ConstantR2FromArray2D(&builder, *constant_lhs_array);
  auto rhs_constant = ConstantR2FromArray2D(&builder, *constant_rhs_array);
  auto zero = ConstantR0<int32>(&builder, 0);
  auto one = ConstantR0<int32>(&builder, 1);
  auto dynamic_slice = DynamicSlice(lhs_constant, {one, zero}, {1, 6});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  DotGeneral(dynamic_slice, rhs_constant, dot_dnums);

  Array2D<float> expected({{56.0, 168.0, 91.0}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, DotOfGatherOptimizationWithConstLHSCols) {
  std::unique_ptr<Array2D<float>> constant_lhs_array(new Array2D<float>(
      {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}));
  std::unique_ptr<Array2D<float>> constant_rhs_array(
      new Array2D<float>({{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                          {7.0, 8.0, 9.0, 9.0, 8.0, 7.0},
                          {6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}));
  // Dot result to slice from: {{91, 168, 56}, {56, 168, 91}}

  XlaBuilder builder(TestName());
  auto lhs_constant = ConstantR2FromArray2D(&builder, *constant_lhs_array);
  auto rhs_constant = ConstantR2FromArray2D(&builder, *constant_rhs_array);
  auto zero = ConstantR0<int32>(&builder, 0);
  auto one = ConstantR0<int32>(&builder, 1);
  auto dynamic_slice = DynamicSlice(rhs_constant, {one, zero}, {1, 6});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  DotGeneral(lhs_constant, dynamic_slice, dot_dnums);

  Array2D<float> expected({{168.0}, {168.0}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, DotRank2AndRank2NonDefaultContractionDims) {
  XlaBuilder builder(TestName());

  Array2D<float> lhs_array({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto lhs_constant = ConstantR2FromArray2D(&builder, lhs_array);

  Array2D<float> rhs_array({{5.0f, 6.0f}, {7.0f, 8.0f}});
  auto rhs_constant = ConstantR2FromArray2D(&builder, rhs_array);

  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  DotGeneral(lhs_constant, rhs_constant, dot_dnums);

  Array2D<float> expected({
      {26.f, 30.f},
      {38.f, 44.f},
  });

  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

using EinsumParamType =
    std::tuple<std::vector<int64>, std::vector<int64>, string>;
class EinsumTest : public DotOperationTest,
                   public ::testing::WithParamInterface<EinsumParamType> {};
XLA_TEST_P(EinsumTest, SimpleEinsumTest) {
  XlaBuilder builder(TestName());
  auto x = AddParam(
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, std::get<0>(GetParam())))
          .ValueOrDie(),
      &builder);
  auto y = AddParam(
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, std::get<1>(GetParam())))
          .ValueOrDie(),
      &builder);
  Einsum(x, y, std::get<2>(GetParam()));
  ComputeAndCompare(&builder, {}, ErrorSpec{1e-3, 1e-3});
}

std::vector<EinsumParamType> GetEinsumTestCases() {
  using v = std::vector<int64>;
  using p = EinsumParamType;
  std::vector<p> test_cases = {
      p{v{5, 6}, v{6, 7}, "mk,kn->mn"},
      p{v{5, 6}, v{6, 7}, "mk,kn->nm"},
      p{v{5, 6, 11}, v{6, 11, 7}, "mkB,kBn->nmB"},
      p{v{31, 55, 11}, v{55, 11, 29}, "mkB,kBn->nmB"},
      p{v{31, 55, 11}, v{55, 11, 29}, "mkB,kBn->Bnm"},
      p{v{8, 55, 11, 3}, v{55, 11, 3, 29}, "mkBC,kBCn->BCnm"},
      p{v{5, 6}, v{6, 7}, "ab,cd->dcba"},
      p{v{6}, v{6, 7}, "b,bc->c"},
      p{v{5, 6, 7}, v{5, 6, 7}, "abc,abc->ab"},
      p{v{5, 6, 7}, v{7, 6, 5}, "abc,cba->ca"},
      p{v{77}, v{77}, "a,a->a"},
      p{v{77}, v{77, 55}, "a,ab->ba"},
      p{v{2, 3, 77}, v{77, 2, 3, 55}, "ija,aijb->baij"},
      p{v{55}, v{}, "a,->a"},
      p{v{11, 111}, v{11}, "ab,a->ab"},
      p{v{16, 34}, v{16, 34}, "ab,ab->ab"},
      p{v{16, 3, 34}, v{3, 16, 34}, "abc,bac->abc"},
      p{v{5, 19}, v{}, "ab,->ab"},
      p{v{8, 1, 16, 64}, v{8, 12, 16, 64}, "bqhf,bkhf->bhqk"},
  };
  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(Einsum, EinsumTest,
                         ::testing::ValuesIn(GetEinsumTestCases()));

using BatchDotParamType =
    std::tuple<std::vector<int64>, std::vector<int64>, std::vector<int64>>;
class BatchDotTest : public DotOperationTest,
                     public ::testing::WithParamInterface<BatchDotParamType> {};
XLA_TEST_P(BatchDotTest, BroadcastingBatchDotTest) {
  XlaBuilder builder(TestName());
  auto x = AddParam(
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, std::get<0>(GetParam())))
          .ValueOrDie(),
      &builder);
  auto y = AddParam(
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, std::get<1>(GetParam())))
          .ValueOrDie(),
      &builder);
  auto batch_dot = BatchDot(x, y);
  auto output_shape = builder.GetShape(batch_dot).ValueOrDie();
  EXPECT_EQ(output_shape.dimensions(), std::get<2>(GetParam()));
  ComputeAndCompare(&builder, {}, ErrorSpec{1e-3, 1e-3});
}

std::vector<BatchDotParamType> GetBatchDotTestCases() {
  using v = std::vector<int64>;
  using p = BatchDotParamType;
  std::vector<p> test_cases = {
      p{v{5, 6}, v{6, 7}, v{5, 7}},
      p{v{5, 6, 11}, v{5, 11, 7}, v{5, 6, 7}},
      p{v{5, 6, 11}, v{11, 7}, v{5, 6, 7}},
      p{v{5, 6, 11}, v{1, 11, 7}, v{5, 6, 7}},
      p{v{6, 11}, v{5, 11, 7}, v{5, 6, 7}},
      p{v{1, 6, 11}, v{5, 11, 7}, v{5, 6, 7}},
      p{v{8, 1, 2, 3}, v{8, 3, 4}, v{8, 8, 2, 4}},
      p{v{8, 8, 2, 3}, v{8, 1, 3, 2}, v{8, 8, 2, 2}},
  };
  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(BatchDot, BatchDotTest,
                         ::testing::ValuesIn(GetBatchDotTestCases()));

class DotOperationTextTest : public HloTestBase {};

XLA_TEST_F(DotOperationTextTest, DotReorderedDotDims) {
  absl::string_view hlo_string =
      R"(
HloModule ComplexDotMultipleNonContracting

ENTRY %test {
  %lhs = f32[7,17,10,13]{3,2,1,0} parameter(0)
  %rhs = f32[7,9,10,13,6]{4,3,2,1,0} parameter(1)
  ROOT %dot = f32[10,7,17,9,6]{4,3,2,1,0} dot(%lhs, %rhs), lhs_batch_dims={2,0}, rhs_batch_dims={2,0}, lhs_contracting_dims={3}, rhs_contracting_dims={3}
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-3, 1e-3}));
}

XLA_TEST_F(DotOperationTextTest, DotReorderedDotDimsAndMultipleContracting) {
  absl::string_view hlo_string =
      R"(
HloModule ComplexDotMultipleNonContracting

ENTRY %test {
  %lhs = f32[7,5,17,10,13]{4,3,2,1,0} parameter(0)
  %rhs = f32[7,9,10,13,6,5]{5,4,3,2,1,0} parameter(1)
  ROOT %dot = f32[10,7,17,9,6]{4,3,2,1,0} dot(%lhs, %rhs), lhs_batch_dims={3,0}, rhs_batch_dims={2,0}, lhs_contracting_dims={1,4}, rhs_contracting_dims={5,3}
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-3, 1e-3}));
}

XLA_TEST_F(DotOperationTextTest, DotWithNoDnums) {
  absl::string_view hlo_string =
      R"(
HloModule DotWithNoDnums

ENTRY %test {
  %lhs = f32[2,3]{1,0} parameter(0)
  %rhs = f32[4,5]{1,0} parameter(1)
  ROOT %dot = f32[2,3,4,5]{3,2,1,0} dot(%lhs, %rhs)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-3, 1e-3}));
}

XLA_TEST_F(DotOperationTextTest, Einsum) {
  absl::string_view hlo_string =
      R"(
HloModule Einsum

ENTRY %test {
  %lhs = f32[8,64,96]{2,1,0} parameter(0)
  %rhs = f32[96,32,4]{2,1,0} parameter(1)
  ROOT %dot = f32[8,64,32,4]{3,2,1,0}  dot(%lhs, %rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{4e-3, 4e-3}));
}

XLA_TEST_F(DotOperationTextTest, CpuTiledDotEmitterCachingBug_1) {
  // Tests for a caching bug in the XLA CPU backend.
  absl::string_view hlo_string =
      R"(
HloModule CpuTiledDotEmitterCachingBug

ENTRY main {
  lhs = f32[20,40] parameter(0)
  rhs_0 = f32[40,1] parameter(2)
  rhs_1 = f32[1,40] parameter(1)

  dot_0 = f32[20,1] dot(lhs, rhs_0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_1 = f32[20,1] dot(lhs, rhs_1), lhs_contracting_dims={1}, rhs_contracting_dims={1}

  ROOT result = f32[20,1] divide(dot_0, dot_1)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{4e-3, 4e-3}));
}

XLA_TEST_F(DotOperationTextTest, CpuTiledDotEmitterCachingBug_2) {
  // Tests for a caching bug in the XLA CPU backend.
  absl::string_view hlo_string =
      R"(
HloModule CpuTiledDotEmitterCachingBug

ENTRY main {
  lhs_0 = f32[20,40] parameter(0)
  rhs_0 = f32[40,1] parameter(1)
  lhs_1 = f32[1,40] parameter(2)
  rhs_1 = f32[20,40] parameter(3)

  dot_0 = f32[20,1] dot(lhs_0, rhs_0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_1 = f32[1,20] dot(lhs_1, rhs_1), lhs_contracting_dims={1}, rhs_contracting_dims={1}

  dot_0_reshaped = f32[20] reshape(dot_0)
  dot_1_reshaped = f32[20] reshape(dot_1)

  ROOT result = f32[20] divide(dot_0_reshaped, dot_1_reshaped)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{4e-3, 4e-3}));
}

XLA_TEST_F(DotOperationTextTest, DISABLED_ON_CPU(GpuIntegerDotCodegen)) {
  absl::string_view hlo_string =
      R"(
HloModule SmallIntegerDot

ENTRY SmallIntegerDot {
  arg0 = s32[1,2,2] parameter(0)
  arg1 = s32[1,2,1] parameter(1)
  ROOT dot = s32[1,2,1] dot(arg0, arg1), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{4e-3, 4e-3}));
}

XLA_TEST_F(DotOperationTextTest, DISABLED_ON_CPU(GpuTransposeOutput)) {
  absl::string_view hlo_string =
      R"(
HloModule TransposeOutput

ENTRY TransposeOutput {
  p0 = f32[32,32] parameter(0)
  p1 = f32[32,64] parameter(1)
  dot = f32[32,64] dot(p0, p1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  ROOT tr = f32[64,32] transpose(dot), dimensions={1,0}
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{4e-3, 4e-3}));
}

XLA_TEST_F(DotOperationTextTest, MatrixVectorComplex) {
  absl::string_view hlo_string =
      R"(
HloModule MatrixVectorComplex

ENTRY MatrixVectorComplex {
  p0 = c64[5,5] parameter(0)
  p1 = c64[5,1] parameter(1)
  p2 = c64[5,1] parameter(2)
  dot = c64[5,1] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT add = c64[5,1] add(dot, p2)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{4e-3, 4e-3}));
}

// Regression test for b/138155357, where we were incorrectly creating a dot-add
// fusion where the dot had a batch dimension.  This isn't supported on the CPU
// backend.
XLA_TEST_F(DotOperationTextTest, FusedBatchDotRegressionTest) {
  absl::string_view module_string = R"(
HloModule jaxpr_computation__5.33

jaxpr_computation__6.8 {
  tuple.9 = () tuple()
  parameter.14 = () parameter(4)
  parameter.13 = (f32[2]{0}) parameter(3)
  get-tuple-element.15 = f32[2]{0} get-tuple-element(parameter.13), index=0
  reshape.16 = f32[1,2]{1,0} reshape(get-tuple-element.15)
  parameter.10 = f32[2,2]{1,0} parameter(0)
  reshape.17 = f32[2,1]{1,0} reshape(get-tuple-element.15)
  dot.18 = f32[2,1]{1,0} dot(parameter.10, reshape.17), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reshape.19 = f32[2]{0} reshape(dot.18)
  reshape.20 = f32[2,1]{1,0} reshape(reshape.19)
  dot.21 = f32[1,1]{1,0} dot(reshape.16, reshape.20), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reshape.22 = f32[] reshape(dot.21)
  parameter.11 = f32[2,1,2]{2,1,0} parameter(1)
  broadcast.23 = f32[2,2,1]{2,1,0} broadcast(reshape.20), dimensions={1,2}
  dot.24 = f32[2,1,1]{2,1,0} dot(parameter.11, broadcast.23), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  broadcast.25 = f32[2,1,2]{2,1,0} broadcast(reshape.16), dimensions={1,2}
  parameter.12 = f32[2,2,1]{2,1,0} parameter(2)
  dot.26 = f32[2,1,1]{2,1,0} dot(broadcast.25, parameter.12), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  add.27 = f32[2,1,1]{2,1,0} add(dot.24, dot.26)
  reshape.28 = f32[2]{0} reshape(add.27)
  ROOT tuple.29 = (f32[], f32[2]{0}) tuple(reshape.22, reshape.28)
}

ENTRY jaxpr_computation__5.33 {
  constant.2 = f32[] constant(1)
  broadcast.3 = f32[2,2]{1,0} broadcast(constant.2), dimensions={}
  constant.5 = f32[2,1,2]{2,1,0} constant({ { { 1, 0 } }, { { 0, 1 } } })
  constant.4 = f32[2,2,1]{2,1,0} constant({ { {1}, {1} }, { {1}, {1} } })
  parameter.6 = f32[2]{0} parameter(0)
  tuple.7 = (f32[2]{0}) tuple(parameter.6)
  tuple.1 = () tuple()
  call.30 = (f32[], f32[2]{0}) call(broadcast.3, constant.5, constant.4, tuple.7, tuple.1), to_apply=jaxpr_computation__6.8
  get-tuple-element.31 = f32[] get-tuple-element(call.30), index=0
  ROOT get-tuple-element.32 = f32[2]{0} get-tuple-element(call.30), index=1
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  EXPECT_TRUE(RunAndCompare(std::move(module), /*error=*/absl::nullopt));
}

XLA_TEST_F(DotOperationTest, ReorderContractingDimsConstLHS_RL) {
  Array3D<float> input_arr(2, 3, 2);
  Array2D<float> const_arr(2, 6);
  input_arr.FillIota(0);
  const_arr.FillIota(0);

  XlaBuilder builder(TestName());
  auto t0 =
      AddParam(LiteralUtil::CreateR3FromArray3D<float>(input_arr), &builder);
  auto t1 = Transpose(t0, {1, 0, 2});
  auto rhs = Reshape(t1, {6, 2});
  auto lhs = ConstantR2FromArray2D(&builder, const_arr);
  Dot(lhs, rhs);

  ComputeAndCompare(&builder, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, ReorderContractingDimsConstRHS_LR) {
  Array3D<float> input_arr(2, 3, 2);
  Array2D<float> const_arr(2, 6);
  input_arr.FillIota(0);
  const_arr.FillIota(0);

  XlaBuilder builder(TestName());
  auto t0 =
      AddParam(LiteralUtil::CreateR3FromArray3D<float>(input_arr), &builder);
  auto t1 = Transpose(t0, {1, 0, 2});
  auto lhs = Reshape(t1, {6, 2});
  auto rhs = ConstantR2FromArray2D(&builder, const_arr);

  DotDimensionNumbers dims;
  dims.add_lhs_contracting_dimensions(0);
  dims.add_rhs_contracting_dimensions(1);
  DotGeneral(lhs, rhs, dims);

  ComputeAndCompare(&builder, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, ReorderContractingDimsConstRHS_RL) {
  Array4D<float> input_arr(2, 2, 3, 4);
  Array2D<float> const_arr(24, 2);
  input_arr.FillIota(0);
  const_arr.FillIota(0);

  XlaBuilder builder(TestName());
  auto t0 =
      AddParam(LiteralUtil::CreateR4FromArray4D<float>(input_arr), &builder);
  auto t1 = Transpose(t0, {0, 2, 3, 1});
  auto lhs = Reshape(t1, {2, 24});
  auto rhs = ConstantR2FromArray2D(&builder, const_arr);
  Dot(lhs, rhs);

  ComputeAndCompare(&builder, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, ReorderContractingDimsConstRHS_MM) {
  Array3D<float> input_arr(2, 6, 2);
  Array3D<float> const_arr(2, 6, 3);
  input_arr.FillIota(0);
  const_arr.FillIota(0);

  XlaBuilder builder(TestName());
  auto t0 =
      AddParam(LiteralUtil::CreateR3FromArray3D<float>(input_arr), &builder);
  auto t1 = Reshape(t0, {2, 2, 3, 2});
  auto t2 = Transpose(t1, {0, 2, 1, 3});
  auto lhs = Reshape(t2, {2, 6, 2});
  auto rhs = ConstantR3FromArray3D(&builder, const_arr);

  DotDimensionNumbers dims;
  dims.add_lhs_contracting_dimensions(1);
  dims.add_rhs_contracting_dimensions(1);
  dims.add_lhs_batch_dimensions(0);
  dims.add_rhs_batch_dimensions(0);
  DotGeneral(lhs, rhs, dims);

  ComputeAndCompare(&builder, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, ReorderContractingDims_Multipass) {
  Array4D<float> input_arr(2, 2, 3, 5);
  Array2D<float> const_arr(2, 30);
  input_arr.FillIota(0);
  const_arr.FillIota(0);

  XlaBuilder builder(TestName());
  auto t0 =
      AddParam(LiteralUtil::CreateR4FromArray4D<float>(input_arr), &builder);
  auto t1 = Transpose(t0, {0, 2, 1, 3});
  auto t2 = Reshape(t1, {2, 6, 5});
  auto t3 = Transpose(t2, {0, 2, 1});
  auto lhs = Reshape(t3, {2, 30});
  auto rhs = ConstantR2FromArray2D(&builder, const_arr);

  DotDimensionNumbers dims;
  dims.add_lhs_contracting_dimensions(1);
  dims.add_rhs_contracting_dimensions(1);
  DotGeneral(lhs, rhs, dims);

  // Constant folding are disabled by default in unit tests. algsimp
  // optimization can be applied multiple times if we fold the transpose
  // and reshape that are moved to the constant side of the dot.
  mutable_debug_options()->clear_xla_disable_hlo_passes();
  ComputeAndCompare(&builder, {}, error_spec_);
}

// This benchmark is to show the performance impact of the following
// transformation:
//   dot(reshape(transpose(A)), Const) ==>
//   dot(reshape(A), reshape(transpose(reshape(Const)))),
// and then fold the reshape and transpose on the Const side.
// We can compare performance with and without algsimp pass to see the impact.
void DOT_ReorderContracting(int num_iters) {
  tensorflow::testing::StopTiming();

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().ValueOrDie();
  auto executors = PlatformUtil::GetStreamExecutors(platform).ValueOrDie();
  se::StreamExecutorMemoryAllocator allocator(platform, executors);

  xla::LocalClientOptions client_options;
  client_options.set_platform(platform);
  auto client =
      ClientLibrary::GetOrCreateLocalClient(client_options).ValueOrDie();

  int device_ordinal = client->default_device_ordinal();

  const int64 d0 = 128;
  const int64 d1 = 128;
  const int64 d2 = 128;
  const int64 d3 = 128;

  Array3D<float> input_arr(d0, d1, d2);
  Array2D<float> const_arr(d1 * d2, d3);
  input_arr.FillIota(0);
  const_arr.FillIota(0);
  XlaBuilder builder("ReorderContracting");
  auto t0 =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {d0, d1, d2}), "param0");
  auto t1 = Transpose(t0, {0, 2, 1});
  auto lhs = Reshape(t1, {d0, d2 * d1});
  auto rhs = ConstantR2FromArray2D(&builder, const_arr);
  Dot(lhs, rhs);
  auto computation = builder.Build().ConsumeValueOrDie();

  auto input_literal = LiteralUtil::CreateR3FromArray3D<float>(input_arr);
  ScopedShapedBuffer buffer0 =
      client->LiteralToShapedBuffer(input_literal, device_ordinal)
          .ConsumeValueOrDie();

  std::unique_ptr<LocalExecutable> executable =
      client
          ->Compile(computation, {&buffer0.on_host_shape()},
                    ExecutableBuildOptions())
          .ConsumeValueOrDie();

  se::Stream stream(executors[device_ordinal]);
  stream.Init();

  ExecutableRunOptions options;
  options.set_allocator(&allocator);

  const int kWarmups = 2;
  for (int i = 0; i < kWarmups; ++i) {
    ASSERT_IS_OK(executable->Run({&buffer0}, options));
  }

  const int64 total_bytes = d0 * d1 * d2 + d1 * d2 * d3 + d0 * d3;
  tensorflow::testing::BytesProcessed(static_cast<int64>(num_iters) *
                                      total_bytes * sizeof(float));
  tensorflow::testing::UseRealTime();
  tensorflow::testing::StartTiming();
  for (int i = 0; i < num_iters; ++i) {
    ASSERT_IS_OK(executable->Run({&buffer0}, options));
  }
}

BENCHMARK(DOT_ReorderContracting);

}  // namespace
}  // namespace xla
