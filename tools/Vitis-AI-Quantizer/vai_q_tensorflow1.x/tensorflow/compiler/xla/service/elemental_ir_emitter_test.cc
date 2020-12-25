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

#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

using absl::nullopt;

class ElementalIrEmitterExecutionTest : public HloTestBase {
 protected:
  void RunTest(const string& hlo_text, absl::Span<Literal* const> args) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), args, nullopt));
  }
};

XLA_TEST_F(ElementalIrEmitterExecutionTest, DotFusion) {
  const string hlo_text = R"(
HloModule FusedDot

fused_computation {
  arg0 = s32[1,2,1]{2,1,0} parameter(0)
  reshape.lhs = s32[2,1]{1,0} reshape(arg0)
  arg1 = s32[1,2,1]{2,1,0} parameter(1)
  reshape.rhs = s32[2,1]{1,0} reshape(arg1)
  ROOT dot = s32[1,1]{1,0} dot(reshape.lhs, reshape.rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY main {
  entry_arg0 = s32[1,2,1]{2,1,0} parameter(0)
  entry_arg1 = s32[1,2,1]{2,1,0} parameter(1)
  ROOT fusion = s32[1,1]{1,0} fusion(entry_arg0, entry_arg1), kind=kLoop, calls=fused_computation
}
)";

  Literal lhs = LiteralUtil::CreateR3<int32>({{{1}, {2}}});
  Literal rhs = LiteralUtil::CreateR3<int32>({{{3}, {4}}});
  RunTest(hlo_text, {&lhs, &rhs});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ScalarDotFusion) {
  const char* hlo_text = R"(
HloModule ScalarDotFusion

fused_computation {
  arg0 = s32[2,2]{1,0} parameter(0)
  reshape.lhs = s32[4]{0} reshape(arg0)
  arg1 = s32[2,2]{1,0} parameter(1)
  reshape.rhs = s32[4]{0} reshape(arg1)
  ROOT dot = s32[] dot(reshape.lhs, reshape.rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY main {
  entry_arg0 = s32[2,2]{1,0} parameter(0)
  entry_arg1 = s32[2,2]{1,0} parameter(1)
  ROOT fusion = s32[] fusion(entry_arg0, entry_arg1), kind=kLoop, calls=fused_computation
}
)";

  Literal lhs = LiteralUtil::CreateR2<int32>({{1, 2}, {3, 4}});
  Literal rhs = LiteralUtil::CreateR2<int32>({{10, 20}, {30, 40}});
  RunTest(hlo_text, {&lhs, &rhs});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, BatchDot) {
  const char* hlo_text = R"(
HloModule BatchDot

fused_computation.1 {
  param_0 = f64[1,1,8]{2,1,0} parameter(0)
  r.1 = f64[2,4]{1,0} reshape(param_0)
  param_1 = f64[1,2,2,2,1]{4,3,2,1,0} parameter(1)
  r.2 = f64[2,4,1]{2,1,0} reshape(param_1)
  ROOT dot = f64[2,1]{1,0} dot(r.1, r.2), lhs_batch_dims={0},
                                          lhs_contracting_dims={1},
                                          rhs_batch_dims={0},
                                          rhs_contracting_dims={1}
}

ENTRY resampler_Resampler.49 {
  p0 = f64[1,1,8]{2,1,0} parameter(0)
  p1 = f64[1,2,2,2,1]{4,3,2,1,0} parameter(1)
  ROOT f = f64[2,1]{1,0} fusion(p0, p1), kind=kLoop, calls=fused_computation.1
}
)";

  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  // Disable the layout assignment pass because it would throw away the layouts
  // in the fusion computation, but not recreate them.
  debug_options.add_xla_disable_hlo_passes("layout-assignment");
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{4e-3, 4e-3}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest,
           DivideComplexNumbersWithInfiniteNormRhs) {
  constexpr char hlo_text[] = R"(
    HloModule DivideComplexNumbers
    ENTRY DivideComplexNumbers {
      constant.1 = c64[8]{0} constant({
        (1, 1),     (1, inf),   (1, inf),   (nan, 1),
        (inf, inf), (inf, nan), (nan, nan), (1, 2)})
      real = f32[8]{0} constant({nan, nan, inf, inf, inf, 1, inf, 3})
      imag = f32[8]{0} constant({inf, inf, inf, inf, 1, inf, inf, 4})
      complex.2 = c64[8]{0} complex(real, imag)
      ROOT divide.1 = c64[8]{0} divide(constant.1, complex.2)
    }
  )";
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_fast_math_honor_nans(true);
  debug_options.set_xla_cpu_fast_math_honor_infs(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest,
           DivideComplexNumbersWithFiniteNormRhs) {
  constexpr char hlo_text[] = R"(
    HloModule DivideComplexNumbers
    ENTRY DivideComplexNumbers {
      constant.1 = c64[5]{0} constant({
        (1, inf), (inf, 1), (inf, nan), (inf, inf), (nan, inf)})
      real = f32[5]{0} constant({1, 1, 1, 1, 1})
      imag = f32[5]{0} constant({1, 1, 1, 1, 1})
      complex.2 = c64[5]{0} complex(real, imag)
      ROOT divide.1 = c64[5]{0} divide(constant.1, complex.2)
    }
  )";
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_fast_math_honor_nans(true);
  debug_options.set_xla_cpu_fast_math_honor_infs(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest,
           DivideComplexNumbersWithZeroNormRhs) {
  constexpr char hlo_text[] = R"(
    HloModule DivideComplexNumbers
    ENTRY DivideComplexNumbers {
      constant.1 = c64[9]{0} constant({
        (1, 1),     (1, nan), (1, inf),   (inf, inf), (inf, 1),
        (inf, nan), (nan, 1), (nan, inf), (nan, nan)})
      real = f32[9]{0} constant({0, 0, 0, 0, 0, 0, 0, 0, 0})
      imag = f32[9]{0} constant({0, 0, 0, 0, 0, 0, 0, 0, 0})
      complex.2 = c64[9]{0} complex(real, imag)
      ROOT divide.1 = c64[9]{0} divide(constant.1, complex.2)
    }
  )";
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_fast_math_honor_nans(true);
  debug_options.set_xla_cpu_fast_math_honor_infs(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
}

}  // namespace
}  // namespace xla
