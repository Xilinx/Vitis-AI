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
#include <cstdarg>
#include <limits>
#include <random>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {

namespace ops {
namespace builtin {

// Tanh kernel registrations.
TfLiteRegistration* Register_TANH_REF();
TfLiteRegistration* Register_TANH_GENERIC_OPT();
TfLiteRegistration* Register_TANH_FIXED_POINT_OPT();

// Logistic kernel registrations.
TfLiteRegistration* Register_LOGISTIC_REF();
TfLiteRegistration* Register_LOGISTIC_GENERIC_OPT();
TfLiteRegistration* Register_LOGISTIC_FIXED_POINT_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

class BaseActivationsOpModel : public SingleOpModel {
 public:
  // Most activations don't take any options, so this constructor works for
  // them.
  BaseActivationsOpModel(BuiltinOperator type, TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else if (input.type == TensorType_INT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256, -128});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  BaseActivationsOpModel(TfLiteRegistration* registration, BuiltinOperator type,
                         TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else if (input.type == TensorType_INT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256, -128});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    resolver_ = absl::make_unique<SingleOpResolver>(type, registration);
    BuildInterpreter({GetShape(input_)});
  }

  // A dedicated constructor for SOFTMAX, which does some options.
  BaseActivationsOpModel(float softmax_beta, TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else if (input.type == TensorType_INT8) {
      output_ = AddOutput({TensorType_INT8, {}, 0, 0, 1. / 256, -128});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, softmax_beta).Union());
    BuildInterpreter({GetShape(input_)});
  }

  // A dedicated constructor for LeakyRelu, which does some options.
  BaseActivationsOpModel(TensorData input, float alpha) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, input.min, input.max});
    } else if (input.type == TensorType_INT8) {
      output_ = AddOutput({TensorType_INT8, {}, input.min, input.max});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(BuiltinOperator_LEAKY_RELU, BuiltinOptions_LeakyReluOptions,
                 CreateLeakyReluOptions(builder_, alpha).Union());
    BuildInterpreter({GetShape(input_)});
  }

  BaseActivationsOpModel(BuiltinOperator type, const TensorData& input,
                         const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  BaseActivationsOpModel(TfLiteRegistration* registration, BuiltinOperator type,
                         const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    resolver_ = absl::make_unique<SingleOpResolver>(type, registration);
    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  int output_;
};

class FloatActivationsOpModel : public BaseActivationsOpModel {
 public:
  using BaseActivationsOpModel::BaseActivationsOpModel;

  void SetInput(const std::vector<float>& data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

// Our fixed-point math function implementations have roughly 12 bits of
// accuracy, when specialized to 16-bit fixed-point arithmetic.
// That is purely an implementation compromise, it would have been possible
// to get closer to 16 bits of accuracy but that would be more expensive,
// and not needed for our purposes as ultimately the output is either
// immediately down-quantized to 8 bits, or will typically be at the output
// of the surrounding LSTM cell.
// So we can require roughly 2^-12 accuracy when the output is 16-bit, and
// we can more or less expect the full 2^-8 accuracy when the output is 8-bit.
//
// However, the representable output interval is often [-1, 1]  (it has to be
// for tanh, and even for logistic, when we implement it in fixed-point, we
// typically have to do so on such a symmetric interval, e.g. ARM NEON only
// has signed fixed-point arithmetic (SQRDMULH)).  As the width of [-1, 1]
// is 2, our representable values are often diluted by a factor of 2, whence
// the factor of 2 below.
const float kQuantizedTolerance = 2 * (1. / 256);
const float kQuantizedToleranceInt16 = 2 * (1. / 4096);

class QuantizedActivationsOpModel : public BaseActivationsOpModel {
 public:
  using BaseActivationsOpModel::BaseActivationsOpModel;

  template <typename T>
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }
  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }
};

const auto kTanhKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_TANH_REF()},
    {"GenericOptimized", ops::builtin::Register_TANH_GENERIC_OPT()},
    {"FixedPointOptimized", ops::builtin::Register_TANH_FIXED_POINT_OPT()},
});

class TanhOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kTanhKernelMap;
  }
};

const auto kLogisticKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_LOGISTIC_REF()},
    {"GenericOptimized", ops::builtin::Register_LOGISTIC_GENERIC_OPT()},
    {"FixedPointOptimized", ops::builtin::Register_LOGISTIC_FIXED_POINT_OPT()},
});

class LogisticOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kLogisticKernelMap;
  }
};

TEST(FloatActivationsOpTest, Elu) {
  FloatActivationsOpModel m(BuiltinOperator_ELU,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, -4,     //
      3, -2, 10, -0.1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.0, -0.997521, 2.0, -0.981684,    //
                                 3.0, -0.864665, 10.0, -0.0951626,  //
                             })));
}

TEST(FloatActivationsOpTest, Relu) {
  FloatActivationsOpModel m(BuiltinOperator_RELU,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,   //
                                 3, 0, 10, 1,  //
                             }));
}

TEST(FloatActivationsOpTest, Relu1) {
  FloatActivationsOpModel m(BuiltinOperator_RELU_N1_TO_1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0, -0.6, 0.2, -0.4,  //
                                 0.3, -1.0, 1.0, -0.1,  //
                             }));
}

TEST(FloatActivationsOpTest, Relu6) {
  FloatActivationsOpModel m(BuiltinOperator_RELU6,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,  //
                                 3, 0, 6, 1,  //
                             }));
}

void GenerateUniformRandomVector(int size, float min, float max,
                                 std::minstd_rand* random_engine,
                                 std::vector<float>* result) {
  // Never use std::uniform_*_distribution in tests, it's
  // implementation-defined. Likewise, don't use std::default_random_engine,
  // implementation-defined. Implementation-defined is bad because it means that
  // any toolchain update or new platform may run into test failures.
  // std::minstd_rand is a standard instantiation of
  // std::linear_congruential_engine, the cheapest generator in c++11 stdlib,
  // it's good enough here.
  result->resize(size);
  for (int i = 0; i < size; i++) {
    // We don't care whether the `max` value may ever be produced exactly.
    // It may actually be thanks to rounding, as std::minstd_rand::modulus
    // is 2^31 - 1 is greater than the inverse float epsilon.
    float random_value_scaled_0_1 =
        (*random_engine)() *
        (1.0f / static_cast<float>(std::minstd_rand::modulus));
    (*result)[i] = min + (max - min) * random_value_scaled_0_1;
  }
}

void EvalTestReferenceHardSwish(int size, const std::vector<float>& input,
                                std::vector<float>* result) {
  result->resize(size);
  for (int i = 0; i < size; i++) {
    const float in = input[i];
    (*result)[i] = in * std::min(6.0f, std::max(0.0f, in + 3)) * (1.0f / 6.0f);
  }
}

void TestFloatHardSwish(int size, std::minstd_rand* random_engine) {
  std::vector<float> float_input_values;
  const float kMin = -10.0f;
  const float kMax = 10.0f;
  GenerateUniformRandomVector(size, kMin, kMax, random_engine,
                              &float_input_values);
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  FloatActivationsOpModel m(BuiltinOperator_HARD_SWISH,
                            /*input=*/{TensorType_FLOAT32, {1, 1, 1, size}},
                            /*output=*/{TensorType_FLOAT32, {1, 1, 1, size}});
  m.SetInput(float_input_values);

  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(float_ref_output_values)));
}

template <typename QuantizedType>
void TestQuantizedHardSwish(TensorType tensor_type, int size, float input_min,
                            float input_max, float output_min, float output_max,
                            std::minstd_rand* random_engine) {
  std::vector<float> float_input_values;
  GenerateUniformRandomVector(size, input_min, input_max, random_engine,
                              &float_input_values);
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  for (float& val : float_ref_output_values) {
    val = std::min(output_max, std::max(output_min, val));
  }
  QuantizedActivationsOpModel m(
      BuiltinOperator_HARD_SWISH,
      /*input=*/{tensor_type, {1, 1, 1, size}, input_min, input_max},
      /*output=*/{tensor_type, {1, 1, 1, size}, output_min, output_max});
  m.SetInput<QuantizedType>(float_input_values);

  m.Invoke();
  const std::vector<float>& dequantized_output =
      m.GetDequantizedOutput<QuantizedType>();
  // The numerical error for any 8bit quantized function is at least one half
  // times the quantization step: 0.5 * (kOutMax - kOutMin) / 256.
  // To that we add again the quantization step (kOutMax - kOutMin) / 256
  // to allow for an off-by-one rounding error.
  const float kTolerance =
      std::max(input_max - input_min, output_max - output_min) * (1.5f / 256.f);
  EXPECT_THAT(dequantized_output, ElementsAreArray(ArrayFloatNear(
                                      float_ref_output_values, kTolerance)));
}

template <typename QuantizedType>
void TestQuantizedHardSwishBias(TensorType tensor_type, float input_min,
                                float input_max, float output_min,
                                float output_max, float tolerated_bias) {
  const float quantized_type_range =
      static_cast<float>(std::numeric_limits<QuantizedType>::max()) -
      static_cast<float>(std::numeric_limits<QuantizedType>::min());
  const float input_scale = (input_max - input_min) / quantized_type_range;
  const float output_scale = (output_max - output_min) / quantized_type_range;
  const float max_scale = std::max(output_scale, input_scale);

  // In this bias-focused test case, no need for randomly generated input
  // values.
  ASSERT_LE(input_min, -3.0f);
  ASSERT_GE(input_max, 3.0f);
  const int quantized_input_negative_three =
      std::round(std::numeric_limits<QuantizedType>::min() +
                 (-3.0f - input_min) / input_scale);
  const int quantized_input_positive_three =
      std::round(std::numeric_limits<QuantizedType>::min() +
                 (3.0f - input_min) / input_scale);
  std::vector<float> float_input_values;
  for (int i = quantized_input_negative_three;
       i <= quantized_input_positive_three; i++) {
    float_input_values.push_back(
        input_min +
        (i - std::numeric_limits<QuantizedType>::min()) * input_scale);
  }
  const int size = float_input_values.size();
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  for (float& val : float_ref_output_values) {
    val = std::min(output_max, std::max(output_min, val));
  }
  QuantizedActivationsOpModel m(
      BuiltinOperator_HARD_SWISH,
      /*input=*/{tensor_type, {1, 1, 1, size}, input_min, input_max},
      /*output=*/{tensor_type, {1, 1, 1, size}, output_min, output_max});
  m.SetInput<QuantizedType>(float_input_values);

  m.Invoke();
  const std::vector<float>& dequantized_output =
      m.GetDequantizedOutput<QuantizedType>();

  float sum_diff = 0;
  for (int i = 0; i < size; i++) {
    sum_diff += dequantized_output[i] - float_ref_output_values[i];
  }
  const float bias = sum_diff / (size * max_scale);
  EXPECT_LE(std::abs(bias), tolerated_bias);
}

TEST(FloatActivationsOpTest, HardSwish) {
  std::minstd_rand random_engine;
  for (int size : {1, 2, 3, 4, 10, 20, 30, 40, 100}) {
    TestFloatHardSwish(size, &random_engine);
  }
}

TEST(QuantizedActivationsOpTest, HardSwish) {
  std::minstd_rand random_engine;
  std::vector<std::pair<float, float>> minmax_pairs{
      {0.f, 1.f}, {-2.f, 1.f}, {-5.f, 10.f}, {-40.f, 60.f}};
  for (const auto& input_minmax : minmax_pairs) {
    for (const auto& output_minmax : minmax_pairs) {
      float input_min = input_minmax.first;
      float input_max = input_minmax.second;
      float output_min = output_minmax.first;
      float output_max = output_minmax.second;
      for (int size : {1, 3, 10, 100}) {
        TestQuantizedHardSwish<uint8_t>(TensorType_UINT8, size, input_min,
                                        input_max, output_min, output_max,
                                        &random_engine);
        TestQuantizedHardSwish<int8_t>(TensorType_INT8, size, input_min,
                                       input_max, output_min, output_max,
                                       &random_engine);
      }
    }
  }
}

// See the comment in the reference implementation of quantized HardSwish:
// A numerical issue significantly affecting ImageNet classification accuracy
// with MobileNet v3 is only observable at the scale of HardSwish unit tests
// if we monitor specifically bias. This testcase is extracted from one of the
// HardSwish nodes in that MobileNet v3 that exhibited this issue.
TEST(QuantizedActivationsOpTest, HardSwishBias) {
  TestQuantizedHardSwishBias<uint8_t>(TensorType_UINT8, -11.654928f, 25.036512f,
                                      -0.3905796f, 24.50887f, 0.035);
}

TEST_P(TanhOpTest, Tanh) {
  FloatActivationsOpModel m(GetRegistration(), BuiltinOperator_TANH,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0, -0.9999877, 0.9640275, 0.999329,    //
                                 0.99505475, -0.9640275, 1, 0.7615941,  //
                             })));
}

TEST(QuantizedActivationsOpTest, Relu6Uint8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU6,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_UINT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0, 0, 2, 4,  //
                      3, 0, 6, 1,  //
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({128, 128, 160, 192, 176, 128, 224, 144}));
}

TEST(QuantizedActivationsOpTest, LeakyReluUint8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      /*input=*/{TensorType_UINT8, {2, 3}, 8 * kMin, 8 * kMax}, 0.5);

  m.SetInput<uint8_t>({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 1.0f, 3.0f,    // Row 1
                      1.0f, -0.5f, -1.0f,  // Row 2
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({
                                          128,
                                          144,
                                          176,
                                          144,
                                          120,
                                          112,
                                      }));
}

TEST(QuantizedActivationsOpTest, Relu1Int8) {
  const float kMin = -1;
  const float kMax = 1;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU_N1_TO_1,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 2 * kMin, kMax},
      /*output=*/{TensorType_INT8, {1, 2, 4, 1}, 2 * kMin, kMax});

  m.SetInput<int8_t>({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0, -0.6, 0.2, -0.4,  //
                      0.3, -1.0, 1.0, -0.1,  //
                  },
                  kQuantizedTolerance)));
}

TEST(QuantizedActivationsOpTest, Relu1UInt8) {
  const float kMin = -1;
  const float kMax = 1;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU_N1_TO_1,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 2 * kMin, kMax},
      /*output=*/{TensorType_UINT8, {1, 2, 4, 1}, 2 * kMin, kMax});

  m.SetInput<uint8_t>({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0, -0.6, 0.2, -0.4,  //
                      0.3, -1.0, 1.0, -0.1,  //
                  },
                  kQuantizedTolerance)));
}

TEST(QuantizedActivationsOpTest, Relu6Int8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU6,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                                    {
                                                        0, 0, 2, 4,  //
                                                        3, 0, 6, 1,  //
                                                    },
                                                    kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 0, 32, 64, 48, 0, 96, 16}));
}

TEST_P(TanhOpTest, TanhUint8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_TANH,
      /*input=*/{TensorType_UINT8, {2, 6, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_UINT8, {2, 6, 4, 1}, kMin, kMax});
  m.SetInput<uint8_t>({
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                  },
                  kQuantizedTolerance)));
  if (GetParam() == "Reference") {
    EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({
                                            128, 0, 251, 255,  //
                                            0,   5, 255, 225,  //
                                            128, 0, 251, 255,  //
                                            0,   5, 255, 225,  //
                                            128, 0, 251, 255,  //
                                            0,   5, 255, 225,  //
                                            128, 0, 251, 255,  //
                                            0,   5, 255, 225,  //
                                            128, 0, 251, 255,  //
                                            0,   5, 255, 225,  //
                                            128, 0, 251, 255,  //
                                            0,   5, 255, 225,  //
                                        }));
  }
}

TEST_P(TanhOpTest, TanhInt8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_TANH,
      /*input=*/{TensorType_INT8, {2, 6, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT8, {2, 6, 4, 1}, kMin, kMax});
  m.SetInput<int8_t>({
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
      0,  -6, 2, 4,  //
      -4, -2, 8, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                      0.0,       -0.999987, 0.964027, 0.999329,  //
                      -0.999329, -0.96402,  0.99999,  0.76159,   //
                  },
                  kQuantizedTolerance)));
  if (GetParam() == "Reference") {
    EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                           0,    -128, 123, 127,  //
                                           -128, -123, 127, 97,   //
                                           0,    -128, 123, 127,  //
                                           -128, -123, 127, 97,   //
                                           0,    -128, 123, 127,  //
                                           -128, -123, 127, 97,   //
                                           0,    -128, 123, 127,  //
                                           -128, -123, 127, 97,   //
                                           0,    -128, 123, 127,  //
                                           -128, -123, 127, 97,   //
                                           0,    -128, 123, 127,  //
                                           -128, -123, 127, 97,   //
                                       }));
  }
}

TEST_P(TanhOpTest, TanhInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_TANH,
      /*input=*/{TensorType_INT16, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT16, {1, 2, 4, 1}, kMin, kMax});
  m.SetInput<int16_t>({
      0, -6, 2, 4,   //
      -4, -2, 8, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0, -0.999987, 0.964027, 0.999329,     //
                      -0.999329, -0.96402, 0.99999, 0.76159,  //
                  },
                  kQuantizedToleranceInt16)));
}

TEST_P(LogisticOpTest, Sigmoid) {
  FloatActivationsOpModel m(GetRegistration(), BuiltinOperator_LOGISTIC,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.5, 0.002473, 0.880797, 0.982014,       //
                                 0.952574, 0.119203, 0.999955, 0.731059,  //
                             })));
}

TEST_P(LogisticOpTest, SigmoidUint8) {
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_UINT8, {2, 6, 4, 1}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                  },
                  kQuantizedTolerance)));
  if (GetParam() == "Reference") {
    EXPECT_THAT(m.GetOutput<uint8_t>(),
                ElementsAreArray({
                    128, 1, 227, 251, 244, 32, 255, 188,  //
                    128, 1, 227, 251, 244, 32, 255, 188,  //
                    128, 1, 227, 251, 244, 32, 255, 188,  //
                    128, 1, 227, 251, 244, 32, 255, 188,  //
                    128, 1, 227, 251, 244, 32, 255, 188,  //
                    128, 1, 227, 251, 244, 32, 255, 188,  //
                }));
  }
}

TEST_P(LogisticOpTest, SigmoidInt8) {
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_INT8, {2, 6, 4, 1}, -10, 10});
  m.SetInput<int8_t>({
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                      0.5,      0.002473, 0.880797, 0.982014,  //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                  },
                  kQuantizedTolerance)));
  if (GetParam() == "Reference") {
    EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                           0,   -127, 99,  123,  //
                                           116, -99,  127, 60,   //
                                           0,   -127, 99,  123,  //
                                           116, -99,  127, 60,   //
                                           0,   -127, 99,  123,  //
                                           116, -99,  127, 60,   //
                                           0,   -127, 99,  123,  //
                                           116, -99,  127, 60,   //
                                           0,   -127, 99,  123,  //
                                           116, -99,  127, 60,   //
                                           0,   -127, 99,  123,  //
                                           116, -99,  127, 60,   //
                                       }));
  }
}

TEST_P(LogisticOpTest, SigmoidInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_INT16, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT16, {1, 2, 4, 1}, kMin, kMax});
  m.SetInput<int16_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.5, 0.002473, 0.880797, 0.982014,       //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                  },
                  kQuantizedToleranceInt16)));
}

TEST(FloatActivationsOpTest, Softmax4D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 1, 4}});
  m.SetInput({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(0.1,
                             /*input=*/{TensorType_FLOAT32, {4, 1, 1, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST(QuantizedActivationsOpTest, Softmax4DUint8) {
  QuantizedActivationsOpModel m(
      0.1,
      /*input=*/{TensorType_UINT8, {1, 2, 1, 4}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      0.1,
      /*input=*/{TensorType_UINT8, {4, 1, 1, 2}, -10, 10});
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax1D, the dequantized output is identical.
TEST(QuantizedActivationsOpTest, Softmax1DInt8) {
  QuantizedActivationsOpModel m(0.1,
                                /*input=*/{TensorType_INT8, {8}, -10, 10});
  m.SetInput<int8_t>({0, -6, 2, 4, 3, -2, 10, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({0.09766, 0.05469, 0.12109, 0.14453,
                                       0.13281, 0.07813, 0.26563, 0.10938},
                                      kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax2D, the dequantized output is identical.
TEST(QuantizedActivationsOpTest, Softmax2DInt8) {
  QuantizedActivationsOpModel m(0.1,
                                /*input=*/{TensorType_INT8, {2, 4}, -10, 10});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(0.1,
                                 /*input=*/{TensorType_INT8, {4, 2}, -10, 10});
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax3D, the dequantized output is identical.
TEST(QuantizedActivationsOpTest, Softmax3DInt8) {
  QuantizedActivationsOpModel m(
      0.1,
      /*input=*/{TensorType_INT8, {1, 2, 4}, -10, 10});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      0.1,
      /*input=*/{TensorType_INT8, {4, 1, 2}, -10, 10});
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax4D, the dequantized output is identical.
TEST(QuantizedActivationsOpTest, Softmax4DInt8) {
  QuantizedActivationsOpModel m(
      0.1,
      /*input=*/{TensorType_INT8, {1, 2, 1, 4}, -10, 10});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                         -68, -95, -54, -38,  //
                                         -70, -93, -12, -81,  //
                                     }));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      0.1,
      /*input=*/{TensorType_INT8, {4, 1, 1, 2}, -10, 10});
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

TEST(FloatActivationsOpTest, Softmax3D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4}});
  m.SetInput({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(0.1,
                             /*input=*/{TensorType_FLOAT32, {4, 1, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST(QuantizedActivationsOpTest, Softmax3DUint8) {
  QuantizedActivationsOpModel m(
      0.1,
      /*input=*/{TensorType_UINT8, {1, 2, 4}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      0.1,
      /*input=*/{TensorType_UINT8, {4, 1, 2}, -10, 10});
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

TEST(FloatActivationsOpTest, Softmax1D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {8}});
  m.SetInput({0, -6, 2, 4, 3, -2, 10, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {.09752, .05352, .11911, .14548, .13164, .07984, .26509, .10778})));
}

TEST(QuantizedActivationsOpTest, Softmax1DUint8) {
  QuantizedActivationsOpModel m(0.1,
                                /*input=*/{TensorType_UINT8, {8}, -10, 10});
  m.SetInput<uint8_t>({0, -6, 2, 4, 3, -2, 10, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({0.09766, 0.05469, 0.12109, 0.14453,
                                       0.13281, 0.07813, 0.26563, 0.10938},
                                      kQuantizedTolerance)));
}

TEST(FloatActivationsOpTest, Softmax2D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {2, 4}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(0.1,
                             /*input=*/{TensorType_FLOAT32, {4, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST(QuantizedActivationsOpTest, Softmax2DUint8) {
  QuantizedActivationsOpModel m(0.1,
                                /*input=*/{TensorType_UINT8, {2, 4}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(0.1,
                                 /*input=*/{TensorType_UINT8, {4, 2}, -10, 10});
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// This contains the same test values as the Softmax test, but reference answer
// generated via the following snippet of python:
//   logits1 = tf.constant([[0, -6, 2, 4],[3, -2, 10, 1]], dtype=tf.float32)
//   logits2 = tf.constant([[0,-6],[2,4],[3,-2],[10,1]], dtype=tf.float32)
//   lsm1 = tf.nn.log_softmax(logits1)
//   lsm2 = tf.nn.log_softmax(logits2)
//   with tf.Session() as sess:
//     print('lsm1', sess.run(lsm1))
//     print('lsm2', sess.run(lsm2))

TEST(FloatActivationsOpTest, LogSoftmax) {
  FloatActivationsOpModel m(BuiltinOperator_LOG_SOFTMAX,
                            /*input=*/{TensorType_FLOAT32, {2, 4}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 -4.14297, -10.14297, -2.14297, -.142971,    //
                                 -7.00104, -12.00104, -.00104087, -9.00104,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(BuiltinOperator_LOG_SOFTMAX,
                             /*input=*/{TensorType_FLOAT32, {4, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  -.00247565, -6.00247,   //
                                  -2.12692, -.126928,     //
                                  -.00671534, -5.00671,   //
                                  -.000123374, -9.00012,  //
                              })));
}

TEST(QuantizedActivationsOpTest, LogSoftmaxUint8) {
  const float kLogSoftmaxQuantizedTolerance = 16 / 256.0;
  QuantizedActivationsOpModel m(
      BuiltinOperator_LOG_SOFTMAX,
      /*input=*/{TensorType_UINT8, {2, 4}, -10, 10},
      /*output=*/{TensorType_UINT8, {}, 0, 0, 16. / 256, 255});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      -4.14297, -10.14297, -2.14297, -.142971,    //
                      -7.00104, -12.00104, -.00104087, -9.00104,  //
                  },
                  kLogSoftmaxQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({189, 93, 221, 253, 142, 63, 255, 111}));
}

TEST(QuantizedActivationsOpTest, LogSoftmaxInt8) {
  const float kLogSoftmaxQuantizedTolerance = 0.06355;
  QuantizedActivationsOpModel m(
      BuiltinOperator_LOG_SOFTMAX,
      /*input=*/{TensorType_INT8, {2, 4}, -10, 10},
      /*output=*/{TensorType_INT8, {}, 0, 0, 16. / 256, 127});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      -4.14297, -10.14297, -2.14297, -.142971,    //
                      -7.00104, -12.00104, -.00104087, -9.00104,  //
                  },
                  kLogSoftmaxQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                         61, -36, 93, 125,   //
                                         15, -65, 127, -16,  //
                                     }));
}

// A base class of PRelu op model. It provides the constructor for
// FloatPReluOpModel and QuantizedPReluOpModel.
class BasePReluOpModel : public SingleOpModel {
 public:
  BasePReluOpModel(const TensorData& input, const TensorData& alpha) {
    input_ = AddInput(input);
    alpha_ = AddInput(alpha);
    output_ = AddOutput({input.type, input.shape, input.min, input.max});
    SetBuiltinOp(BuiltinOperator_PRELU, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_), GetShape(alpha_)});
  }

 protected:
  int input_;
  int alpha_;
  int output_;
};

// The FloatPReluOpModel class handles float input and output.
class FloatPReluOpModel : public BasePReluOpModel {
 public:
  using BasePReluOpModel::BasePReluOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  void SetAlpha(std::initializer_list<float> data) {
    PopulateTensor(alpha_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

// The QuantizedPReluOpModel class handles quantized input and output.
class QuantizedPReluOpModel : public BasePReluOpModel {
 public:
  using BasePReluOpModel::BasePReluOpModel;

  template <typename T>
  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }
  template <typename T>
  void SetAlpha(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(alpha_, data);
  }
  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }
};

TEST(FloatActivationsOpTest, PRelu) {
  FloatPReluOpModel m({TensorType_FLOAT32, {1, 2, 2, 3}},
                      {TensorType_FLOAT32, {1, 1, 3}});

  m.SetInput({
      0.0f, 0.0f, 0.0f,     // Row 1, Column 1
      1.0f, 1.0f, 1.0f,     // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,  // Row 2, Column 1
      -2.0f, -2.0f, -2.0f,  // Row 1, Column 2
  });
  m.SetAlpha({0.0f, 1.0f, 2.0f});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0f, 0.0f, 0.0f,    // Row 1, Column 1
                                 1.0f, 1.0f, 1.0f,    // Row 1, Column 2
                                 0.0f, -1.0f, -2.0f,  // Row 2, Column 1
                                 0.0f, -2.0f, -4.0f,  // Row 1, Column 2
                             }));
}

TEST(QuantizedActivationsOpTest, PRelu) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedPReluOpModel m({TensorType_UINT8, {1, 2, 2, 3}, kMin, kMax},
                          {TensorType_UINT8, {1, 1, 3}, kMin, kMax});
  m.SetInput<uint8_t>({
      0.0f, 0.0f, 0.0f,        // Row 1, Column 1
      0.5f, 0.5f, 0.5f,        // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,     // Row 2, Column 1
      -0.25f, -0.25f, -0.25f,  // Row 1, Column 2
  });
  m.SetAlpha<uint8_t>({0.0f, 0.5f, -0.5f});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 0.0f, 0.0f,       // Row 1, Column 1
                      0.5f, 0.5f, 0.5f,       // Row 1, Column 2
                      0.0f, -0.5f, 0.5f,      // Row 2, Column 1
                      0.0f, -0.125f, 0.125f,  // Row 1, Column 2
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({
                                          128, 128, 128,  // Row 1, Column 1
                                          192, 192, 192,  // Row 1, Column 2
                                          128, 64, 192,   // Row 2, Column 1
                                          128, 112, 144,  // Row 1, Column 2
                                      }));
}

class LeakyReluOpModel : public SingleOpModel {
 public:
  LeakyReluOpModel(const TensorData& input, float alpha) {
    input_ = AddInput(input);
    output_ = AddOutput(input);
    SetBuiltinOp(BuiltinOperator_LEAKY_RELU, BuiltinOptions_LeakyReluOptions,
                 CreateLeakyReluOptions(builder_, alpha).Union());
    BuildInterpreter({GetShape(input_)});
  }
  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int output_;
};

TEST(FloatActivationsOpTest, LeakyRelu) {
  LeakyReluOpModel m({TensorType_FLOAT32, {2, 3}}, 0.5f);

  m.SetInput({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0f, 1.0f, 3.0f,    // Row 1
                                 1.0f, -0.5f, -1.0f,  // Row 2
                             }));
}

INSTANTIATE_TEST_SUITE_P(
    TanhOpTest, TanhOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kTanhKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    LogisticOpTest, LogisticOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kLogisticKernelMap)));

}  // namespace
}  // namespace tflite
