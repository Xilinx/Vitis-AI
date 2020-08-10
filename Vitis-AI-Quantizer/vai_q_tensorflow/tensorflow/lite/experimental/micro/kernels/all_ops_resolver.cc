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

#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"

namespace tflite {
namespace ops {
namespace micro {

TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
TfLiteRegistration* Register_FULLY_CONNECTED();
TfLiteRegistration* Register_SOFTMAX();
TfLiteRegistration* Register_CONV_2D();
TfLiteRegistration* Register_AVERAGE_POOL_2D();
TfLiteRegistration* Register_MAX_POOL_2D();
TfLiteRegistration* Register_ABS();
TfLiteRegistration* Register_SIN();
TfLiteRegistration* Register_COS();
TfLiteRegistration* Register_LOG();
TfLiteRegistration* Register_SQRT();
TfLiteRegistration* Register_RSQRT();
TfLiteRegistration* Register_SQUARE();
TfLiteRegistration* Register_PRELU();
TfLiteRegistration* Register_FLOOR();
TfLiteRegistration* Register_MAXIMUM();
TfLiteRegistration* Register_MINIMUM();
TfLiteRegistration* Register_ARG_MAX();
TfLiteRegistration* Register_ARG_MIN();
TfLiteRegistration* Register_LOGICAL_OR();
TfLiteRegistration* Register_LOGICAL_AND();
TfLiteRegistration* Register_LOGICAL_NOT();
TfLiteRegistration* Register_RESHAPE();
TfLiteRegistration* Register_EQUAL();
TfLiteRegistration* Register_NOT_EQUAL();
TfLiteRegistration* Register_GREATER();
TfLiteRegistration* Register_GREATER_EQUAL();
TfLiteRegistration* Register_LESS();
TfLiteRegistration* Register_LESS_EQUAL();
TfLiteRegistration* Register_CEIL();
TfLiteRegistration* Register_ROUND();
TfLiteRegistration* Register_STRIDED_SLICE();
TfLiteRegistration* Register_PACK();
TfLiteRegistration* Register_SPLIT();
TfLiteRegistration* Register_UNPACK();
TfLiteRegistration* Register_NEG();

AllOpsResolver::AllOpsResolver() {
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED(),
             /* min_version */ 1,
             /* max_version */ 3);
  AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D());
  AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX());
  AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D());
  AddBuiltin(BuiltinOperator_ABS, Register_ABS());
  AddBuiltin(BuiltinOperator_SIN, Register_SIN());
  AddBuiltin(BuiltinOperator_COS, Register_COS());
  AddBuiltin(BuiltinOperator_LOG, Register_LOG());
  AddBuiltin(BuiltinOperator_SQRT, Register_SQRT());
  AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT());
  AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE());
  AddBuiltin(BuiltinOperator_PRELU, Register_PRELU());
  AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR());
  AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM());
  AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM());
  AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX());
  AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN());
  AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR());
  AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND());
  AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT());
  AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
  AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL());
  AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL());
  AddBuiltin(BuiltinOperator_GREATER, Register_GREATER());
  AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL());
  AddBuiltin(BuiltinOperator_LESS, Register_LESS());
  AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL());
  AddBuiltin(BuiltinOperator_CEIL, Register_CEIL());
  AddBuiltin(BuiltinOperator_ROUND, Register_ROUND());
  AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE());
  AddBuiltin(BuiltinOperator_PACK, Register_PACK());
  AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT(),
             /* min_version */ 1,
             /* max_version */ 3);
  AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK());
  AddBuiltin(BuiltinOperator_NEG, Register_NEG());
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
