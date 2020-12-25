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

#include "tensorflow/lite/delegates/gpu/metal/kernels/add.h"

#import <XCTest/XCTest.h>

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

using ::tflite::gpu::AddAttributes;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::DataType;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::OperationType;

@interface AddTest : XCTestCase
@end

@implementation AddTest
- (void)setUp {
  [super setUp];
}

- (void)testTwoInputTensorsOfTheSameShape {
  TensorRef<BHWC> augend, addend, output;
  augend.type = DataType::FLOAT32;
  augend.ref = 0;
  augend.shape = BHWC(1, 2, 2, 1);

  addend.type = DataType::FLOAT32;
  addend.ref = 1;
  addend.shape = BHWC(1, 2, 2, 1);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  AddAttributes attr;
  SingleOpModel model({ToString(OperationType::ADD), std::move(attr)}, {augend, addend}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {-2.0, 0.2, 0.7, 0.8}));
  XCTAssertTrue(model.PopulateTensor(1, {0.1, 0.2, 0.3, 0.5}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.ToString().c_str());
  status = CompareVectors({-1.9, 0.4, 1.0, 1.3}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.ToString().c_str());
}

- (void)testInputTensorAndScalar {
  AddAttributes attr;
  attr.param = 0.1f;
  TensorRef<BHWC> input, output;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 1, 2);

  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 3, 1, 2);

  SingleOpModel model({ToString(OperationType::ADD), std::move(attr)}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.ToString().c_str());
  status = CompareVectors({-1.9, 0.3, 0.8, 0.9, 1.2, 2.1}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.ToString().c_str());
}

@end
