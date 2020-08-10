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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/concat_xy.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/concat_z.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, ConcatWidth) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {half(0.0f), half(-1.0f), half(-0.05f), half(0.045f)};
  src1.shape = BHWC(1, 2, 2, 2);
  src1.data = {half(1.0f), half(-1.2f), half(-0.45f), half(1.045f),
               half(1.1f), half(-1.3f), half(-0.55f), half(2.045f)};

  ConcatAttributes attr;
  attr.axis = Axis::WIDTH;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      ConcatXY operation = CreateConcatXY(op_def, attr, 2);
      ASSERT_OK(ExecuteGPUOperation({src0, src1}, creation_context_, &operation,
                                    BHWC(1, 2, 3, 2), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(0.0f),
                    {half(0.0f), half(-1.0f), half(1.0f), half(-1.2f),
                     half(-0.45f), half(1.045f), half(-0.05f), half(0.045f),
                     half(1.1f), half(-1.3f), half(-0.55f), half(2.045f)}));
    }
  }
}

TEST_F(OpenCLOperationTest, ConcatHeight) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {half(0.0f), half(-1.0f), half(-0.05f), half(0.045f)};
  src1.shape = BHWC(1, 1, 1, 2);
  src1.data = {half(1.0f), half(-1.2f)};

  ConcatAttributes attr;
  attr.axis = Axis::HEIGHT;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      ConcatXY operation = CreateConcatXY(op_def, attr, 2);
      ASSERT_OK(ExecuteGPUOperation({src0, src1}, creation_context_, &operation,
                                    BHWC(1, 3, 1, 2), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(0.0f), {half(0.0f), half(-1.0f), half(-0.05f),
                                      half(0.045f), half(1.0f), half(-1.2f)}));
    }
  }
}

TEST_F(OpenCLOperationTest, ConcatChannels) {
  TensorFloat32 src0, src1, src2;
  src0.shape = BHWC(1, 2, 1, 1);
  src0.data = {half(0.0f), half(-1.0f)};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {half(1.0f), half(2.0f), half(3.0f), half(4.0f)};
  src2.shape = BHWC(1, 2, 1, 3);
  src2.data = {half(5.0f), half(6.0f), half(7.0f),
               half(8.0f), half(9.0),  half(10.0f)};

  ConcatAttributes attr;
  attr.axis = Axis::CHANNELS;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.src_tensors.push_back({data_type, storage});
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      ConcatZ operation = CreateConcatZ(op_def, {1, 2, 3});
      ASSERT_OK(ExecuteGPUOperation({src0, src1, src2}, creation_context_,
                                    &operation, BHWC(1, 2, 1, 6), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(0.0f),
                            {half(0.0f), half(1.0f), half(2.0f), half(5.0f),
                             half(6.0f), half(7.0f), half(-1.0f), half(3.0f),
                             half(4.0f), half(8.0f), half(9.0), half(10.0f)}));
    }
  }
}

TEST_F(OpenCLOperationTest, ConcatChannelsAlignedx4) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 4);
  src0.data = {half(-1.0f), half(-2.0f), half(-3.0f), half(-4.0f),
               half(1.0f),  half(2.0f),  half(3.0f),  half(4.0f)};
  src1.shape = BHWC(1, 2, 1, 4);
  src1.data = {half(5.0f),  half(6.0f),  half(7.0f),  half(8.0f),
               half(-5.0f), half(-6.0f), half(-7.0f), half(-8.0f)};

  ConcatAttributes attr;
  attr.axis = Axis::CHANNELS;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      ConcatZ operation = CreateConcatZ(op_def, {4, 4});
      ASSERT_OK(ExecuteGPUOperation({src0, src1}, creation_context_, &operation,
                                    BHWC(1, 2, 1, 8), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(0.0f),
                    {half(-1.0f), half(-2.0f), half(-3.0f), half(-4.0f),
                     half(5.0f), half(6.0f), half(7.0f), half(8.0f), half(1.0f),
                     half(2.0f), half(3.0f), half(4.0f), half(-5.0f),
                     half(-6.0f), half(-7.0f), half(-8.0f)}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
