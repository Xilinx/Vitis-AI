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

#include "tensorflow/compiler/xla/service/gpu/hlo_algorithm_blacklist.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/dnn.h"

namespace xla {
namespace gpu {
namespace {

class BlacklistTest : public testing::Test {
 protected:
  BlacklistTest() {
    setenv("XLA_FLAGS",
           absl::StrCat(
               "--xla_gpu_algorithm_blacklist_path=",
               tensorflow::io::JoinPath(
                   tensorflow::testing::TensorFlowSrcRoot(), "compiler", "xla",
                   "service", "gpu", "data", "hlo_algorithm_blacklist.pbtxt"))
               .data(),
           0);
  }
};

TEST_F(BlacklistTest, DefaultTest) {
  tensorflow::ComputeCapability cc;
  cc.set_major(7);
  cc.set_minor(0);
  tensorflow::CudnnVersion cudnn_version;
  cudnn_version.set_major(7);
  cudnn_version.set_minor(6);
  cudnn_version.set_patch(2);
  auto list = GetBlacklistedConvAlgorithms(
      cc, cudnn_version, /*blas_version=*/"9000",
      R"((f16[256,112,112,64]{3,2,1,0}, u8[0]{0}) custom-call(f16[256,224,224,4]{3,2,1,0}, f16[7,7,4,64]{2,1,0,3}), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convForward", backend_config="{conv_result_scale:1}")");
  ASSERT_EQ(4, list.size());
  EXPECT_EQ(stream_executor::dnn::AlgorithmDesc(0, false), list[0]);
  EXPECT_EQ(stream_executor::dnn::AlgorithmDesc(0, true), list[1]);
  EXPECT_EQ(stream_executor::dnn::AlgorithmDesc(1, false), list[2]);
  EXPECT_EQ(stream_executor::dnn::AlgorithmDesc(1, true), list[3]);
}

TEST_F(BlacklistTest, NegativeTest) {
  tensorflow::ComputeCapability cc;
  cc.set_major(7);
  cc.set_minor(0);
  tensorflow::CudnnVersion cudnn_version;
  cudnn_version.set_major(7);
  cudnn_version.set_minor(6);
  cudnn_version.set_minor(2);
  auto list =
      GetBlacklistedConvAlgorithms(cc, cudnn_version, "9000", R"(invalid hlo)");
  ASSERT_EQ(0, list.size());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
