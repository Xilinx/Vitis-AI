/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef DEEPHI_UltraFastPostIMP_HPP_
#define DEEPHI_UltraFastPostIMP_HPP_

#include <vitis/ai/image_util.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <vitis/ai/nnpp/ultrafast.hpp>


using std::shared_ptr;
using std::vector;

namespace vitis {
namespace ai {

class UltraFastPostImp : public UltraFastPost{
 public:

  UltraFastPostImp(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config,
      int batch_size,
      int& real_batch_size,
      std::vector<cv::Size>& pic_size );
  virtual ~UltraFastPostImp();

  virtual UltraFastResult post_process(unsigned int idx) override;
  virtual std::vector<UltraFastResult> post_process() override;

 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  int batch_size;
  int& real_batch_size;
  std::vector<cv::Size>& pic_size;

  std::vector<float> softmax_data;
  std::vector<float> row_anchor_orig;
  std::vector<float> row_anchor;
  int o_height, o_width,  o_channel;
  int i_height, i_width;
  float o_scale;
  std::vector<int8_t*> o_cls;
  int griding_num=200;
  float col_sample_w=0.0;
};

} // namespace ai
} // namespace vitis

#endif

