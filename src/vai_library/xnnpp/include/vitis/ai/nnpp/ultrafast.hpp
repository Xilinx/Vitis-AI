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
#pragma once

#include <vitis/ai/image_util.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/proto/dpu_model_param.pb.h>

using std::shared_ptr;
using std::vector;

namespace vitis {
namespace ai {

/**
 * @struct UltraFastResult
 * @brief Struct of the result returned by the ultrafast neural network.
 */
struct UltraFastResult{
  /// Width of input image.
  int width = 0;
  /// Height of input image.
  int height = 0;
  /// vector of lanes information. each lane is a vector holding pair structure.
  std::vector<std::vector<std::pair<float, float>>> lanes;
};

/**
 * @class UltraFastPost
 * @brief Class of the UltraFast post-process. 
 * */
class UltraFastPost {
 public:
  /**
   * @brief Create an UltraFastPost object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param config The DPU model configuration information.
   * @param batch_size the model batch information
   * @param real_batch_size the real batch information of the model
   * @param pic_size vector holding the size information of input pics
   * @return A unique pointer of UltraFastPost.
   */
  static std::unique_ptr<UltraFastPost> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config,
      int batch_size,
      int& real_batch_size,
      std::vector<cv::Size>& pic_size
     );

  /**
   * @brief Post-process the UltraFast result.
   * @param idx  batch index.
   * @return UltraFastResult.
   */
  virtual UltraFastResult post_process(unsigned int idx) =0;
  /**
   * @brief Post-process the UltraFast result.
   * @return vector of UltraFastResult.
   */
  virtual std::vector<UltraFastResult> post_process() =0;
  /**
   * @cond NOCOMMENTS
   */
  virtual ~UltraFastPost();

 protected:
  explicit UltraFastPost();
  UltraFastPost(const UltraFastPost&) = delete;
  UltraFastPost& operator=(const UltraFastPost&) = delete;
  /**
   * @endcond
   */
};

} // namespace ai
} // namespace vitis

