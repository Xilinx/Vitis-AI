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
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <string>
#include <utility>
#include <vector>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

/**
 *@struct PPBbox
 *@brief Struct of an object coordinate, confidence and classification.
 */
struct PPBbox {
  /// Confidence
  float score;
  /// Bounding box: x, y, z, x-size, y-size, z-size, yaw, custom value and so
  /// on.
  std::vector<float> bbox;
  /// Classification
  uint32_t label;
};

/**
 *@struct PointPillarsNuscenesResult
 *@brief Struct of the result returned by the PointPillarsNuscenes network.
 */
struct PointPillarsNuscenesResult {
  /// All bounding boxes
  std::vector<PPBbox> bboxes;
};

class PointPillarsNuscenesPostProcess {
 public:
  /**
   * @brief Create an PointPillarsNuscenesPostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param config The DPU model configuration information.
   * @return An unique pointer of PointPillarsNuscenesPostProcess.
   */
  static std::unique_ptr<PointPillarsNuscenesPostProcess> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);
  /**
   * @brief The batch mode post-processing function of the PointPillarsNuscenes
   * network.
   * @return The vector of struct of PointPillarsNuscenesResult.
   */
  virtual std::vector<PointPillarsNuscenesResult> postprocess(
      size_t batch_size) = 0;

  /**
   * @cond NOCOMMENTS
   */
  virtual ~PointPillarsNuscenesPostProcess();

 protected:
  explicit PointPillarsNuscenesPostProcess();
  PointPillarsNuscenesPostProcess(const PointPillarsNuscenesPostProcess&) =
      delete;
  PointPillarsNuscenesPostProcess& operator=(
      const PointPillarsNuscenesPostProcess&) = delete;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
