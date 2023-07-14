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

#include <vector>
#include <vitis/ai/library/tensor.hpp>
namespace vitis {
namespace ai {

/**
 *@struct FaceDetectResult
 *@brief Struct of the result with the facedetect network.
 *
 */
struct FaceDetectResult {
  /**
   *@struct BoundingBox
   *@brief The coordinate and confidence of a face.
   */
  struct BoundingBox {
    /// x-coordinate. x is normalized relative to the input image columns.
    /// Range from 0 to 1.
    float x;
    /// y-coordinate. y is normalized relative to the input image rows.
    /// Range from 0 to 1.
    float y;
    /// face width. Width is normalized relative to the input image columns,
    /// Range from 0 to 1.
    float width;
    /// face height. Heigth is normalized relative to the input image rows,
    /// Range from 0 to 1.
    float height;
    /// face confidence, the value range from 0 to 1.
    float score;
  };
  /// Width of an input image.
  int width;
  /// Height of an input image.
  int height;
  /// All faces, filtered by confidence >= detect threshold.
  std::vector<BoundingBox> rects;
};
/**
 *@brief The post-processing function of the facedetect network.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@param det_threshold The results will be filtered by score >= det_threshold.
 *@return the result of the facedetect.
 */
std::vector<FaceDetectResult> face_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const float det_threshold);

}  // namespace ai
}  // namespace vitis
