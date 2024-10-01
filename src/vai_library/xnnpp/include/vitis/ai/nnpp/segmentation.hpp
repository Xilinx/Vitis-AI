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

#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

/**
 *@struct SegmentationResult
 *@brief Struct of the result returned by the segmentation network.
 */
/// FPN
/// Num of segmentation classes
/// \li \c 0 : "unlabeled"
/// \li \c  1 : "ego vehicle"
/// \li \c  2 : "rectification border"
/// \li \c  3 : "out of roi"
/// \li \c  4 : "static"
/// \li \c  5 : "dynamic"
/// \li \c  6 : "ground"
/// \li \c  7 : "road"
/// \li \c  8 : "sidewalk"
/// \li \c  9 : "parking"
/// \li \c  10 : "rail track"
/// \li \c  11 : "building"
/// \li \c  12 : "wall"
/// \li \c  13 : "fence"
/// \li \c  14 : "guard rail"
/// \li \c  15 : "bridge"
/// \li \c  16 : "tunnel"
/// \li \c  17 : "pole"
/// \li \c  18 : "polegroup"
struct SegmentationResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Segmentation result. The cv::Mat type is CV_8UC1 or CV_8UC3.
  cv::Mat segmentation;
};

struct Covid19SegmentationResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Positive detection result. The cv::Mat type is CV_8UC1 or CV_8UC3.
  cv::Mat positive_classification;
  /// Infected area detection result. The cv::Mat type is CV_8UC1 or CV_8UC3.
  cv::Mat infected_area_classification;
};

/**
 * @brief The post-processing function of the segmentation which stored the
 * original segmentation classes.
 * @param input_tensors A vector of all input-tensors in the network.
 *   Usage: input_tensors[input_tensor_index].
 * @param output_tensors A vector of all output-tensors in the network.
 *   Usage: output_tensors[output_index].
 * @return Struct of SegmentationResult.
 */
std::vector<SegmentationResult> segmentation_post_process_8UC1(
    const vitis::ai::library::InputTensor& input_tensors,
    const vitis::ai::library::OutputTensor& output_tensors);

/**
 * @brief The post-processing function of the segmentation which returns an image
 *mapped to color.
 * @param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 * @param output_tensors A vector of all output-tensors in the network.
 *  Usage: output_tensors[output_index].
 * @param config The dpu model configuration information ().
 * @return Struct of SegmentationResult.
 */
std::vector<SegmentationResult> segmentation_post_process_8UC3(
    const vitis::ai::library::InputTensor& input_tensors,
    const vitis::ai::library::OutputTensor& output_tensors,
    const vitis::ai::proto::DpuModelParam& config);

}  // namespace ai
}  // namespace vitis
