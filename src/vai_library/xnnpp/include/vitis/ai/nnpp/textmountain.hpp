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
#include <vitis/ai/library/tensor.hpp>
#include <array>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

namespace vitis {
namespace ai {

typedef cv::Point2d arr4_point2d[4]; 

struct TextMountainResult{
  /// width of network input.
  int width=0;
  /// height of network input.
  int height=0;
  /// Struct to hold each textmountain detected result 
  struct tmitem{
    /// construct function;
    tmitem(arr4_point2d& inbox, float inscore): box(inbox), score(inscore){}
    /// 4 Point2f to hold the box coordinate. sequence is clock-wise
    arr4_point2d box;
    /// scores for each box
    float score;
  };
  /// vector to hold the detected result
  std::vector<tmitem> res;
};

class TextMountainPost {
 public:
  /**
   * @brief Create an TextMountainPost object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param batch_size the model batch information
   * @param real_batch_size the real batch information of the model
   * @param scale_h: the array to hold the height scale for each input img
   * @param scale_w: the array to hold the width scale for each input img
   * @return An unique pointer of TextMountainPost
   */
  static std::unique_ptr<TextMountainPost> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      int batch_size,
      int& real_batch_size,
      float* scale_h,
      float* scale_w
  );

  /**
   * @brief Post-process the textmountain result.
   * @param idx  batch index.
   * @return TextMountainResult.
   */
  virtual TextMountainResult process(int idx)=0;
  /**
   * @brief Post-process the textmountain result.
   * @return vector of TextMountainResult.
   */
  virtual std::vector<TextMountainResult> process()=0;
  /**
   * @cond NOCOMMENTS
   */
  virtual ~TextMountainPost();

 protected:
  explicit TextMountainPost();
  TextMountainPost(const TextMountainPost&) = delete;
  TextMountainPost& operator=(const TextMountainPost&) = delete;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis

