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

namespace vitis {
namespace ai {

using V1I = std::vector<int>;
using V2I = std::vector<V1I>;
using V1F = std::vector<float>;
using V2F = std::vector<V1F>;

struct OCRResult{
  /// width of network input.
  int width=0;
  /// Height of network input.
  int height=0;

  /// vector of recognized words in input pic
  std::vector<std::string> words;
  /// vector of box information of the recognized words in input pic
  std::vector<std::vector<cv::Point>> box;
};

class OCRPost {
 public:
  /**
   * @brief Create an OCRPost object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param cfgpath configuration file path (*_officialcfg.prototxt )
   * @param batch_size the model batch information
   * @param real_batch_size the real batch information of the model
   * @param target_h8 inner data structure
   * @param target_w8 inner data structure
   * @param ratioh inner data structure for height ratio
   * @param ratiow inner data structure for width ratio
   * @param oriimg original image
   * @return An unique pointer of OCRPostProcess.
   */
  static std::unique_ptr<OCRPost> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const std::string& cfgpath,
      int batch_size,
      int& real_batch_size,
      std::vector<int>& target_h8,
      std::vector<int>& target_w8,
      std::vector<float>& ratioh,
      std::vector<float>& ratiow,
      std::vector<cv::Mat>& oriimg
  );

  /**
   * @brief Post-process the ocr result.
   * @param idx  batch index.
   * @return OCRResult.
   */
  virtual OCRResult process(int idx)=0;
  /**
   * @brief Post-process the ocr result.
   * @return vector of OCRResult.
   */
  virtual std::vector<OCRResult> process()=0;
  /**
   * @cond NOCOMMENTS
   */
  virtual ~OCRPost();

 protected:
  explicit OCRPost();
  OCRPost(const OCRPost&) = delete;
  OCRPost& operator=(const OCRPost&) = delete;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis

