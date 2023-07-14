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

#include <memory>
#include <opencv2/core.hpp>
namespace vitis {
namespace ai {

class General {
 public:
  static std::unique_ptr<General> create(const std::string& model_name,
                                         bool need_preprocess = true);

 public:
  explicit General();
  General(const General&) = delete;
  virtual ~General();

 public:
  virtual vitis::ai::proto::DpuModelResult run(const cv::Mat& image) = 0;

  virtual std::vector<vitis::ai::proto::DpuModelResult> run(
      const std::vector<cv::Mat>& image) = 0;
  /**
   * @brief Function to get InputWidth of the network (input image
   *cols).
   *
   * @return InputWidth of the network
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the network (input image
   *rows).
   *
   *@return InputHeight of the network.
   */
  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   * @return Batch size.
   */

  virtual size_t get_input_batch() const = 0;
};

}  // namespace ai
}  // namespace vitis
