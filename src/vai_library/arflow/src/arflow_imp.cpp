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
#include "arflow_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

DEF_ENV_PARAM(DEBUG_RGBDSEGMENTATION, "0")

using namespace std;
namespace vitis {
namespace ai {

ARFlowImp::ARFlowImp(const string& model_name, bool need_preprocess)
    : ARFlow(model_name, need_preprocess) {}

ARFlowImp::~ARFlowImp() {}

vector<float> get_means(const vitis::ai::proto::DpuKernelParam& c) {
  return vector<float>(c.mean().begin(), c.mean().end());
}
vector<float> get_scales(const vitis::ai::proto::DpuKernelParam& c) {
  return vector<float>(c.scale().begin(), c.scale().end());
}
std::vector<vitis::ai::library::OutputTensor> ARFlowImp::run(
    const cv::Mat& image_1, const cv::Mat& image_2) {
  return run(vector<cv::Mat>(1, image_1), vector<cv::Mat>(1, image_2));
}
std::vector<vitis::ai::library::OutputTensor> ARFlowImp::run(
    const std::vector<cv::Mat>& image_1, const std::vector<cv::Mat>& image_2) {
  auto config = configurable_dpu_task_->getConfig();
  std::vector<std::string> input_name{config.kernel(0).name(),
                                      config.kernel(1).name()};
  auto unsort_inputtensor = configurable_dpu_task_->getInputTensor()[0];
  std::vector<int> inputtensor_idxs;
  for (auto&& name : input_name) {
    for (size_t i = 0; i < unsort_inputtensor.size(); i++) {
      if (unsort_inputtensor[i].name.find(name) != std::string::npos)
        inputtensor_idxs.push_back(i);
    }
  }
  CHECK_EQ(inputtensor_idxs.size(), unsort_inputtensor.size());
  std::vector<std::vector<cv::Mat>> images;
  images.emplace_back(image_1);
  images.emplace_back(image_2);
  auto outputs = configurable_dpu_task_->getOutputTensor()[0];
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  for (size_t i = 0; i < inputtensor_idxs.size(); i++) {
    std::vector<cv::Mat> inputs;
    auto size = cv::Size(sWidth, sHeight);
    for (auto&& im : images[i]) {
      cv::Mat image;
      if (size != im.size()) {
        cv::resize(im, image, size);
      } else {
        image = im;
      }
      inputs.push_back(image);
    }
    configurable_dpu_task_->setInputImageRGB(inputs, inputtensor_idxs[i]);
  }
  configurable_dpu_task_->run(0);
  std::stable_sort(
      outputs.begin(), outputs.end(),
      [](const auto& ls, const auto& rs) { return ls.size > rs.size; });

  return outputs;
}

}  // namespace ai
}  // namespace vitis
