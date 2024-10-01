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
#include "./pmrid_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

#include "utils.hpp"
using namespace std;
namespace vitis {
namespace ai {

PMRIDImp::PMRIDImp(const std::string& model_name, bool need_preprocess)
    : PMRID(model_name, need_preprocess) {}
PMRIDImp::PMRIDImp(const std::string& model_name, xir::Attrs* attrs,
                   bool need_preprocess)
    : PMRID(model_name, attrs, need_preprocess) {}

PMRIDImp::~PMRIDImp() {}

std::vector<float> PMRIDImp::run(const cv::Mat& input_images, float iso) {
  return run(vector<cv::Mat>(1, input_images), std::vector<float>(1, iso))[0];
}
std::vector<std::vector<float>> PMRIDImp::run(
    const std::vector<cv::Mat>& input_images, const std::vector<float>& isos) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  vector<cv::Mat> images(input_images.size());
  vector<float> scale(input_images.size());
  float input_fixed_scale =
      tensor_scale(configurable_dpu_task_->getInputTensor()[0][0]);
  // preprocessing and set the input
  __TIC__(PRE_PROCESS)
  // raw image shape and pad parameter
  constexpr int raw_height = 3000;
  constexpr int raw_width = 4000;
  constexpr int ph = (32 - ((raw_height / 2) % 32)) / 2;
  constexpr int pw = (32 - ((raw_width / 2) % 32)) / 2;
  CHECK((raw_height % 2) || (raw_width % 2) == 0)
      << "raw_height is not 2N, or raw_width is not 2N.";
  CHECK((raw_height / 2 + ph * 2 == sHeight) &&
        (raw_width / 2 + pw * 2 == sWidth))
      << "raw image size not match with xmodel input(N*1504*2016*4).";
  // read raw data
  for (auto i = 0u; i < input_images.size(); ++i) {
    set_input(input_images[i], isos[i],
              configurable_dpu_task_->getInputTensor()[0][0], i);
  }
  __TOC__(PRE_PROCESS)

  __TIC__(DPU)
  configurable_dpu_task_->run(0);
  __TOC__(DPU)

  // postprocessing
  __TIC__(POST_PROCESS)
  float output_fixed_scale =
      tensor_scale(configurable_dpu_task_->getOutputTensor()[0][0]);
  auto result = std::vector<std::vector<float>>(input_images.size());
  for (auto i = 0u; i < input_images.size(); ++i) {
    result[i] = invKSigma_unpad_rggb2bayer(
        configurable_dpu_task_->getOutputTensor()[0][0].get_data(i),
        configurable_dpu_task_->getInputTensor()[0][0].get_data(i),
        output_fixed_scale, 1.0f / input_fixed_scale, raw_height, raw_width,
        sWidth, configurable_dpu_task_->getOutputTensor()[0][0].channel, ph, pw,
        isos[i], 256.0f);
  }
  __TOC__(POST_PROCESS)
  return result;
}

}  // namespace ai
}  // namespace vitis
