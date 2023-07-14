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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include <vitis/ai/ultrafast.hpp>
#include "./ultrafast_imp.hpp"

using namespace std;
namespace vitis {
namespace ai {

DEF_ENV_PARAM(ENABLE_UF_DEBUG, "0");

UltraFastImp::UltraFastImp(const std::string& model_name,
                           bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<UltraFast>(model_name, need_preprocess),
      batch_size(get_input_batch()),
      processor_{vitis::ai::UltraFastPost::create(
          configurable_dpu_task_->getInputTensor()[0],
          configurable_dpu_task_->getOutputTensor()[0],
          configurable_dpu_task_->getConfig(),
          batch_size, real_batch_size, pic_size) } 
{
  pic_size.resize(batch_size);
}

UltraFastImp::~UltraFastImp() {}

UltraFastResult UltraFastImp::run(const cv::Mat& input_img) 
{
  if (ENV_PARAM(ENABLE_UF_DEBUG) == 1) {
    {
      std::vector<vitis::ai::library::InputTensor> inputs = configurable_dpu_task_->getInputTensor()[0];
      const auto& layer_data = inputs[0];
      int sWidth = layer_data.width;
      int sHeight= layer_data.height;
      float scale =  tensor_scale(layer_data);
      auto channels = layer_data.channel;
      std::cout <<"net0in: width height scale channel:  " << sWidth << "  " << sHeight << "  " << scale << "  " << channels << "\n";  //  
    }
    {
      std::vector<vitis::ai::library::OutputTensor> outputs = configurable_dpu_task_->getOutputTensor()[0];
      const auto& layer_datao = outputs[0];
      int sWidth = layer_datao.width;
      int sHeight= layer_datao.height;
      float scale =  tensor_scale(layer_datao);
      auto channels = layer_datao.channel;
      std::cout <<"net0out: width height scale channel:  " << sWidth << "  " << sHeight << "  " << scale << "  " << channels << "\n";  //  
    }
  }

  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }
  __TIC__(DET_total)
  __TIC__(DET_setimg)

  real_batch_size = 1;
  pic_size[0] = input_img.size();

  configurable_dpu_task_->setInputImageRGB(img);

  __TOC__(DET_setimg)
  __TIC__(DET_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(DET_dpu)

  __TIC__(DET_post)
  auto results = processor_->post_process(0);
  __TOC__(DET_post)

  __TOC__(DET_total)
  return results;
}

std::vector<UltraFastResult> UltraFastImp::run(
    const std::vector<cv::Mat>& input_img) {
  auto size = cv::Size(getInputWidth(), getInputHeight());

  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  std::vector<cv::Mat> vimg(real_batch_size);
  for (auto i = 0; i < real_batch_size; i++) {
    if (size != input_img[i].size()) {
      cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_LINEAR);
    } else {
      vimg[i] = input_img[i];
    }
    pic_size[i] = input_img[i].size();
  }
  __TIC__(DET_total)
  __TIC__(DET_setimg)

  configurable_dpu_task_->setInputImageRGB(vimg);

  __TOC__(DET_setimg)
  __TIC__(DET_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(DET_dpu)

  __TIC__(DET_post)
  auto results = processor_->post_process();
  __TOC__(DET_post)

  __TOC__(DET_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
