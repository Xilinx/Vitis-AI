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
#include "./mnistclassification_imp.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

MnistClassificationImp::MnistClassificationImp(const std::string &model_name,
                                               bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<MnistClassification>(model_name, need_preprocess),
     input_tensors_(configurable_dpu_task_->getInputTensor()[0]),
     output_tensors_(configurable_dpu_task_->getOutputTensor()[0]),
     cfg_(configurable_dpu_task_->getConfig())
{
  scale_conf_ = vitis::ai::library::tensor_scale(output_tensors_[0]);
  float input_fixed_scale_ = vitis::ai::library::tensor_scale(input_tensors_[0]);
  input_scale_ = input_fixed_scale_*cfg_.kernel(0).scale()[0];
}

MnistClassificationImp::~MnistClassificationImp() {}

MnistClassificationResult MnistClassificationImp::post_process(int idx)
{
  MnistClassificationResult result{getInputWidth(), getInputHeight(), 0};

  auto size_ = (int)output_tensors_[0].width * (int)output_tensors_[0].height * (int)output_tensors_[0].channel;
  int8_t* conf = (int8_t*)(output_tensors_[0].get_data(idx));
  result.classIdx = std::max_element(conf, conf+size_) - conf;
  return result;
}

void MnistClassificationImp::pre_process(int idx, cv::Mat& img){ 
  const auto& layer_data = input_tensors_[0];
  auto rows = layer_data.height;
  auto cols = layer_data.width;
  auto channels = layer_data.channel;
  auto data = (int8_t*)layer_data.get_data(idx);
  uint8_t* input = img.data;
  // std::cout << "input_scale:" << input_scale_ << "  hwc " << rows << cols << channels << "\n"; // 64  28 28 1 // out_scale:0.25

  for (auto h = 0u; h < rows; ++h) {
    for (auto w = 0u; w < cols; ++w) {
      for (auto c = 0u; c < channels; ++c) {
        auto value =
            (int)((input[h * cols * channels + w * channels + c] * 1.0f ) * input_scale_);
        data[h * cols * channels + w * channels ] = (char)value;
      }
    }
  }
}

MnistClassificationResult MnistClassificationImp::run(
    const cv::Mat &input_img) {
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }

  __TIC__(CLS_total)
  real_batch_size = 1;
  // preprocess :
  __TIC__(CLS_pre)
  pre_process(0, img);
  __TOC__(CLS_pre)

  __TIC__(CLS_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(CLS_dpu)

  __TIC__(CLS_post)
  MnistClassificationResult  result = post_process( 0 ) ;
  __TOC__(CLS_post)

  __TOC__(CLS_total)
  return result;
}

std::vector<MnistClassificationResult> MnistClassificationImp::run(
    const std::vector<cv::Mat> &input_img) {
  auto size = cv::Size(getInputWidth(), getInputHeight());
  auto batch_size = get_input_batch();
  real_batch_size = std::min((int)batch_size, (int)input_img.size());
  std::vector<cv::Mat> vimg(real_batch_size);

  for (auto i = 0; i < real_batch_size; i++) {
    if (size != input_img[i].size()) {
      cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_LINEAR);
    } else {
      vimg[i] = input_img[i];
    }
  }
  __TIC__(CLS_total)

  // preprocess :
  __TIC__(CLS_pre)
  for (auto i = 0; i < real_batch_size; i++) {
     pre_process(i, vimg[i]);
  }
  __TOC__(CLS_pre)
  
  __TIC__(CLS_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(CLS_dpu)

  __TIC__(CLS_post)
  std::vector<MnistClassificationResult>  results(batch_size);
  for (auto i = 0; i < real_batch_size; i++) {
    results[i] = this->post_process(i);
  }
  __TOC__(CLS_post)

  __TOC__(CLS_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
