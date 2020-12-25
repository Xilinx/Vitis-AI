/*
 * Copyright 2019 Xilinx Inc.
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
#include "./cifar10classification_imp.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/math.hpp>

namespace vitis {
namespace ai {

Cifar10ClassificationImp::Cifar10ClassificationImp(const std::string &model_name,
                                               bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Cifar10Classification>(model_name, need_preprocess),
     input_tensors_(configurable_dpu_task_->getInputTensor()[0]),
     output_tensors_(configurable_dpu_task_->getOutputTensor()[0]) 
{
  scale_conf_ = vitis::ai::library::tensor_scale(output_tensors_[0]);
}

Cifar10ClassificationImp::~Cifar10ClassificationImp() {}

Cifar10ClassificationResult Cifar10ClassificationImp::post_process(int idx)
{
  Cifar10ClassificationResult result{getInputWidth(), getInputHeight(), 0};
  auto size_ = (int)output_tensors_[0].width * (int)output_tensors_[0].height * (int)output_tensors_[0].channel;
  std::vector<float> vconf( size_ );
  int8_t* conf = (int8_t*)(output_tensors_[0].get_data(idx));

  __TIC__(CLS_softmax)
  vitis::ai::softmax((int8_t*)conf, scale_conf_, size_, 1, vconf.data());
  result.classIdx = std::max_element(vconf.begin(), vconf.end()) - vconf.begin();
  __TOC__(CLS_softmax)
  return result;
}

Cifar10ClassificationResult Cifar10ClassificationImp::run(
    const cv::Mat &input_img) {
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }

  __TIC__(CLS_total)
  __TIC__(CLS_setimg)
  configurable_dpu_task_->setInputImageRGB(img);
  __TOC__(CLS_setimg)

  __TIC__(CLS_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(CLS_dpu)

  __TIC__(CLS_post)
  auto result = post_process(0);
  __TOC__(CLS_post)

  __TOC__(CLS_total)
  return result;
}

std::vector<Cifar10ClassificationResult> Cifar10ClassificationImp::run(
    const std::vector<cv::Mat> &input_img) {
  auto size = cv::Size(getInputWidth(), getInputHeight());
  auto batch_size = get_input_batch();

  std::vector<cv::Mat> vimg(batch_size);

  for (auto i = 0ul; i < batch_size; i++) {
    if (size != input_img[i].size()) {
      cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_LINEAR);
    } else {
      vimg[i] = input_img[i];
    }
  }
  __TIC__(CLS_total)

  __TIC__(CLS_setimg)
  configurable_dpu_task_->setInputImageBGR(vimg);
  __TOC__(CLS_setimg)
  __TIC__(CLS_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(CLS_dpu)

  __TIC__(CLS_post)
  std::vector<Cifar10ClassificationResult> results(batch_size);
  for (auto i = 0ul; i < batch_size; i++) {
    results[i] = this->post_process(i);
  }
  __TOC__(CLS_post)

  __TOC__(CLS_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
