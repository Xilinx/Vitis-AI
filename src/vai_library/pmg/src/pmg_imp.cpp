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
#include "./pmg_imp.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

PMGImp::PMGImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<PMG>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0])
{}

PMGImp::~PMGImp() {}

std::vector<PMGResult> PMGImp::pmg_post_process() {

  auto ret = std::vector<vitis::ai::PMGResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(pmg_post_process(i));
  }
  return ret;
}

PMGResult PMGImp::pmg_post_process(int idx) {
  int8_t* p = (int8_t*)output_tensors_[0].get_data(idx);
  int len =  output_tensors_[0].width * output_tensors_[0].height* output_tensors_[0].channel;
  /*
  std::cout << " output: " << output_tensors_[0].width << " " << output_tensors_[0].height << " " <<  output_tensors_[0].channel << "\n";
  float scale =  tensor_scale( output_tensors_[0] );    std::cout << "scale :"  << scale <<"\n";
  for(int i=0; i<len; i++) { 
    std::cout << scale * p[i] << "  ";
    if ((i+1)%30==0) std::cout <<"\n";
  } std::cout <<"\n";
  */
  auto max_it = std::max_element(p, p+len);
  int index = std::distance(p, max_it);

  return PMGResult{int(input_tensors_[0].width), int(input_tensors_[0].height) , index};
}

PMGResult PMGImp::run( const cv::Mat &input_img) {
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_AREA);
  } else {
    img = input_img;
  }
  __TIC__(PMG_total)
  __TIC__(PMG_setimg)
  real_batch_size = 1;
  configurable_dpu_task_->setInputImageRGB(img);

  __TOC__(PMG_setimg)
  __TIC__(PMG_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(PMG_dpu)

  __TIC__(PMG_post)
  auto results = pmg_post_process();
  __TOC__(PMG_post)

  __TOC__(PMG_total)
  return results[0];
}

std::vector<PMGResult> PMGImp::run( const std::vector<cv::Mat> &input_img) {
  auto size = cv::Size(getInputWidth(), getInputHeight());
  auto batch_size = get_input_batch();
  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  std::vector<cv::Mat> vimg(real_batch_size);
  for (auto i = 0; i < real_batch_size; i++) {
    if (size != input_img[i].size()) {
      cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_AREA);
    } else {
      vimg[i] = input_img[i];
    }
  }
  __TIC__(PMG_total)
  __TIC__(PMG_setimg)

  configurable_dpu_task_->setInputImageRGB(vimg);

  __TOC__(PMG_setimg)
  __TIC__(PMG_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(PMG_dpu)

  __TIC__(PMG_post)
  auto results = pmg_post_process();
  __TOC__(PMG_post)

  __TOC__(PMG_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
