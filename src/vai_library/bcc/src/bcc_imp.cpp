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
#include "./bcc_imp.hpp"

#include <numeric>
#include <fstream>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include <sys/stat.h>

namespace vitis {
namespace ai {
DEF_ENV_PARAM(ENABLE_BCC_DEBUG, "0");

BCCImp::BCCImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<BCC>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0]),
      cfg_(configurable_dpu_task_->getConfig()),
      need_preprocess_(need_preprocess)
{
  batch_size = get_input_batch();
  new_height.resize(batch_size);
  new_width.resize(batch_size);

  std::vector<float> vmean(cfg_.kernel(0).mean().begin(), cfg_.kernel(0).mean().end());
  mean.swap(vmean);
  std::vector<float> vscale(cfg_.kernel(0).scale().begin(), cfg_.kernel(0).scale().end());
  scale.swap(vscale);

  size = cv::Size(getInputWidth(), getInputHeight());
  scale_i = tensor_scale( input_tensors_[0] ); 
  for(unsigned int i=0; i<scale.size(); i++) {
     scale[i]*=scale_i;
  }
  scale_o = tensor_scale( output_tensors_[0] );
}

BCCImp::~BCCImp() {}

std::vector<BCCResult> BCCImp::bcc_post_process() {
  auto ret = std::vector<vitis::ai::BCCResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(bcc_post_process(i));
  }
  return ret;
}

BCCResult BCCImp::bcc_post_process(int idx) {
  int8_t* p = (int8_t*)output_tensors_[0].get_data(idx);

  int vaild_height = new_height[idx] % 8 ?  new_height[idx] / 8 + 1 : new_height[idx] / 8;
  int vaild_width  = new_width[idx]  % 8 ?  new_width[idx]  / 8 + 1 : new_width[idx]  / 8;

  float count = 0.0;
  for(int i=0; i<vaild_height; i++) {
     for(int j=0; j<vaild_width; j++) {
        count += std::abs( p[ i*output_tensors_[0].width+j] );
     }
  }
  count *= scale_o;

  return BCCResult{int(input_tensors_[0].width), int(input_tensors_[0].height) , int(count)};
}

void BCCImp::cleanmem()
{
  for(unsigned int i=0; i<batch_size; i++) { 
    cleanmem(i);
  }
}

void BCCImp::cleanmem(unsigned int idx)
{
  int8_t* p = (int8_t*)input_tensors_[0].get_data(idx);
  memset(p, 0, input_tensors_[0].width * input_tensors_[0].height* input_tensors_[0].channel );
}

void BCCImp::preprocess(const cv::Mat& input_img, int idx) {
  cv::Mat img;
  __TIC__(resize)
  if (cv::Size(new_width[idx], new_height[idx]) != input_img.size()) {
    cv::resize(input_img, img, cv::Size(new_width[idx], new_height[idx]), 0, 0, cv::INTER_LANCZOS4);
  } else {
    img = input_img;
  }
  __TOC__(resize)

  int channels =  input_tensors_[0].channel;
  uint8_t* input = img.data;
  int8_t* dest = (int8_t*)input_tensors_[0].get_data(idx);
  int rows1 = img.rows;
  int cols1 = img.cols;
  int cols =  input_tensors_[0].width;
  int cols1_channels = img.step; //cols1*channels ;
  int cols_channels  = cols*channels ;
  std::vector<float> mean_scale = {mean[0]*scale[0], mean[1]*scale[1], mean[2]*scale[2] };
  __TIC__(imgtodpu)
  for (auto h = 0; h < rows1; ++h) {
    for (auto w = 0; w < cols1; ++w) {
        dest[h * cols_channels + w * channels + 2] = int(round(input[h * cols1_channels + w * channels + 0] * scale[0] - mean_scale[0]));
        dest[h * cols_channels + w * channels + 1] = int(round(input[h * cols1_channels + w * channels + 1] * scale[1] - mean_scale[1]));
        dest[h * cols_channels + w * channels + 0] = int(round(input[h * cols1_channels + w * channels + 2] * scale[2] - mean_scale[2]));
    }
  }
  __TOC__(imgtodpu)

}

void BCCImp::setVarForPostProcess(const cv::Mat& input_img, int idx)
{
    float ratio = float(input_img.rows)/float(input_img.cols);
    if (ratio < float(size.height)/float(size.width)) {
       new_height[idx] = int(size.width*ratio);
       new_width[idx] = size.width;
    } else {
       new_height[idx] = size.height;
       new_width[idx] = int(size.height/ratio);
    }
}

BCCResult BCCImp::run( const cv::Mat &input_img) {

  __TIC__(BCC_total)
  __TIC__(BCC_setimg)

  real_batch_size = 1;


  setVarForPostProcess(input_img, 0);
  if (need_preprocess_) {
    cleanmem(0);
    preprocess(input_img, 0);
  } else {
    configurable_dpu_task_->setInputImageRGB(input_img);
  }

  __TOC__(BCC_setimg)
  __TIC__(BCC_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(BCC_dpu)

  __TIC__(BCC_post)
  auto results = bcc_post_process();
  __TOC__(BCC_post)

  __TOC__(BCC_total)
  return results[0];
}

std::vector<BCCResult> BCCImp::run( const std::vector<cv::Mat> &input_img) {

  __TIC__(BCC_total)
  __TIC__(BCC_setimg)

  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  for (auto i = 0; i < real_batch_size; i++) {
    setVarForPostProcess(input_img[i], i);
  }
  if (need_preprocess_) {
    cleanmem();
    for (auto i = 0; i < real_batch_size; i++) {
      preprocess(input_img[i], i);
    }
  } else {
    configurable_dpu_task_->setInputImageRGB(input_img);
  }
  __TOC__(BCC_setimg)
  __TIC__(BCC_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(BCC_dpu)

  __TIC__(BCC_post)
  auto results = bcc_post_process();
  __TOC__(BCC_post)

  __TOC__(BCC_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
