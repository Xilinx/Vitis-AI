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
#include "./brtseg_imp.hpp"
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/nnpp/apply_nms.hpp>
#include <vitis/ai/math.hpp>
using namespace std;
namespace vitis {
namespace ai {

DEF_ENV_PARAM(ENABLE_BRTSEG_DEBUG, "0");
DEF_ENV_PARAM(ENABLE_BRTSEG_SOFTMAX, "1");

BrtsegImp::BrtsegImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Brtseg>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0])
{
  batch_size = get_input_batch();
  iData.resize(batch_size);
  oData.resize(batch_size);

  for(int j=0; j<batch_size; j++) {
      oData[j] = (int8_t*)output_tensors_[0].get_data(j);
      iData[j] = (int8_t*)input_tensors_[0].get_data(j);
  }
  sWidth   = output_tensors_[0].width;
  sHeight  = output_tensors_[0].height;
  sChannel = output_tensors_[0].channel;
  sScaleo  = tensor_scale(output_tensors_[0]);
  // std::cout <<"whc : " << sWidth << " " << sHeight << " " << sChannel << " " << sScaleo <<"\n";
}

BrtsegImp::~BrtsegImp() {}

std::vector<BrtsegResult> BrtsegImp::brtseg_post_process() {
  auto ret = std::vector<vitis::ai::BrtsegResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(brtseg_post_process(i));
  }
  return ret;
}

BrtsegResult BrtsegImp::brtseg_post_process(int idx) {

  BrtsegResult  ret{int(input_tensors_[0].width), int(input_tensors_[0].height), cv::Mat(sHeight, sWidth, CV_8UC1)};
  int8_t* p = oData[idx];
  uint8_t* pmat = ret.mat.ptr<uint8_t>(0);

  if (ENV_PARAM(ENABLE_BRTSEG_SOFTMAX)) {
     __TIC__(softmax)
     ret.softmax_data.reserve(sHeight * sWidth * sChannel);
     vitis::ai::softmax((int8_t*)p, sScaleo, 4, sHeight*sWidth, ret.softmax_data.data());
     __TOC__(softmax)
  }
  __TIC__(argmax)
  int8_t max_v, max_pos;
  int i_all = sHeight*sWidth, i=0; 
  int k=0; (void)k;
  for ( i = 0; i < i_all; ++i) {
      max_v = p[0];
      max_pos = 0;
      for( k=1;k<4;++k) {
        if( p[k] > max_v ) {
           max_v = p[k];  max_pos = k;
        }
      }
      pmat[i] = max_pos;
      p += 4; 
  }
  __TOC__(argmax)
  return ret;
}

BrtsegResult BrtsegImp::run( const cv::Mat &input_img) {
  __TIC__(Brtseg_total)
  __TIC__(Brtseg_setimg)
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  __TIC__(resize)
  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
    // cv::resize(input_img, img, size, 0, 0, cv::INTER_CUBIC);
  } else {
    img = input_img;
  }
  __TOC__(resize)
  real_batch_size = 1;
  configurable_dpu_task_->setInputImageRGB(img);

  __TOC__(Brtseg_setimg)
  __TIC__(Brtseg_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Brtseg_dpu)

  __TIC__(Brtseg_post)
  auto results = brtseg_post_process();
  __TOC__(Brtseg_post)

  __TOC__(Brtseg_total)
  return results[0];
}

std::vector<BrtsegResult> BrtsegImp::run( const std::vector<cv::Mat> &input_img) {
  __TIC__(Brtseg_total)
  __TIC__(Brtseg_setimg)
  auto size = cv::Size(getInputWidth(), getInputHeight());
  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  std::vector<cv::Mat> vimg(real_batch_size);
  for (auto i = 0; i < real_batch_size; i++) {
    if (size != input_img[i].size()) {
      cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_LINEAR);
      // cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_CUBIC);
    } else {
      vimg[i] = input_img[i];
    }
  }

  configurable_dpu_task_->setInputImageRGB(vimg);

  __TOC__(Brtseg_setimg)
  __TIC__(Brtseg_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Brtseg_dpu)

  __TIC__(Brtseg_post)
  auto results = brtseg_post_process();
  __TOC__(Brtseg_post)

  __TOC__(Brtseg_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
