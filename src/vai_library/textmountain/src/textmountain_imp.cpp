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
#include "./textmountain_imp.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

// NP.round(22.5)==22   NP.round(23.5)==24
// this macro control if fix c++ behavior to align with python np.round
DEF_ENV_PARAM(XLNX_TEXTMOUNTAIN_FIX_NPROUND, "0");

using namespace std;
using namespace cv;

namespace vitis {
namespace ai {

// Round x to the nearest multiple of p and x' >= x
int TextMountainImp::round2nearest_multiple(int x, int p){
  float tmp=(float)x/p;
  if ( !XLNX_TEXTMOUNTAIN_FIX_NPROUND) {
    return std::max(p, int(std::round(tmp)*p));
  }

  if (tmp-int(tmp)==0.5 && (int(tmp) % 2) ==0 ) {  // 22.5 should be 22
    //std::cout << "round2nearest :" << x << " " << p << " " << (float)x/(float)p << " " << std::round((float)x/(float)p) << "  " << int((std::round(tmp)-1)*p)   <<  "\n";
    return  std::max(p, int((std::round(tmp)-1)*p));
  } else {
    return std::max(p, int(std::round(tmp)*p));
  }
}

TextMountainImp::TextMountainImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<TextMountain>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0])
{
  XLNX_TEXTMOUNTAIN_FIX_NPROUND = ENV_PARAM(XLNX_TEXTMOUNTAIN_FIX_NPROUND);
  batch_size = get_input_batch();
  post_ = vitis::ai::TextMountainPost::create( 
            input_tensors_, 
            output_tensors_,
            batch_size, real_batch_size,
            scale_h, scale_w
          );
}

TextMountainImp::~TextMountainImp() {}

cv::Mat TextMountainImp::textmountain_pre_process(  const cv::Mat &input_img, int idx ) {
 cv::Mat imgout = cv::Mat::zeros( getInputHeight(), getInputWidth(), CV_8UC3);
 float scale_resize =  float(getInputWidth())/std::max(input_img.cols, input_img.rows);
 float w_resize = round2nearest_multiple(scale_resize*input_img.cols, 32);
 float h_resize = round2nearest_multiple(scale_resize*input_img.rows, 32);
 // std::cout <<"wh _resize : " << w_resize << " " << h_resize <<"\n"; // 704 960
 cv::Mat img(imgout, cv::Rect(0,0, w_resize, h_resize ));
 cv::resize(input_img, img, cv::Size( w_resize, h_resize), 0, 0, cv::INTER_LINEAR);
 // std::cout << "img resize size : " << img.cols << " " << img.rows << "\n";
 scale_w[idx] = float(w_resize*output_tensors_[0].width/ (  getInputWidth()*input_img.cols));
 scale_h[idx] = float(h_resize*output_tensors_[0].height/( getInputHeight()*input_img.rows));

 // std::cout <<" scale_wh " << scale_w[idx] << " " << scale_h[idx] << "\n";
 return imgout;
}

TextMountainResult TextMountainImp::run( const cv::Mat &input_img) {
  __TIC__(TextMountain_total)

  __TIC__(TextMountain_setimg)
  real_batch_size = 1;
  cv::Mat img = textmountain_pre_process(input_img, 0);
  configurable_dpu_task_->setInputImageRGB(img);
  __TOC__(TextMountain_setimg)

  __TIC__(TextMountain_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(TextMountain_dpu)

  __TIC__(TextMountain_post)
  auto results = post_->process();
  __TOC__(TextMountain_post)

  __TOC__(TextMountain_total)
  return results[0];
}

std::vector<TextMountainResult> TextMountainImp::run( const std::vector<cv::Mat> &input_img) {
  real_batch_size = std::min(int(input_img.size()), int(batch_size));

  __TIC__(TextMountain_total)
  __TIC__(TextMountain_setimg)
  std::vector<cv::Mat> vimg(real_batch_size);
  for (auto i = 0; i < real_batch_size; i++) {
    vimg[i] = textmountain_pre_process(input_img[i], i);
  }
  configurable_dpu_task_->setInputImageRGB(vimg);
  __TOC__(TextMountain_setimg)

  __TIC__(TextMountain_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(TextMountain_dpu)

  __TIC__(TextMountain_post)
  auto results = post_->process();
  __TOC__(TextMountain_post)

  __TOC__(TextMountain_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
