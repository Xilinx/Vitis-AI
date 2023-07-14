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
#include "./monodepth2_imp.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

Monodepth2Imp::Monodepth2Imp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Monodepth2>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0])
{
   scale_o = tensor_scale( output_tensors_[0] );
}

Monodepth2Imp::~Monodepth2Imp() {}

std::vector<Monodepth2Result> Monodepth2Imp::monodepth2_post_process() {

  auto ret = std::vector<vitis::ai::Monodepth2Result>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(monodepth2_post_process(i));
  }
  return ret;
}

Monodepth2Result Monodepth2Imp::monodepth2_post_process(int idx) {
  int8_t* p = (int8_t*)output_tensors_[0].get_data(idx);
  int width =  output_tensors_[0].width;  // 640
  int height =  output_tensors_[0].height;  // 192
  //   min_disp = 1 / 100 --> 0.01
  //   max_disp = 1 / 0.1 --> 10
  //   scaled_disp = min_disp + (max_disp - min_disp) * disp  -->  9.99*disp+0.01 
  // std::cout << "scale out whc :"<<scale_o<<" "<< width << " " << height << " " <<  output_tensors_[0].channel << "  input wh "<<input_tensors_[0].width<< " " << input_tensors_[0].height << "\n"; 
  // scale in/out whc :32 0.0078125 640 192 1  input wh 640 192
  cv::Mat mat( height, width, CV_32FC1);
  for(int i=0; i<height; i++) {
    for(int j=0; j<width; j++) {
      mat.ptr<float>(i)[j] = 9.99*scale_o*p[i*width+j] + 0.01;
    }
  }

  return Monodepth2Result{int(input_tensors_[0].width), int(input_tensors_[0].height), mat};
}

Monodepth2Result Monodepth2Imp::run( const cv::Mat &input_img) {
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }
  __TIC__(Monodepth2_total)
  __TIC__(Monodepth2_setimg)
  real_batch_size = 1;
  configurable_dpu_task_->setInputImageRGB(img);

  __TOC__(Monodepth2_setimg)
  __TIC__(Monodepth2_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Monodepth2_dpu)

  __TIC__(Monodepth2_post)
  auto results = monodepth2_post_process();
  __TOC__(Monodepth2_post)

  __TOC__(Monodepth2_total)
  return results[0];
}

std::vector<Monodepth2Result> Monodepth2Imp::run( const std::vector<cv::Mat> &input_img) {
  auto size = cv::Size(getInputWidth(), getInputHeight());
  auto batch_size = get_input_batch();
  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  std::vector<cv::Mat> vimg(real_batch_size);
  for (auto i = 0; i < real_batch_size; i++) {
    if (size != input_img[i].size()) {
      cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_LINEAR);
    } else {
      vimg[i] = input_img[i];
    }
  }
  __TIC__(Monodepth2_total)
  __TIC__(Monodepth2_setimg)

  configurable_dpu_task_->setInputImageRGB(vimg);

  __TOC__(Monodepth2_setimg)
  __TIC__(Monodepth2_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Monodepth2_dpu)

  __TIC__(Monodepth2_post)
  auto results = monodepth2_post_process();
  __TOC__(Monodepth2_post)

  __TOC__(Monodepth2_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
