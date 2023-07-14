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
#include "./medicalsegcell_imp.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

MedicalSegcellImp::MedicalSegcellImp(const std::string& model_name,
                                     bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<MedicalSegcell>(model_name,
                                                      need_preprocess),
      input_tensors_(configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0]) {}

MedicalSegcellImp::~MedicalSegcellImp() {}

MedicalSegcellResult MedicalSegcellImp::run(const cv::Mat& input_img) {
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }
  __TIC__(SEG_total)
  __TIC__(SEG_setimg)
  real_batch_size = 1;
  configurable_dpu_task_->setInputImageRGB(img);

  __TOC__(SEG_setimg)
  __TIC__(SEG_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(SEG_dpu)

  __TIC__(SEG_post)
  auto results = post_process();
  __TOC__(SEG_post)

  __TOC__(SEG_total)
  return results[0];
}

std::vector<MedicalSegcellResult> MedicalSegcellImp::run(
    const std::vector<cv::Mat>& input_img) {
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
  __TIC__(SEG_total)
  __TIC__(SEG_setimg)

  configurable_dpu_task_->setInputImageRGB(vimg);

  __TOC__(SEG_setimg)
  __TIC__(SEG_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(SEG_dpu)

  __TIC__(SEG_post)
  auto results = post_process();
  __TOC__(SEG_post)

  __TOC__(SEG_total)
  return results;
}

MedicalSegcellResult MedicalSegcellImp::post_process(unsigned int idx) {
  // in:  sWidth heiht scale channel:  128  128  64   3
  // out: sWidth heiht scale channel:  128  128  0.5  1

  unsigned int col_ind = 0;
  unsigned int row_ind = 0;
  auto output_layer = output_tensors_[0];
  float output_scale = tensor_scale(output_layer);
  cv::Mat segMat(output_layer.height, output_layer.width, CV_8UC1);
  int8_t* data = (int8_t*)output_layer.get_data(idx);

  for (size_t i = 0; i < output_layer.height * output_layer.width; ++i) {
    segMat.at<uchar>(row_ind, col_ind) =
        output_scale <= 0.0078125*2 ? 
            float(*(data + i)) * output_scale > 0.5 ? 255 : 0
        :
            float(*(data + i)) > 0 ? 255 : 0;
    col_ind++;
    if (col_ind > output_layer.width - 1) {
      row_ind++;
      col_ind = 0;
    }
  }

  return MedicalSegcellResult{(int)input_tensors_[0].width,
                              (int)input_tensors_[0].height, segMat};
}

std::vector<MedicalSegcellResult> MedicalSegcellImp::post_process() {
  auto ret = std::vector<vitis::ai::MedicalSegcellResult>{};
  ret.reserve(real_batch_size);

  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(post_process(i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
