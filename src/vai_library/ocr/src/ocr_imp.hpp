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
#pragma once 

#include <thread>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/ocr.hpp>
#include <xir/graph/graph.hpp>
#include "vitis/ai/nnpp/ocr.hpp"

using namespace std;
using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class OCRImp
    : public vitis::ai::TConfigurableDpuTask<OCR> {
 public:
  OCRImp(const std::string &model_name,
         bool need_preprocess = true);
  virtual ~OCRImp();

 private:
  virtual OCRResult run(const cv::Mat &img) override;
  virtual std::vector<OCRResult> run(
      const std::vector<cv::Mat> &img) override;
  void preprocess(const cv::Mat& input_img, int idx);
  void preprocess_thread(int start, int len,  uint8_t* input, int8_t* dest, int cols1_channels, int cols_channels, int cols1, int channels);

 void resize_oriimg(cv::Mat img, int idx);
 void cleanmem(unsigned int idx);
 void cleanmem();
 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  const vitis::ai::proto::DpuModelParam cfg_;

  std::unique_ptr<OCRPost> post_;

  std::vector<float> mean;
  std::vector<float> scale;
  std::vector<float> mean_scale;
  std::vector<int> target_h8, target_w8;
  std::vector<float> ratioh;
  std::vector<float> ratiow;
  std::vector<cv::Mat> oriimg;

  std::vector<std::thread> resize_oriimg_t;

  float scale_i = 0.0;
  int batch_size;
  int real_batch_size = 1;

  int XLNX_OCR_PRE_THREAD = 2;
  int XLNX_OCR_PRE_ROUND =1;
  int XLNX_OCR_PRE_CVRESIZE = 0;

};
}  // namespace ai
}  // namespace vitis

