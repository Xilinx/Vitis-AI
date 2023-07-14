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
#ifndef DEEPHI_BCC_HPP_
#define DEEPHI_BCC_HPP_

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/bcc.hpp>

namespace vitis {

namespace ai {

class BCCImp
    : public vitis::ai::TConfigurableDpuTask<BCC> {
 public:
  BCCImp(const std::string &model_name,
         bool need_preprocess = true);
  virtual ~BCCImp();

 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  const vitis::ai::proto::DpuModelParam cfg_;
  std::vector<float> mean;
  std::vector<float> scale;
  bool need_preprocess_= true;
  void setVarForPostProcess(const cv::Mat& img, int idx);
  void preprocess(const cv::Mat& img, int idx);
  virtual BCCResult run(const cv::Mat &img) override;
  virtual std::vector<BCCResult> run( const std::vector<cv::Mat> &img) override;

  std::vector<BCCResult> bcc_post_process();
  BCCResult bcc_post_process(int idx);

  std::vector<int> new_height, new_width;
  float scale_i = 0.0f;
  float scale_o = 0.0f;
  cv::Size size;
  unsigned int batch_size;
  int real_batch_size = 1;
  void cleanmem(unsigned int idx);
  void cleanmem();

};
}  // namespace ai
}  // namespace vitis

#endif
