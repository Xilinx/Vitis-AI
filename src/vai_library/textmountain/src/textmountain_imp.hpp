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
#ifndef DEEPHI_TextMountain_HPP_
#define DEEPHI_TextMountain_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include "vitis/ai/textmountain.hpp"

namespace vitis {

namespace ai {

class TextMountainImp : public vitis::ai::TConfigurableDpuTask<TextMountain> {
 public:
  TextMountainImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~TextMountainImp();

 private:

  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;

  virtual TextMountainResult run(const cv::Mat &img) override;
  virtual std::vector<TextMountainResult> run( const std::vector<cv::Mat> &img) override;

  std::unique_ptr<TextMountainPost> post_;
  int batch_size = 1, real_batch_size = 1;

  float scale_h[20], scale_w[20]; // max batch set to
  cv::Mat textmountain_pre_process(  const cv::Mat &input_img, int idx );
  int XLNX_TEXTMOUNTAIN_FIX_NPROUND=0;
  int round2nearest_multiple(int x, int p);
};
}  // namespace ai
}  // namespace vitis

#endif
