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
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/nnpp/textmountain.hpp>

namespace vitis {
namespace ai {

using namespace std;

class TextMountainPostImp : public TextMountainPost{
 public:

  TextMountainPostImp(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      int batch_size,
      int& real_batch_size,
      float* scale_h,
      float* scale_w
  );

  virtual ~TextMountainPostImp();

  virtual TextMountainResult process(int idx) override;
  virtual std::vector<TextMountainResult> process() override;

 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  int batch_size_;
  int& real_batch_size_;
  float* scale_h_;
  float* scale_w_;
  int output_w;
  int output_h;
  int output_c;

  int XLNX_TEXTMOUNTAIN_FILTERIN=1;
  int XLNX_TEXTMOUNTAIN_IMPORT_POST=0;
  int kernel_size = 3;
  float scale_o;
  float score_thres=0.75, thres_center=0.6, score_thres_pixel=0.6;
  void get_cc_eval(cv::Mat& shrink_scores,
                 std::vector<float>& pred0_sft,
                 std::vector<bool>& valid,
                 cv::Mat& score_i,
                 std::vector<float>& score_mean);
  void maxpool( int8_t* input, std::vector<int>& output , cv::Mat& shrink_scores );
  void groupSearch(
           std::vector<std::pair<int,int>>& points_ptr ,
           std::vector<int>& next_ptr,  // next is indices_2d also is mpool
           cv::Mat & instance_ptr,       // instance_ptr is score_i
           std::vector<float> pred0_sft  // prob_ptr is pred0_sft
        );
  void fix_scale(cv::Point2f* vertices, cv::Point2d* dest, int idx);
  std::vector<float> pred0_sft;
};

}  // namespace ai
}  // namespace vitis

