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
#ifndef DEEPHI_Brtseg_HPP_
#define DEEPHI_Brtseg_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/brtseg.hpp>

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class BrtsegImp : public vitis::ai::TConfigurableDpuTask<Brtseg> {
 public:
  BrtsegImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~BrtsegImp();

 private:

  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;

  virtual BrtsegResult run(const cv::Mat &img) override;
  virtual std::vector<BrtsegResult> run( const std::vector<cv::Mat> &img) override;
  std::vector<BrtsegResult> brtseg_post_process();
  BrtsegResult brtseg_post_process(int idx);

  int batch_size = 1;
  int real_batch_size = 1;

  int sWidth=0;
  int sHeight=0;
  int sChannel=0;
  float sScaleo=0.0;
  int o_idx=0;
  std::vector<int8_t*> oData;
  std::vector<int8_t*> iData;
};
}  // namespace ai
}  // namespace vitis

#endif
