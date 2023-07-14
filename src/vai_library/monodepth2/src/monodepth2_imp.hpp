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
#ifndef DEEPHI_Monodepth2_HPP_
#define DEEPHI_Monodepth2_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/monodepth2.hpp>

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class Monodepth2Imp : public vitis::ai::TConfigurableDpuTask<Monodepth2> {
 public:
  Monodepth2Imp(const std::string &model_name, bool need_preprocess = true);
  virtual ~Monodepth2Imp();

 private:

  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;

  virtual Monodepth2Result run(const cv::Mat &img) override;
  virtual std::vector<Monodepth2Result> run( const std::vector<cv::Mat> &img) override;
  std::vector<Monodepth2Result> monodepth2_post_process();
  Monodepth2Result monodepth2_post_process(int idx);
  int real_batch_size = 1;

  float MIN_DEPTH=0.001;
  float MAX_DEPTH=80.0;
  float scale_o;
  bool ifStereo = false;
};
}  // namespace ai
}  // namespace vitis

#endif
