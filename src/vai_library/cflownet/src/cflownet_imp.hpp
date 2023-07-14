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
#ifndef DEEPHI_Cflownet_HPP_
#define DEEPHI_Cflownet_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/cflownet.hpp>
#include <random>
using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class CflownetImp : public vitis::ai::TConfigurableDpuTask<Cflownet> {
 public:
  CflownetImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~CflownetImp();

 private:

  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;

  virtual CflownetResult run(const float* p) override;
  virtual std::vector<CflownetResult> run( const std::vector<const float*> ps) override;
  std::vector<CflownetResult> cflownet_post_process();
  CflownetResult cflownet_post_process(int idx);
  void cflownet_pre_process(int idx, const float*);

  int batch_size = 1;
  int real_batch_size = 1;

  int sWidth;
  int sHeight;
  int sChannel;
  float sScaleo, sScalei0, sScalei1;
  int o_idx;
  std::vector<int8_t*> oData;
  std::vector<int8_t*> iData0, iData1;
  std::mt19937 gen;
};
}  // namespace ai
}  // namespace vitis

#endif
