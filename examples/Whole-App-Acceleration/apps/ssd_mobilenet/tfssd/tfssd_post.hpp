/*
 * Copyright 2019 Xilinx Inc.
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
#include "tfssd_detector.hpp"
#include "tfssd.hpp"
#include "sort_wrapper.h"
namespace vitis {
namespace ai {

using namespace dptfssd;

class TFSSDPost : public vitis::ai::TFSSDPostProcess {
 public:
  TFSSDPost(
      int inWidth, int inHeight,float conf_scale, float loc_scale,
      const vitis::ai::proto::DpuModelParam& config);
  virtual ~TFSSDPost();

  virtual TFSSDResult ssd_post_process(int8_t* conf, int8_t* loc,unsigned int idx, bool en_swpost, AcceleratorHandle* postproc_handle) override;
  virtual std::vector<TFSSDResult> ssd_post_process(int8_t* conf, int8_t* loc, bool en_swpost, AcceleratorHandle* postproc_handle) override;
  
 private:
  int num_classes_;
  float scale_conf_;
  float scale_loc_;
  int inWidth_;
  int inHeight_;
  // Prior Box
  std::vector<std::shared_ptr<std::vector<float>>> priors_;
  const short* fx_priors_;
  // DetectorImp
  std::unique_ptr<vitis::ai::dptfssd::TFSSDdetector> detector_;
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  SCORE_CONVERTER score_converter_;

  int CONF_IDX = 0;
  int LOC_IDX = 1;
};
}  // namespace ai
}  // namespace vitis
