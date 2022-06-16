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
/*
 * Filename: dpu_sfm.hpp
 *
 * Description:
 * This network is used to getting position and score of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <stdio.h>
#include <xrt/experimental/xrt_bo.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <vitis/ai/library/tensor.hpp>

#include "./my_xrt_bo.hpp"
#include "./vai_graph.hpp"

class DpuSfm {
 public:
  explicit DpuSfm(const char* dpuxclbin,
                  vitis::ai::library::OutputTensor& input);
  ~DpuSfm();
  void run_with();
  float* get_output() { return out_; }

 private:
  std::shared_ptr<vai_graph> graph_;
  // one bo per batch
  std::vector<vitis::ai::ImportedXrtBo> inBO_;
  void* in_ptr_;
  xrtBufferHandle inBO;
  xrtBufferHandle outBO_;
  //xrtBufferHandle outBO_;
  std::vector<float> exp_lut_;
  float* out_;
  //void* g_sfm_;
};
