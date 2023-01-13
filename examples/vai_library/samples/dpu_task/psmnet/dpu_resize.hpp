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
 * Filename: dpu_resize.hpp
 *
 * Description:
 * This network is used to getting position and score of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <xrt/experimental/xrt_bo.h>

#include <iostream>
#include <vector>
#include <vitis/ai/library/tensor.hpp>

#include "./my_xrt_bo.hpp"
#include "./vai_graph.hpp"

class vai_resize {
 public:
  vai_resize(const char* xclbin_path,
             vitis::ai::library::OutputTensor&
                 input,  // output of DPU is input of resize
             vitis::ai::library::InputTensor&
                 output,  // input of next DPU is output of resize
             std::vector<size_t>& channels
  );
  ~vai_resize();

  void run();
  void run_internal(xrtBufferHandle input_xrt_bo,
                    xrtBufferHandle output_xrt_bo,
		    size_t offset, size_t stride);

 private:
  std::shared_ptr<vai_graph> graph_;
  std::vector<vitis::ai::ImportedXrtBo> in_;
  std::vector<vitis::ai::ImportedXrtBo> out_;
  int ih_, iw_, ic_, oh_, ow_, oc_;
  int input_fix_point_;
  int output_fix_point_;
  //const int bytes_of_value = 1;
  xrtBufferHandle inBO_;
  void* in_ptr_;
  xrtBufferHandle outBO_;
  void* out_ptr_;

  std::vector<size_t> channels_;

  std::shared_ptr<vai_pl_kernel> mm2s_;
  std::shared_ptr<vai_pl_kernel> s2mm_;
  //xrtKernelHandle mm2s_khdl_;
  //xrtRunHandle mm2s_;
  //xrtKernelHandle s2mm_khdl_;
  //xrtRunHandle s2mm_;
};
