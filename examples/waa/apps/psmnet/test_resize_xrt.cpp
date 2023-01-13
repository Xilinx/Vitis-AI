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
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <xrt/experimental/xrt_aie.h>
#include <xrt/experimental/xrt_bo.h>
#include <xrt/experimental/xrt_device.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/xrt.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/profiling.hpp>

#include "dpu_resize.hpp"

DEF_ENV_PARAM_2(PSMNET_MODEL_DIR, "./PSMnet", std::string);

DEF_ENV_PARAM_2(PSMNET_MODEL_0, "PSMnet_0.xmodel", std::string);
DEF_ENV_PARAM_2(PSMNET_MODEL_1, "PSMnet_1.xmodel", std::string);
DEF_ENV_PARAM_2(PSMNET_MODEL_2, "PSMnet_2.xmodel", std::string);
DEF_ENV_PARAM(USE_REAL_SIZE, "0");
vector<vitis::ai::library::InputTensor> sort_tensors(
    const vector<vitis::ai::library::InputTensor>& tensors,
    vector<string>& layer_names) {
  vector<vitis::ai::library::InputTensor> ordered_tensors;
  for (auto i = 0u; i < layer_names.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].name.find(layer_names[i]) != std::string::npos) {
        ordered_tensors.push_back(tensors[j]);
        break;
      }
  return ordered_tensors;
}

vector<vitis::ai::library::OutputTensor> sort_tensors(
    const vector<vitis::ai::library::OutputTensor>& tensors,
    vector<string>& layer_names) {
  vector<vitis::ai::library::OutputTensor> ordered_tensors;
  for (auto i = 0u; i < layer_names.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].name.find(layer_names[i]) != std::string::npos) {
        ordered_tensors.push_back(tensors[j]);
        break;
      }
  return ordered_tensors;
}

int main(int argc, char** argv) {
  const int iw = 30;
  const int ih = 18;
  const int ic = 32;

  const int ow = 240;
  const int oh = 144;
  const int oc = 32;

  int input_size = iw * ih * ic;
  int output_size = ow * oh * oc;
  if (ENV_PARAM(USE_REAL_SIZE)) {
    input_size = 6659072;
    output_size = 11059200;
  }
  auto h_xcl = xclOpen(0, NULL, XCL_INFO);

  auto bo_xcl_in = xclAllocBO(h_xcl, input_size, 0, XCL_BO_FLAGS_CACHEABLE);
  auto bo_xcl_out = xclAllocBO(h_xcl, output_size, 0, XCL_BO_FLAGS_CACHEABLE);

  // auto bo_xcl_in = xclAllocBO(h_xcl, input_size, 0, 0);
  // auto bo_xcl_out = xclAllocBO(h_xcl, output_size, 0, 0);

  vitis::ai::library::OutputTensor input;
  auto batch = 3u;
  input.size = input_size;
  input.batch = batch;
  input.height = ih;
  input.width = iw;
  input.channel = ic;
  input.name = "input";
  for (auto b = 0u; b < batch; ++b) {
    input.xcl_bo[b].xcl_handle = h_xcl;
    input.xcl_bo[b].bo_handle = bo_xcl_in;
    input.xcl_bo[b].offset = 0;  // 6635520;
  }
  vitis::ai::library::InputTensor output;
  output.size = output_size;
  output.batch = batch;
  output.height = oh;
  output.width = ow;
  output.channel = oc;
  output.name = "output";
  for (auto b = 0u; b < batch; ++b) {
    output.xcl_bo[b].xcl_handle = h_xcl;
    output.xcl_bo[b].bo_handle = bo_xcl_out;
    output.xcl_bo[b].offset = 0;  // 9953280;
  }

  auto resize = std::make_unique<vai_resize>("/media/sd-mmcblk0p1/dpu.xclbin",
                                             input, output);
  LOG(INFO) << "input " << input << " "    //
            << "output " << output << " "  //
      ;
  __TIC__(PSMNET_RESIZE_LEFT)
  for (auto i = 0; i < 4; ++i) {
    // vai_res_[0]->run();
    resize->run();
  }
  LOG(INFO) << "ENV_PARAM(USE_REAL_SIZE) " << ENV_PARAM(USE_REAL_SIZE)
            << " "  //
      ;
  __TOC__(PSMNET_RESIZE_LEFT)

  // xrtBOFree(inBO);
  // xrtBOFree(outBO);
  // xrtDeviceClose(dhdl);
};
