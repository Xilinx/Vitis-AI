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
#include "./dpu_sfm.hpp"

#include <glog/logging.h>
#include <stdio.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>

DEF_ENV_PARAM(DEBUG_SFM_X8, "0");
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using namespace std;
using int8 = uint8_t;
using int32 = uint32_t;

extern int xrtSyncBOAIENB(xrtDeviceHandle handle, xrtBufferHandle bohdl,
                          const char* gmioName, enum xclBOSyncDirection dir,
                          size_t size, size_t offset);
extern int xrtGMIOWait(xrtDeviceHandle handle, const char* gmioName);

static constexpr int NUM_OF_CORE = 8;
static constexpr int CHANNEL = 192;
static constexpr int WIDTH = 960;
static constexpr int HEIGHT = 576;
static constexpr int SFM_INPUT_SIZE = (WIDTH * HEIGHT * CHANNEL / NUM_OF_CORE);

static constexpr auto SFM_INPUT_BYTE = (SFM_INPUT_SIZE * sizeof(int8));
static constexpr auto SFM_OUTPUT_BYTE =
    (SFM_INPUT_SIZE / CHANNEL * sizeof(float));

DpuSfm::DpuSfm(const char* xclbin, vitis::ai::library::OutputTensor& input) {
  auto batch = input.batch;
  graph_ = vitis::ai::WeakStore<std::string, vai_graph>::create(
      "graph_sfm", std::string(xclbin), std::string("graph_sfm"));
  inBO_ = vitis::ai::ImportedXrtBo::create(graph_->h_->dhdl, input);
  outBO_ =
      xrtBOAlloc(graph_->h_->dhdl, SFM_OUTPUT_BYTE * NUM_OF_CORE * batch, 0, 0);
  out_ = ((float*)xrtBOMap(outBO_));
  int32 control_params[6] = {0};
  control_params[0] = 1;
  control_params[1] = 192;
  control_params[2] = 1;
  control_params[3] = 1;
  control_params[4] = 0;
  control_params[5] = 0;
  for (auto c = 0; c < NUM_OF_CORE; ++c) {
    xrtGraphUpdateRTP(
        graph_->g_,
        (std::string("graph_sfm.superkernel[") + std::to_string(c) + "].in[1]")
            .c_str(),
        (char*)control_params, 6 * sizeof(int32));
  }
  // xrtGraphRun(graph_->g_, 0);
}

DpuSfm::~DpuSfm() {
  //  xrtGraphClose(g_sfm_);
  inBO_.clear();
  xrtBOFree(outBO_);
  //  h_ = nullptr;
}

void DpuSfm::run_with() {
  static std::shared_ptr<std::mutex> mtx =
      vitis::ai::WeakStore<std::string, std::mutex>::create("aie-resize");
  std::lock_guard<std::mutex> lock(*mtx);
  auto batch = inBO_.size();
  // LOG(INFO) << "input = " << *input;
  for (auto b = 0u; b < batch; ++b) {
    for (auto c = 0; c < NUM_OF_CORE; ++c) {
      xrtSyncBOAIENB(graph_->h_->dhdl, inBO_[b].real->bo,
                     (std::string("gmio_sfm_a") + std::to_string(c)).c_str(),
                     XCL_BO_SYNC_BO_GMIO_TO_AIE, SFM_INPUT_BYTE,
                     inBO_[b].offset + SFM_INPUT_BYTE * c);
    }
    auto batch_offset = b * SFM_OUTPUT_BYTE * NUM_OF_CORE;
    for (auto c = 0; c < NUM_OF_CORE; ++c) {
      xrtSyncBOAIENB(graph_->h_->dhdl, outBO_,
                     (std::string("gmio_sfm_c") + std::to_string(c)).c_str(),
                     XCL_BO_SYNC_BO_AIE_TO_GMIO, SFM_OUTPUT_BYTE,
                     batch_offset + SFM_OUTPUT_BYTE * c);
    }
    for (auto c = 0; c < NUM_OF_CORE; ++c) {
      xrtGMIOWait(graph_->h_->dhdl,
                  (std::string("gmio_sfm_c") + std::to_string(c)).c_str());
    }
  }
}
