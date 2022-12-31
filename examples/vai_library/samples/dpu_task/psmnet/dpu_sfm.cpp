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
#include <mutex>

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

static void fill_exp_lut(std::vector<float>& exp_lut, int8_t input_fix_point) {
    float scale = pow(2.0, float(-1.0 * input_fix_point));
    const int offset = -127;
    for (int i = 0; i < 256; i++) {
        exp_lut[i] = expf((i + offset) * scale);
	//cout << exp_lut[i] << " ";
    }
    //cout << endl;
}

DpuSfm::DpuSfm(const char* xclbin, vitis::ai::library::OutputTensor& input) {
  auto batch = input.batch;
  graph_ = vitis::ai::WeakStore<std::string, vai_graph>::create(
      //"graph_softmax_conv1d", std::string(xclbin), std::string("graph_softmax_conv1d")); /*for vai3.0*/
      "graph_sfm", std::string(xclbin), std::string("graph_sfm")); /*for vai2.5*/
  inBO_ = vitis::ai::ImportedXrtBo::create(graph_->h_->dhdl, input);
  inBO = xrtBOAlloc(graph_->h_->dhdl, SFM_INPUT_BYTE, XCL_BO_FLAGS_CACHEABLE, 0);
  in_ptr_ = xrtBOMap(inBO);
  outBO_ =
      xrtBOAlloc(graph_->h_->dhdl, SFM_OUTPUT_BYTE * NUM_OF_CORE * batch, 0, 0);
  out_ = ((float*)xrtBOMap(outBO_));
  int32 control_params[6] = {1, 192, 1, 0, 0, 0};
  exp_lut_.resize(256);
  fill_exp_lut(exp_lut_, 1);
  for (auto c = 0; c < NUM_OF_CORE; ++c) {
    char cfg_rtp_name[256] = {0};
    char lut_rtp_name[256] = {0};
    //sprintf(cfg_rtp_name, "graph_softmax_conv1d.superkernel[%d].in[1]", c); /*for vai3.0*/
    sprintf(cfg_rtp_name, "graph_sfm.superkernel[%d].in[1]", c); /*for vai2.5*/
    //sprintf(lut_rtp_name, "graph_softmax_conv1d.superkernel[%d].in[2]", c);
    sprintf(lut_rtp_name, "graph_sfm.superkernel[%d].in[2]", c); /*for vai2.5*/
    xrtGraphUpdateRTP(graph_->g_, cfg_rtp_name, (char*)control_params, 6*sizeof(int32_t));
    xrtGraphUpdateRTP(graph_->g_, lut_rtp_name, (char*)exp_lut_.data(), 256*sizeof(float));
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
    if(ENV_PARAM(DEBUG_SFM_X8)) {
      memcpy(in_ptr_, inBO_[b].ptr_, SFM_INPUT_BYTE);
      ofstream fout("x2_" + to_string(b) + ".out", ios::binary);
      fout.write((char*)in_ptr_, SFM_INPUT_BYTE);
      fout.close();
    }
    for (auto c = 0; c < NUM_OF_CORE; ++c) {
      xrtSyncBOAIENB(graph_->h_->dhdl, inBO_[b].real->bo,
                     //(std::string("gmio_sfm_a") + std::to_string(c)).c_str(),
                     //(std::string("graph_softmax_conv1d.datain[") + std::to_string(c)+"]").c_str(), /*for vai3.0*/
                     (std::string("graph_sfm.datain[") + std::to_string(c)+"]").c_str(), /*for vai2.5*/
                     XCL_BO_SYNC_BO_GMIO_TO_AIE, SFM_INPUT_BYTE,
                     inBO_[b].offset + SFM_INPUT_BYTE * c);
    }

//    for (auto c = 0; c < NUM_OF_CORE; ++c) {
//      char sfm_inport_name[256] = {0};
//      sprintf(sfm_inport_name, "gmio_sfm_a%d", c);
//      if(ENV_PARAM(DEBUG_SFM_X8)==0){
//        xrtSyncBOAIENB(graph_->h_->dhdl, inBO_[b].real->bo, sfm_inport_name,
//          	      XCL_BO_SYNC_BO_GMIO_TO_AIE, SFM_INPUT_BYTE,
//          	      inBO_[b].offset);
//      } else {
//        xrtSyncBOAIENB(graph_->h_->dhdl, inBO, sfm_inport_name,
//          	      XCL_BO_SYNC_BO_GMIO_TO_AIE, SFM_INPUT_BYTE,
//          	      0);
//      }
//    }
    auto batch_offset = b * SFM_OUTPUT_BYTE * NUM_OF_CORE;
    for (auto c = 0; c < NUM_OF_CORE; ++c) {
      xrtSyncBOAIENB(graph_->h_->dhdl, outBO_,
                     //(std::string("gmio_sfm_c") + std::to_string(c)).c_str(),
                     //(std::string("graph_softmax_conv1d.dataout[") + std::to_string(c)+"]").c_str(), /*for vai3.0*/
                     (std::string("graph_sfm.dataout[") + std::to_string(c)+"]").c_str(), /*for vai2.5*/
                     XCL_BO_SYNC_BO_AIE_TO_GMIO, SFM_OUTPUT_BYTE,
                     batch_offset + SFM_OUTPUT_BYTE * c);
    }
    for (auto c = 0; c < NUM_OF_CORE; ++c) {
      xrtGMIOWait(graph_->h_->dhdl,
                  //(std::string("gmio_sfm_c") + std::to_string(c)).c_str());
                  //(std::string("graph_softmax_conv1d.dataout[") + std::to_string(c)+"]").c_str()); /*for vai3.0*/
                  (std::string("graph_sfm.dataout[") + std::to_string(c)+"]").c_str()); /*for vai2.5*/
    }
  }
}
