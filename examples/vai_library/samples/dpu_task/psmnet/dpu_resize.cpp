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
#include "./dpu_resize.hpp"

#include <glog/logging.h>
#include <xrt_mem.h>

#include <cstring>
#include <fstream>
#include <vitis/ai/env_config.hpp>

DEF_ENV_PARAM(DEBUG_DPU_RESIZE, "0")
#include "./my_xrt_bo.hpp"

extern int xrtSyncBOAIENB(xrtDeviceHandle handle, xrtBufferHandle bohdl,
                          const char* gmioName, enum xclBOSyncDirection dir,
                          size_t size, size_t offset);
extern int xrtGMIOWait(xrtDeviceHandle handle, const char* gmioName);

vai_resize::vai_resize(const char* xclbin,
                       vitis::ai::library::OutputTensor&
                           input,  // output of DPU is input of resize
                       vitis::ai::library::InputTensor&
                           output,  // input of next DPU is output of resize
                       std::vector<size_t>& channels
) {
  //  h_ = vitis::ai::WeakStore<std::string, vai_aie_task_handler>::create(
  //      std::string(xclbin), xclbin);
  //  g_resize = xrtGraphOpen(h_->dhdl, h_->uuid, "graph_resize");
  graph_ = vitis::ai::WeakStore<std::string, vai_graph>::create(
      "graph_upsample", std::string(xclbin), std::string("graph_upsample"));
  auto batch = input.batch;
  CHECK_EQ(batch, output.batch);
  in_ = vitis::ai::ImportedXrtBo::create(graph_->h_->dhdl, input);
  out_ = vitis::ai::ImportedXrtBo::create(graph_->h_->dhdl, output);
  iw_ = input.width;
  ih_ = input.height;
  ic_ = input.channel;

  ow_ = output.width;
  oh_ = output.height;
  oc_ = output.channel;

  input_fix_point_ = input.fixpos;
  output_fix_point_ = output.fixpos;
  channels_ = channels;
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RESIZE))
      << "input fixpos: " << input_fix_point_ << " "
      << "output fixpos: " << output_fix_point_ << " "
      << "resize offset: " << channels_[0] << " "
      << "resize stride: " << channels_[1] << " "
      ;

  std::vector<int> config = {input_fix_point_, output_fix_point_};
  xrtGraphUpdateRTP(graph_->g_, "graph_upsample.k_upsample_aie.in[1]", (char *)(config.data()), 2*sizeof(int));

  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RESIZE))
      << "input.name " << input.name << " "    //
      << "output.name " << output.name << " "  //
      << "ih_ " << ih_ << " "                  //
      << "iw_ " << iw_ << " "                  //
      << "ic_ " << ic_ << " "                  //
      << "ow_ " << ow_ << " "                  //
      << "oh_ " << oh_ << " "                  //
      << "oc_ " << oc_ << " "                  //
      ;
  const int input_size = iw_ * ih_ * ic_;
  const int input_bytes = input_size;
  const int output_size = ow_ * oh_ * channels_[1];
  const int output_bytes = output_size;
  inBO_ = xrtBOAlloc(graph_->h_->dhdl, input_bytes, XCL_BO_FLAGS_CACHEABLE, 0);
  in_ptr_ = xrtBOMap(inBO_);
  outBO_ =
      xrtBOAlloc(graph_->h_->dhdl, output_bytes, XCL_BO_FLAGS_CACHEABLE, 0);
  out_ptr_ = xrtBOMap(outBO_);

  mm2s_ = vitis::ai::WeakStore<std::string, vai_pl_kernel>::create(
      "mm2s", std::string(xclbin), std::string("mm2s"));
  s2mm_ = vitis::ai::WeakStore<std::string, vai_pl_kernel>::create(
      "s2mm", std::string(xclbin), std::string("s2mm"));
//  mm2s_khdl_ = xrtPLKernelOpen(graph_->h_->dhdl, graph_->h_->uuid, "mm2s");
//  mm2s_ = xrtRunOpen(mm2s_khdl_);
//  s2mm_khdl_ = xrtPLKernelOpen(graph_->h_->dhdl, graph_->h_->uuid, "s2mm");
//  s2mm_ = xrtRunOpen(s2mm_khdl_);
//  xrtGraphRun(graph_->g_, -1);
}

void vai_resize::run() {
  auto batch = in_.size();
  for (auto b = 0u; b < batch; ++b) {
    if (1) {
      run_internal(in_[b].real->bo, out_[b].real->bo,
                   channels_[0], channels_[1]);
    } else {
      // LOG(INFO) << "resize run";
      memcpy(in_ptr_, in_[2].ptr_, ih_ * iw_ * ic_);
      CHECK(std::ofstream("xxx_input_" + std::to_string(b) + ".bin")
              .write((char*)in_ptr_, ih_ * iw_ * ic_)
              .good())
        << "fail to write! filename=";
      // LOG(INFO) << "resize copy finish";
      run_internal(inBO_, outBO_, channels_[0], channels_[1]);
      // LOG(INFO) << "resize run finish";
      memcpy(out_[b].ptr_, out_ptr_, oh_ * ow_ * channels_[1]);
      auto file = "xxx_output_" + std::to_string(b) + ".bin";
      CHECK(std::ofstream(file)
              .write((char*)out_ptr_, oh_ * ow_ * channels_[1])
              .good())
        << "fail to write! filename=";

    }
  }
}

void vai_resize::run_internal(xrtBufferHandle inBO, 
                              xrtBufferHandle outBO,
                              size_t off, size_t strd) {
  xrtRunSetArg(mm2s_->r_, 0, inBO);
  xrtRunSetArg(mm2s_->r_, 2, ih_);
  xrtRunSetArg(mm2s_->r_, 3, iw_);
  xrtRunSetArg(mm2s_->r_, 4, ic_);
  xrtRunSetArg(mm2s_->r_, 5, oh_);
  xrtRunSetArg(mm2s_->r_, 6, ow_);
  xrtRunSetArg(mm2s_->r_, 7, oc_);
  xrtRunStart(mm2s_->r_);

  xrtRunSetArg(s2mm_->r_, 0, outBO);
  xrtRunSetArg(s2mm_->r_, 2, ih_);
  xrtRunSetArg(s2mm_->r_, 3, iw_);
  xrtRunSetArg(s2mm_->r_, 4, ic_);
  xrtRunSetArg(s2mm_->r_, 5, oh_);
  xrtRunSetArg(s2mm_->r_, 6, ow_);
  xrtRunSetArg(s2mm_->r_, 7, oc_);
  xrtRunSetArg(s2mm_->r_, 8, off);
  xrtRunSetArg(s2mm_->r_, 9, strd);
  xrtRunStart(s2mm_->r_);

  xrtRunWait(mm2s_->r_);
  xrtRunWait(s2mm_->r_);
}

vai_resize::~vai_resize() {
  // terminate called after throwing an instance of 'xrt_core::error'
  // what():  failed to unmap BO
  // avoid xrt_core exception, TODO: resource leak.
  xrtBOFree(inBO_);
  xrtBOFree(outBO_);
}
