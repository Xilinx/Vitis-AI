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
#include <vitis/ai/env_config.hpp>

DEF_ENV_PARAM(DEBUG_DPU_RESIZE, "0")
#include "./my_xrt_bo.hpp"

extern int xrtSyncBOAIENB(xrtDeviceHandle handle, xrtBufferHandle bohdl,
                          const char* gmioName, enum xclBOSyncDirection dir,
                          size_t size, size_t offset);
extern int xrtGMIOWait(xrtDeviceHandle handle, const char* gmioName);

const char* gmio_in_names[] = {"gmio_resize_in[0]", "gmio_resize_in[1]",

                               "gmio_resize_in[2]", "gmio_resize_in[3]",

                               "gmio_resize_in[4]", "gmio_resize_in[5]",

                               "gmio_resize_in[6]", "gmio_resize_in[7]"};
const char* gmio_out_names[] = {"gmio_resize_out[0]", "gmio_resize_out[1]",
                                "gmio_resize_out[2]", "gmio_resize_out[3]",
                                "gmio_resize_out[4]", "gmio_resize_out[5]",
                                "gmio_resize_out[6]", "gmio_resize_out[7]"};
const char* rtp_names[] = {
    "graph_resize.k[0].in[1]", "graph_resize.k[0].in[2]",

    "graph_resize.k[1].in[1]", "graph_resize.k[1].in[2]",

    "graph_resize.k[2].in[1]", "graph_resize.k[2].in[2]",

    "graph_resize.k[3].in[1]", "graph_resize.k[3].in[2]",

    "graph_resize.k[4].in[1]", "graph_resize.k[4].in[2]",

    "graph_resize.k[5].in[1]", "graph_resize.k[5].in[2]",

    "graph_resize.k[6].in[1]", "graph_resize.k[6].in[2]",

    "graph_resize.k[7].in[1]", "graph_resize.k[7].in[2]"};

vai_resize::vai_resize(const char* xclbin,
                       vitis::ai::library::OutputTensor&
                           input,  // output of DPU is input of resize
                       vitis::ai::library::InputTensor&
                           output  // input of next DPU is output of resize
) {
  //  h_ = vitis::ai::WeakStore<std::string, vai_aie_task_handler>::create(
  //      std::string(xclbin), xclbin);
  //  g_resize = xrtGraphOpen(h_->dhdl, h_->uuid, "graph_resize");
  graph_ = vitis::ai::WeakStore<std::string, vai_graph>::create(
      "graph_resize", std::string(xclbin), std::string("graph_resize"));
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
  const int output_size = ow_ * oh_ * oc_;
  const int output_bytes = output_size;
  inBO_ = xrtBOAlloc(graph_->h_->dhdl, input_bytes, XCL_BO_FLAGS_CACHEABLE, 0);
  in_ptr_ = xrtBOMap(inBO_);
  outBO_ =
      xrtBOAlloc(graph_->h_->dhdl, output_bytes, XCL_BO_FLAGS_CACHEABLE, 0);
  out_ptr_ = xrtBOMap(outBO_);
}

void vai_resize::run() {
  auto batch = in_.size();
  for (auto b = 0u; b < batch; ++b) {
    if (1) {
      run_internal(in_[b].real->bo, in_[b].offset, out_[b].real->bo,
                   out_[b].offset);
    } else {
      memcpy(in_ptr_, in_[b].ptr_, ih_ * iw_ * ic_);
      run_internal(inBO_, 0, outBO_, 0);
      memcpy(out_[b].ptr_, out_ptr_, oh_ * ow_ * oc_);
    }
  }
}

void vai_resize::run_internal(xrtBufferHandle input_xrt_bo, size_t input_offset,
                              xrtBufferHandle output_xrt_bo,
                              size_t output_offset) {
  int input_block_size = iw_ * ic_;
  int raw_bytes = input_block_size * bytes_of_value;
  int output_block_size = ow_ * oc_;
  int output_block_bytes = output_block_size * bytes_of_value;
  auto indexer = [this](auto index) { return (index + 0.5) / oh_ * ih_ - 0.5; };
  float h_lerp = 0.0f;
  for (auto hight = 0; hight < std::ceil(oh_ / 8.0f); hight++) {
    for (int i = 0; i < 8; i++) {
      int config[6] = {ic_, iw_, ow_, 0, input_fix_point_, output_fix_point_};
      auto oh_index = hight * 8 + i;
      if (oh_index >= oh_) break;
      float ih_index = indexer(oh_index);
      int h_lower = std::floor(ih_index);
      h_lower = h_lower < 0 ? 0 : h_lower;

      int h_upper = std::ceil(ih_index);
      h_upper = h_upper > ih_ - 1 ? ih_ - 1 : h_upper;
      h_lerp = ih_index - h_lower;
      xrtGraphUpdateRTP(graph_->g_, rtp_names[2 * i + 1], (char*)&h_lerp, 4);

      int input_block_bytes = raw_bytes;
      if (h_lower == h_upper) {
        config[3] = 1;
      } else {
        input_block_bytes = input_block_bytes * 2;
        config[3] = 0;
      }
      // The follow 6 is a magic number which need use a graceful way to
      // modify
      xrtGraphUpdateRTP(graph_->g_, rtp_names[2 * i], (char*)config,
                        6 * sizeof(int));

      xrtSyncBOAIENB(graph_->h_->dhdl, input_xrt_bo, gmio_in_names[i],
                     XCL_BO_SYNC_BO_GMIO_TO_AIE, input_block_bytes,
                     input_offset + raw_bytes * h_lower);
      xrtSyncBOAIENB(graph_->h_->dhdl, output_xrt_bo, gmio_out_names[i],
                     XCL_BO_SYNC_BO_AIE_TO_GMIO, output_block_bytes,
                     output_offset + output_block_size * oh_index);
    }
    for (int i = 0; i < 8; i++) {
      xrtGMIOWait(graph_->h_->dhdl, gmio_out_names[i]);
    }
  }
}

vai_resize::~vai_resize() {
  // terminate called after throwing an instance of 'xrt_core::error'
  // what():  failed to unmap BO
  // avoid xrt_core exception, TODO: resource leak.
  xrtBOFree(inBO_);
  xrtBOFree(outBO_);
}
