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
 * Filename: facedetect.hpp
 *
 * Description:
 * This network is used to getting position and score of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <string>
#include <vitis/ai/weak.hpp>
#include "vai_aie_task_handler.hpp"

const std::string dpuxclbin = "/media/sd-mmcblk0p1/dpu.xclbin";

class vai_graph {
 public:
  vai_graph(const std::string filename, const std::string cate) {
    // LOG(INFO) << filename << " " << cate;
    h_ = vitis::ai::WeakStore<std::string, vai_aie_task_handler>::create(
      filename, filename.c_str());
    g_ = xrtGraphOpen(h_->dhdl, h_->uuid, cate.c_str());
  }
  ~vai_graph() {
    xrtGraphClose(g_);
    h_ = nullptr;
  }

  std::shared_ptr<vai_aie_task_handler> h_;
  xrtGraphHandle g_;
};

