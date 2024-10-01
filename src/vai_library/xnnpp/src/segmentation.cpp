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

#include "vitis/ai/nnpp/segmentation.hpp"
#include <vitis/ai/max_index.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/env_config.hpp>

#include <queue>
#include <vector>
#include <mutex>

DEF_ENV_PARAM(HARDWARE_ARGMAX, "0");

namespace vitis {
namespace ai {

static void convert_color(const std::vector<std::string>& src, std::vector<uint32_t>& dest) {
  std::vector<std::vector<uint8_t>> tv(3);
  for(int i=0; i<3; i++) {
    size_t pos = 0;
    while ((pos = src[i].find_first_of("0123456789", pos)) != std::string::npos) {
      tv[i].emplace_back(std::stoi(std::string(src[i], pos)));
      pos = src[i].find_first_of(" ", pos);
    }
  }
  dest.resize(tv[0].size(), 0);
  uint8_t* pd = (uint8_t*)dest.data();
  for(int j=0; j<(int)tv[0].size(); j++) {
    for(int i=0; i<3; i++) {
      // dest[j*3+i] = tv[i][j];
      *(pd+j*4+i) = tv[i][j];
    }
  }
}

SegmentationResult segmentation_post_process_8UC1(
    const vitis::ai::library::InputTensor& input_tensors,
    const vitis::ai::library::OutputTensor& output_layer,
    size_t batch_idx) {

  if (ENV_PARAM(HARDWARE_ARGMAX) == 0) {
  __TIC__(SEG_MAX_VALUE)
    std::vector<uint8_t> output =
             vitis::ai::max_index((int8_t*)output_layer.get_data(batch_idx),
                                  output_layer.width, 
                                  output_layer.height,
                                  output_layer.channel);
    __TOC__(SEG_MAX_VALUE)

    __TIC__(SEG_COPY_IMAGE)
    cv::Mat segMat = cv::Mat(output_layer.height, output_layer.width, CV_8UC1,
                   output.data()).clone();
    __TOC__(SEG_COPY_IMAGE)
    return SegmentationResult{(int)input_tensors.width,
                              (int)input_tensors.height, segMat};
  } else {
    __TIC__(SEG_COPY_IMAGE)
    cv::Mat segMat = cv::Mat(output_layer.height, output_layer.width, CV_8UC1,
                   (uint8_t*)output_layer.get_data(batch_idx));
    __TOC__(SEG_COPY_IMAGE)
    return SegmentationResult{(int)input_tensors.width,
                              (int)input_tensors.height, segMat};
  }
}

std::vector<SegmentationResult> segmentation_post_process_8UC1(
    const vitis::ai::library::InputTensor& input_tensors,
    const vitis::ai::library::OutputTensor& output_tensors) {
  auto batch_size = input_tensors.batch;
  auto ret = std::vector<SegmentationResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.emplace_back(
        segmentation_post_process_8UC1(input_tensors, output_tensors, i));
  }
  return ret;
}


SegmentationResult segmentation_post_process_8UC3(
    const vitis::ai::library::InputTensor& input_tensors,
    const vitis::ai::library::OutputTensor& output_layer,
    const vitis::ai::proto::DpuModelParam& config,
    size_t batch_idx) {

  SegmentationResult ret{(int)input_tensors.width,
                         (int)input_tensors.height,
                         cv::Mat(output_layer.height, output_layer.width, CV_8UC3)};

  static std::vector<uint32_t> color_c;
  static std::mutex mtx_init;
  if (color_c.empty()) {
    mtx_init.lock();
    if(color_c.empty() ) {
      std::vector<std::string> scolor(3);
      scolor[0] = std::string{config.segmentation_param().color1()};
      scolor[1] = std::string{config.segmentation_param().color2()};
      scolor[2] = std::string{config.segmentation_param().color3()};
      convert_color(scolor, color_c);
    }
    mtx_init.unlock();
  }

  int i_all = output_layer.height * output_layer.width;
  uint8_t* pmat = ret.segmentation.ptr<uint8_t>(0);

  if (ENV_PARAM(HARDWARE_ARGMAX) == 0) {
    __TIC__(SEG_MAX_VALUE)
    std::vector<uint8_t> output;
    output = vitis::ai::max_index((int8_t*)output_layer.get_data(batch_idx),
                                  output_layer.width,
                                  output_layer.height,
                                  output_layer.channel);
    __TOC__(SEG_MAX_VALUE)
    __TIC__(SEG_CREATE_COLOR_IMG)
    for(int i=0; i<i_all-1; ++i, pmat+=3 )  {
       *(uint32_t*)pmat = color_c[output[i]];
    }
    memcpy(pmat, (uint8_t*)color_c.data()+output[i_all-1]*4, 3);
    __TOC__(SEG_CREATE_COLOR_IMG)
  } else {
    int8_t* output_p = (int8_t*)output_layer.get_data(batch_idx);
    __TIC__(SEG_CREATE_COLOR_IMG)
    for(int i=0; i<i_all-1; ++i, pmat+=3 )  {
        *(uint32_t*)(pmat) = color_c[output_p[i]];
    }
    memcpy(pmat, (uint8_t*)color_c.data()+output_p[i_all-1]*4, 3);
    __TOC__(SEG_CREATE_COLOR_IMG)
  }
  return ret;
}

std::vector<SegmentationResult> segmentation_post_process_8UC3(
    const vitis::ai::library::InputTensor& input_tensors,
    const vitis::ai::library::OutputTensor& output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  auto batch_size = input_tensors.batch;
  auto ret = std::vector<SegmentationResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.emplace_back(segmentation_post_process_8UC3(input_tensors,
                                                    output_tensors, config, i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
