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
#include "vitis/ai/nnpp/facedetect.hpp"
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_XNNPP, "0")
using namespace std;
namespace vitis {
namespace ai {

static float my_div(float a, int b) { return a / static_cast<float>(b); };

static vector<vector<float>> FilterBox(const float bb_out_scale,
                                       const float det_threshold, int8_t* bbout,
                                       int w, int h, float* pred) {
  vector<vector<float>> boxes;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int position = i * w + j;
      vector<float> box;
      if (pred[position * 2 + 1] > det_threshold) {
        box.push_back(bbout[position * 4 + 0] * bb_out_scale + j * 4);
        box.push_back(bbout[position * 4 + 1] * bb_out_scale + i * 4);
        box.push_back(bbout[position * 4 + 2] * bb_out_scale + j * 4);
        box.push_back(bbout[position * 4 + 3] * bb_out_scale + i * 4);
        box.push_back(pred[position * 2 + 1]);
        boxes.push_back(box);
      }
    }
  }
  return boxes;
}

static void getResult(const vector<vector<float>>& box, const vector<int>& keep,
                      vector<vector<float>>& results) {
  results.clear();
  results.reserve(keep.size());
  for (auto i = 0u; i < keep.size(); ++i) {
    auto b = box[keep[i]];
    b[2] -= b[0];
    b[3] -= b[1];
    results.emplace_back(b);
  }
}

static void NMS(const float nms_threshold, const vector<vector<float>>& box,
                vector<vector<float>>& results) {
  auto count = box.size();
  vector<pair<size_t, float>> order(count);
  for (auto i = 0u; i < count; ++i) {
    order[i].first = i;
    order[i].second = box[i][4];
  }
  sort(order.begin(), order.end(),
       [](const pair<int, float>& ls, const pair<int, float>& rs) {
         return ls.second > rs.second;
       });

  vector<int> keep;
  vector<bool> exist_box(count, true);
  for (auto i = 0u; i < count; ++i) {
    auto idx = order[i].first;
    if (!exist_box[idx]) continue;
    keep.emplace_back(idx);
    for (auto j = i + 1; j < count; ++j) {
      auto kept_idx = order[j].first;
      if (!exist_box[kept_idx]) continue;
      auto x1 = max(box[idx][0], box[kept_idx][0]);
      auto y1 = max(box[idx][1], box[kept_idx][1]);
      auto x2 = min(box[idx][2], box[kept_idx][2]);
      auto y2 = min(box[idx][3], box[kept_idx][3]);
      auto intersect = max(0.f, x2 - x1 + 1) * max(0.f, y2 - y1 + 1);
      auto sum_area =
          (box[idx][2] - box[idx][0] + 1) * (box[idx][3] - box[idx][1] + 1) +
          (box[kept_idx][2] - box[kept_idx][0] + 1) *
              (box[kept_idx][3] - box[kept_idx][1] + 1);
      auto overlap = intersect / (sum_area - intersect);
      if (overlap >= nms_threshold) exist_box[kept_idx] = false;
    }
  }
  getResult(box, keep, results);
}

FaceDetectResult face_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>& input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const float det_threshold, size_t batch_idx) {
  // int num_of_classes = 2;
  const auto out_tensors00_channel_ = output_tensors[0][0].channel;
  const auto out_tensors01_channel_ = output_tensors[0][1].channel;
  auto conv_idx = 0;
  auto bbox_idx = 1;
  if(out_tensors00_channel_ == 2 && out_tensors01_channel_ == 4){
    conv_idx = 0;
    bbox_idx = 1;
    }
  else if(out_tensors00_channel_ == 4 && out_tensors01_channel_ == 2){
    conv_idx = 1;
    bbox_idx = 0;
    }
  else{
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP))
        << "output_tensors channel is error "  //
        << "output_tensors[0][0] channel is " << out_tensors00_channel_ << " "  //
        << "output_tensors[0][1] channel is " << out_tensors01_channel_ << " "  //
        << std::endl;
    }
  __TIC__(DET_SOFTMAX)

  const auto batch_size_ = input_tensors[0][0].batch;
  const auto conv_out_size_ = output_tensors[0][conv_idx].size / batch_size_;
  const auto conv_out_addr_ = (int8_t*)output_tensors[0][conv_idx].get_data(batch_idx);
  const auto conv_out_scale_ = vitis::ai::library::tensor_scale(output_tensors[0][conv_idx]);
  const auto conv_out_channel_ = output_tensors[0][conv_idx].channel;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP))
      << "output_tensors[0][0] " << output_tensors[0][0] << " "  //
      << "output_tensors[0][1] " << output_tensors[0][1] << " "  //
      ;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP))
      << "conv_out_size " << (int)conv_out_size_ << " "     //
      << "conv_out_addr_ " << (void*)conv_out_addr_ << " "  //
      << "conv_out_scale_ " << conv_out_scale_ << " "       //
      << "conv_out_channle " << conv_out_channel_ << " "    //
      << std::endl;
  vector<float> conf(conv_out_size_);
  vitis::ai::softmax(conv_out_addr_, conv_out_scale_, conv_out_channel_,
                      conv_out_size_ / conv_out_channel_, &conf[0]);
  __TOC__(DET_SOFTMAX)

  __TIC__(DET_FILTER)

  int8_t* bbout = (int8_t*)output_tensors[0][bbox_idx].get_data(batch_idx);
  const auto bb_out_width = output_tensors[0][bbox_idx].width;
  const auto bb_out_height = output_tensors[0][bbox_idx].height;
  const auto bb_out_scale = vitis::ai::library::tensor_scale(output_tensors[0][bbox_idx]);
  if (0)
    std::cout << "bbout " << (void*)bbout << " "                //
              << "bb_out_width " << (int)bb_out_width << " "    //
              << "bb_out_height " << (int)bb_out_height << " "  //
              << "bb_out_scale " << bb_out_scale << " "         //
              << std::endl;
  vector<vector<float>> boxes =
      FilterBox(bb_out_scale, det_threshold, bbout, bb_out_width, bb_out_height,
                conf.data());
  __TOC__(DET_FILTER)

  __TIC__(DET_NMS)
  vector<vector<float>> results;
  NMS(config.dense_box_param().nms_threshold(), boxes, results);
  __TOC__(DET_NMS)

  const int input_width = input_tensors[0][0].width;
  const int input_height = input_tensors[0][0].height;

  auto ret = FaceDetectResult{input_width, input_height};
  for (const auto r : results) {
    ret.rects.push_back(FaceDetectResult::BoundingBox{
        my_div(r[0], input_width),   //
        my_div(r[1], input_height),  //
        my_div(r[2], input_width),   //
        my_div(r[3], input_height),  //
        r[4]                         //
    });
  }
  return ret;
}

std::vector<FaceDetectResult> face_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>& input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, 
    const float det_threshold) {
    auto batch_size = input_tensors[0][0].batch;  
    auto ret = std::vector<FaceDetectResult>{};
    ret.reserve(batch_size);  
    for (auto i = 0u; i < batch_size; i++) {    
        ret.emplace_back(face_detect_post_process(input_tensors, output_tensors, config, det_threshold, i));
    }
    return ret;
}

}  // namespace ai
}  // namespace vitis
