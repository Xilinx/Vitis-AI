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
#include "vitis/ai/nnpp/facedetect.hpp"

#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/nnpp/apply_nms.hpp"
DEF_ENV_PARAM(DEBUG_XNNPP, "0")
using namespace std;
namespace vitis {
namespace ai {

static float my_div(float a, int b) { return a / static_cast<float>(b); };

//# Post process for DPUV1
#ifdef ENABLE_DPUCADX8G_RUNNER
//# GSTiling layer used for DPUV1
static void GSTilingLayer_forward_c(const void* top_np, const void* bottom_np,
                                    const int input_batch,
                                    const int input_channels,
                                    const int input_height,
                                    const int input_width, const int stride) {
  // cout << "in gstiling layer : " << input_channels << "\t" << input_height <<
  // "\t" <<  input_width << "\t" << stride << "\n";

  int stride_sq = stride * stride;

  const float* bottom = (float*)bottom_np;
  float* top = (float*)top_np;

  int output_channels = input_channels / stride_sq;
  int output_height = input_height * stride;
  int output_width = input_width * stride;
  float* tmp = new float[output_channels * output_height * output_width];

  int count_per_output_map = output_width * output_height * output_channels;
  int count_per_input_map = input_width * input_height * input_channels;

  const float* bottom_data = bottom;
  float* top_data = top;
  int ox, oy, oc, oi;
  int n, ic, iy, ix;

  for (n = 0; n < input_batch; ++n) {
    int ii = 0;
    for (ic = 0; ic < input_channels; ++ic) {
      for (iy = 0; iy < input_height; ++iy) {
        for (ix = 0; ix < input_width; ++ix) {
          int off = ic / output_channels;
          ox = ix * stride + off % stride;
          oy = iy * stride + off / stride;
          oc = ic % output_channels;
          oi = (oc * output_height + oy) * output_width + ox;

          tmp[oi] = bottom[ii];
          ++ii;
        }
      }
    }
    bottom_data += count_per_input_map;
    top_data += count_per_output_map;
  }

  int cnt = 0;
  for (int h = 0; h < output_height; h++) {
    for (int w = 0; w < output_width; w++) {
      for (int c = 0; c < output_channels; c++) {
        top[cnt++] =
            tmp[c * (output_height * output_width) + h * (output_width) + w];
      }
    }
  }
}
#endif

//# Templatize input datatype
template <typename T>
static vector<vector<float>> FilterBox(const float bb_out_scale,
                                       const float det_threshold, T* bbout,
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

//# Post process for DPUV1
#ifdef ENABLE_DPUCADX8G_RUNNER
FaceDetectResult face_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const float det_threshold,
    size_t batch_idx) {
  //# Get HW outputs
  const auto conv_in = (float*)output_tensors[0][1].get_data(batch_idx);
  const auto bb_in = (float*)output_tensors[0][0].get_data(batch_idx);
  int stride = 8;
  const auto batch_size_ = input_tensors[0][0].batch;

  //# Read HW conv output params
  auto conv_in_channel = output_tensors[0][1].channel;
  auto conv_in_height = output_tensors[0][1].height;
  auto conv_in_width = output_tensors[0][1].width;

  //# Compute output channels for ixel conv GSTiling layer
  int stride_sq = stride * stride;
  auto conv_out_channel_ = conv_in_channel / stride_sq;
  auto pixel_conv_tiling_height = conv_in_height * stride;
  auto pixel_conv_tiling_width = conv_in_width * stride;
  auto pixel_conv_tiling_size =
      conv_out_channel_ * pixel_conv_tiling_height * pixel_conv_tiling_width;

  //# perform GSTiling on pixel_conv
  void* pixel_conv_tiling = new float[batch_size_ * pixel_conv_tiling_size];
  GSTilingLayer_forward_c((void*)pixel_conv_tiling, (void*)conv_in, batch_size_,
                          conv_in_channel, conv_in_height, conv_in_width,
                          stride);

  //# Read HW bb output params
  auto bb_in_channel = output_tensors[0][0].channel;
  auto bb_in_height = output_tensors[0][0].height;
  auto bb_in_width = output_tensors[0][0].width;

  //# Compute output channels for ixel conv GSTiling layer
  auto bb_out_channel = bb_in_channel / stride_sq;
  const auto bb_out_width = bb_in_width * stride;
  const auto bb_out_height = bb_in_height * stride;
  const auto bb_tiling_size = bb_out_channel * bb_out_width * bb_out_height;
  const auto bb_out_scale =
      vitis::ai::library::tensor_scale(output_tensors[0][0]);

  //# perform GSTiling on bb_output
  const auto bb_tiling = new float[batch_size_ * bb_tiling_size];
  GSTilingLayer_forward_c(
      (void*)bb_tiling, (void*)bb_in, batch_size_, output_tensors[0][0].channel,
      output_tensors[0][0].height, output_tensors[0][0].width, stride);

  __TIC__(DET_SOFTMAX)
  const auto conv_out_size_ = pixel_conv_tiling_size / batch_size_;
  const auto conv_out_addr_ = (float*)pixel_conv_tiling;
  const auto conv_out_scale_ = 1.0f;

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
  vitis::ai::softmax((float*)conv_out_addr_, conv_out_scale_, conv_out_channel_,
                     conv_out_size_ / conv_out_channel_, &conf[0]);
  __TOC__(DET_SOFTMAX)

  __TIC__(DET_FILTER)

  float* bbout = (float*)bb_tiling;

  // if (1)
  //   std::cout << "bbout " << (void*)bbout << " "                //
  //             << "bb_out_width " << (int)bb_out_width << " "    //
  //             << "bb_out_height " << (int)bb_out_height << " "  //
  //             << "bb_out_scale " << bb_out_scale << " "         //
  //             << std::endl;

  vector<vector<float>> boxes =
      FilterBox(bb_out_scale, det_threshold, bbout, bb_out_width, bb_out_height,
                conf.data());
  __TOC__(DET_FILTER)

  __TIC__(DET_NMS)
  vector<float> scores(boxes.size());
  transform(boxes.begin(), boxes.end(), scores.begin(),
            [](auto& b) { return b[4]; });
  transform(boxes.begin(), boxes.end(), boxes.begin(), [](auto b) {
    b[0] = 0.5f * (b[0] + b[2]);
    b[1] = 0.5f * (b[1] + b[3]);
    b[2] = (b[2] - b[0]) * 2.0 + 1;
    b[3] = (b[3] - b[1]) * 2.0 + 1;
    return b;
  });
  vector<size_t> res_k;
  applyNMS(boxes, scores, config.dense_box_param().nms_threshold(), 0, res_k);
  vector<vector<float>> results;
  for (auto& k : res_k) {
    boxes[k][2] -= 1;
    boxes[k][3] -= 1;
    boxes[k][0] -= 0.5f * boxes[k][2];
    boxes[k][1] -= 0.5f * boxes[k][3];
    results.push_back(boxes[k]);
  }
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
#else
FaceDetectResult face_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const float det_threshold,
    size_t batch_idx) {
  // int num_of_classes = 2;
  const auto out_tensors00_channel_ = output_tensors[0][0].channel;
  const auto out_tensors01_channel_ = output_tensors[0][1].channel;
  auto conv_idx = 0;
  auto bbox_idx = 1;
  if (out_tensors00_channel_ == 2 && out_tensors01_channel_ == 4) {
    conv_idx = 0;
    bbox_idx = 1;
  } else if (out_tensors00_channel_ == 4 && out_tensors01_channel_ == 2) {
    conv_idx = 1;
    bbox_idx = 0;
  } else {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP))
        << "output_tensors channel is error "  //
        << "output_tensors[0][0] channel is " << out_tensors00_channel_
        << " "  //
        << "output_tensors[0][1] channel is " << out_tensors01_channel_
        << " "  //
        << std::endl;
  }
  __TIC__(DET_SOFTMAX)

  const auto batch_size_ = input_tensors[0][0].batch;
  const auto conv_out_size_ = output_tensors[0][conv_idx].size / batch_size_;
  const auto conv_out_addr_ =
      (int8_t*)output_tensors[0][conv_idx].get_data(batch_idx);
  const auto conv_out_scale_ =
      vitis::ai::library::tensor_scale(output_tensors[0][conv_idx]);
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
  const auto bb_out_scale =
      vitis::ai::library::tensor_scale(output_tensors[0][bbox_idx]);
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

  vector<float> scores(boxes.size());
  transform(boxes.begin(), boxes.end(), scores.begin(),
            [](auto& b) { return b[4]; });
  transform(boxes.begin(), boxes.end(), boxes.begin(), [](auto b) {
    b[0] = 0.5f * (b[0] + b[2]);
    b[1] = 0.5f * (b[1] + b[3]);
    b[2] = (b[2] - b[0]) * 2.0 + 1;
    b[3] = (b[3] - b[1]) * 2.0 + 1;
    return b;
  });
  vector<size_t> res_k;
  applyNMS(boxes, scores, config.dense_box_param().nms_threshold(), 0, res_k);
  vector<vector<float>> results;
  for (auto& k : res_k) {
    boxes[k][2] -= 1;
    boxes[k][3] -= 1;
    boxes[k][0] -= 0.5f * boxes[k][2];
    boxes[k][1] -= 0.5f * boxes[k][3];
    results.push_back(boxes[k]);
  }
  __TOC__(DET_NMS)

  const int input_width = input_tensors[0][0].width;
  const int input_height = input_tensors[0][0].height;

  auto ret = FaceDetectResult{input_width, input_height};
  for (const auto &r : results) {
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
#endif

std::vector<FaceDetectResult> face_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const float det_threshold) {
  auto batch_size = input_tensors[0][0].batch;
  auto ret = std::vector<FaceDetectResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.emplace_back(face_detect_post_process(input_tensors, output_tensors,
                                              config, det_threshold, i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
