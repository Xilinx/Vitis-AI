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
#include "vitis/ai/nnpp/platedetect.hpp"

#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_XNNPP, "0")
using namespace std;
namespace vitis {
namespace ai {
int GLOBAL_ENABLE_TEST_ACC = 0;
static PlateDetectResult FilterBox(float scale_box, float scale_conf,
                                   int input_width, int input_height,
                                   int8_t* bbout, int8_t* conf) {
  __TIC__(PLATEDETECT_POST)
  std::vector<float> box;
  auto prob_max = conf[64];
  int cc = 0, ii = 0, jj = 0;
  for (size_t c = 0; c < 64; c++) {
    for (size_t i = 0; i < 10; i++) {
      for (size_t j = 0; j < 10; j++) {
        int dpu_index = i * 10 * 128 + j * 128 + (c + 64);
        if (conf[dpu_index] >= prob_max) {
          cc = c;
          ii = i;  // hei
          jj = j;  // wid

          prob_max = conf[dpu_index];
          // cout << (int)prob_max << endl;
        }
      }
    }
  }
  prob_max *= scale_conf;
  float prob_max_neg = conf[ii * 10 * 128 + jj * 128 + cc] * scale_conf;
  float prob_softmax = exp(prob_max) / (exp(prob_max) + exp(prob_max_neg));

  int row = 8 * ii + cc / 8;
  int col = 8 * jj + cc % 8;
  // cout << row << " " << col << " " << ii << " " << jj << " " << cc << " "
  // << prob_max_neg << " " << prob_softmax << " " << prob_max << endl;
  int bbox_index = ii * 10 * 512 + jj * 512 + cc;
  int x1 = (int)((col * 4 + bbout[bbox_index] * scale_box));
  int y1 = (int)((row * 4 + bbout[bbox_index + 64] * scale_box));
  int x2 = (int)((col * 4 + bbout[bbox_index + 128] * scale_box));
  int y2 = (int)((row * 4 + bbout[bbox_index + 192] * scale_box));
  int x3 = (int)((col * 4 + bbout[bbox_index + 256] * scale_box));
  int y3 = (int)((row * 4 + bbout[bbox_index + 320] * scale_box));
  int x4 = (int)((col * 4 + bbout[bbox_index + 384] * scale_box));
  int y4 = (int)((row * 4 + bbout[bbox_index + 448] * scale_box));

  if (GLOBAL_ENABLE_TEST_ACC == 1) {
    cout << x1 << " " << y1 << " " << x2 << " " << y2 << " " << x3 << " " << y3
         << " " << x4 << " " << y4 << endl;
  }

  int x = std::min(x1, x4);
  int y = std::min(y1, y2);
  int width = std::max(x2, x3) - x;
  int height = std::max(y3, y4) - y;
  //  cout << "xywh" << x << " " << y << " " << width << " " << height << endl;
  __TOC__(PLATEDETECT_POST)
  return PlateDetectResult{
      input_width,
      input_height,
      PlateDetectResult::BoundingBox{prob_softmax,                   //
                                     (float)x / input_width,         //
                                     (float)y / input_height,        //
                                     (float)width / input_width,     //
                                     (float)height / input_height},  //
      PlateDetectResult::Point{(float)x1 / input_width,
                               (float)y1 / input_height},  //
      PlateDetectResult::Point{(float)x2 / input_width,
                               (float)y2 / input_height},  //
      PlateDetectResult::Point{(float)x4 / input_width,
                               (float)y4 / input_height},  //
      PlateDetectResult::Point{(float)x3 / input_width,
                               (float)y3 / input_height}};
}

std::vector<PlateDetectResult> plate_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors) {
  auto results = std::vector<PlateDetectResult>{};
  auto batch_size = input_tensors[0][0].batch;
  results.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    results.emplace_back(
        FilterBox(vitis::ai::library::tensor_scale(output_tensors[0][0]),
                  vitis::ai::library::tensor_scale(output_tensors[0][1]),
                  input_tensors[0][0].width, input_tensors[0][0].height,
                  (int8_t*)output_tensors[0][0].get_data(i),
                  (int8_t*)output_tensors[0][1].get_data(i)));
  }
  return results;
}

}  // namespace ai
}  // namespace vitis
