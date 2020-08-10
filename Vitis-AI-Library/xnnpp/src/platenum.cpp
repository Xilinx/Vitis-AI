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
#include "vitis/ai/nnpp/platenum.hpp"
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/image_util.hpp>
DEF_ENV_PARAM(DEBUG_XNNPP, "0")
using namespace std;
int enable_platenum_acc = 0;
namespace vitis {
namespace ai {
std::vector<PlateNumResult> plate_num_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors
) {
  auto results = std::vector<PlateNumResult>{};
  auto batch_size = input_tensors[0][0].batch;
  results.reserve(batch_size);
  softmax_output_t softmax_output;
  softmax_output = new float[8][35];
  std::vector<std::string> charactor;
  if (enable_platenum_acc == 1) {
    charactor = charactor_ch_;
  } else {
    charactor = charactor_py_;
  }
  for (auto batch_ind = 0u; batch_ind < batch_size; batch_ind++) {
    for (int i = 0; i < 8; i++) {
      vitis::ai::softmax((int8_t*)output_tensors[1][i].get_data(batch_ind),
                                std::get<0>(v_maxparam[i]),
                                std::get<1>(v_maxparam[i]),
                                std::get<2>(v_maxparam[i]), softmax_output[i]);
    }

    int index[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float score[8];
    for (int i = 0; i < 8; i++) {
      score[i] = softmax_output[i][0];
    }
    for (int j = 1; j < 32; j++) {
      if (softmax_output[0][j] > score[0]) {
        index[0] = j - 1;
        score[0] = softmax_output[0][j];
      }
    }

    for (int j = 1; j < 26; j++) {
      if (softmax_output[1][j] > score[1]) {
        index[1] = j - 1;
        score[1] = softmax_output[1][j];
      }
    }

    index[1] += 41;

    for (int i = 2; i < 7; i++) {
      for (int j = 1; j < 35; j++) {
        if (softmax_output[i][j] > score[i]) {
          index[i] = j - 1;
          score[i] = softmax_output[i][j];
        }
      }
      index[i] = index[i] < 23 ? (index[i] + 31) : (index[i] + 32);
    }

    for (int j = 1; j < 2; j++) {
      if (softmax_output[7][j] > score[7]) {
        index[7] = j;
        score[7] = softmax_output[7][j];
      }
    }

    // if (0) {
    //   std::cout << "plate_number " << index[0] << index[1] << index[2] << index[3]
    //             << index[4] << index[5] << index[6] << " "
    //             << "plate_color " << index[7] << " " //
    //             << std::endl;
    // }

    std::string plate_number = "";
    std::string plate_color = "";

    plate_color = color_[index[7]];
    for (int i = 0; i < 7; i++) {
      plate_number += charactor[index[i]];
    }

    // DLOG(INFO) << "Plate number & color: " << plate_number << " " << plate_color;
    results.emplace_back(PlateNumResult{(int)input_tensors[0][0].width, 
                          (int)input_tensors[0][0].height,
                          plate_number, plate_color });
  }
  delete[] softmax_output;
  return results;   
}


}  // namespace ai
}  // namespace vitis
