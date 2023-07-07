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
 * Filename: cpu_op.hpp
 *
 * Description:
 * This network is used to getting position and score of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <vitis/ai/library/tensor.hpp>
#include <vector>

class Resize {
 public:
  Resize(vitis::ai::library::OutputTensor& input,
         vitis::ai::library::InputTensor& output);
  ~Resize();
  void run();
 private:
  std::vector<size_t> i_shape_;
  std::vector<size_t> o_shape_;
  bool align_corners_;
  bool half_pixel_centers_;
  float fix_scale;

  std::vector<int8_t*> data_in_ptr_;
  std::vector<float> output_f_;
  std::vector<int8_t*> data_out_ptr_;
};

class CPUsfm {
 public:
  CPUsfm(vitis::ai::library::OutputTensor& tensor);
  ~CPUsfm();
  void run();
  float* get_output();
 private:
  std::vector<size_t> i_shape_;
  float fix_scale;
  std::vector<int8_t*> inputs_;
  std::vector<float> outputs_;
  std::vector<float> disp_;
};
