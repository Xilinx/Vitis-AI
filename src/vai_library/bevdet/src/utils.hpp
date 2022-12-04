/*
 * Copyright 2019 xilinx Inc.
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
#include <string>
#include <vector>
#include <vitis/ai/library/tensor.hpp>

#include "vitis/ai/bevdet.hpp"
#pragma once

std::string find_bevdet_1_pt_file(const std::string& name);

std::vector<vitis::ai::CenterPointResult> post_process(
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    float score_threshold);

cv::Mat resize_and_crop_image(const cv::Mat& image);
bool filesize(const std::string& filename);
int8_t float2fix(float data);
void my_softmax(int8_t* input, int begin_idx, float* output);