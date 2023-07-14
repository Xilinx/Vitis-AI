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
#include <vart/tensor_buffer.hpp>
#include <vector>

#include "vitis/ai/bevdet.hpp"
#pragma once

std::string find_bevdet_1_pt_file(const std::string& name);

std::vector<vitis::ai::CenterPointResult> post_process(
    const std::vector<vart::TensorBuffer*>& output_tensors,
    float score_threshold);

void copy_input_from_image(const std::vector<cv::Mat>& imgs,
                           vart::TensorBuffer* input, std::vector<float> mean,
                           std::vector<float> scale);

void copy_input_from_bin(const std::vector<char>& bin,
                         vart::TensorBuffer* input);