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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/library/tensor.hpp>

void set_input(const cv::Mat& raw, const float iso,
               vitis::ai::library::InputTensor& tensor, int batch_idx);

std::vector<float> invKSigma_unpad_rggb2bayer(
    void* output_data, void* input_data, const float output_scale,
    const float input_scale, const int rows, const int cols,
    const int input_width, const int channels, const int ph, const int pw,
    const float iso, const float scale);
