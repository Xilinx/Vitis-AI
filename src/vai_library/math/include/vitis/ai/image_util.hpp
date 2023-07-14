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
#ifndef DPMAP_IMAGE_UTIL_HPP_
#define DPMAP_IMAGE_UTIL_HPP_

#include <opencv2/core/core.hpp>

using std::pair;
using std::tuple;
using std::vector;

namespace vitis {
namespace ai {

/// All the normalize input API only support channels = 3

//# Method for float data type and NCHW format
void NormalizeInputData(const uint8_t* input, int rows, int cols, int channels,
                        int stride, const std::vector<float>& mean,
                        const std::vector<float>& scale, float* data);

void NormalizeInputData(const uint8_t* input, int rows, int cols, int channels,
                        int stride, const std::vector<float>& mean,
                        const std::vector<float>& scale, int8_t* data);

void NormalizeInputData(const float* input, int rows, int cols, int channels,
                        int stride, const std::vector<float>& mean,
                        const std::vector<float>& scale, int8_t* data);

void NormalizeInputData(const cv::Mat& img, const std::vector<float>& mean,
                        const std::vector<float>& scale, int8_t* data);

void NormalizeInputDataRGB(const cv::Mat& img, const std::vector<float>& mean,
                           const std::vector<float>& scale, int8_t* data);

//# Method for float data type and NCHW format
void NormalizeInputDataRGB(const cv::Mat& img, const std::vector<float>& mean,
                           const std::vector<float>& scale, float* data);

void NormalizeInputDataRGB(const uint8_t* input, int rows, int cols,
                           int channels, int stride,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale, int8_t* data);

//# Method for float data type and NCHW format
void NormalizeInputDataRGB(const uint8_t* input, int rows, int cols,
                           int channels, int stride,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale, float* data);
}  // namespace ai
}  // namespace vitis

#endif
