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
#pragma once

#include <opencv2/imgproc/types_c.h>

#include <vector>

using namespace std;
namespace vitis {
namespace ai {
namespace rgbdsegmentation {
cv::Mat pad_image_to_shape(const cv::Mat& img, size_t shape, int border_mode,
                           std::vector<size_t>& margin);
void process_image_rgbd(const cv::Mat& img, const cv::Mat& disp,
                        const std::vector<float>& mean,
                        const std::vector<float>& img_scale,
                        const std::vector<float>& disp_scale, int8_t* img_data,
                        int8_t* disp_data);

}  // namespace rgbdsegmentation
}  // namespace ai
}  // namespace vitis
