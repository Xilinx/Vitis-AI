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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

cv::Mat process_result(cv::Mat &image,
                       const vitis::ai::FaceFeatureFloatResult &result,
                       bool is_jpeg) {
  auto features = *result.feature;

  LOG_IF(INFO, is_jpeg) << "float features :";  //

  for (float f : features) {
    std::cout << f << " ";  //
  }
  std::cout << std::endl;

  return image;
}
