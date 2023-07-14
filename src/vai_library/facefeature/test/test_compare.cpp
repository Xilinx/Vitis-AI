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

#include <stdio.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "vitis/ai/facefeature.hpp"

float feature_norm(const int8_t *feature) {
  int sum = 0;
  for (int i = 0; i < 512; ++i) {
    sum += feature[i] * feature[i];
  }
  return 1.f / sqrt(sum);
}

/// This function is used for computing dot product of two vector
static float feature_dot(const int8_t *f1, const int8_t *f2) {
  int dot = 0;
  for (int i = 0; i < 512; ++i) {
    dot += f1[i] * f2[i];
  }
  return (float)dot;
}

float feature_compare(const int8_t *feature, const int8_t *feature_lib){
  float norm = feature_norm(feature);
  float feature_norm_lib = feature_norm(feature_lib);
  return feature_dot(feature, feature_lib) * norm * feature_norm_lib;
}

float score_map(float score) { return 1.0 / (1 + exp(-12.4 * score + 3.763)); }

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "usage : " << argv[0] << "<model_name>"
              << " <img_url1> <img_url2> " << std::endl;
  }

  auto facefeature = vitis::ai::FaceFeature::create(argv[1], true);
  if (!facefeature) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  int width = facefeature->getInputWidth();
  int height = facefeature->getInputHeight();

  // get frist feature
  cv::Mat image = cv::imread(argv[2]);
  if (image.empty()) {
    std::cout << "cannot load " << argv[2] << std::endl;
    return -1;
  }
  cv::Mat img_resize;
  cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
             cv::INTER_LINEAR);
  auto result_fixed = facefeature->run_fixed(image);

  // get second feature
  cv::Mat image2 = cv::imread(argv[3]);
  if (image2.empty()) {
    std::cout << "cannot load " << argv[3] << std::endl;
    return -1;
  }
  cv::Mat img_resize2;
  cv::resize(image2, img_resize2, cv::Size(width, height), 0, 0,
             cv::INTER_LINEAR);
  auto result_fixed2 = facefeature->run_fixed(image2);

  auto similarity = feature_compare(result_fixed.feature->data(),
                                    result_fixed2.feature->data());
  float similarity_mapped = score_map(similarity);
  std::cout << "similarity :" << similarity << " after " << similarity_mapped
            << std::endl;
}
