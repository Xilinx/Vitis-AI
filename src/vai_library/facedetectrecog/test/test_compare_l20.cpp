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

#include <cstdint>
#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>
#include <fstream>

using namespace std;
using namespace cv;

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

float feature_compare(const int8_t *feature, const int8_t *feature_lib) {
  float norm = feature_norm(feature);
  float feature_norm_lib = feature_norm(feature_lib);
  return feature_dot(feature, feature_lib) * norm * feature_norm_lib;
}

float score_map(float score) { return 1.0 / (1 + exp(-12.4 * score + 3.763)); }

void read_bin(int8_t* dst, int size, const char* file_path) {
  std::ifstream in(file_path, ios::in | ios::binary);
  if (!in.is_open()) {
    std::cout << "Open file :" << file_path << " error" << std::endl;
    exit(0);
  } else {
    for (auto i = 0; i < size; i++) {
      in.read((char*)dst + i, sizeof(int8_t));
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "usage: " << argv[0] << "<feature1.bin> <feature2.bin>" << std::endl;
    exit(0);
  }
  
  auto feature_1_name = argv[1];
  auto feature_2_name = argv[2];

  std::vector<int8_t> feature1(512);
  std::vector<int8_t> feature2(512);
  read_bin(&feature1[0], 512, feature_1_name);
  read_bin(&feature2[0], 512, feature_2_name);
  
  for (auto i = 0u; i < feature1.size(); ++i) {
    std::cout << (int)(feature1[i]) << ",";
  }
  std::cout << std::endl;

  auto similarity = feature_compare(&feature1[0], &feature2[0]);
  auto mapped = score_map(similarity);
  std::cout << "mapped score:" << mapped << std::endl;
  return 0;
}
