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
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>

#include <cstdint>
#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>

#include <vitis/ai/facedetectrecog.hpp>
#include <vitis/ai/profiling.hpp>

using namespace vitis::ai;
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

void write_bin(const int8_t *src, int size, const char * file_path) {
  std::cout << "out path: " << file_path << "\n";
  std::ofstream out(file_path, ios::out|ios::binary);
  out.write((char *)src, sizeof(int8_t) * size);
  out.close();
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "usage: " << argv[0] << "<model_name> <image>" << std::endl;
    exit(0);
  }

  cv::Mat image_normal = cv::imread(argv[2]);
  if (image_normal.empty()) {
    cout << "cannot imread " << argv[2] << endl;
    exit(2);
  }
  //auto detectrecog = FaceDetectRecog::create("densebox_640_360",
  ////auto detectrecog = FaceDetectRecog::create("densebox_320_320",
  //                                           "face_landmark",
  //                                           "facerec_resnet20",
  //                                           true);
  auto detectrecog = FaceDetectRecog::create(argv[1], true);
  if (!detectrecog) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  __TIC__(RECOG_MAT_FIXED_NORMAL)
  auto result = detectrecog->run_fixed(image_normal);
  __TOC__(RECOG_MAT_FIXED_NORMAL)

  auto image_e = image_normal.clone();
  std::cout << "image: " << std::endl;
  for (auto i = 0u; i < result.rects.size(); ++i) {
    cv::Rect bbox{(int)(result.rects[i].x * image_normal.cols), (int)(result.rects[i].y * image_normal.rows),
                  (int)(result.rects[i].width * image_normal.cols), (int)(result.rects[i].height * image_normal.rows)};
    std::cout << "rect: " << bbox << std::endl;
    cv::rectangle(image_e, bbox, 0xff);
  }

  for (auto i = 0u; i < result.features.size(); ++i) {
    std::cout << "feature " << i << ":";
    for (auto j = 0u; j < 512; ++j) {
      std::cout << (int)(result.features[i][j]) << "," ;
    } 
    std::cout << std::endl;
    write_bin(&result.features[i][0], 512, 
              std::string(std::string("feature-") + std::to_string(i) + std::string(".bin")).c_str()); 
  }

  //auto result_float = detectrecog->run(image_normal);
  //if (result.features.size() >= 1) {
  //  std::cout << "feature scale: " << result.feature_scale << std::endl;
  //  std::cout << "feature float " << 0 << ":";
  //  for (auto j = 0u; j < 512; ++j) {
  //    std::cout << result_float.features[0][j] << "," ;
  //  }
  //  std::cout << std::endl;
  //  std::cout << "feature fixed" << 0 << ":";
  //  for (auto j = 0u; j < 512; ++j) {
  //    std::cout << (int)result.features[0][j] << "," ;
  //  }
  //  std::cout << std::endl;
  //}

  cv::imwrite("recog_normal_e.jpg", image_e);
  return 0;
}
