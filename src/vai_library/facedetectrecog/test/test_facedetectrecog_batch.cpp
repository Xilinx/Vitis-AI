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
  if (argc < 2) {
    cout << "usage: " << argv[0] << "<image1> <image2> ..." << std::endl;
    exit(0);
  }

  std::vector<cv::Mat> images;
  for (auto i = 1; i < argc; ++i) {
    cv::Mat image_normal = cv::imread(argv[i]);
    if (image_normal.empty()) {
      cout << "cannot imread " << argv[i] << endl;
      exit(2);
    }
    images.push_back(image_normal);
  }
  auto detectrecog = FaceDetectRecog::create("densebox_640_360",
  //auto detectrecog = FaceDetectRecog::create("densebox_320_320",
                                             "face_landmark",
                                             "facerec_resnet20",
                                             true);
  if (!detectrecog) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  __TIC__(RECOG_MAT_FIXED_NORMAL)
  auto result_batch = detectrecog->run_fixed(images);
  __TOC__(RECOG_MAT_FIXED_NORMAL)


  for (auto n = 0u; n < result_batch.size(); ++n) {
    auto image_e = images[n].clone();
    std::cout << "image: " << n << std::endl;
    for (auto i = 0u; i < result_batch[n].rects.size(); ++i) {
      cv::Rect bbox{(int)(result_batch[n].rects[i].x * images[n].cols), 
                    (int)(result_batch[n].rects[i].y * images[n].rows),
                    (int)(result_batch[n].rects[i].width * images[n].cols), 
                    (int)(result_batch[n].rects[i].height * images[n].rows)};
      std::cout << "rect: " << bbox << std::endl;
      cv::rectangle(image_e, bbox, 0xff);
    }

    for (auto i = 0u; i < result_batch[n].features.size(); ++i) {
      std::cout << "feature " << i << ":";
      for (auto j = 0u; j < 512; ++j) {
        std::cout << (int)(result_batch[n].features[i][j]) << "," ;
      } 
      std::cout << std::endl;
      write_bin(&result_batch[n].features[i][0], 512, 
                std::string(std::string("feature-") + 
                std::to_string(n) + "-" +
                std::to_string(i) + std::string(".bin")).c_str()); 
    }
    cv::imwrite(std::string("recog_normal_e-") + std::to_string(n) + ".jpg", image_e);
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

  return 0;
}
