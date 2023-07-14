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

#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/facerecog.hpp>
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

void write_bin(const int8_t *src, int size, const char * file_path) {
  std::cout << "out path: " << file_path << "\n";
  std::ofstream out(file_path, ios::out|ios::binary);
  out.write((char *)src, sizeof(int8_t) * size);
  out.close();
}


int main(int argc, char *argv[]) {
  auto recog = FaceRecog::create("facerec_resnet20", true);
  auto densebox_detect =
      vitis::ai::FaceDetect::create("densebox_320_320", true);
  if (!densebox_detect) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  cv::Mat image_normal = cv::imread(argv[1]);

  if (image_normal.empty()) {
    cout << "cannot imread " << argv[1] << endl;
    exit(2);
  }
  __TIC__(RECOG_MAT_FIXED_NORMAL)
  auto r1 = densebox_detect->run(image_normal).rects;
  auto e1 = FaceRecog::expand_and_align(
      image_normal.cols, image_normal.rows, //
      r1[0].x * image_normal.cols, r1[0].y * image_normal.rows,
      r1[0].width * image_normal.cols, r1[0].height * image_normal.rows, //
      0.2, 0.2, 16, 8);
  auto result_tuple =
      recog->run_fixed(image_normal(e1.first), e1.second.x, e1.second.y,
                       e1.second.width, e1.second.height);
  auto image_e = image_normal(e1.first).clone();
  cv::rectangle(image_e, e1.second, 0xff);
  cv::imwrite("recog_normal_e.jpg", image_e);
  __TOC__(RECOG_MAT_FIXED_NORMAL)

  auto features_normal = std::move(result_tuple.feature);

  cout << "Image normal:";
  std::cout << "feature :";
  for (auto j = 0u; j < 512; ++j) {
    std::cout << (int)((*features_normal)[j]) << "," ;
  } 
  std::cout << std::endl;
  write_bin(&((*features_normal)[0]), 512, 
            std::string(std::string("feature-normal") + std::string(".bin")).c_str()); 


  for (int i = 2; i < argc; ++i) {
    auto filename = string{argv[i]};
    cv::Mat image_test = cv::imread(filename);
    if (image_test.empty()) {
      cout << "cannot imread " << filename << endl;
      continue;
    }
    __TIC__(RECOG_MAT_FIXED_TEST)
    auto r2 = densebox_detect->run(image_test).rects;
    auto e2 = FaceRecog::expand_and_align(
        image_test.cols, image_test.rows, //
        r2[0].x * image_test.cols, r2[0].y * image_test.rows,
        r2[0].width * image_test.cols, r2[0].height * image_test.rows, //
        0.2, 0.2, 16, 8);
    auto result_tuple2 = recog->run_fixed(image_test(e2.first), e2.second.x, e2.second.y,
                                          e2.second.width, e2.second.height);
    __TOC__(RECOG_MAT_FIXED_TEST)
    auto features_test = std::move(result_tuple2.feature);
    cout << filename << " ";
    cout << "Image test:";

    std::cout << "feature " << i << ":";
    for (auto j = 0u; j < 512; ++j) {
      std::cout << (int)((*features_test)[j]) << "," ;
    } 
    std::cout << std::endl;
    write_bin(&((*features_test)[0]), 512, 
                std::string(std::string("feature-test") + std::string(".bin")).c_str()); 

    auto similarity =
        feature_compare(features_normal->data(), features_test->data());
    float similarity_mapped = score_map(similarity);
    cout << "similarity raw:" << similarity << ", "
         << "similarity mapped:" << similarity_mapped << endl;
    auto image_e2 = image_test(e2.first).clone();
    cv::rectangle(image_e2, e2.second, 0xff);
    cv::imwrite(string{"compare_"} + std::to_string(i) + ".jpg", image_e2);
  }
}
