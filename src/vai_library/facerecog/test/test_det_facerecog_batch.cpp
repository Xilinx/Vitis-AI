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
      vitis::ai::FaceDetect::create("densebox_640_360", true);
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
  auto batch = recog->get_input_batch(); 
  cout << "recog batch: " << batch << endl;

  std::vector<cv::Mat> face_batch(batch);
  std::vector<cv::Rect> face_expanded_rects(batch);
  auto cnt = 0u;
  for (auto i = 0u; i < r1.size(); i++) {
    cnt = i % batch + 1; 
    auto e1 = FaceRecog::expand_and_align(
      image_normal.cols, image_normal.rows, //
      r1[i].x * image_normal.cols, r1[i].y * image_normal.rows,
      r1[i].width * image_normal.cols, r1[i].height * image_normal.rows, //
      0.2, 0.2, 16, 8);
    face_expanded_rects[i % batch] = e1.second; 
    face_batch[i % batch] = image_normal(e1.first);  
    if (cnt == batch) {
      auto results = recog->run_fixed(face_batch, face_expanded_rects); 
      for (auto c = 0u; c < cnt; ++c) {
        std::cout << "feature " <<  (int)(i / batch * batch + c) << ":";
        for (auto j = 0u; j < 512; ++j) {
          std::cout << (int)((*(results[c].feature))[j]) << "," ;
        } 
        std::cout << std::endl;
      }
    }
  } 
  return 0;
}
