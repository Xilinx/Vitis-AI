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

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <vitis/ai/facefeature.hpp>
using namespace std;

void LoadImageNames(std::string const& filename,
                    std::vector<std::string>& images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE* fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }

  fclose(fp);
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "usage : " << argv[0] << " <model_name>"
              << " <image_list_file>  <output_file>" << std::endl;
    return -1;
  }
  string image_list_name = argv[2];
  string feature_output_name = argv[3];

  bool preprocess = !(getenv("PRE") != nullptr);
  auto facefeature = vitis::ai::FaceFeature::create(argv[1], preprocess);
  if (!facefeature) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  int width = facefeature->getInputWidth();
  int height = facefeature->getInputHeight();

  std::vector<string> names;
  LoadImageNames(image_list_name, names);

  ofstream out_file(feature_output_name);
  for (auto& name : names) {
    cv::Mat image = cv::imread(name);
    if (image.empty()) {
      std::cout << "cannot load " << name << std::endl;
      continue;
    }
    cv::Mat img_resize;

    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);

    auto result = facefeature->run(img_resize);
    out_file << "image " << name << " float_features:";  //
    for (auto feature : *result.feature) {
      out_file << feature << " ";  //
    }
    cv::Mat img_flip;
    cv::flip(img_resize, img_flip, 1);
    auto result_flip = facefeature->run(img_flip);
    for (auto feature : *result_flip.feature) {
      out_file << feature << " ";  //
    }

    out_file << std::endl;
  }

  return 0;
}
