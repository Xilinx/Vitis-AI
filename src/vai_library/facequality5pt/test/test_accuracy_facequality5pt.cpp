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
#include <string>
#include <vector>
#include <vitis/ai/facequality5pt.hpp>
using namespace std;

void LoadImageNames(std::string const &filename,
                    std::vector<std::string> &images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE *fp = fopen(filename.c_str(), "r");
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

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "usage : " << argv[0] << " <model_name>"
              << " <image_list_file> <output_file> "
              << std::endl;
    return -1;
  }
  string image_list_name = argv[2];
  string output_name = argv[3];

  bool preprocess = !(getenv("PRE") != nullptr);
  auto facequality5pt= vitis::ai::FaceQuality5pt::create(argv[1], preprocess);
  int width = facequality5pt->getInputWidth();
  int height = facequality5pt->getInputHeight();

  std::vector<string> image_names;
  LoadImageNames(image_list_name, image_names);

  ofstream output_file(output_name);
  for (auto name : image_names) {
    cv::Mat image = cv::imread(name);
    if (image.empty()) {
      std::cout << "cannot load " << name << std::endl;
      continue;
    }
    cv::Mat img_resize;

    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);

    auto result = facequality5pt->run(img_resize);
    output_file << "image " << name << " quality :" << result.score;  //
    output_file << std::endl;
  }

  return 0;
}
