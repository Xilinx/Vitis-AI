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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/c2d2_lite.hpp>

using namespace std;
using namespace cv;

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
    std::cerr << "usage : " << argv[0]
              << "<model_name0> <model_name1> <image_path_file> " << std::endl;
    abort();
  }
  std::vector<std::string> names;
  LoadImageNames(argv[3], names);
  std::vector<cv::Mat> images;
  for (auto& name : names) {
    images.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
  }
  auto model = vitis::ai::C2D2_lite::create(argv[1], argv[2]);
  if (!model) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto result = model->run(images);
  cout << result << "\n";
  return 0;
}
