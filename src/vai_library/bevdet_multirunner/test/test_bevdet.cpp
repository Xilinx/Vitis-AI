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
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/bevdet.hpp>

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
              << "<model_name> <imgfile_names> <binfile_names> " << std::endl;
    abort();
  }

  auto model = vitis::ai::BEVdet::create(argv[1]);
  std::vector<std::string> names;
  LoadImageNames(argv[2], names);
  std::vector<cv::Mat> images;
  for (auto&& i : names) {
    images.emplace_back(cv::imread(i));
  }
  std::vector<std::vector<char>> bins;
  std::vector<std::string> bin_names;
  LoadImageNames(argv[3], bin_names);
  for (auto&& i : bin_names) {
    auto infile = std::ifstream(i, std::ios_base::binary);
    bins.emplace_back(std::vector<char>(std::istreambuf_iterator<char>(infile),
                                        std::istreambuf_iterator<char>()));
  }
  std::vector<vitis::ai::CenterPointResult> res;

  res = model->run(images, bins);
  for (size_t i = 0; i < 32 && i < res.size(); i++) {
    const auto& r = res[i];
    cout << "label: " << r.label << " score: " << r.score
         << " bbox: " << r.bbox[0] << " " << r.bbox[1] << " " << r.bbox[2]
         << " " << r.bbox[3] << " " << r.bbox[4] << " " << r.bbox[5] << " "
         << r.bbox[6] << " " << r.bbox[7] << " " << r.bbox[8] << endl;
  }
  return 0;
}
