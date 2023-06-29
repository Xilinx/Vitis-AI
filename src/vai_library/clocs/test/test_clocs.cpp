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

#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/profiling.hpp>
#include "vitis/ai/clocs.hpp"

using namespace vitis::ai;
using namespace vitis::ai::clocs;

static void read_points_file(const std::string& points_file_name,
                             std::vector<float>& points) {
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  }
  auto file_size = file_stat.st_size;
  LOG(INFO) << "input file:" << points_file_name << " size:" << file_size;
  // points_info.points.resize(file_size / 4);
  points.resize(file_size / 4);
  // CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char
  // *>(points_info.points.data()), file_size).good());
  CHECK(std::ifstream(points_file_name)
            .read(reinterpret_cast<char*>(points.data()), file_size)
            .good());
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    // std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0]
              << " [yolo_model] [pointpillars_model_0] [pointpillars_model_1] "
                 "[fusion_model] [idx]"
              << std::endl;
    exit(0);
  }

  auto idx = std::atoi(argv[5]);

  if (idx >= 10) {
    std::cerr << "invalid idx:" << idx << std::endl;
    exit(0);
  }

  auto points_file_name = std::string("00000") + std::to_string(idx) + ".bin";
  auto image_file_name = std::string("00000") + std::to_string(idx) + ".png";

  // 1. read float bin
  // input_file : 000001.bin
  auto input_file = std::string(points_file_name);
  std::vector<float> input;
  read_points_file(input_file, input);
  // int read_last_dim = 7;
  int last_dim = 4;
  int points_num = input.size() / last_dim;

  std::cout << "input shape: " << last_dim << " * " << points_num << std::endl;

  // 2. read image
  // image: 000001.png
  auto input_image = cv::imread(image_file_name);
  std::cout << "input_image rows:" << input_image.rows
            << ", cols:" << input_image.cols << std::endl;

  std::string yolo_model_name = argv[1];
  std::string pp_model_0 = argv[2];
  std::string pp_model_1 = argv[3];
  std::string fusion_model_name = argv[4];

  ClocsInfo clocs_info;
  std::vector<float> p2;
  std::vector<float> trv2c;
  std::vector<float> rect;
  read_points_file(std::to_string(idx) + "_P2.bin", p2);
  read_points_file(std::to_string(idx) + "_Trv2c.bin", trv2c);
  read_points_file(std::to_string(idx) + "_rect.bin", rect);
  clocs_info.calib_P2.assign(p2.begin(), p2.end());
  clocs_info.calib_Trv2c.assign(trv2c.begin(), trv2c.end());
  clocs_info.calib_rect.assign(rect.begin(), rect.end());
  clocs_info.image = input_image;
  clocs_info.points = input;

  auto clocs = vitis::ai::Clocs::create(yolo_model_name, pp_model_0, pp_model_1,
                                        fusion_model_name, true);
  if (!clocs) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  // auto ret = pointpillars->run(input);
  auto ret = clocs->run(clocs_info);

  auto size = ret.bboxes.size();
  for (auto i = 0u; i < size; ++i) {
    auto r = ret.bboxes[i];
    std::cout << "label:" << r.label << " ";
    std::cout << "bbox:"
              << " ";
    for (auto j = 0u; j < r.bbox.size(); ++j) {
      std::cout << r.bbox[j] << " ";
    }
    std::cout << "score:" << r.score;
    std::cout << std::endl;
  }
  // LOG(INFO) << "input width:" << ret.width
  //          << " input height: " << ret.height;
  // std::cout << "result size:" << ret.bboxes.size();
  // for (auto c = 0u; c < 10; ++c) {
  //  for (auto i = 0u; i < ret.bboxes.size(); ++i) {
  //    if (ret.bboxes[i].label != c) {
  //      continue;
  //    }
  //    std::cout << "label: " << ret.bboxes[i].label;
  //    std::cout << " bbox: ";
  //    for (auto j = 0u; j < ret.bboxes[i].bbox.size(); ++j) {
  //      std::cout << ret.bboxes[i].bbox[j] << " ";
  //    }
  //    std::cout << "score: " << ret.bboxes[i].score;
  //    std::cout << std::endl;
  //  }
  //}
  return 0;
}

