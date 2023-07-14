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
#include <vitis/ai/profiling.hpp>
#include "../src/clocs_pointpillars.hpp"

using namespace vitis::ai;
// using namespace vitis::ai::clocs;

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
  if (argc < 4) {
    // std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0] << " [model_0] [model_1] [input_file]"
              << std::endl;
    exit(0);
  }

  // 1. read float bin
  auto input_file = std::string(argv[3]);
  std::vector<float> input;
  read_points_file(input_file, input);
  // int read_last_dim = 7;
  int last_dim = 4;
  int points_num = input.size() / last_dim;

  std::cout << "input shape: " << last_dim << " * " << points_num << std::endl;
  // std::string model_0 = "pointpillars_nuscenes_quant_v2_0";
  std::string model_0 = argv[1];
  // std::string model_1 = "pointpillars_nuscenes_quant_v2_1";
  std::string model_1 = argv[2];

  auto pointpillars = vitis::ai::ClocsPointPillars::create(model_0, model_1);
  if (!pointpillars) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  auto ret = pointpillars->run(input);

  auto size = ret.bboxes.size();
  for (auto i = 0u; i < size; ++i) {
    std::cout << "label:" << ret.bboxes[i].label << " ";
    std::cout << "bbox:"
              << " ";
    for (auto j = 0u; j < ret.bboxes[i].bbox.size(); ++j) {
      std::cout << ret.bboxes[i].bbox[j] << " ";
    }
    std::cout << "score:" << ret.bboxes[i].score;
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

