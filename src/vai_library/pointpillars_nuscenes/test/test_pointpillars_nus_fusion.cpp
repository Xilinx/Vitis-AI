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
//#include <vitis/ai/pointpillars_nuscenes.hpp>
#include "../src/multi_frame_fusion.hpp"
#include "../src/utils.hpp"
#include "../src/voxelize.hpp"

using namespace vitis::ai;
using namespace vitis::ai::pointpillars_nus;

void debug_vector(const std::vector<int8_t>& v, const std::string& name,
                  int dim) {
  auto lines = v.size() / dim;
  std::cout << name << " dim:" << dim << ", lines: " << lines << std::endl;
  int size = v.size();
  int head = std::min(5, size);
  int tail = std::min(5, size);
  std::cout << "first lines: " << std::endl;
  for (auto i = 0; i < head; ++i) {
    for (auto j = 0; j < dim; ++j) {
      std::cout << (int)(*(v.data() + dim * i + j)) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "last lines: " << std::endl;
  for (auto i = lines - tail; i < lines; ++i) {
    for (auto j = 0; j < dim; ++j) {
      std::cout << (int)(*(v.data() + dim * i + j)) << " ";
    }
    std::cout << std::endl;
  }
}

void save_vector(const std::vector<int8_t>& v, const std::string& name,
                 int dim) {
  auto lines = v.size() / dim;
  std::cout << "saving vector" << name << " dim:" << dim << " lines:" << lines
            << std::endl;

  std::ofstream o(name + ".txt");
  for (auto i = 0u; i < lines; ++i) {
    for (auto j = 0; j < dim; ++j) {
      o << (int)(*(v.data() + dim * i + j)) << " ";
    }
    o << std::endl;
  }
  o.close();
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "usage:" << argv[0] << " [anno_file_name]" << std::endl;
    exit(0);
  }

  std::string anno_file_name = argv[1];
  struct stat file_stat{};
  if (stat(anno_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << anno_file_name << " state error!" << std::endl;
    exit(-1);
  }
  auto points_info = PointsInfo();
  auto sweeps = std::vector<SweepInfo>();
  read_inno_file_pp_nus(anno_file_name, points_info, 5, sweeps);

  auto result =
      vitis::ai::pointpillars_nus::multi_frame_fusion(points_info, sweeps);
  LOG(INFO) << "result size:" << result.size();
  auto valid_dim = points_info.points.dim;
  LOG(INFO) << "result shape:" << result.size() / valid_dim << " * "
            << valid_dim;

  std::vector<float> range{-50.0, -50.0, -5.0, 50.0, 50.0, 3.0};
  result = points_filter(result, valid_dim, range);
  LOG(INFO) << "result size:" << result.size();
  LOG(INFO) << "result shape:" << result.size() / valid_dim << " * "
            << valid_dim;
  std::vector<int8_t> preprocess_buffer(40000 * 64 * 5);

  auto input_scale =
      std::vector<float>{0.02, 0.02, 0.25, 0.0078, 3.6364};  // read from config
  for (auto i = 0u; i < input_scale.size(); ++i) {
    input_scale[i] *= 64;
  }
  auto input_mean =
      std::vector<float>{0, 0, -1, 127.5, 0.275};  // read from config
  // auto preprocess_result = preprocess(result, valid_dim, input_mean,
  // input_scale, preprocess_buffer.data());
  auto voxelizer = Voxelization::create(input_mean, input_scale, 64, 40000);
  auto preprocess_result = voxelizer->voxelize(
      result, valid_dim, preprocess_buffer.data(), preprocess_buffer.size());
  LOG(INFO) << "preprocess result size:" << preprocess_result.size();
  std::string output_file = "preprocess.bin";
  CHECK(std::ofstream(output_file)
            .write(reinterpret_cast<char*>(preprocess_buffer.data()),
                   preprocess_buffer.size())
            .good());

  return 0;
}
