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
#include <sys/stat.h>
#include <vitis/ai/profiling.hpp>
#include "../src/voxelize.hpp"

int main( int argc, char *argv[])
{
  if (argc < 2) {
    std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    exit(0);
  }

  // 1. read float bin
  auto input_file = std::string(argv[1]);
  struct stat file_stat;
  if (stat(input_file.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << input_file << " state error!" << std::endl;
    exit(-1);
  } 
  auto file_size = file_stat.st_size; 
  auto len = file_size / 4;
  LOG(INFO) << "input file size:" << file_size;

  auto input = vitis::ai::pointpillars_nus::DataContainer<float>({uint32_t(len / 4), 4}, 0.f);
  LOG(INFO) << "input data shape: [" << input.shape[0]
            << ", " << input.shape[1] << "]";
  LOG(INFO) << "input data size: " << input.data.size();
  CHECK(std::ifstream(input_file).read(reinterpret_cast<char *>(input.data.data()), input.data.size() * 4).good()); 

  std::vector<int8_t> output_buffer(40000 *64 *4);
  float scale = 64;
  auto coors = vitis::ai::pointpillars_nus::preprocess(input, output_buffer.data(), scale); 

  
  CHECK(std::ofstream("out.bin").write(reinterpret_cast<char *>(output_buffer.data()), output_buffer.size()).good()); 
  return 0;
}

