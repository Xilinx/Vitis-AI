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
#include "../src/scatter.hpp"

int main( int argc, char *argv[])
{
  if (argc < 3) {
    std::cout << "usage: " << argv[0] << " <input_file> <coors_file>" << std::endl;
    exit(0);
  }

  // 1. read int8 bin
  auto input_file = std::string(argv[1]);
  struct stat file_stat;
  if (stat(input_file.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << input_file << " state error!" << std::endl;
    exit(-1);
  } 
  auto file_size = file_stat.st_size; 
  auto len = file_size;
  LOG(INFO) << "input file size:" << file_size;

  auto input = std::vector<int8_t>(len);
  LOG(INFO) << "input data size: " << input.size();
  CHECK(std::ifstream(input_file).read(reinterpret_cast<char *>(input.data()), input.size()).good()); 

  // 2. read coors file
  auto coors_file = std::string(argv[2]);
  if (stat(coors_file.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << coors_file << " state error!" << std::endl;
    exit(-1);
  } 
  file_size = file_stat.st_size;
  len = file_size / 4;
  LOG(INFO) << "coors file size:" << file_size;
  //auto coors = vitis::ai::pointpillars_nus::DataContainer<int>({uint32_t(len / 4), 4}, 0);
  std::vector<int> coors(len, 0);
  LOG(INFO) << "coors data size: " << coors.size();
  CHECK(std::ifstream(coors_file).read(reinterpret_cast<char *>(coors.data()), coors.size() * 4).good()); 


  std::vector<int8_t> output(400*400*64);
  vitis::ai::pointpillars_nus::scatter(coors, 4, input.data(), 1.0, output.data(), 1.0, 64, 400, 400);

  LOG(INFO) << "output data size:" << output.size();
  
  CHECK(std::ofstream("out.bin").write(reinterpret_cast<char *>(output.data()), output.size()).good()); 
  return 0;
}

