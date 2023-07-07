/*
 * Copyright 2019 Xilinx Inc.
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
#include <vitis/ai/centerpoint.hpp>
#include <fstream>
#include <string>
using namespace std;


template <class T>
static void readfile(string& filename, vector<T>& data) {
  ifstream input_file(filename);
  std::string line;
  while (std::getline(input_file, line)) {
    istringstream ss(line);
    T num;
    ss >> num;
    data.push_back(num);
  }
  //cout << filename << " " << data.size() << endl;
}



int main( int argc, char *argv[])
{
  if (argc < 4) {
    std::cout << "usage: " << argv[0] << " <model1> <model2> <input_file...>" << std::endl;
    exit(0);
  }

  // 1. read float bin

  std::string model_0 = argv[1];
  std::string model_1 = argv[2];
  auto centerpoint = vitis::ai::CenterPoint::create(
          model_0, model_1);
  std::vector<std::vector<float>> inputs(centerpoint->get_input_batch());
  for (auto i = 0u; i < centerpoint->get_input_batch(); i++) {
    auto input_file = std::string(argv[i % (argc - 3) + 3]);
    std::cout << "input file  " << input_file << endl;
    readfile(input_file, inputs[i]);	
  }
  auto results = centerpoint->run(inputs);
  for (auto& result:results) {
    cout << "======================================" << endl;
    for (auto& i:result) {
      cout << "bbox:     ";
      for (auto& j:i.bbox)
        cout << j << "    ";
      cout << "score:    " << i.score << endl; 
    }
  }
  return 0;
}

