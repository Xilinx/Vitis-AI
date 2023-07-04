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
    std::cout << "usage: " << argv[0] << " <model1> <model2> <input_file>" << std::endl;
    exit(0);
  }

  // 1. read float bin
  auto input_file = std::string(argv[3]);
  std::vector<float> input;
  readfile(input_file, input);	

  std::string model_0 = argv[1];
  std::string model_1 = argv[2];
  auto centerpoint = vitis::ai::CenterPoint::create(
          model_0, model_1);
  if (!centerpoint) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }   

  auto result = centerpoint->run(input);
  for (auto& i:result) {
    cout << "bbox:     ";
    for (auto& j:i.bbox)
      cout << j << "    ";
    cout << "score:    " << i.score << endl; 
  }
  return 0;
}

