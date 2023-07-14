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
#include <string>
#include <vitis/ai/benchmark.hpp>
#include <vitis/ai/centerpoint.hpp>
#include <fstream>
#include <iostream>
std::string path;

static void readfile(std::string& filename, std::vector<float>& data) {
  std::ifstream input_file(filename);
  std::string line;
  while (std::getline(input_file, line)) {
    std::istringstream ss(line);
    float num;
    ss >> num;
    data.push_back(num);
  }
  std::cout << filename << " " << data.size() << std::endl;
}

namespace vitis {
namespace ai {

class CenterpointPerf {
public:
  CenterpointPerf(std::string input_n, std::string model_1, std::string model_2)
      :  input_name(input_n), kernel_name_1(model_1),  kernel_name_2(model_2), det(CenterPoint::create(kernel_name_1, kernel_name_2)) {
    batch_size = get_input_batch();
    all_arrays.resize(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
      std::vector<float> array;
      readfile(input_name, array);
      all_arrays[i] = array;
    }
  }
  int getInputWidth() {return det->getInputWidth();}
  int getInputHeight() {return det->getInputHeight();}
  size_t get_input_batch() {return (size_t)det->get_input_batch();}
  std::vector<std::vector<CenterPointResult>> run(const std::vector<cv::Mat> & image) {
    auto res = det->run(all_arrays);
    return res;
  }

private:
  std::string input_name;
  std::string kernel_name_1;
  std::string kernel_name_2;
  size_t batch_size;
  std::unique_ptr<CenterPoint> det;
  std::vector<std::vector<float>> all_arrays;


};

} //namespace ai
} //namespace vitis


int main(int argc, char *argv[]) {
  std::string input = argv[3];
  std::string model_1 = argv[1];
  std::string model_2 = argv[2];
  return vitis::ai::main_for_performance(argc, argv, [input, model_1, model_2] {
    { return std::unique_ptr<vitis::ai::CenterpointPerf>(new vitis::ai::CenterpointPerf(input, model_1, model_2)); }
  });
}
