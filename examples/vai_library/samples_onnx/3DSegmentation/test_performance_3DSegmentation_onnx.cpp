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

#include <vitis/ai/benchmark.hpp>
#include "3DSegmentation_onnx.hpp"

using namespace ns_OnnxSegmentation3D;

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

class Segmentation3DPerf {
 public:
  Segmentation3DPerf(std::string model)
      : kernel_name(model), det(OnnxSegmentation3D::create(kernel_name)) {
    batch_size = get_input_batch();
    all_arrays.resize(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
      std::vector<std::vector<float>> arrays(4);
      std::string scan_x = path + "scan_x.txt";
      std::string scan_y = path + "scan_y.txt";
      std::string scan_z = path + "scan_z.txt";
      std::string remission = path + "scan_remission.txt";
      readfile(scan_x, arrays[0]);
      readfile(scan_y, arrays[1]);
      readfile(scan_z, arrays[2]);
      readfile(remission, arrays[3]);
      all_arrays[i] = arrays;
    }
  }
  int getInputWidth() { return det->getInputWidth(); }
  int getInputHeight() { return det->getInputHeight(); }
  size_t get_input_batch() { return (size_t)det->get_input_batch(); }
  std::vector<OnnxSegmentation3DResult> run(const std::vector<cv::Mat>& image) {
    return det->run(all_arrays);
  }

 private:
  std::string kernel_name;
  size_t batch_size;
  std::unique_ptr<OnnxSegmentation3D> det;
  std::vector<std::vector<std::vector<float>>> all_arrays;
};

int main(int argc, char* argv[]) {
  auto model = std::string(argv[1]);
  path = "./salsanext_input/";
  return vitis::ai::main_for_performance(argc, argv, [model] {
    return std::unique_ptr<Segmentation3DPerf>(new Segmentation3DPerf(model));
  });
}

