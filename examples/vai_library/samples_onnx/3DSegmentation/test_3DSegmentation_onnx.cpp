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

#include <fstream>
#include <sstream>
#include "3DSegmentation_onnx.hpp"

using namespace ns_OnnxSegmentation3D;

void readfile(string& filename, vector<float>& data) {
  ifstream input_file(filename);
  std::string line;
  while (std::getline(input_file, line)) {
    istringstream ss(line);
    float num;
    ss >> num;
    data.push_back(num);
  }
  cout << filename << " " << data.size() << endl;
}

template <typename T>
void writefilebin1(const string& filename, vector<T>& data) {
  ofstream output_file(filename, ios::binary);
  output_file.write(reinterpret_cast<char*>(data.data()),
                    sizeof(T) * data.size());
}

template <typename T>
void writefile1(const string& filename, vector<T>& data) {
  ofstream output_file(filename);
  for (size_t i = 0; i < data.size(); i++) output_file << data[i] << endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << "  < dir 4 data_files in> " << std::endl;
    abort();
  }
  auto model_name = std::string(argv[1]);

  auto det = OnnxSegmentation3D::create(model_name);

  int width = det->getInputWidth();
  int height = det->getInputHeight();
  auto batch = det->get_input_batch();
  std::cout << "width " << width << " "    //
            << "height " << height << " "  //
            << std::endl;

  V3F v3f;

  V2F arrays(4);
  string name = argv[2];
  auto path = name + "/";
  string scan_x = path + "scan_x.txt";
  string scan_y = path + "scan_y.txt";
  string scan_z = path + "scan_z.txt";
  string remission = path + "scan_remission.txt";
  cout << "load file " << endl;
  readfile(scan_x, arrays[0]);
  readfile(scan_y, arrays[1]);
  readfile(scan_z, arrays[2]);
  readfile(remission, arrays[3]);

  for (auto i = 0u; i < batch; i++) {
    v3f.push_back(arrays);
  }

  cout << "start running " << endl;
  std::vector<OnnxSegmentation3DResult> res = det->run(v3f);
  for (int i = 0; i < (int)res.size(); i++) {
    std::cout << "batch " << i << std::endl;
    std::stringstream ss;
    ss << "result_" << i << ".txt";
    writefile1(ss.str(), res[i].array);
    std::cout << "write " << ss.str() << std::endl;
    ss.str("");
    ss << "result_" << i << ".bin";
    writefilebin1(ss.str(), res[i].array);
    std::cout << "write " << ss.str() << std::endl;
  }
  return 0;
}

