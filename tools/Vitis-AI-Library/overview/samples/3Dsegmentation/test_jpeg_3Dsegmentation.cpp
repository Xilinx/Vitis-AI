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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/3Dsegmentation.hpp>
#include <vitis/ai/profiling.hpp>
#include <fstream>

using namespace std;
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

template<typename T> 
void writefile(string& filename, vector<T>& data) {
  ofstream output_file(filename);
  for(size_t i = 0; i < data.size(); i++) 
    output_file << data[i] << endl;
}


using namespace vitis::ai;
int main(int argc, char *argv[]) {
  // bool preprocess = !(getenv("PRE") != nullptr);
  auto det = vitis::ai::Segmentation3D::create(argv[1], false);
  int width = det->getInputWidth();
  int height = det->getInputHeight();
  std::cout << "width " << width << " "    //
            << "height " << height << " "  //
            << std::endl;
  vector<vector<float>> arrays(4);
  string scan_x = argv[2];
  string scan_y = argv[3];
  string scan_z = argv[4];
  string remission = argv[5];
  readfile(scan_x, arrays[0]);
  readfile(scan_y, arrays[1]);
  readfile(scan_z, arrays[2]);
  readfile(remission, arrays[3]);

  vitis::ai::Segmentation3DResult res = det->run(arrays);
  string result_name = "result.txt";
  writefile(result_name, res.array);

  return 0;
}
