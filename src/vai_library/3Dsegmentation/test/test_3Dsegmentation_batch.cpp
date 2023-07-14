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
void writefilebin(string& filename, vector<T>& data) {
  ofstream output_file(filename, ios::binary);
  output_file.write(reinterpret_cast<char *>(data.data()), sizeof(T) * data.size());
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
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  size_t batch_size = det->get_input_batch();
  vector<vector<vector<float>>> all_arrays(batch_size);
  vector<string> all_path;
  for(int i = 2; i < argc; i++) {
    string name = argv[i];
    all_path.push_back(name + "/");
  }
  for (size_t i = 0; i < batch_size; i++) {
    string path = all_path[i % all_path.size()];
    vector<vector<float>> arrays(4);
    string scan_x = path + "scan_x.txt";
    string scan_y = path + "scan_y.txt";
    string scan_z = path + "scan_z.txt";
    string remission = path + "scan_remission.txt";
    readfile(scan_x, arrays[0]);
    readfile(scan_y, arrays[1]);
    readfile(scan_z, arrays[2]);
    readfile(remission, arrays[3]);
    all_arrays[i] = arrays;
  }
  
  std::vector<vitis::ai::Segmentation3DResult> res = det->run(all_arrays);
  for(size_t i = 0; i < res.size(); i++) {
    string result_name = "batch_result_" + to_string(i) + ".txt";
    writefile(result_name, res[i].array);
    string result_bin_name = "batch_result_" + to_string(i) + ".bin";
    writefilebin(result_bin_name, res[i].array);
  }

  return 0;
}
