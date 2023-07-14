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
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/3Dsegmentation.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

DEF_ENV_PARAM(SAMPLES_ENABLE_BATCH, "1");
DEF_ENV_PARAM(SAMPLES_BATCH_NUM, "0");

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


template <typename T>
void writefile(string& filename, vector<T>& data) {
  ofstream output_file(filename);
  for (size_t i = 0; i < data.size(); i++) output_file << data[i] << endl;
}

using namespace vitis::ai;
int main(int argc, char* argv[]) {
  if (argc < 6) {
    cerr << "need at least 4 files" << endl;
    return -1;
  }
  // bool preprocess = !(getenv("PRE") != nullptr);
  auto det = vitis::ai::Segmentation3D::create(argv[1], false);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  int width = det->getInputWidth();
  int height = det->getInputHeight();
  std::cout << "width " << width << " "    //
            << "height " << height << " "  //
            << std::endl;
  if (ENV_PARAM(SAMPLES_ENABLE_BATCH)) {
    std::vector<std::string> input_x_files;
    std::vector<std::string> input_y_files;
    std::vector<std::string> input_z_files;
    std::vector<std::string> input_remission_files;

    for (int i = 2; i < argc - 3; i = i + 4) {
      input_x_files.push_back(std::string(argv[i]));
      input_y_files.push_back(std::string(argv[i + 1]));
      input_z_files.push_back(std::string(argv[i + 2]));
      input_remission_files.push_back(std::string(argv[i + 3]));
    }

    if (input_x_files.size() != input_y_files.size() ||
        input_y_files.size() != input_z_files.size() ||
        input_z_files.size() != input_remission_files.size()) {
      std::cerr << "input files should be pair" << std::endl;
      exit(1);
    }

    if (input_x_files.empty()) {
      cerr << "can't load files! " << endl;
      return -1;
    }

    auto batch = det->get_input_batch();
    if (ENV_PARAM(SAMPLES_BATCH_NUM)) {
      unsigned int batch_set = ENV_PARAM(SAMPLES_BATCH_NUM);
      assert(batch_set <= batch);
      batch = batch_set;
    }

    vector<vector<vector<float>>> batch_arrays(batch);
    for (auto index = 0u; index < batch; ++index) {
      vector<vector<float>> arrays(4);
      readfile(input_x_files[index % input_x_files.size()], arrays[0]);
      readfile(input_y_files[index % input_y_files.size()], arrays[1]);
      readfile(input_z_files[index % input_z_files.size()], arrays[2]);
      readfile(input_remission_files[index % input_remission_files.size()],
               arrays[3]);
      batch_arrays[index] = arrays;
    }

    auto results = det->run(batch_arrays);
    assert(results.size() == batch);
    string model_name;
    model_name=argv[1];
    for (auto i = 0u; i < results.size(); i++) {
      LOG(INFO) << "batch: " << i;
      string result_name = model_name + "_batch_" + std::to_string(i)  +".bin";
      writefilebin(result_name, results[i].array);
      std::cout << std::endl;
    }
  } else {
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
    string result_name = "result.bin";
    writefilebin(result_name, res.array);
  }
  return 0;
}
