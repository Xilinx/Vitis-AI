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
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/vehicleclassification.hpp>

using namespace std;
using namespace cv;

static std::string get_single_name(const std::string& line) {
  std::size_t found = line.rfind('/');
  if (found != std::string::npos) {
    return line.substr(found + 1);
  }
  return line;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    cout << "Please input a model name as the first param!" << endl;
    cout << "Please input a images list file as the second param!" << endl;
    cout << "The third param is a txt to store results!" << endl;
  }
  string model = argv[1] + string("_acc");
  string images_list_file = argv[2];
  std::ofstream out_fs(argv[3], std::ofstream::out);
  auto det = vitis::ai::VehicleClassification::create(model, true);
  if (!det) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }
  std::ifstream fs(images_list_file);
  std::string line;
  std::string single_name;
  while (getline(fs, line)) {
    auto image = cv::imread(line);
    if (image.empty()) {
      std::cerr << "cannot read image: " << line;
      continue;
    }
    single_name = get_single_name(line);
    auto res = det->run(image);
    for (size_t j = 0; j < res.scores.size(); ++j) {
      int index = res.scores[j].index;
      out_fs << single_name << " " << index << endl;
    }
  }
  out_fs.close();
}
