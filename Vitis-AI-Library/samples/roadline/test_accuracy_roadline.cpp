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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <xilinx/ai/roadline.hpp>

using namespace cv;
using namespace std;

#define DATASERESULT_FILE "datase.txt"
#define SEEDXRESULT_FILE "seedx.txt"
#define SEEDYRESULT_FILE "seedy.txt"

int main(int argc, char *argv[]) {
  // argv[1] is the file list name
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " image_list_file" << std::endl;
    return -1;
  }

  auto roadline = xilinx::ai::RoadLine::create("vpgnet_pruned_0_99");

  std::ifstream fs(argv[1]);
  std::string line;
  std::string single_name;
  while (getline(fs, line)) {
    // LOG(INFO) << "line = [" << line << "]";
    auto image = cv::imread(line);
    if (image.empty()) {
      cout << "cannot read image: " << line;
      continue;
    }
    auto mt_results = roadline->run(image);
  }
  fs.close();
  return 0;
}
