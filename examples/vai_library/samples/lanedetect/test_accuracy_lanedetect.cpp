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
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/lanedetect.hpp>

using namespace cv;
using namespace std;
//DEF_ENV_PARAM_2(XLNX_ROADLINE_ACCURACY_OUTDIR, "", std::string);
extern string g_roadline_acc_outdir;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cout << "Please input a model name as the first param!" << endl;
    cout << "Please input your image path as the second param!" << endl;
    cout << "The third param is a dir path for outputs!" << endl;
  }

  auto roadline = vitis::ai::RoadLine::create(argv[1]);
  if (!roadline) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  std::ifstream fs(argv[2]);
  g_roadline_acc_outdir = argv[3];
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
