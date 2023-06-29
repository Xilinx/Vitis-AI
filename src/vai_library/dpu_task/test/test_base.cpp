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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "vitis/ai/configurable_dpu_task.hpp"
using namespace std;

int main(int argc, char *argv[]) {
  auto model_name = std::string(argv[1]);
  auto image_file = std::string(argv[2]);
  auto dpu_task = vitis::ai::ConfigurableDpuTask::create(model_name, true);
  if (!dpu_task) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  auto image = cv::imread(image_file);
  if (image.empty()) {
    cerr << "cannot load " << image_file << endl;
    abort();
  }

  dpu_task->setInputImageBGR(image);
  dpu_task->run(0);

  std::cout << model_name << " run success" << std::endl;
  return 0;
}
