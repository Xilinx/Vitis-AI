/*
 * Copyright 2019 xilinx Inc.
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
#include <google/protobuf/text_format.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "vitis/ai/general.hpp"
using namespace std;

std::string g_model_name = "resnet50";
std::string g_image_file = "";

static void usage() {
  std::cout << "usage: test_general <model_name> <img_file> " << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    usage();
    return 1;
  }
  g_model_name = argv[1];
  g_image_file = argv[2];
  auto image = cv::imread(g_image_file);
  if (image.empty()) {
    std::cerr << "cannot load " << g_image_file << std::endl;
    abort();
  }
  auto model = vitis::ai::General::create(g_model_name, true);
  if (model) {
    auto result = model->run(image);
    cerr << "result = " << result.DebugString() << endl;
  } else {
    cerr << "no such model, ls -l /usr/share/vitis-ai-library to see available "
            "models"
         << endl;
  }
  return 0;
}
