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

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

#include "vitis/ai/general.hpp"
//
using namespace std;
#include "./process_image.hpp"
#include "./xmodel_result_to_string.hpp"

static std::vector<cv::Mat> read_images(const std::vector<std::string>& files,
                                        size_t batch) {
  std::vector<cv::Mat> images(batch);
  for (auto index = 0u; index < batch; ++index) {
    const auto& file = files[index % files.size()];
    images[index] = cv::imread(file);
    CHECK(!images[index].empty()) << "cannot read image from " << file;
  }
  return images;
}

std::vector<std::string> g_image_files;
std::string g_xmodel_file;
static inline void parse_opt(int argc, char* argv[]) {
  int opt = 0;
  auto usage = [=] {
    std::cout
        << argv[0] << "\n"  //
        << "-m <xmodel_file> : set a xmodel file for testing,  "
           "there must be  a "
           "a py file with sample base file name in the same directory.\n"  //
        << "-h : for help\n"                                                //
        << std::endl;
  };
  while ((opt = getopt(argc, argv, "m:h")) != -1) {
    switch (opt) {
      case 'm':
        g_xmodel_file = optarg;
        break;
      case 'h':
        usage();
        exit(0);
      default:
        std::cerr << "unknown arguments: " << opt << std::endl;
        usage();
        exit(1);
    }
  }

  for (auto i = optind; i < argc; i++) {
    g_image_files.push_back(std::string(argv[i]));
  }

  if (g_image_files.empty()) {
    std::cerr << "no input file" << std::endl;
    exit(1);
  }
  if (g_xmodel_file.empty()) {
    std::cerr << "no input model" << std::endl;
    exit(1);
  }
  return;
}

int main(int argc, char* argv[]) {
  parse_opt(argc, argv);
  auto xmodel = vitis::ai::General::create(g_xmodel_file, true);
  auto images = read_images(g_image_files, xmodel->get_input_batch());
  auto results = xmodel->run(images);
  int c = 0;
  for (const auto& r : results) {
    LOG(INFO) << "xmodel result:\n" << to_string(r);
    c = c + 1;
  }
  c = 0;
  for (const auto& r : results) {
    auto img = process_image(images[c], r);
    if (!img.empty()) {
      auto out_file = std::string("test_xmodel_" + std::to_string(c) + ".jpg");
      cv::imwrite(out_file, img);
    }
    c = c + 1;
  }
  LOG(INFO) << "BYEBYE";
  return 0;
}
