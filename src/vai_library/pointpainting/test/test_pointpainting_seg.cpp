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
#include <fstream>
#include <sys/stat.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/pointpainting.hpp>
#include "../src/utils.hpp"

using namespace vitis::ai::pointpillars_nus;
using namespace vitis::ai::pointpainting;

int main( int argc, char *argv[]) {
  if (argc < 5) {
    //std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0] << " [segmentation_model] [pointpillars_model0] [pointpillars_model1] [image_name]" << std::endl;
    exit(0);
  }

  std::string image_name = argv[4];
  struct stat file_stat;
  if (stat(image_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << image_name << " state error!" << std::endl;
    exit(-1);
  }


  std::string seg_model = argv[1];
  std::string model_0 = argv[2];
  std::string model_1 = argv[3];

  auto pointpainting = vitis::ai::PointPainting::create(
          seg_model, model_0, model_1);
  auto batch = pointpainting->get_segmentation_batch();
  std::vector<cv::Mat> images(batch);
  for (auto i = 0u; i < batch; ++i) {
    images[i] = cv::imread(image_name); 
    std::cout << "image[" << i << "] size:" << images[i].cols << " * " << images[i].rows << std::endl;
  }

  auto result = pointpainting->runSegmentation(images);
  for (auto i = 0u; i < batch; ++i) {
    auto name = image_name;
    if (name.find_last_of('/') != std::string::npos) {
      name = name.substr(name.find_last_of('/') + 1);
    }
    name = name.substr(0, name.find_last_of('.')) + "_" + std::to_string(i);
    auto png_name = name  + ".png";
    std::cout << "save name:" << name << std::endl;
    cv::imwrite(png_name, result[i]);
    auto out = std::ofstream(name + ".txt");
    if (out.is_open()) {
      for (auto m = 0; m < result[i].rows; m++) {
        for (auto n = 0; n < result[i].cols; n++) {
          out << (int)result[i].at<uchar>(m, n) << " "; 
        }
        out << "\n";
      }
      out.close();
    }
  }
  return 0;
}

