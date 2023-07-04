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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/pointpainting.hpp>

using namespace vitis::ai;
using namespace vitis::ai::pointpillars_nus;

std::vector<std::string> get_image_file_name(const std::string &list_name) {
  struct stat file_stat;
  if (stat(list_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << list_name << " state error!" << std::endl;
    exit(-1);
  } 
  auto list = std::ifstream(list_name);
  char line[1024];
  std::vector<std::string> file_names;
  while (list.getline(line, 1024, '\n')) {
    file_names.emplace_back(line);  
  } 
  list.close();
  return file_names; 
}

int main( int argc, char *argv[])
{
  if (argc < 6) {
    //std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0] << " [seg_model] [pointpillars_model_0] [pointpillars_model_1] [segmatation_image_list] [out_path]" << std::endl;
    exit(0);
  }

  std::string seg_model = argv[1];
  std::string model_0 = argv[2];
  std::string model_1 = argv[3];

  seg_model += std::string("_acc");
  model_0 += std::string("_acc");
  model_1 += std::string("_acc");

  auto pointpainting = vitis::ai::PointPainting::create(
          seg_model, model_0, model_1);
  if (!pointpainting) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  auto batch = pointpainting->get_segmentation_batch();

  std::string list_name = argv[4];
  std::string list_path;
  if (list_name.find_last_of('/') != std::string::npos) {
    list_path = list_name.substr(0, list_name.find_last_of('/') +1);
  }

  auto list = get_image_file_name(list_name);

  std::string out_path = argv[5];
  std::string cmd = "mkdir -p " + out_path;
  if (system(cmd.c_str()) == -1) {
    std::cerr << "command: " << cmd << " error!" << std::endl;
    exit(-1);
  }
  
  std::vector<std::string> batch_image_names;
  std::vector<cv::Mat> batch_images;
  for (auto n = 0u; n < list.size(); n+=batch) {
    auto start = n;
    auto end = std::min(n + batch , list.size());
    auto input_num = end - start;
    batch_image_names.clear();
    batch_image_names.resize(input_num);
    batch_images.clear();
    batch_images.resize(input_num);
    for (auto i = 0u; i < input_num; ++i) {
      batch_image_names[i] = list[start + i];
      auto full_name = list_path + batch_image_names[i]; 
      struct stat file_stat;
      if (stat(full_name.c_str(), &file_stat) != 0) {
        std::cerr << "file:" << full_name << " state error!" << std::endl;
        exit(-1);
      }
      batch_images[i] =  cv::imread(full_name);
    }
    auto batch_ret = pointpainting->runSegmentation(batch_images);
    for (auto b = 0u; b < batch_ret.size(); ++b) {
      auto full_name = batch_image_names[b];
      std::string name = full_name;
      if (name.find_last_of('/') != std::string::npos) {
        name = name.substr(name.find_last_of('/') + 1); 
      }
      if (name.find_last_of('.') != std::string::npos) {
        name = name.substr(0, name.find_last_of('.')); 
      }
      //std::string out_name = out_path + "/" + name + ".jpg"; 
      std::string out_name = out_path + "/" + name + ".png"; 
      cv::imwrite(out_name, batch_ret[b]);
    }
  }
  return 0;
}

