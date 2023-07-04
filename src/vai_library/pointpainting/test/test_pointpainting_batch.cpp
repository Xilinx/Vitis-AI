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
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/pointpainting.hpp>
#include "../src/utils.hpp"

using namespace vitis::ai;
using namespace vitis::ai::pointpillars_nus;
using namespace vitis::ai::pointpainting;

int main( int argc, char *argv[]) {
  if (argc < 5) {
    //std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0] << " [segmentation_model] [pointpillars_model0] [pointpillars_model1] [anno_file_name]" << std::endl;
    exit(0);
  }

  int input_num = argc - 4;
  std::vector<std::string> anno_file_names(input_num);
  std::vector<PointsInfo> points_infos(input_num);
  std::vector<std::vector<cv::Mat>> images(input_num);
  for (auto i = 0; i < input_num; ++i) {
    anno_file_names[i] = argv[4 + i];
    struct stat file_stat;
    if (stat(anno_file_names[i].c_str(), &file_stat) != 0) {
      std::cerr << "file:" << anno_file_names[i] << " state error!" << std::endl;
      exit(-1);
    }
    read_inno_file_pointpainting(anno_file_names[i], points_infos[i], 5, points_infos[i].sweep_infos, 16, images[i]); 
  }

  //std::string seg_model = "pointpainting_segmentation";
  std::string seg_model = argv[1];
  //std::string model_0 = "pointpainting_pointpillars_0";
  std::string model_0 = argv[2];
  //std::string model_1 = "pointpainting_pointpillars_1";
  std::string model_1 = argv[3];

  auto pointpainting = vitis::ai::PointPainting::create(
          seg_model, model_0, model_1);
  if (!pointpainting) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  //auto batch = pointpainting->get_pointpillars_batch();
  //std::vector<PointsInfo> batch_points_info(batch, points_info);

  auto batch_ret = pointpainting->run(images, points_infos);
  //LOG(INFO) << "input width:" << ret.width
  //          << " input height: " << ret.height;
  for (auto b = 0u; b < batch_ret.size(); ++b) {
    std::cout << "batch : " << b << std::endl; 
    auto &ret = batch_ret[b];
    for (auto c = 0u; c < 10; ++c) { 
      for (auto i = 0u; i < ret.bboxes.size(); ++i) {
        if (ret.bboxes[i].label != c) {
          continue;
        }
        std::cout << "label: " << ret.bboxes[i].label;
        std::cout << " bbox: ";
        for (auto j = 0u; j < ret.bboxes[i].bbox.size(); ++j) {
          std::cout << ret.bboxes[i].bbox[j] << " ";
        }
        std::cout << "score: " << ret.bboxes[i].score;
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }

  return 0;
}

