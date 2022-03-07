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

#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>
#include "vitis/ai/clocs.hpp"
#include "vitis/ai/yolovx.hpp"

using std::vector;
using namespace vitis::ai;
using namespace vitis::ai::clocs;

static void read_points_file(const std::string& points_file_name,
                             std::vector<float>& points) {
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  }
  auto file_size = file_stat.st_size;
  LOG(INFO) << "input file:" << points_file_name << " size:" << file_size;
  // points_info.points.resize(file_size / 4);
  points.resize(file_size / 4);
  // CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char
  // *>(points_info.points.data()), file_size).good());
  CHECK(std::ifstream(points_file_name)
            .read(reinterpret_cast<char*>(points.data()), file_size)
            .good());
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    // std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0]
              << " [yolo_model] [pointpillars_model_0] [pointpillars_model_1] "
                 "[fusion_model] [input_file1] ..."
              << std::endl;
    exit(0);
  }

  std::string yolo_model_name = argv[1];
  std::string pp_model_0 = argv[2];
  std::string pp_model_1 = argv[3];
  std::string fusion_model_name = argv[4];

  auto yolo = vitis::ai::YOLOvX::create(yolo_model_name, true);
  auto clocs = vitis::ai::Clocs::create(yolo_model_name, pp_model_0, pp_model_1,
                                        fusion_model_name, true);

  int batch = clocs->get_input_batch();
  int input_num = argc - 5;

  input_num = input_num > batch ? batch : input_num;
  vector<ClocsInfo> batch_clocs_info(input_num);
  for (auto i = 0; i < input_num; ++i) {
    auto input_file_name = std::string(argv[i + 5]);
    std::string input_path;
    auto pos = input_file_name.find_last_of('/');
    std::string idx_str = input_file_name;
    if (pos != std::string::npos) {
      input_path = input_file_name.substr(0, pos + 1);
      idx_str = input_file_name.substr(pos + 1);
    }
    idx_str = idx_str.substr(0, idx_str.find_last_of('.'));

    auto points_file_name = input_path + idx_str + ".bin";
    auto image_file_name = input_path + idx_str + ".png";

    // 1. read float bin
    // input_file : 000001.bin
    auto input_file = std::string(points_file_name);
    std::vector<float> input;
    read_points_file(input_file, input);
    // int last_dim = 4;
    // int points_num = input.size() / last_dim;

    // std::cout << "input shape: " << last_dim << " * " << points_num
    //          << std::endl;

    // 2. read image
    // image: 000001.png
    auto input_image = cv::imread(image_file_name);
    // std::cout << "input_image rows:" << input_image.rows
    //          << ", cols:" << input_image.cols << std::endl;

    auto& clocs_info = batch_clocs_info[i];
    std::vector<float> p2;
    std::vector<float> trv2c;
    std::vector<float> rect;

    read_points_file(input_path + idx_str + "_P2.bin", p2);
    read_points_file(input_path + idx_str + "_Trv2c.bin", trv2c);
    read_points_file(input_path + idx_str + "_rect.bin", rect);
    clocs_info.calib_P2.assign(p2.begin(), p2.end());
    clocs_info.calib_Trv2c.assign(trv2c.begin(), trv2c.end());
    clocs_info.calib_rect.assign(rect.begin(), rect.end());
    clocs_info.image = input_image;
    clocs_info.points = input;
  }

  std::vector<cv::Mat> yolo_input(input_num);
  for (auto i = 0; i < input_num; ++i) {
    yolo_input[i] = batch_clocs_info[i].image;
  }

  auto yolo_ret = yolo->run(yolo_input);
  std::vector<std::vector<float>> yolo_result_for_clocs(yolo_ret.size());

  for (auto i = 0u; i < yolo_ret.size(); ++i) {
    yolo_result_for_clocs[i].resize(yolo_ret[i].bboxes.size() * 5);
    for (auto j = 0u; j < yolo_ret[i].bboxes.size(); ++j) {
      yolo_result_for_clocs[i][j * 5 + 0] = yolo_ret[i].bboxes[j].box[0];
      yolo_result_for_clocs[i][j * 5 + 1] = yolo_ret[i].bboxes[j].box[1];
      yolo_result_for_clocs[i][j * 5 + 2] = yolo_ret[i].bboxes[j].box[2];
      yolo_result_for_clocs[i][j * 5 + 3] = yolo_ret[i].bboxes[j].box[3];
      yolo_result_for_clocs[i][j * 5 + 4] = yolo_ret[i].bboxes[j].score;
    }
  }
  auto batch_ret = clocs->run(yolo_result_for_clocs, batch_clocs_info);

  LOG(INFO) << "ret batch:" << batch_ret.size();
  for (auto b = 0u; b < batch_ret.size(); ++b) {
    std::cout << "batch:" << b << std::endl;
    auto& ret = batch_ret[b];
    auto size = ret.bboxes.size();
    for (auto i = 0u; i < size; ++i) {
      auto r = ret.bboxes[i];
      std::cout << "label:" << r.label << " ";
      std::cout << "bbox:"
                << " ";
      for (auto j = 0u; j < r.bbox.size(); ++j) {
        std::cout << r.bbox[j] << " ";
      }
      std::cout << "score:" << r.score;
      std::cout << std::endl;
    }
  }
  return 0;
}

