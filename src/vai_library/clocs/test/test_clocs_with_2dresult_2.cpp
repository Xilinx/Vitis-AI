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

static std::vector<std::string> split(const char* s, const char* delim) {
  std::vector<std::string> result;
  if (s && strlen(s)) {
    int len = strlen(s);
    char* src = new char[len + 1];
    strcpy(src, s);
    src[len] = '\0';
    char* tokenptr = strtok(src, delim);
    while (tokenptr != NULL) {
      std::string tk = tokenptr;
      result.emplace_back(tk);
      tokenptr = strtok(NULL, delim);
    }
    delete[] src;
  }
  return result;
}

static void read_2d_result_txt(const std::string& file_name,
                               std::vector<float>& result) {
  std::ifstream in(file_name);
  if (!in) {
    std::cerr << "error opening:" << file_name << std::endl;
    exit(0);
  }

  std::vector<char> buffer(1024);
  std::string tok = " ";

  while (!in.eof()) {
    in.getline(buffer.data(), buffer.size());
    // std::cout << "buffer: " << buffer.data() << std::endl;
    std::vector<float> bbox(5);
    auto split_result = split(buffer.data(), tok.c_str());
    if (split_result.size() < 16) {
      continue;
    }
    // std::cout << "bbox:";
    for (auto i = 0; i < 4; ++i) {
      bbox[i] = std::atof(split_result[4 + i].data());
      result.push_back(bbox[i]);
      // std::cout << bbox[i] << " ";
    }
    bbox[4] = std::atof(split_result[15].data());
    result.push_back(bbox[4]);
    // std::cout << bbox[4] << std::endl;
  }
  in.close();
  // exit(0);
}

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
  std::string path_2d_result = "./detect_2d_result/";

  // auto yolo = vitis::ai::YOLOvX::create(yolo_model_name, true);
  auto clocs = vitis::ai::Clocs::create(yolo_model_name, pp_model_0, pp_model_1,
                                        fusion_model_name, true);

  int batch = clocs->get_input_batch();
  int input_num = argc - 5;

  input_num = input_num > batch ? batch : input_num;
  vector<ClocsInfo> batch_clocs_info(input_num);
  vector<vector<float>> batch_2d_result(input_num);
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
    //
    // 3. read 2d result
    read_2d_result_txt(path_2d_result + idx_str + ".txt", batch_2d_result[i]);

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

  auto batch_ret = clocs->run(batch_2d_result, batch_clocs_info);

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

