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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>
#include "vitis/ai/clocs.hpp"

using std::string;
using std::vector;
using namespace vitis::ai;
using namespace vitis::ai::clocs;

DEF_ENV_PARAM(DEBUG_CLOCS_ACC, "0");

static void read_points_file(const std::string& points_file_name,
                             std::vector<float>& points) {
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  }
  auto file_size = file_stat.st_size;
  // LOG(INFO) << "input file:" << points_file_name << " size:" << file_size;
  // points_info.points.resize(file_size / 4);
  points.resize(file_size / 4);
  // CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char
  // *>(points_info.points.data()), file_size).good());
  CHECK(std::ifstream(points_file_name)
            .read(reinterpret_cast<char*>(points.data()), file_size)
            .good());
}

void LoadImageNames(std::string const& filename,
                    std::vector<std::string>& images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE* fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }

  fclose(fp);
}

int main(int argc, char* argv[]) {
  if (argc < 8) {
    // std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0]
              << " [yolo_model] [pointpillars_model_0] [pointpillars_model_1] "
                 "[fusion_model] [input_list] [dataset_path] [output_path] "
              << std::endl;
    exit(0);
  }

  std::string yolo_model_name = argv[1];
  std::string pp_model_0 = argv[2];
  std::string pp_model_1 = argv[3];
  std::string fusion_model_name = argv[4];

  std::string input_file_list = argv[5];
  std::string base_path = argv[6];
  std::string output_path = argv[7];

  std::string cmd = "mkdir -p " + output_path;
  if (system(cmd.c_str()) == -1) {
    std::cerr << "command: " << cmd << " error!" << std::endl;
    exit(-1);
  }

  // std::string path_2d_result = base_path + "/../yolox-kitti-det2d/";
  // if (argc >= 9) {
  //  path_2d_result = argv[8];
  //}

  vector<string> names;
  LoadImageNames(input_file_list.c_str(), names);

  // auto yolo = vitis::ai::YOLOvX::create(yolo_model_name, true);
  auto clocs = vitis::ai::Clocs::create(yolo_model_name, pp_model_0, pp_model_1,
                                        fusion_model_name, true);
  if (!clocs) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  size_t batch = clocs->get_input_batch();

  auto group = 0u;
  if (names.size() % batch) {
    group = names.size() / batch + 1;
  } else {
    group = names.size() / batch;
  }

  for (auto i = 0u; i < group; ++i) {
    std::string input_path;
    size_t input_num = names.size() - i * batch;
    if (input_num > batch) {
      input_num = batch;
    }

    vector<ClocsInfo> batch_clocs_info(input_num);
    // vector<vector<float>> batch_2d_result(input_num);
    for (auto j = 0u; j < input_num; ++j) {
      std::string idx_str = names[i * batch + j];
      LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_ACC)) << "load " << idx_str;
      auto points_file_name =
          base_path + "/velodyne_reduced/" + idx_str + ".bin";
      auto image_file_name = base_path + "/image_2/" + idx_str + ".png";

      // 1. read float bin
      // input_file : 000001.bin
      auto input_file = std::string(points_file_name);
      std::vector<float> input;
      read_points_file(input_file, input);

      // 2. read image
      // image: 000001.png
      auto input_image = cv::imread(image_file_name);

      // 3. read 2d result
      // read_2d_result_txt(path_2d_result + "/" + idx_str + ".txt",
      //                   batch_2d_result[j]);

      auto& clocs_info = batch_clocs_info[j];
      std::vector<float> p2;
      std::vector<float> trv2c;
      std::vector<float> rect;

      read_points_file(base_path + "/P2/" + idx_str + "_P2.bin", p2);
      read_points_file(base_path + "/Trv2c/" + idx_str + "_Trv2c.bin", trv2c);
      read_points_file(base_path + "/rect/" + idx_str + "_rect.bin", rect);
      clocs_info.calib_P2.assign(p2.begin(), p2.end());
      clocs_info.calib_Trv2c.assign(trv2c.begin(), trv2c.end());
      clocs_info.calib_rect.assign(rect.begin(), rect.end());
      clocs_info.image = input_image;
      clocs_info.points = input;
    }

    auto batch_ret = clocs->run(batch_clocs_info);
    for (auto j = 0u; j < input_num; ++j) {
      std::string output_file_name =
          output_path + "/" + names[i * batch + j] + ".txt";
      LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_ACC))
          << "output: " << output_file_name;
      std::ofstream out(output_file_name);
      auto& ret = batch_ret[j];
      for (auto& r : ret.bboxes) {
        out << r.label << " ";
        for (auto& b : r.bbox) {
          out << b << " ";
        }
        out << r.score << std::endl;
      }
      out.close();
    }
  }

  return 0;
}

