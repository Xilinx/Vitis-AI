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

#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>
#include <vitis/ai/profiling.hpp>
#include "../src/multi_frame_fusion.hpp"

using namespace vitis::ai::pointpainting;
constexpr int DIM = 16;

void read_s2lr_from_line(const std::string &line, PointsInfo &points_info) {
  auto s = line;
  auto cnt = 0u;
  while(cnt < points_info.cam_info.s2l_r.size()) {
    auto f = std::atof(s.c_str()); 
    points_info.cam_info.s2l_r[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}

void read_s2lt_from_line(const std::string &line, PointsInfo &points_info) {
  auto s = line;
  auto cnt = 0u;
  while(cnt < points_info.cam_info.s2l_t.size()) {
    auto f = std::atof(s.c_str()); 
    points_info.cam_info.s2l_t[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}

void read_points_file(const std::string &points_file_name, PointsInfo &points_info) {
  points_info.dim = DIM; 
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  } 
  auto file_size = file_stat.st_size; 
  LOG(INFO) << "input file:" << points_file_name << " size:" << file_size;
  points_info.points.resize(file_size / 4);
  CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char *>(points_info.points.data()), file_size).good()); 
}

void read_inno_file(const std::string &file_name, 
                    PointsInfo &points_info,
                    std::vector<PointsInfo> &sweeps) {
  auto anno_file = std::ifstream(file_name);
  if (!anno_file) {
    std::cerr << "open:" << file_name << " fail!" << std::endl;
    exit(-1);
  }
  char line[1024]; 
  while(anno_file.getline(line, 1024, '\n')) {
    auto len = std::strlen(line);
    if (len == 0) {
      continue;
    }

    if (std::strncmp(line, "lidar_path:", 11) == 0) {
      auto lidar_path = std::string(line).substr(11); 
      //std::cout << "lidar_path:" << lidar_path << std::endl;
      read_points_file(lidar_path, points_info);
    } else {
      break;
    }

    // read timestamp
    if (anno_file.getline(line, 1024, '\n') && std::strncmp(line, "timestamp:", 10) == 0) {
      auto timestamp = std::atoll(std::string(line).substr(10).c_str());
      //std::cout << "timestamp:" << timestamp<< std::endl;
      points_info.cam_info.timestamp = timestamp;
    } else {
      break;
    }

    // read sweeps 
    if (anno_file.getline(line, 1024, '\n') && std::strncmp(line, "sweeps:", 7) == 0) {
      auto s = std::string(line + 7);
      int num = std::atoi(s.c_str());
      //std::cout << "sweep num:" << num << std::endl;
      if (num == 0) {
        continue;
      } 
      int cnt = 0;
      sweeps.clear();
      sweeps.resize(num);
      while(cnt < num) {
        // read data path
        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "data_path:", 10) == 0) {
          auto data_path = std::string(line + 1).substr(10);
          //std::cout << "data_path:" << data_path << std::endl;
          read_points_file(data_path, sweeps[cnt]);
        } else {
          break;
        }

        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "timestamp:", 10) == 0) {
          auto timestamp = std::atoll(std::string(line + 1).substr(10).c_str());
          //std::cout << "timestamp:" << timestamp<< std::endl;
          sweeps[cnt].cam_info.timestamp = timestamp; 
        } else {
          break;
        }

        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "sensor2lidar_rotation:", 22) == 0) {
           auto l = std::string(line + 1).substr(22);
           //std::cout << "s2lr:" << l << std::endl;
           read_s2lr_from_line(l, sweeps[cnt]); 
        } else {
          break;
        }

        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "sensor2lidar_translation:", 25) == 0) {
           auto l = std::string(line + 1).substr(25);
           //std::cout << "s2lt:" << l << std::endl;
           read_s2lt_from_line(l, sweeps[cnt]); 
        } else {
          break;
        }
        cnt++;
      }
    } else {
      break;
    }
    break;
  }
}

void print_points_info(const PointsInfo &points_info) {
  std::cout << "points info: " << std::endl; 
  std::cout << "  cam_info:" << std::endl;
  std::cout << "    timestamp:" << points_info.cam_info.timestamp << std::endl;
  std::cout << "    s2l_t:";
  for (auto i = 0u; i < points_info.cam_info.s2l_t.size(); ++i) {
    std::cout << points_info.cam_info.s2l_t[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "    s2l_r:";

  for (auto i = 0u; i < points_info.cam_info.s2l_r.size(); ++i) {
    std::cout << points_info.cam_info.s2l_r[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "  dim:" << points_info.dim << std::endl;
  std::cout << "  points size:" << points_info.points.size() << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "usage:" << argv[0] << " [anno_file_name]" << std::endl;
    exit(0);
  }

  std::string anno_file_name = argv[1];
  struct stat file_stat;
  if (stat(anno_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << anno_file_name << " state error!" << std::endl;
    exit(-1);
  }
  PointsInfo points_info;  
  std::vector<PointsInfo> sweeps;
  read_inno_file(anno_file_name, points_info, 5, sweeps, 16); 
  //print_points_info(points_info);
  points_info = remove_useless_dim(points_info, 3);
  std::deque<PointsInfo> sweeps_deque;
  std::vector<float> range{-50.0, -50.0, -5.0, 50.0, 50.0, 3.0};
  for (auto i = 0u; i < sweeps.size(); ++i) {
    //print_points_info(sweeps[i]);
    sweeps[i] = remove_useless_dim(sweeps[i], 3);
    //sweeps[i] = transform_points(sweeps[i]);
    //sweeps[i] = points_filter(sweeps[i], range);
    sweeps_deque.push_back(sweeps[i]);
  }
  for (auto i = 0u; i < sweeps.size(); ++i) {
    LOG(INFO) << "sweeps[" << i << "] size:" << sweeps[i].points.size();
    int num = sweeps[i].points.size() / sweeps[i].dim;
    LOG(INFO) << "sweeps[" << i << "] shape:" << num << " * " << sweeps[i].dim; 
  }
  auto result = vitis::ai::pointpainting::multi_frame_fusion(
                points_info, sweeps_deque);
  LOG(INFO) << "result size:" << result.size();
  auto valid_dim = points_info.dim;
  LOG(INFO) << "result shape:" << result.size() / valid_dim << " * " << valid_dim;
  result = points_filter(result, valid_dim, range);
  LOG(INFO) << "result size:" << result.size();
  LOG(INFO) << "result shape:" << result.size() / valid_dim << " * " << valid_dim;
  debug_vector(result, "result", valid_dim);
  save_vector(result, "multi_frame_fusion_result", valid_dim);
  
  return 0;
}
