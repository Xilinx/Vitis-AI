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

#include "./utils.hpp"
#include <iostream>
#include <fstream>
#include <vitis/ai/profiling.hpp>
#include <sys/stat.h>

namespace vitis { namespace ai {
namespace pointpillars_nus {

static void read_s2lr_from_line(const std::string &line, SweepInfo &sweep_info) {
  auto s = line;
  auto cnt = 0u;
  while(cnt < sweep_info.cam_info.s2l_r.size()) {
    auto f = std::atof(s.c_str()); 
    sweep_info.cam_info.s2l_r[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}

static void read_s2lt_from_line(const std::string &line, SweepInfo &sweep_info) {
  auto s = line;
  auto cnt = 0u;
  while(cnt < sweep_info.cam_info.s2l_t.size()) {
    auto f = std::atof(s.c_str()); 
    sweep_info.cam_info.s2l_t[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}

//void read_points_file(const std::string &points_file_name, PointsInfo &points_info) {
static void read_points_file(const std::string &points_file_name, std::vector<float> &points) {
  //int DIM = 5; 
  //points_info.dim = DIM; 
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  } 
  auto file_size = file_stat.st_size; 
  //LOG(INFO) << "input file:" << points_file_name << " size:" << file_size;
  //points_info.points.resize(file_size / 4);
  points.resize(file_size / 4);
  //CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char *>(points_info.points.data()), file_size).good()); 
  CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char *>(points.data()), file_size).good()); 
}

static void read_sweeps(std::ifstream &anno_file, const std::string &path_prefix, std::vector<SweepInfo> &sweeps, int points_dim) {
    char line[1024];
    // read sweeps 
    if (anno_file.getline(line, 1024, '\n') && std::strncmp(line, "sweeps:", 7) == 0) {
      auto s = std::string(line + 7);
      int num = std::atoi(s.c_str());
      //std::cout << "sweep num:" << num << std::endl;
      if (num == 0) {
        return;
      } 
      int cnt = 0;
      sweeps.clear();
      sweeps.resize(num);
      while(cnt < num) {
        // read data path
        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "data_path:", 10) == 0) {
          auto data_path = std::string(line + 1).substr(10);
          data_path = path_prefix + data_path;
          //std::cout << "data_path:" << data_path << std::endl;
          if (!sweeps[cnt].points.points) {
            sweeps[cnt].points.points.reset(new std::vector<float>);
          }
          read_points_file(data_path, *(sweeps[cnt].points.points));
          sweeps[cnt].points.dim = points_dim;
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
    }
}

void read_inno_file_pp_nus(const std::string &file_name, PointsInfo &points_info,
                           int points_dim, std::vector<SweepInfo> &sweeps) {

  std::string path_prefix; 
  if (file_name.find_last_of('/') != std::string::npos) {
    path_prefix = file_name.substr(0, file_name.find_last_of('/') + 1);
    //std::cout << "path_prefix:" << path_prefix << std::endl;
  }
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
      //read_points_file(lidar_path, points_info);
      if (!points_info.points.points) {
        points_info.points.points.reset(new std::vector<float>);
      }
      read_points_file(path_prefix + lidar_path, *(points_info.points.points));
      points_info.points.dim = points_dim;
    } else {
      break;
    }

    // read timestamp
    if (anno_file.getline(line, 1024, '\n') && std::strncmp(line, "timestamp:", 10) == 0) {
      auto timestamp = std::atoll(std::string(line).substr(10).c_str());
      //std::cout << "timestamp:" << timestamp<< std::endl;
      //points_info.cam_info.timestamp = timestamp;
      points_info.timestamp = timestamp;
    } else {
      break;
    }

    // read sweeps 
    read_sweeps(anno_file, path_prefix, sweeps, points_dim);

    // read cams
    //read_cams(anno_file, sweeps);
    break;
  }
}


}}}
