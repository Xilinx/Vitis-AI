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
#include <string>
#include <vitis/ai/benchmark.hpp>
#include <vitis/ai/pointpainting.hpp>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
using namespace vitis::ai::pointpillars_nus;

namespace vitis {
namespace ai {

void read_cam_intr_from_line(const std::string &line, CamInfo &cam_info) {
  auto s = line;
  auto cnt = 0u;
  while(cnt < cam_info.cam_intr.size()) {
    auto f = std::atof(s.c_str()); 
    cam_info.cam_intr[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}


void read_s2lr_from_line(const std::string &line, CamInfo &cam_info) {
  auto s = line;
  auto cnt = 0u;
  while(cnt < cam_info.s2l_r.size()) {
    auto f = std::atof(s.c_str()); 
    cam_info.s2l_r[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}

void read_s2lt_from_line(const std::string &line, CamInfo &cam_info) {
  auto s = line;
  auto cnt = 0u;
  while(cnt < cam_info.s2l_t.size()) {
    auto f = std::atof(s.c_str()); 
    cam_info.s2l_t[cnt] = f;
    cnt++;
    auto n = s.find_first_of(' ');
    if (n == std::string::npos) {
      break;
    }
    s = s.substr(n + 1);
  }
}

//void read_points_file(const std::string &points_file_name, PointsInfo &points_info) {
void read_points_file(const std::string &points_file_name, std::vector<float> &points) {
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

void read_cams(std::ifstream &anno_file, const std::string &path_prefix, std::vector<CamInfo> &cam_infos, std::vector<cv::Mat> &images) {
    char line[1024];
    // read sweeps 
    if (anno_file.getline(line, 1024, '\n') && std::strncmp(line, "cams:", 5) == 0) {
      auto s = std::string(line + 5);
      int num = std::atoi(s.c_str());
      //std::cout << "sweep num:" << num << std::endl;
      if (num == 0) {
        return;
      } 
      int cnt = 0;
      images.clear();
      cam_infos.clear();
      images.resize(num);
      cam_infos.resize(num);
      while(cnt < num) {
        // read data path
        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "data_path:", 10) == 0) {
          auto data_path = std::string(line + 1).substr(10);
          data_path = path_prefix + data_path;
          //std::cout << "data_path:" << data_path << std::endl;
          images[cnt] = cv::imread(data_path);
        } else {
          break;
        }

        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "sensor2lidar_rotation:", 22) == 0) {
           auto l = std::string(line + 1).substr(22);
           //std::cout << "s2lr:" << l << std::endl;
           read_s2lr_from_line(l, cam_infos[cnt]); 
        } else {
          break;
        }

        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "sensor2lidar_translation:", 25) == 0) {
           auto l = std::string(line + 1).substr(25);
           //std::cout << "s2lt:" << l << std::endl;
           read_s2lt_from_line(l, cam_infos[cnt]); 
        } else {
          break;
        }
        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "cam_intrinsic:", 14) == 0) {
           auto l = std::string(line + 1).substr(14);
           //std::cout << "s2lt:" << l << std::endl;
           read_cam_intr_from_line(l, cam_infos[cnt]); 
        } else {
          break;
        }

        cnt++;
      }
    }
}


void read_sweeps(std::ifstream &anno_file, const std::string &path_prefix, std::vector<SweepInfo> &sweeps, int points_dim) {
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
           read_s2lr_from_line(l, sweeps[cnt].cam_info); 
        } else {
          break;
        }

        if (anno_file.getline(line, 1024, '\n') && std::strncmp(line + 1, "sensor2lidar_translation:", 25) == 0) {
           auto l = std::string(line + 1).substr(25);
           //std::cout << "s2lt:" << l << std::endl;
           read_s2lt_from_line(l, sweeps[cnt].cam_info); 
        } else {
          break;
        }
        cnt++;
      }
    }
}

void read_inno_file(const std::string &file_name, 
                    PointsInfo &points_info, int points_dim,
                    std::vector<SweepInfo> &sweeps, int sweeps_points_dim,
                    std::vector<cv::Mat> &images) {
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
    read_sweeps(anno_file, path_prefix, sweeps, sweeps_points_dim);

    // read cams
    read_cams(anno_file, path_prefix, points_info.cam_info, images);
    break;
  }
  anno_file.close();
}


class PointPaintingPerf {
public:
  PointPaintingPerf(const std::string &anno_file, const std::string &model_0, const std::string &model_1, const std::string &model_2)
      :  anno_file_name(anno_file), 
         kernel_name_0(model_0),  
         kernel_name_1(model_1), 
         kernel_name_2(model_2), 
         det(PointPainting::create(kernel_name_0, kernel_name_1, kernel_name_2)) {
    batch_size = get_input_batch();
    points_infos.resize(batch_size);
    images.resize(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
      read_inno_file(anno_file_name, points_infos[i], 5, points_infos[i].sweep_infos, 16, images[i]);
    }
  }
  int getInputWidth() {return det->getInputWidth();}
  int getInputHeight() {return det->getInputHeight();}
  size_t get_input_batch() {return (size_t)det->get_pointpillars_batch();}
  std::vector<PointPaintingResult> run(const std::vector<cv::Mat> & fake_image) {
    auto res = det->run(images, points_infos);
    return res;
  }

private:
  std::string anno_file_name;
  std::string kernel_name_0;
  std::string kernel_name_1;
  std::string kernel_name_2;
  size_t batch_size;
  std::unique_ptr<PointPainting> det;
  std::vector<PointsInfo> points_infos;
  std::vector<std::vector<cv::Mat>> images;

};

} //namespace ai
} //namespace vitis


int main(int argc, char *argv[]) {
  std::string image_name = argv[4];
  std::string model_1 = argv[1];
  std::string model_2 = argv[2];
  std::string model_3 = argv[3];
  std::string anno_name = "./1531281442299946.info"; 
  if (image_name.find_last_of('/') != std::string::npos) {
    anno_name = image_name.substr(0, image_name.find_last_of('/') + 1) + anno_name;
  }
   
  return vitis::ai::main_for_performance(argc, argv, [anno_name, model_1, model_2, model_3] {
    { return std::unique_ptr<vitis::ai::PointPaintingPerf>(new vitis::ai::PointPaintingPerf(anno_name, model_1, model_2, model_3)); }
  });
}
