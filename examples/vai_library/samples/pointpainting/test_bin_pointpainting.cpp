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
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/pointpainting.hpp>

using namespace vitis::ai;
using namespace vitis::ai::pointpillars_nus;

DEF_ENV_PARAM(SAMPLES_BATCH_NUM, "0");
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

void print_points_info(const PointsInfo &points_info) {
  std::cout << "points info: " << std::endl; 
  std::cout << "  cam_info:" << std::endl;
  for (auto n = 0u; n < points_info.cam_info.size(); ++n) {
    std::cout << "    timestamp:" << points_info.cam_info[n].timestamp << std::endl;
    std::cout << "    s2l_t:";
    for (auto i = 0u; i < points_info.cam_info[n].s2l_t.size(); ++i) {
      std::cout << points_info.cam_info[n].s2l_t[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "    s2l_r:";

    for (auto i = 0u; i < points_info.cam_info[n].s2l_r.size(); ++i) {
      std::cout << points_info.cam_info[n].s2l_r[i] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "  dim:" << points_info.points.dim << std::endl;
  if (points_info.points.points) {
    std::cout << "  points size:" << points_info.points.points->size() << std::endl;
  }
}
int main( int argc, char *argv[]) {
  if (argc < 5) {
    //std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0] << " [segmentation_model] [pointpillars_model0] [pointpillars_model1] [anno_file_name]" << std::endl;
    exit(0);
  }

  int input_num = argc - 4;
  if (ENV_PARAM(SAMPLES_BATCH_NUM)) {
    input_num = std::min(ENV_PARAM(SAMPLES_BATCH_NUM), input_num);
    //std::cout << "set batch num :" << input_num << std::endl;
  }

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
    read_inno_file(anno_file_names[i], points_infos[i], 5,  points_infos[i].sweep_infos, 16, images[i]); 
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

