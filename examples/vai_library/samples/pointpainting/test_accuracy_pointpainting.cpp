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
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/pointpainting.hpp>

using namespace vitis::ai;
using namespace vitis::ai::pointpillars_nus;

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

void read_sweeps(std::ifstream &anno_file, const std::string &path_prefix, std::vector<SweepInfo> &sweeps) {
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
          sweeps[cnt].points.dim = 16;
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
                    PointsInfo &points_info,
                    std::vector<SweepInfo> &sweeps,
                    std::vector<cv::Mat> &images,
                    const std::string &points_info_path) {

  std::string path_prefix; 
  if (!points_info_path.empty()) {
    path_prefix = points_info_path + "/";
  } else if (file_name.find_last_of('/') != std::string::npos) {
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
      points_info.points.dim = 5;
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
    read_sweeps(anno_file, path_prefix, sweeps);

    // read cams
    read_cams(anno_file, path_prefix, points_info.cam_info, images);
    break;
  }
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

std::vector<std::string> get_anno_file_name(const std::string &anno_list_name) {
  struct stat file_stat;
  if (stat(anno_list_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << anno_list_name << " state error!" << std::endl;
    exit(-1);
  } 
  auto anno_list = std::ifstream(anno_list_name);
  char line[1024];
  std::vector<std::string> file_names;
  while (anno_list.getline(line, 1024, '\n')) {
    file_names.emplace_back(line);  
  } 
  anno_list.close();
  return file_names; 
}

int main( int argc, char *argv[])
{
  if (argc < 6) {
    //std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0] << " [segmentation_model] [pointpillars_model0] [pointpillars_model1] [anno_file_list] [out_path]" << std::endl;
    exit(0);
  }

  
  std::string seg_model = argv[1];
  std::string model_0 = argv[2];
  std::string model_1 = argv[3];

  seg_model += std::string("_acc");
  model_0 += std::string("_acc");
  model_1 += std::string("_acc");

  auto pointpainting = vitis::ai::PointPainting::create(seg_model,
          model_0, model_1);
  if (!pointpainting) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto batch = pointpainting->get_pointpillars_batch();

  std::string anno_list_name = argv[4];
  std::string anno_list_path;
  if (anno_list_name.find_last_of('/') != std::string::npos) {
    anno_list_path = anno_list_name.substr(0, anno_list_name.find_last_of('/') +1);
  }

  auto anno_file_names = get_anno_file_name(anno_list_name);
  std::string out_path = argv[5];
  std::string cmd = "mkdir -p " + out_path;
  if (system(cmd.c_str()) == -1) {
    std::cerr << "command: " << cmd << " error!" << std::endl;
    exit(-1);
  }
  
  std::vector<std::string> batch_anno_file_names;
  std::vector<PointsInfo> batch_points_infos;
  std::vector<std::vector<cv::Mat>> batch_images;
  for (auto n = 0u; n < anno_file_names.size(); n+=batch) {
    auto start = n;
    auto end = std::min(n + batch , anno_file_names.size());
    auto input_num = end - start;
    batch_points_infos.clear();
    batch_points_infos.resize(input_num);
    batch_anno_file_names.clear();
    batch_anno_file_names.resize(input_num);
    batch_images.clear();
    batch_images.resize(input_num);
    for (auto i = 0u; i < input_num; ++i) {
      batch_anno_file_names[i] = anno_file_names[start + i];
      auto full_name = anno_list_path + batch_anno_file_names[i]; 
      struct stat file_stat;
      if (stat(full_name.c_str(), &file_stat) != 0) {
        std::cerr << "file:" << full_name << " state error!" << std::endl;
        exit(-1);
      }
      read_inno_file(full_name, batch_points_infos[i], batch_points_infos[i].sweep_infos, batch_images[i], anno_list_path); 
    }
    auto batch_ret = pointpainting->run(batch_images, batch_points_infos);
    for (auto b = 0u; b < batch_ret.size(); ++b) {
      auto full_name = batch_anno_file_names[b];
      std::string name = full_name;
      if (name.find_last_of('/') != std::string::npos) {
        name = name.substr(name.find_last_of('/') + 1); 
      }
      if (name.find_last_of('.') != std::string::npos) {
        name = name.substr(0, name.find_last_of('.')); 
      }
      std::string out_name = out_path + "/" + name + ".txt"; 
      std::cout << "out_name : " << out_name << std::endl;
      auto out = std::ofstream(out_name);
      auto &ret = batch_ret[b];
      for (auto i = 0u; i < ret.bboxes.size(); ++i) {
        for (auto j = 0u; j < ret.bboxes[i].bbox.size(); ++j) {
          //std::cout << ret.bboxes[i].bbox[j] << " ";
          out << ret.bboxes[i].bbox[j] << " ";
        }
        out << ret.bboxes[i].label << " "; 
        out << ret.bboxes[i].score << "\n"; 
      }
      out.close();
    }
  }
  return 0;
}

