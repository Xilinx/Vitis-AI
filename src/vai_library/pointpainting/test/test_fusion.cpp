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
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>
#include <vitis/ai/profiling.hpp>
#include "../src/fusion.hpp"

using namespace vitis::ai;
using namespace vitis::ai::pointpillars_nus;
using namespace vitis::ai::pointpainting;

int main() {
  std::vector<CamInfo> cam_infos(6);
  std::vector<cv::Mat> images(6);
  std::string root_dir = "./pointpainting/";
  std::vector<std::string> image_names{
         "./data/image_seg_mask/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470_mask.png",
         "./data/image_seg_mask/n015-2018-07-18-11-07-57+0800__CAM_FRONT_RIGHT__1531883530420339_mask.png",
         "./data/image_seg_mask/n015-2018-07-18-11-07-57+0800__CAM_FRONT_LEFT__1531883530404844_mask.png",
         "./data/image_seg_mask/n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883530437525_mask.png",
         "./data/image_seg_mask/n015-2018-07-18-11-07-57+0800__CAM_BACK_LEFT__1531883530447423_mask.png",
         "./data/image_seg_mask/n015-2018-07-18-11-07-57+0800__CAM_BACK_RIGHT__1531883530427893_mask.png"};

  struct stat file_stat;
  for (auto i = 0u; i < image_names.size(); ++i) {
    auto name = root_dir + image_names[i];
    if (stat(name.c_str(), &file_stat) != 0) {
      std::cerr << "file:" << name << " state error!" << std::endl;
      exit(-1);
    } 
    images[i] = cv::imread(name.c_str());
  }
  
  std::string bin_name = root_dir + "./data/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin";
  if (stat(bin_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << bin_name<< " state error!" << std::endl;
    exit(-1);
  } 

  auto file_size = file_stat.st_size; 
  auto len = file_size / 4;
  LOG(INFO) << "input file size:" << file_size;
  std::vector<float> points(len);
  CHECK(std::ifstream(bin_name).read(reinterpret_cast<char *>(points.data()), points.size() * 4).good()); 

  // cam front
  cam_infos[0] = CamInfo{0,
                   {0.00072265, 0.60818175, -0.31034774},
                   {0.99995012, 0.00730543, 0.00681137,
                   -0.00694924, 0.01901527, 0.99979504,
                    0.00717441,-0.9997925,  0.01906509},
                   {1266.41720, 0.0,            816.267020,
                    0,          1266.41720,     491.507066,
                    0,          0,              1}};
  // cam front right
  cam_infos[1] = CamInfo{0,
                         {0.50605516,  0.48573333, -0.32626257},
                         {0.5447327 , -0.01001924,  0.83854988,
                          -0.83820065,  0.02472418,  0.54480124,
                          -0.02619095, -0.9996441 ,  0.00506993},
                         {1260.84744, 0.0, 	807.968245,
                          0.0, 1260.84744, 495.334427,
                          0.0, 0.0, 1.0}};
  // cam front left
  cam_infos[2] = CamInfo{0,
                         {-0.47811395,  0.40702043, -0.31929113},
                         {0.58312896,  0.00348856, -0.81237211,
                          0.81208429,  0.02445774,  0.58302738,
                          0.02190271, -0.99969478,  0.01142902},
                         {1272.59795, 0.0, 826.615493,
                          0.0, 1272.59795, 479.751654,
                          0.0, 0.0, 1.0}};

  // cam back
  cam_infos[3] = CamInfo{0,
                         {-0.00562831, -0.96006092, -0.28334762},
                         {-0.99991364,  0.01038427, -0.00805477,
                           0.0081334 ,  0.00755213, -0.9999384,
                          -0.0103228 , -0.99991756, -0.00763594},
                         {809.22099057,   0.0        , 829.21960033,
                          0.0        , 809.22099057, 481.77842385,
                          0.0, 0.0, 1.0}};
  // cam back left
  cam_infos[4] = CamInfo{0,
                         {-0.48305756,  0.09083851, -0.24988307},
                         {-0.31651335,  0.02013239, -0.94837439,
                           0.94826328,  0.03287486, -0.31577839,
                           0.02482031, -0.99925669, -0.02949614},
                         {1256.74148, 0.0, 792.112574,
                          0.0, 1256.74148, 492.775747,
                          0.0, 0.0, 1.0}};

  // cam back right 
  cam_infos[5] = CamInfo{0,
                         {0.48553667, -0.01920347, -0.2728415},
                         {-0.36268682, -0.00462228,  0.93189962,
                          -0.93121605,  0.04041165, -0.36222033,
                          -0.03598531, -0.99917242, -0.01896112},
                         {1259.51374, 0.0, 807.252905,
                          0.0, 1259.51374, 501.195799,
                          0.0, 0.0, 1.0}};
__TIC__(FUSION)
  auto result = fusion(cam_infos, points, 5, images, 11);
__TOC__(FUSION)
  std::cout << "result shape: " << result.size() / 16
            << " * " << 16;
  return 0;
}
