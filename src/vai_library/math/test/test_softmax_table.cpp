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
using namespace std;
#include <fstream>

#include "../include/vitis/softmax.hpp"
//#include "../src/softmax_table.hpp"

#include <unistd.h>

#include <chrono>
#include <random>
#include <string>

using Clock = std::chrono::high_resolution_clock;
#define __TIC__(tag) auto __##tag##_start_time = Clock::now();

#define __TOC__(tag)                                                           \
  auto __##tag##_end_time = Clock::now();                                      \
  cout << #tag << " : "                                                        \
       << std::chrono::duration_cast<std::chrono::microseconds>(               \
              __##tag##_end_time - __##tag##_start_time)                       \
              .count()                                                         \
       << endl;

extern int GLOBAL_ENABLE_C_SOFTMAX;

float err(const float* a, const float* b, int cls) {
  float ret = 0.0f;
  for (int i = 0; i < cls; ++i) {
    auto d = (a[i] - b[i]);
    ret = ret + d * d;
  }
  ret = sqrtf(ret);
  return ret;
}

/*int cls_ = 4;
int group_ = 8000;
float scale_ = 0.0625f;
*/
int model_ = 0;
int show_ = 0;  // 显示多少条数据

int cls = 4;
int group = 16436;
int total;
float scale = 0.0625f;
string filename = "/home/liumingyue/ssd_adas_vehicle_v3_480x360.bin";
int8_t* d;

static void parse_opt(int argc, char* argv[]) {
  int opt = 0;

  while ((opt = getopt(argc, argv, "m:c:g:s:l:")) != -1) {
    switch (opt) {
      case 'm':
        model_ = std::stoi(optarg);

        break;
      case 'l':
        show_ = std::stoi(optarg);
        break;
      case 'c':
        cls = std::stoi(optarg);
        break;
      case 'g':
        group = std::stoi(optarg);
        break;
      case 's':
        scale = std::stof(optarg);
        break;
      default:
        break;
    }
  }

  return;
}

void getdata() {
  switch (model_) {
    case 1:  // face_detect
      cls = 2;
      group = 6400;
      scale = 0.0625f;
      filename = "/home/liumingyue/face_detect.bin";
      break;
    case 2:  // ssd_pedestrian_640x360
      cls = 2;
      group = 50696;
      scale = 0.03125f;
      filename = "/home/liumingyue/ssd_adas_pedestrian_640x360.bin";
      break;
    case 3:
      cls = 4;
      group = 27236;
      scale = 0.25f;
      filename = "/home/liumingyue/ssd_traffic_480x360.bin";
      break;
    case 0:  // 配合其他参数使用
      break;
    default:  // ssd_adas_vehicle_v3_480x360
      // ssd_pedestrian_640x360
      break;
  }
  total = cls * group;
  d = new int8_t[total];
  if (model_ == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(101);
    std::uniform_int_distribution<> dis(-128, 127);
    for (int i = 0; i < total; ++i) {
      d[i] = dis(gen);
    }

  } else {
    ifstream file_in(filename, ios::in | ios::binary);
    char* buffer = new char[total];
    file_in.read(buffer, total);
    file_in.close();

    int max = 0;
    int min = 0;

    for (int i = 0; i < total; ++i) {
      // d[i] = dis(gen);
      d[i] = buffer[i];
      if (max < d[i]) max = d[i];
      if (min > d[i]) min = d[i];
    }
    std::cout << "max :" << max << " "
              << "min : " << min << " " << std::endl;
    delete []buffer;
  }
}

int main(int argc, char* argv[]) {
  parse_opt(argc, argv);

  getdata();

  /* std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(101);
  std::uniform_int_distribution<> dis(-128, 127);

  int cls = cls_;
  int group = group_;

  float scale = scale_;*/
  // GLOBAL_ENABLE_C_SOFTMAX = 2;
  cout << "cls " << cls << " "      //
       << "group " << group << " "  //
       << "scale " << scale << " "  //
       << std::endl;
  /*
  int total = cls * group;
  int8_t *d = new int8_t[total];
  for (int i = 0; i < total; ++i) {
    d[i] = dis(gen);
  }
  */
  float* output = new float[total];
  float* output_neon = new float[total];
  float* output_neon_table = new float[total];

  GLOBAL_ENABLE_C_SOFTMAX = 0;
  // dpuOpen();
  __TIC__(softmax_neon_0)
  vitis::ai::softmax(d, scale, cls, group, output_neon);
  __TOC__(softmax_neon_0)
  // dpuClose();

  GLOBAL_ENABLE_C_SOFTMAX = 1;
  __TIC__(softmax_neon_1)
  vitis::ai::softmax(d, scale, cls, group, output_neon);
  __TOC__(softmax_neon_1)

  GLOBAL_ENABLE_C_SOFTMAX = 2;
  __TIC__(softmax_c)
  vitis::ai::softmax(d, scale, cls, group, output);
  __TOC__(softmax_c)

  GLOBAL_ENABLE_C_SOFTMAX = 3;
  __TIC__(softmax_neon_table)
  vitis::ai::softmax(d, scale, cls, group, output_neon_table);
  __TOC__(softmax_neon_table)

  if (1)
    for (int i = 0; i < show_; ++i) {
      cout << "input g=" << i << ":";
      for (int j = 0; j < cls; ++j) {
        cout << " " << (int)d[i * cls + j];
      }
      cout << endl;

      cout << "input g=" << i << ":";
      for (int j = 0; j < cls; ++j) {
        cout << " " << ((int)(d[i * cls + j])) * scale;
      }
      cout << endl;

      cout << "output_c g=" << i << ":";
      float s = 0.0f;
      float s_neon = 0.0f;
      float s_neon_table = 0.0f;
      for (int j = 0; j < cls; ++j) {
        s = s + output[i * cls + j];
        cout << " " << output[i * cls + j];
      }
      cout << " " << s << endl;

      cout << "output_neon g=" << i << ":";
      for (int j = 0; j < cls; ++j) {
        s_neon = s_neon + output_neon[i * cls + j];
        cout << " " << output_neon[i * cls + j];
      }
      cout << " " << s_neon << endl;

      cout << "        err="
           << err(&output[i * cls], &output_neon[i * cls], cls) << endl;

      cout << "output_neon_table g=" << i << ":";
      for (int j = 0; j < cls; ++j) {
        s_neon_table = s_neon_table + output_neon_table[i * cls + j];
        cout << " " << output_neon_table[i * cls + j];
      }
      cout << " " << s_neon_table << endl;

      cout << "        err="
           << err(&output[i * cls], &output_neon_table[i * cls], cls) << endl;

      cout << "========================" << endl;
    }
  if(output != nullptr) delete []output;
  if(output_neon != nullptr) delete []output_neon;
  if(output_neon_table != nullptr) delete []output_neon_table;

  return 0;
}
