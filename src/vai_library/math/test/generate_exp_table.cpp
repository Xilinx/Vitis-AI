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
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

std::vector<uint8_t> table_;
int offset_ = 0;
float miny_;
float maxy_;
float step_;
float scale_ = 0.03125f;
int step = 4;
// 4,2,1
void init() {
  switch (step) {
    case 4:
      miny_ = expf((-64 - offset_) * scale_);
      maxy_ = expf((60 - offset_) * scale_);
      break;
    case 2:
      miny_ = expf((-32 - offset_) * scale_);
      maxy_ = expf((30 - offset_) * scale_);
      break;
    case 1:
      miny_ = expf((96 - offset_) * scale_);
      maxy_ = expf((127 - offset_) * scale_);
      break;
    default:
      miny_ = expf((-128 - offset_) * scale_);
      maxy_ = expf((127 - offset_) * scale_);
  }

  step_ = (maxy_ - miny_) / 255.0f;
}

// 0:4:124
/*static int offset() { return 0; }
static float scale() { return 0.0625f; }
static float miny() { return expf((0 - offset()) * scale()); }
static float maxy() { return expf((127 - offset()) * scale()); }
static float step() {
    return (maxy() - miny()) / 256.0f;
}
*/

static void softmax_c(const int8_t *input, float scale, unsigned int cls,
                      float *output) {
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    auto x = input[i] * scale;
    output[i] = exp(x);
    // cout << output[i] << " ";
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] /= sum;
    // cout << output[i] << " ";
  }
  // cout << endl;
}

static void softmax_c(const int8_t *input, float scale, unsigned int cls,
                      unsigned int group, float *output) {
  for (unsigned int i = 0; i < group; ++i) {
    softmax_c(input, scale, cls, output);
    // cout << "softmax_c g=" << i << " ";
    input += cls;
    output += cls;
  }
}

static inline unsigned char float2char(float x) {
  auto y = (unsigned int)((x - miny_) / step_);
  y = min(y, 255u);
  if (0)
    cout << "x " << x << " "  //
         << "y " << y << " "  //
         << std::endl;
  return y;
}
static inline float char2float(unsigned char x) {
  auto y = ((unsigned int)x) * step_ + miny_;
  return y;
}
static uint8_t exp1(int8_t x, float scale) {
  int x1 = x - offset_;
  // float fx = (float)((x1 >> 3) << 3) * scale;
  float fx = (float)x1 * scale;
  float fy = expf(fx);
  /*std::cerr <<  __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__<<"]"//
            << "x "  << (int)x << " " //
            << "x1 "  << x1 << " " //
            << "fx "  << fx << " " //
            << "fy "  << fy << " " //
            << std::endl;*/
  return float2char(fy);
}

static void softmax_neon(const int8_t *input, float scale, unsigned int cls,
                         float *output) {
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    //    auto y = exp1(input[i], scale);

    int index = 0;
    switch (step) {
      case 4:

        if (input[i] >= -64 && input[i] < 64) {
          index = ((int)input[i] + 64) >> 2;
        }
        if (input[i] >= 64) {
          index = 31;
        }
        break;
      case 2:
        if (input[i] >= -32 && input[i] < 32) {
          index = ((int)(input[i] + 32)) >> 1;
        }
        if (input[i] >= 32) {
          index = 31;
        }
        break;
      case 1:
        if (input[i] > 96) {
          index = (int)input[i] - 96;
        }
        break;
      default:
        index = ((int)input[i] + 128) >> 3;
    }
    auto y = table_[index];
    output[i] = char2float(y);
    // cout << output[i] << " ";
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] /= sum;
    // cout << output[i] << " ";
  }
  // cout << endl;
}

// static void softmax4_neon(const int8_t *input, float scale, unsigned int
// group, float *output){
//     table_ = table(scale);
// }

static void softmax_neon(const int8_t *input, float scale, unsigned int cls,
                         unsigned int group, float *output) {
  for (unsigned int i = 0; i < group; ++i) {
    softmax_neon(input, scale, cls, output);
    // cout << "softmax_neon g=" << i << " ";
    input += cls;
    output += cls;
  }
}

// static void softmax_f(const int8_t *input, float scale, unsigned int cls,
//                       unsigned int group, float *output) {
//   for (unsigned int i = 0; i < group; ++i) {
//     softmax_c(input, scale, cls, output);
//     input += cls;
//     output += cls;
//   }
// }
// 0::4::124
static std::vector<uint8_t> table_4(float scale) {
  vector<uint8_t> ret;
  for (int i = -64; i < 64; i = i + 4) {
    unsigned int y = exp1(i, scale);
    ret.push_back(y);
  }
  return ret;
}
// 64::2::126
static std::vector<uint8_t> table_2(float scale) {
  vector<uint8_t> ret;
  for (int i = -32; i < 32; i = i + 2) {
    unsigned int y = exp1(i, scale);
    ret.push_back(y);
  }
  return ret;
}

// 96::1::127
static std::vector<uint8_t> table_1(float scale) {
  vector<uint8_t> ret;

  for (int i = 96; i < 0x78 + 8; i = i + 1) {
    unsigned int y = exp1(i, scale);
    ret.push_back(y);
  }

  return ret;
}

static std::vector<uint8_t> table(float scale) {
  // 5bits fix point 0x80 --  0x80
  // -128 -- 120, step = 8;
  // float range = (-1, 1.0)
  vector<uint8_t> ret;
  switch (std::abs((int)std::log2(scale))) {
    case 8:
      ret = {0,   4,   9,   14,  19,  25,  30,  36,  42,  48,  54,
             61,  68,  75,  82,  89,  97,  105, 113, 121, 130, 139,
             148, 157, 167, 177, 187, 198, 209, 221, 232, 245};
      break;
    case 7:
      ret = {0,   2,   5,   8,   11,  14,  18,  22,  26,  30,  35,
             39,  45,  50,  56,  62,  69,  76,  84,  92,  100, 109,
             119, 129, 140, 152, 164, 178, 192, 207, 223, 240};
      break;
    case 6:
      ret = {0,  0,  1,  2,   3,   4,   5,   6,   8,   10, 12,
             14, 16, 19, 23,  26,  31,  35,  41,  47,  54, 62,
             71, 81, 92, 105, 120, 136, 155, 177, 201, 228};
      break;
    case 5:
      ret = {0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  2,   2,   3,
             4, 6, 7, 10, 13, 16, 21, 27, 35, 45, 58, 75, 97, 124, 160, 205};
      break;
    default:
      ret.reserve(32);
      for (int i = -128; i < 0x78 + 8; i = i + 8) {
        unsigned int y = exp1(i, scale);
        // cout << y << " ";
        ret.push_back(y);
      }
      break;
  }

  // cout << endl;
  // 0.00390625 == 0 4 9 14 19 25 30 36 42 48 54 61 68 75 82 89 97 105 113 121
  // 130 139 148 157 167 177 187 198 209 221 232 245
  // 0.078125  == 0 2 5 8 11 14 18 22 26 30 35 39 45 50 56 62 69 76 84 92 100
  // 109 119 129 140 152 164 178 192 207 223 240
  // 0.015625  == 0 0 1 2 3 4 5 6 8 10 12 14 16 19 23 26 31 35 41 47 54 62 71 81
  // 92 105 120 136 155 177 201 228
  // 0.03125 == 0 0 0 0 0 0 0 0 0 0 0 1 1 2 2 3 4 6 7 10 13 16 21 27 35 45 58 75
  // 97 124 160 205

  return ret;
}

static void float_table() {
  cout << "scale =  " << scale_ << " float_table :";
  for (auto i : table_) {
    cout << (unsigned int)i << " : " << char2float(i) << " ";
  }
  cout << endl;
  // scale =  0.00390625 float_table :0.606531 0.622714 0.642944 0.663174
  // 0.683404 0.707679 0.727909 0.752185 0.776461 0.800736 0.825012 0.853334
  // 0.881655 0.909977 0.938298 0.96662
  // 0.998988 1.03136 1.06372 1.09609 1.1325 1.16892 1.20533 1.24174 1.2822 1.32266
  // 1.36312 1.40763 1.45213 1.50069 1.54519 1.59779

  // scale =  0.0078125 float_table :0.367879 0.386077 0.413373 0.440668
  // 0.467964 0.49526 0.531655 0.568049 0.604444 0.640838 0.686331 0.722726
  // 0.777318 0.822811 0.877403 0.931994
  // 0.995685 1.05938 1.13216 1.20495 1.27774 1.35963 1.45062 1.5416 1.64169 1.75087
  // 1.86005 1.98744 2.11482 2.2513 2.39687 2.55155

  // scale =  0.015625 float_table :0.135335 0.135335 0.163223 0.19111 0.218997
  // 0.246885 0.274772 0.302659 0.358434 0.414209 0.469984 0.525758 0.581533
  // 0.665195 0.776745 0.860407
  // 0.999843 1.11139 1.27872 1.44604 1.64125 1.86435 2.11534 2.39421 2.70097 3.06351
  // 3.48182 3.92802 4.45788 5.0714 5.74069 6.49365

  // scale =  0.03125 float_table :0.0183156 0.0183156 0.0183156 0.0183156
  // 0.0183156 0.0183156 0.0183156 0.0183156 0.0183156 0.0183156 0.0183156
  // 0.224956 0.224956 0.431597 0.431597 0.638238
  // 0.844879 1.25816 1.4648 2.08472 2.70465 3.32457 4.35777 5.59762 7.25074 9.31715
  // 12.0035 15.5164 20.0625 25.6418 33.0808 42.3797
}

float err(const float *a, const float *b, int cls) {
  float ret = 0.0f;
  for (int i = 0; i < cls; ++i) {
    auto d = (a[i] - b[i]);
    ret = ret + d * d;
  }
  ret = sqrtf(ret);
  return ret;
}

int main(int argc, char *argv[]) {
  init();
  std::cout << "step " << step << " "     //
            << "scale " << scale_ << " "  //
            << "miny " << miny_ << " "    //
            << "maxy " << maxy_ << " "    //
            << "step_ " << step_ << " "   //
            << std::endl;

  /*std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(101);
  std::uniform_int_distribution<> dis(-128, 127);
  */
  /* int cls = 2;
  int group = 6400;
  ifstream file_in("/home/liumingyue/face_detect.bin",
                   ios::in | ios::binary);
  */
  int cls = 2;
  int group = 50696;
  ifstream file_in("/home/liumingyue/ssd_adas_pedestrian_640x360.bin",
                   ios::in | ios::binary);

  /*
  int cls = 4;
  int group = 16436;
  ifstream file_in("/home/liumingyue/ssd_adas_vehicle_v3_480x360.bin",
                   ios::in | ios::binary);
  */
  /*
  int cls = 4;
  int group = 27236;
  ifstream file_in("/home/liumingyue/ssd_traffic_480x360.bin",
                   ios::in | ios::binary);
  */
  int total = cls * group;
  char *buffer = new char[total];
  file_in.read(buffer, total);
  file_in.close();
  int8_t *d = new int8_t[total];
  int max = 0;
  int min = 0;
  int count_max = 0;
  int count_min = 0;
  for (int i = 0; i < total; ++i) {
    // d[i] = dis(gen);
    d[i] = buffer[i];
    if (d[i] > 64) count_max++;
    if (d[i] < -64) count_min++;
    if (max < d[i]) max = d[i];
    if (min > d[i]) min = d[i];
  }
  cout << "max :" << max << " "
       << "min : " << min << " "
       << "count_max " << count_max << " "  //
       << "count_min " << count_min << " "  //
       << endl;
  switch (step) {
    case 4:
      table_ = table_4(scale_);
      break;
    case 2:
      table_ = table_2(scale_);
      break;
    case 1:
      table_ = table_1(scale_);
      break;
    default:
      table_ = table(scale_);
      break;
  }

  for (unsigned int t : table_) {
    cout << t << " ";
  }
  cout << endl;
  //  return 0;

  // cout << "===========**********=================" << endl;
  // float_table();
  // for (int i = 0; i < 256; ++i) {
  //   d[i] = i;
  // }
  float *output = new float[total];
  float *output_neon = new float[total];
  softmax_c(d, scale_, cls, group, output);
  softmax_neon(d, scale_, cls, group, output_neon);

  group = 100;
  for (int i = 0; i < group; ++i) {
    cout << "input g=" << i << ":";
    for (int j = 0; j < cls; ++j) {
      cout << " " << (int)d[i * cls + j];
    }
    cout << endl;

    cout << "input g=" << i << ":";
    for (int j = 0; j < cls; ++j) {
      cout << " " << ((int)(d[i * cls + j])) * scale_;
    }
    cout << endl;

    cout << "output_c g=" << i << ":";
    float s = 0.0f;
    float s_neon = 0.0f;
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

    cout << "        err=" << err(&output[i * cls], &output_neon[i * cls], cls)
         << endl;
    cout << "========================" << endl;
  }
  return 0;
}
