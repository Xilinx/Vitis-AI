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
#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>
//#include "../include/vitis/softmax.hpp"

using namespace std;

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

/*
const std::vector<uint8_t> char_table_8 = {
  97, 105, 113, 121, 130, 139, 148, 157, 167, 177, 187, 198, 209, 221, 232, 245,
  0,  4,   9,   14,  19,  25,  30,  36,  42,  48,  54,  61,  68,  75,  82,  89
};
const std::vector<uint8_t> char_table_7_ = {
  69, 76, 84, 92, 100, 109, 119, 129, 140, 152, 164, 178, 192, 207, 223, 240,
  0,  2,  5,  8,  11,  14,  18,  22,  26,  30,  35,  39,  45,  50,  56,  62
};
const std::vector<uint8_t> char_table_6 = {
  31, 35, 41, 47, 54, 62, 71, 81, 92, 105, 120, 136, 155, 177, 201, 228,
  0,  0,  1,  2,  3,  4,  5,  6,  8,  10,  12,  14,  16,  19,  23,  26
};
const std::vector<uint8_t> char_table_5 = { 4,  6,  7,  10, 13, 16,  21,  27,
                                            35, 45, 58, 75, 97, 124, 160, 205,
                                            0,  0,  0,  0,  0,  0,   0,   0,
                                            0,  0,  0,  1,  1,  2,   2,   3 };*/

struct Table {
  const float miny_;
  const float step_;
  const uint8x8x4_t char_table_;
};

Table getTable(int point) {
  switch (point) {
    case 8:
      return {0.606531f,    //
              0.00404595f,  //
              {vcreate_u8(0x9D948B8279716961), vcreate_u8(0xF5E8DDD1C6BBB1A7),
               vcreate_u8(0x241E19130E090400), vcreate_u8(0x59524B443D36302A)}};
    case 7:
      return {0.367879f,    //
              0.00909863f,  //
              {vcreate_u8(0x81776D645C544C45), vcreate_u8(0xF0DFCFC0B2A4988C),
               vcreate_u8(0x16120E0B08050200), vcreate_u8(0x3E38322D27231E1A)}};
    case 6:
      return {0.135335f,   //
              0.0278874f,  //
              {vcreate_u8(0x51473E362F29231F), vcreate_u8(0xE4C9B19B8878695C),
               vcreate_u8(0x0605040302010000), vcreate_u8(0x1A1713100E0C0A08)}};
    case 5:
      return {0.0183156f,  //
              0.206641f,   //
              {vcreate_u8(0x1B15100D0A070604), vcreate_u8(0xCDA07C614B3A2D23),
               vcreate_u8(0x0000000000000000), vcreate_u8(0x0302020101000000)}};
    default:
      exit(-1);
  }
}
/*
class ExpTable
{
  public:
    ExpTable(float scale);
    virtual ~ExpTable(){}
    ExpTable(const ExpTable&rhs) = delete;

  public :
    std::vector<uint8_t> getCharTable(){return char_table_;}
    float getMiny(){ return miny_; }
    float getStep(){ return step_; }

  private:
    const float scale_;
    const float miny_;
    const float maxy_;
    const float step_;
    std::vector<uint8_t> char_table_;

  private:
     unsigned char float2char(float x) {
      auto y = (unsigned int)((x - miny_) / step_);
      y = std::min(y, 255u);
      if (0)
          std::cout << "x " << x << " " //
             << "y " << y << " " //
             << std::endl;
      return y;
    }

    uint8_t exp1(int8_t x) {
      int x1 = x;
      float fx = (float)x1 * scale_;
      float fy = expf(fx);
      return float2char(fy);
    }

    void generate_char_table() {
      // 5bits fix point 0x80 --  0x80
      // -128 -- 120, step = 8;
      // float range = (-1, 1.0)
      char_table_.reserve(32);
      for(int i = 0 ; i < 0x78 + 8 ; i = i+ 8) {
        unsigned int y = exp1(i);
        char_table_.push_back(y);
      }
      for (int i = -128; i < 0; i = i + 8) {
        unsigned int y = exp1(i);
        char_table_.push_back(y);
      }
    }

};

ExpTable::ExpTable(float scale):
        scale_(scale),
        miny_(expf(-128 * scale_)),
        maxy_(expf(127 * scale_)),
        step_((maxy_ - miny_) / 256.0f)
{
    generate_char_table();
    //generate_float_table();
    if(1){
      std::cout << "miny_ " << miny_ << " " //
                << "maxy_ " << maxy_ << " " //
                << "step_ " << step_ << " " //
                << std::endl;
      for (unsigned int t : char_table_) {
        std::cout << t << " " //
            ;
      }
      std::cout << std::endl;
    }
}
*/
void print_s8x8(int8x8_t d) {
  std::cout << (int)vget_lane_s8(d, 7) << ", " << (int)vget_lane_s8(d, 6)
            << ", " << (int)vget_lane_s8(d, 5) << ", "
            << (int)vget_lane_s8(d, 4) << ", " << (int)vget_lane_s8(d, 3)
            << ", " << (int)vget_lane_s8(d, 2) << ", "
            << (int)vget_lane_s8(d, 1) << ", " << (int)vget_lane_s8(d, 0)
            << std::endl;
}

void print_u8x8(uint8x8_t d) {
  std::cout << (int)vget_lane_u8(d, 7) << ", " << (int)vget_lane_u8(d, 6)
            << ", " << (int)vget_lane_u8(d, 5) << ", "
            << (int)vget_lane_u8(d, 4) << ", " << (int)vget_lane_u8(d, 3)
            << ", " << (int)vget_lane_u8(d, 2) << ", "
            << (int)vget_lane_u8(d, 1) << ", " << (int)vget_lane_u8(d, 0)
            << std::endl;
}

void print_f32x4(float32x4_t d) {
  std::cout << vgetq_lane_f32(d, 3) << ", " << vgetq_lane_f32(d, 2) << ", "
            << vgetq_lane_f32(d, 1) << ", " << vgetq_lane_f32(d, 0)
            << std::endl;
}
/*
float char2float(unsigned char x， float miny, float step) {
      auto y = ((unsigned int)x) * step + miny;
      return y;
      }*/

static void softmax_c(const int8_t* input, float scale, unsigned int cls,
                      float* output) {
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    auto x = input[i] * scale;
    output[i] = exp(x);
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] /= sum;
  }
  // cout << endl;
}

static void softmax_c(const int8_t* input, float scale, unsigned int cls,
                      unsigned int group, float* output) {
  /*
cout << "group "  << group << " " //
       << "scale "  << scale << " " //
       << std::endl;*/
  for (unsigned int i = 0; i < group; ++i) {
    softmax_c(input, scale, cls, output);
    input += cls;
    output += cls;
  }
}

static inline uint8x8_t value2char(const uint8x8x4_t q_table_,
                                   const int8x8_t q) {
  // vshr_n_u8(vreinterpret_u8_s8(q) , 3)
  // int8x8_t q_0 = vadd_s8(vshr_n_s8(q, 3), d_16);
  return vtbl4_u8(q_table_, vshr_n_u8(vreinterpret_u8_s8(q), 3));
}

static inline float32x4_t char2float(const uint16x4_t d0_l, const float step,
                                     const float32x4_t f_miny) {
  const float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(d0_l));
  return vmlaq_n_f32(f_miny, f0, step);  // ai + bi * c;
  // return vaddq_f32(vmulq_n_f32(f0, step), f_miny);
}

static void __attribute__((noinline))
softmax4_internal(const int8_t* input, float scale, unsigned int group,
                  float* output) {
  unsigned int batch = group / 8;
  // __TIC__(expTable)
  // ExpTable expTable(scale);
  Table table = getTable(std::abs((int)log2(scale)));
  float miny = table.miny_;
  float step = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;
  if (0) {
    print_u8x8(q_table_.val[0]);
    print_u8x8(q_table_.val[1]);
    print_u8x8(q_table_.val[2]);
    print_u8x8(q_table_.val[3]);
  }

  float32x4_t f_miny = vdupq_n_f32(miny);
  // __TOC__(expTable)

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x4_t q01 = vld4_s8(input);
    const uint8x8_t q01_0_tbl = value2char(q_table_, q01.val[0]);
    const uint8x8_t q01_1_tbl = value2char(q_table_, q01.val[1]);
    const uint8x8_t q01_2_tbl = value2char(q_table_, q01.val[2]);
    const uint8x8_t q01_3_tbl = value2char(q_table_, q01.val[3]);

    uint16x8_t d0 = vmovl_u8(q01_0_tbl);
    uint16x8_t d1 = vmovl_u8(q01_1_tbl);
    uint16x8_t d2 = vmovl_u8(q01_2_tbl);
    uint16x8_t d3 = vmovl_u8(q01_3_tbl);

    uint16x4_t d0_l = vget_low_u16(d0);
    uint16x4_t d1_l = vget_low_u16(d1);
    uint16x4_t d2_l = vget_low_u16(d2);
    uint16x4_t d3_l = vget_low_u16(d3);

    float32x4_t f0 = char2float(d0_l, step, f_miny);
    float32x4_t f1 = char2float(d1_l, step, f_miny);
    float32x4_t f2 = char2float(d2_l, step, f_miny);
    float32x4_t f3 = char2float(d3_l, step, f_miny);

    // 求和
    float32x4_t q_sum = vaddq_f32(f0, f1);
    q_sum = vaddq_f32(q_sum, f2);
    q_sum = vaddq_f32(q_sum, f3);
    // 取倒数
    q_sum = vrecpeq_f32(q_sum);
    f0 = vmulq_f32(f0, q_sum);
    f1 = vmulq_f32(f1, q_sum);
    f2 = vmulq_f32(f2, q_sum);
    f3 = vmulq_f32(f3, q_sum);

    float32x4x4_t b0 = {f0, f1, f2, f3};
    vst4q_f32(output, b0);
    output += 16;
    // 取高位进行计算
    d0_l = vget_high_u16(d0);
    d1_l = vget_high_u16(d1);
    d2_l = vget_high_u16(d2);
    d3_l = vget_high_u16(d3);

    //将16位的无符号int 转为 32位float
    f0 = char2float(d0_l, step, f_miny);
    f1 = char2float(d1_l, step, f_miny);
    f2 = char2float(d2_l, step, f_miny);
    f3 = char2float(d3_l, step, f_miny);

    // 求和
    q_sum = vaddq_f32(f0, f1);
    q_sum = vaddq_f32(q_sum, f2);
    q_sum = vaddq_f32(q_sum, f3);
    // 取倒数
    q_sum = vrecpeq_f32(q_sum);
    f0 = vmulq_f32(f0, q_sum);
    f1 = vmulq_f32(f1, q_sum);
    f2 = vmulq_f32(f2, q_sum);
    f3 = vmulq_f32(f3, q_sum);

    b0 = {f0, f1, f2, f3};
    vst4q_f32(output, b0);
    output += 16;

    input += 32;
  }
}

static void softmax4_neon(const int8_t* input, float scale, unsigned int group,
                          float* output) {
  unsigned int aligned = group & (-8);
  softmax4_internal(input, scale, aligned, output);
  unsigned int remain = group - aligned;
  input += (4 * aligned);
  output += (4 * aligned);
  softmax_c(input, scale, 4, remain, output);
}

static void softmax8_internal(const int8_t* input, float scale,
                              unsigned int group, float* output) {
  unsigned int batch = group / 4;

  Table table = getTable(std::abs((int)log2(scale)));
  float miny = table.miny_;
  float step = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x4_t q01 = vld4_s8(input);

    const uint8x8_t q01_0_tbl = value2char(q_table_, q01.val[0]);
    const uint8x8_t q01_1_tbl = value2char(q_table_, q01.val[1]);
    const uint8x8_t q01_2_tbl = value2char(q_table_, q01.val[2]);
    const uint8x8_t q01_3_tbl = value2char(q_table_, q01.val[3]);

    uint16x8_t d0 = vmovl_u8(q01_0_tbl);
    uint16x8_t d1 = vmovl_u8(q01_1_tbl);
    uint16x8_t d2 = vmovl_u8(q01_2_tbl);
    uint16x8_t d3 = vmovl_u8(q01_3_tbl);

    //=====
    // 先取低位进行计算
    uint16x4_t d0_l = vget_low_u16(d0);
    uint16x4_t d1_l = vget_low_u16(d1);
    uint16x4_t d2_l = vget_low_u16(d2);
    uint16x4_t d3_l = vget_low_u16(d3);

    //将16位的无符号int 转为 32位float

    float32x4_t f0 = char2float(d0_l, step, f_miny);
    float32x4_t f1 = char2float(d1_l, step, f_miny);
    float32x4_t f2 = char2float(d2_l, step, f_miny);
    float32x4_t f3 = char2float(d3_l, step, f_miny);

    // 求和
    float32x4_t q_sum = vaddq_f32(f0, f1);
    q_sum = vaddq_f32(q_sum, f2);
    q_sum = vaddq_f32(q_sum, f3);
    float32x4_t q_sum_rev = vrev64q_f32(q_sum);
    q_sum = vaddq_f32(q_sum, q_sum_rev);

    // 取倒数
    q_sum = vrecpeq_f32(q_sum);
    f0 = vmulq_f32(f0, q_sum);
    f1 = vmulq_f32(f1, q_sum);
    f2 = vmulq_f32(f2, q_sum);
    f3 = vmulq_f32(f3, q_sum);

    float32x4x4_t b0 = {f0, f1, f2, f3};
    vst4q_f32(output, b0);
    output += 16;
    // print_s8x8(d_16);
    //=====
    // 取高位进行计算
    d0_l = vget_high_u16(d0);
    d1_l = vget_high_u16(d1);
    d2_l = vget_high_u16(d2);
    d3_l = vget_high_u16(d3);

    //将16位的无符号int 转为 32位float
    f0 = char2float(d0_l, step, f_miny);
    f1 = char2float(d1_l, step, f_miny);
    f2 = char2float(d2_l, step, f_miny);
    f3 = char2float(d3_l, step, f_miny);

    // 求和
    q_sum = vaddq_f32(f0, f1);
    q_sum = vaddq_f32(q_sum, f2);
    q_sum = vaddq_f32(q_sum, f3);

    q_sum_rev = vrev64q_f32(q_sum);
    q_sum = vaddq_f32(q_sum, q_sum_rev);

    // 取倒数
    q_sum = vrecpeq_f32(q_sum);

    f0 = vmulq_f32(f0, q_sum);
    f1 = vmulq_f32(f1, q_sum);
    f2 = vmulq_f32(f2, q_sum);
    f3 = vmulq_f32(f3, q_sum);

    b0 = {f0, f1, f2, f3};
    vst4q_f32(output, b0);
    output += 16;

    input += 32;
  }
}

static void softmax8_neon(const int8_t* input, float scale, unsigned int group,
                          float* output) {
  unsigned int aligned = group & (-4);
  softmax8_internal(input, scale, aligned, output);
  unsigned int remain = group - aligned;
  input += (8 * aligned);
  output += (8 * aligned);
  softmax_c(input, scale, 8, remain, output);
}

static void softmax2_internal(const int8_t* input, float scale,
                              unsigned int group, float* output) {
  unsigned int batch = group / 16;

  Table table = getTable(std::abs((int)log2(scale)));
  float miny = table.miny_;
  float step = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x4_t q01 = vld4_s8(input);

    const uint8x8_t q01_0_tbl = value2char(q_table_, q01.val[0]);
    const uint8x8_t q01_1_tbl = value2char(q_table_, q01.val[1]);
    const uint8x8_t q01_2_tbl = value2char(q_table_, q01.val[2]);
    const uint8x8_t q01_3_tbl = value2char(q_table_, q01.val[3]);

    const uint16x8_t d0 = vmovl_u8(q01_0_tbl);
    const uint16x8_t d1 = vmovl_u8(q01_1_tbl);
    const uint16x8_t d2 = vmovl_u8(q01_2_tbl);
    const uint16x8_t d3 = vmovl_u8(q01_3_tbl);
    //=====
    // 先取低位进行计算
    uint16x4_t d0_l = vget_low_u16(d0);
    uint16x4_t d1_l = vget_low_u16(d1);
    uint16x4_t d2_l = vget_low_u16(d2);
    uint16x4_t d3_l = vget_low_u16(d3);

    //将16位的无符号int 转为 32位float

    float32x4_t f0 = char2float(d0_l, step, f_miny);
    float32x4_t f1 = char2float(d1_l, step, f_miny);
    float32x4_t f2 = char2float(d2_l, step, f_miny);
    float32x4_t f3 = char2float(d3_l, step, f_miny);

    // print_f32x4(f0);

    // 求和
    float32x4_t q_sum = vaddq_f32(f0, f1);

    // 取倒数
    q_sum = vrecpeq_f32(q_sum);
    f0 = vmulq_f32(f0, q_sum);
    f1 = vmulq_f32(f1, q_sum);

    q_sum = vaddq_f32(f2, f3);
    q_sum = vrecpeq_f32(q_sum);
    f2 = vmulq_f32(f2, q_sum);
    f3 = vmulq_f32(f3, q_sum);

    float32x4x4_t b0 = {f0, f1, f2, f3};
    vst4q_f32(output, b0);
    output += 16;
    // print_s8x8(d_16);
    //=====
    // 取高位进行计算
    d0_l = vget_high_u16(d0);
    d1_l = vget_high_u16(d1);
    d2_l = vget_high_u16(d2);
    d3_l = vget_high_u16(d3);

    //将16位的无符号int 转为 32位float
    f0 = char2float(d0_l, step, f_miny);
    f1 = char2float(d1_l, step, f_miny);
    f2 = char2float(d2_l, step, f_miny);
    f3 = char2float(d3_l, step, f_miny);

    // 求和
    q_sum = vaddq_f32(f0, f1);
    // 取倒数
    q_sum = vrecpeq_f32(q_sum);
    f0 = vmulq_f32(f0, q_sum);
    f1 = vmulq_f32(f1, q_sum);

    q_sum = vaddq_f32(f2, f3);
    q_sum = vrecpeq_f32(q_sum);
    f2 = vmulq_f32(f2, q_sum);
    f3 = vmulq_f32(f3, q_sum);

    b0 = {f0, f1, f2, f3};
    vst4q_f32(output, b0);
    output += 16;

    input += 32;
  }
}

static void softmax2_neon(const int8_t* input, float scale, unsigned int group,
                          float* output) {
  unsigned int aligned = group & (-16);
  softmax2_internal(input, scale, aligned, output);
  unsigned int remain = group - aligned;
  input += (2 * aligned);
  output += (2 * aligned);
  softmax_c(input, scale, 2, remain, output);
}

float err(const float* a, const float* b, int cls) {
  float ret = 0.0f;
  for (int i = 0; i < cls; ++i) {
    auto d = (a[i] - b[i]);
    ret = ret + d * d;
  }
  ret = sqrtf(ret);
  return ret;
}

int cls_ = 4;
int group_ = 8000;
float scale_ = 0.0078125f;

static void parse_opt(int argc, char* argv[]) {
  int opt = 0;
  while ((opt = getopt(argc, argv, "c:g:s:")) != -1) {
    switch (opt) {
      case 'c':
        cls_ = std::stoi(optarg);
        break;
      case 'g':
        group_ = std::stoi(optarg);
        break;
      case 's':
        scale_ = std::stof(optarg);
        break;
      default:
        break;
    }
  }
  return;
}

int main(int argc, char* argv[]) {
  parse_opt(argc, argv);
  auto rd = std::random_device();
  std::mt19937 gen(rd());
  gen.seed(101);
  std::uniform_int_distribution<> dis(-128, 127);

  int cls = cls_;
  int group = group_;
  float scale = scale_;
  // GLOBAL_ENABLE_C_SOFTMAX = 2;
  cout << "cls " << cls << " "      //
       << "group " << group << " "  //
       << "scale " << scale << " "  //
       << std::endl;

  int total = cls * group;
  int8_t* d = new int8_t[total];
  for (int i = 0; i < total; ++i) {
    d[i] = dis(gen);
  }

  float* output = new float[total];
  float* output_neon = new float[total];
  memset(output_neon, 0, sizeof(float)*total);
  __TIC__(softmax_c)
  softmax_c(d, scale, cls, group, output);
  __TOC__(softmax_c)
  __TIC__(softmax_neon)
  switch (cls) {
    case 2:
      softmax2_neon(d, scale, group, output_neon);
      break;
    case 4:
      softmax4_neon(d, scale, group, output_neon);
      break;
    case 8:
      softmax8_neon(d, scale, group, output_neon);
      break;
    default:
      break;
  }
  __TOC__(softmax_neon)
  /*
GLOBAL_ENABLE_C_SOFTMAX = 0;
__TIC__(softmax_neon_old_0)
vitis::ai::softmax(d, scale, cls, group, output_neon);
__TOC__(softmax_neon_old_0)

GLOBAL_ENABLE_C_SOFTMAX = 1;
__TIC__(softmax_neon_old_1)
vitis::ai::softmax(d, scale, cls, group, output_neon);
__TOC__(softmax_neon_old_1)

GLOBAL_ENABLE_C_SOFTMAX = 2;
__TIC__(softmax_neon_old_c)
vitis::ai::softmax(d, scale, cls, group, output_neon);
__TOC__(softmax_neon_old_c)
  */

  for (int i = 0; i < group; ++i) {
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

  if (d != nullptr) delete []d;
  if (output != nullptr) delete []output;
  if (output_neon != nullptr) delete []output_neon;

  return 0;
}
