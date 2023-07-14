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
#include <gtest/gtest.h>
#include <math.h>

#include <algorithm>

#include "time.hpp"
#include "vitis/ai/ssd_normalizer.hpp"

using std::fill_n;

using namespace vitis::ai;

class TestNormalizeG : public ::testing::Test {
 public:
  TestNormalizeG() {}
  virtual ~TestNormalizeG() {}

  void SetUp() {}

  void TearDown() {}

 protected:
  int8_t input_int8[27]{58, 21,  0, 83, 17,  0,  119, 41, 0, 78, 0,  0,  64, 0,
                        0,  110, 0, 35, 121, 60, 17,  92, 0, 0,  39, 30, 0};
  float scale[3]{1.325, 0.938, 1.551};
  float output_gt[27]{1.2459,  0.31933, 0,     1.2981, 0.1882, 0,       1.2527,
                      0.30555, 0,       1.325, 0,      0,      1.325,   0,
                      0,       1.2626,  0,     0.4703, 1.1778, 0.41344, 0.1937,
                      1.325,   0,       0,     1.0502, 0.5719, 0};
  float output_gt_1[27]{
      1.2459, 0.45108, 0,       1.2981, 0.26585, 0, 1.2527, 0.43161, 0,
      1.325,  0,       0,       1.325,  0,       0, 1.2626, 0,       0.40177,
      1.1778, 0.58402, 0.16548, 1.325,  0,       0, 1.0502, 0.80785, 0};
  float output_gt_2[27]{0.27214, 0.069754, 0,       0.38944, 0.056468, 0,
                        0.55836, 0.13619,  0,       0.36598, 0,        0,
                        0.30029, 0,        0,       0.51613, 0,        0.19223,
                        0.56774, 0.1993,   0.09337, 0.43167, 0,        0,
                        0.18299, 0.099649, 0};
  float output_gt_3[27]{0.27214, 0.098533, 0,        0.38944, 0.079765, 0,
                        0.55836, 0.19237,  0,        0.36598, 0,        0,
                        0.30029, 0,        0,        0.51613, 0,        0.16422,
                        0.56774, 0.28152,  0.079765, 0.43167, 0,        0,
                        0.18299, 0.14076,  0};
};

// no across spatial, no channel shared
TEST_F(TestNormalizeG, Test_NAS_NCS_float) {
  bool across_spatial = false;
  bool channel_shared = false;
  int height = 3;
  int width = 3;
  int channel = 3;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  float output[27];
  normalizer.normalize(input_int8, output);

  float output_float[27];
  for (unsigned int i = 0; i < 27; ++i) {
    output_float[i] = output[i] / 32.0;
    EXPECT_NEAR(output_gt[i], output_float[i], 1e-4 * output_gt[i]);
  }
}

TEST_F(TestNormalizeG, Test_NAS_NCS_int8) {
  bool across_spatial = false;
  bool channel_shared = false;
  int height = 3;
  int width = 3;
  int channel = 3;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  int8_t output[27];
  normalizer.normalize(input_int8, output);

  float output_float[27];
  for (unsigned int i = 0; i < 27; ++i) {
    output_float[i] = output[i] / 32.0;
    EXPECT_NEAR(output_gt[i], output_float[i], 0.05 * output_gt[i]);
  }
}

// no across spatial, channel shared
TEST_F(TestNormalizeG, Test_NAS_CS_float) {
  bool across_spatial = false;
  bool channel_shared = true;
  int height = 3;
  int width = 3;
  int channel = 3;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  float output[27];
  normalizer.normalize(input_int8, output);

  float output_float[27];
  for (unsigned int i = 0; i < 27; ++i) {
    output_float[i] = output[i] / 32.0;
    EXPECT_NEAR(output_gt_1[i], output_float[i], 1e-4 * output_gt[i]);
  }
}

// across spatial, no channel shared
TEST_F(TestNormalizeG, Test_AS_NCS_float) {
  bool across_spatial = true;
  bool channel_shared = false;
  int height = 3;
  int width = 3;
  int channel = 3;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  float output[27];
  normalizer.normalize(input_int8, output);

  float output_float[27];
  for (unsigned int i = 0; i < 27; ++i) {
    output_float[i] = output[i] / 32.0;
    EXPECT_NEAR(output_gt_2[i], output_float[i], 1e-4 * output_gt[i]);
  }
}

// across spatial, channel shared
TEST_F(TestNormalizeG, Test_AS_CS_float) {
  bool across_spatial = true;
  bool channel_shared = true;
  int height = 3;
  int width = 3;
  int channel = 3;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  float output[27];
  normalizer.normalize(input_int8, output);

  float output_float[27];
  for (int i = 0; i < 27; ++i) output_float[i] = output[i] / 32.0;

  for (unsigned int i = 0; i < 27; ++i) {
    EXPECT_NEAR(output_gt_3[i], output_float[i], 1e-4 * output_gt[i]);
  }
}

#ifdef ENABLE_NEON
extern float sum_neon(const float *input, unsigned int size);
extern float square_sum_neon(const int8_t *input, unsigned int size);
extern void scalar_prod_neon(float scalar, const int8_t *input,
                             unsigned int size, int8_t *output);
// extern void scalar_prod_neon_8(float scalar, const int8_t* input,
//                             unsigned int size, int8_t* output);
extern void dot_prod_neon(const float *scale, const int8_t *input,
                          unsigned int size, int8_t *output);
extern int dot_2d_prod_neon(const float *scale_w, const float *scale_h,
                            const int8_t *input, unsigned int width,
                            unsigned int height, int8_t *output);

// Do not ask for the exact on armv7 because vrndaq_f32 is not defined
template <typename T>
void expect_equal_or_near(T a, T b) {
#ifdef __ARM_ARCH_7A__
  EXPECT_NEAR(a, b, 1);
#else
  EXPECT_EQ(a, b);
#endif
}

TEST(TestNormalizeUnit, Sum_3) {
  float input[3]{1.5, -2.5, 1};

  float sum = sum_neon(input, 3);
  float sum_gt = 0.f;
  for (int i = 0; i < 3; ++i) {
    sum_gt += input[i];
  }
  expect_equal_or_near(sum_gt, sum);
}

TEST(TestNormalizeUnit, Sum_4) {
  float input[4]{1.5, -2.5, 1, 3.2};

  float sum = sum_neon(input, 4);
  float sum_gt = 0.f;
  for (int i = 0; i < 4; ++i) {
    sum_gt += input[i];
  }
  expect_equal_or_near(sum_gt, sum);
}

TEST(TestNormalizeUnit, Sum_9) {
  float input[9]{1.5, -2.5, 1, 3.2, 1.3, 2.7, 3.1, 1.0, -0.01};

  float sum = sum_neon(input, 9);
  float sum_gt = 0.f;
  for (int i = 0; i < 9; ++i) {
    sum_gt += input[i];
  }
  expect_equal_or_near(sum_gt, sum);
}

TEST(TestNormalizeUnit, SquareSum) {
  int8_t input[19]{1,  3,  5,   4,  2,   7, 8,  19, 100, 21,
                   70, 10, 101, 13, 121, 6, 21, 12, 19};

  float sum = square_sum_neon(input, 19);

  float sum_gt = 0;
  for (int i = 0; i < 19; ++i) {
    sum_gt += input[i] * input[i];
  }

  expect_equal_or_near(sum_gt, sum);
}

TEST(TestNormalizeUnit, ScalarProd) {
  int8_t input[19]{29,  41, 59, 39, 32, 55, 61,  46, 19, 10,
                   -20, 30, 0,  15, 25, 31, -50, -7, 11};
  int8_t output[19];
  int8_t output_gt[19];
  float scalar = 1.311;

  scalar_prod_neon(scalar, input, 19, output);
  for (int i = 0; i < 19; ++i) {
    output_gt[i] = round(scalar * input[i]);
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

/*
TEST(TestNormalizeUnit, ScalarProdPerf_2M) {
  int num = 512*64*64;
  int8_t* input = new int8_t[num];
  int8_t* output = new int8_t[num];
  fill_n(input, num, 25);
  float scalar = 1.311;

  scalar_prod_neon_8(scalar, input, num, output);

  delete[] output;
  delete[] input;
}
*/

TEST(TestNormalizeUnit, DotProd) {
  int8_t input[19]{29,  41, 59, 39, 32, 55, 61,  46, 19, 10,
                   -20, 30, 0,  15, 25, 31, -50, -7, 11};
  int8_t output[19];
  int8_t output_gt[19];
  float scale[19] = {1.311, 1.1,  0.9,  -0.766, 0.543, 1.027, 0.989,
                     0.57,  0.87, -0.7, 2.11,   1.3,   1.07,  0.92,
                     0.85,  0.75, 0.99, 1.0,    1.11};

  dot_prod_neon(scale, input, 19, output);
  for (int i = 0; i < 19; ++i) {
    output_gt[i] = round(scale[i] * input[i]);
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

TEST(TestNormalizeUnit, DotProd2D) {
  int8_t input[48];
  int8_t output[48];
  int8_t output_gt[48];
  float scale_w[16] = {1.311, 1.1, 0.9,  -0.766, 0.543, 1.027, 0.989, -0.7,
                       2.11,  1.3, 1.07, 0.92,   0.85,  0.75,  0.99,  1.0};
  float scale_h[3] = {1.1, -0.9, 1.2};

  for (int i = 0; i < 48; ++i) {
    input[i] = 40 - i;
  }
  dot_2d_prod_neon(scale_w, scale_h, input, 16, 3, output);
  for (int h = 0; h < 3; ++h) {
    for (int w = 0; w < 16; ++w) {
      int i = 16 * h + w;
      output_gt[i] = round(scale_h[h] * scale_w[w] * input[i]);
      expect_equal_or_near(output_gt[i], output[i]);
    }
  }
}

class TestNormalizeNEON : public ::testing::Test {
 public:
  TestNormalizeNEON() {}
  virtual ~TestNormalizeNEON() {}

  void SetUp() {
    for (int i = 0; i < 144; ++i) {
      input[i] = (120 - i) / 2;
    }
  }

  void TearDown() {}

 protected:
  int8_t input[144];
  int8_t output[144];
  int8_t output_gt[144];
  float scale[16] = {1.311, 1.1, 0.9,  -0.766, 0.543, 1.027, 0.989, -0.7,
                     2.11,  1.3, 1.07, 0.92,   0.85,  0.75,  0.99,  1.0};
};

// no across spatial, no channel shared
TEST_F(TestNormalizeNEON, Test_NAS_NCS_8x1x1) {
  bool across_spatial = false;
  bool channel_shared = false;
  int height = 1;
  int width = 1;
  int channel = 8;
  int output_fix_pos = 6;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  normalizer.normalize_neon(input, output);
  normalizer.normalize(input, output_gt);

  for (unsigned int i = 0; i < 8; ++i) {
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

TEST_F(TestNormalizeNEON, Test_NAS_NCS_16x1x1) {
  bool across_spatial = false;
  bool channel_shared = false;
  int height = 1;
  int width = 1;
  int channel = 16;
  int output_fix_pos = 6;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  normalizer.normalize_neon(input, output);
  normalizer.normalize(input, output_gt);

  for (unsigned int i = 0; i < 16; ++i) {
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

TEST_F(TestNormalizeNEON, Test_NAS_NCS_16x2x1) {
  bool across_spatial = false;
  bool channel_shared = false;
  int channel = 16;
  int height = 2;
  int width = 1;
  int output_fix_pos = 6;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  normalizer.normalize_neon(input, output);
  normalizer.normalize(input, output_gt);

  for (int i = 0; i < channel * height * width; ++i) {
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

TEST_F(TestNormalizeNEON, Test_NAS_NCS_8x2x2) {
  bool across_spatial = false;
  bool channel_shared = false;
  int channel = 8;
  int height = 2;
  int width = 2;
  int output_fix_pos = 6;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  normalizer.normalize_neon(input, output);
  normalizer.normalize(input, output_gt);

  for (int i = 0; i < channel * height * width; ++i) {
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

TEST_F(TestNormalizeNEON, Test_NAS_NCS_16x3x3) {
  bool across_spatial = false;
  bool channel_shared = false;
  int height = 3;
  int width = 3;
  int channel = 16;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  normalizer.normalize_neon(input, output);
  normalizer.normalize(input, output_gt);

  for (int i = 0; i < channel * height * width; ++i) {
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

// no across spatial, channel shared
TEST_F(TestNormalizeNEON, Test_NAS_CS_8x1x1) {
  bool across_spatial = false;
  bool channel_shared = true;
  int height = 1;
  int width = 1;
  int channel = 8;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  normalizer.normalize_neon(input, output);
  normalizer.normalize(input, output_gt);

  for (int i = 0; i < channel * height * width; ++i) {
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

TEST_F(TestNormalizeNEON, Test_NAS_CS_16x3x3) {
  bool across_spatial = false;
  bool channel_shared = true;
  int height = 3;
  int width = 3;
  int channel = 16;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  normalizer.normalize_neon(input, output);
  normalizer.normalize(input, output_gt);

  for (int i = 0; i < channel * height * width; ++i) {
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

// across spatial, no channel shared
TEST_F(TestNormalizeNEON, Test_AS_NCS_16x3x3) {
  bool across_spatial = true;
  bool channel_shared = false;
  int height = 3;
  int width = 3;
  int channel = 16;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  normalizer.normalize_neon(input, output);
  normalizer.normalize(input, output_gt);

  for (int i = 0; i < channel * height * width; ++i) {
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

// across spatial, channel shared
TEST_F(TestNormalizeNEON, Test_AS_CS_16x3x3) {
  bool across_spatial = true;
  bool channel_shared = true;
  int height = 3;
  int width = 3;
  int channel = 16;
  int output_fix_pos = 5;

  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);

  normalizer.normalize_neon(input, output);
  normalizer.normalize(input, output_gt);

  for (int i = 0; i < channel * height * width; ++i) {
    expect_equal_or_near(output_gt[i], output[i]);
  }
}

class TestNormalizePerf : public ::testing::Test {
 public:
  TestNormalizePerf() {}
  virtual ~TestNormalizePerf() {}

  void SetUp() {
    height = 64;
    width = 64;
    channel = 512;
    output_fix_pos = 5;
    int num = height * width * channel;
    input = new int8_t[num];
    output = new int8_t[num];
    scale = new float[channel];
    fill_n(input, num, 25);
    fill_n(scale, channel, 1.1);
  }

  void TearDown() {
    if (input != nullptr) delete[] input;
    if (output != nullptr) delete[] output;
    if (scale != nullptr) delete[] scale;
  }

 protected:
  int height=0;
  int width=0;
  int channel=0;
  int output_fix_pos=0;
  int8_t *input=nullptr;
  int8_t *output=nullptr;
  float *scale=nullptr;
};

TEST_F(TestNormalizePerf, Test_Perf_NEON_AS_NCS_512x64x64) {
  bool across_spatial = true;
  bool channel_shared = false;
  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);
  __TIC__(neon)
  normalizer.normalize_neon(input, output);
  __TOC__(neon)
}

TEST_F(TestNormalizePerf, Test_Perf_C_AS_NCS_512x64x64) {
  bool across_spatial = true;
  bool channel_shared = false;
  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);
  __TIC__(c)
  normalizer.normalize(input, output);
  __TOC__(c)
}

TEST_F(TestNormalizePerf, Test_Perf_NEON_NAS_CS_512x64x64) {
  bool across_spatial = false;
  bool channel_shared = true;
  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);
  __TIC__(neon)
  normalizer.normalize_neon(input, output);
  __TOC__(neon)
}

TEST_F(TestNormalizePerf, Test_Perf_C_NAS_CS_512x64x64) {
  bool across_spatial = false;
  bool channel_shared = true;
  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);
  __TIC__(c)
  normalizer.normalize(input, output);
  __TOC__(c)
}

TEST_F(TestNormalizePerf, Test_Perf_NEON_AS_CS_512x64x64) {
  bool across_spatial = true;
  bool channel_shared = true;
  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);
  __TIC__(neon)
  normalizer.normalize_neon(input, output);
  __TOC__(neon)
}

TEST_F(TestNormalizePerf, Test_Perf_C_AS_CS_512x64x64) {
  bool across_spatial = true;
  bool channel_shared = true;
  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);
  __TIC__(c)
  normalizer.normalize(input, output);
  __TOC__(c)
}

TEST_F(TestNormalizePerf, Test_Perf_NEON_NAS_NCS_512x64x64) {
  bool across_spatial = false;
  bool channel_shared = false;
  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);
  __TIC__(neon)
  normalizer.normalize_neon(input, output);
  __TOC__(neon)
}

TEST_F(TestNormalizePerf, Test_Perf_C_NAS_NCS_512x64x64) {
  bool across_spatial = false;
  bool channel_shared = false;
  SSDNormalizer normalizer(across_spatial, channel_shared, height, width,
                           channel, output_fix_pos);
  normalizer.loadScaleParam(scale);
  __TIC__(c)
  normalizer.normalize(input, output);
  __TOC__(c)
}

#endif
