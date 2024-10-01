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

#include "../include/vitis/softmax.hpp"

using namespace vitis::ai;

class TestSoftmaxG : public ::testing::Test {
 public:
  TestSoftmaxG() {}
  virtual ~TestSoftmaxG() {}

  void SetUp() {}

  void TearDown() {}

 protected:
  int8_t input_int8[14]{-4, -2, 1, 4, 6, 10, 14, 2, 3, 10, 4, 6, 12, 14};
  float scale = 0.5;
  float input_float[14]{-2, -1, 0.5, 2, 3, 5, 7, 1, 1.5, 5, 2, 3, 6, 7};
  float output_gt[14]{
      1.06172563e-4, 2.88606949e-4, 1.29344661e-3, 5.79682553e-3, 0.0157574055,
      0.116432353,   0.86032519,    0.001615,      0.00266268,    0.08817585,
      0.00439002,    0.0119333,     0.23968682,    0.65153633};
};

TEST_F(TestSoftmaxG, TestInt8_7x2) {
  unsigned int cls = 7;
  unsigned int group = 2;
  unsigned int size = cls * group;

  float output[size];

  softmax_c(input_int8, scale, cls, group, output);

  for (unsigned int i = 0; i < size; ++i) {
    EXPECT_NEAR(output_gt[i], output[i], 1e-5 * output_gt[i]);
  }
}

TEST_F(TestSoftmaxG, TestFloat_7x2) {
  unsigned int cls = 7;
  unsigned int group = 2;
  unsigned int size = cls * group;

  float output[size];

  softmax_c(input_float, cls, group, output);

  for (unsigned int i = 0; i < size; ++i) {
    EXPECT_NEAR(output_gt[i], output[i], 1e-5 * output_gt[i]);
  }
}

#ifdef ENABLE_NEON
class TestSoftmax2 : public ::testing::Test {
 public:
  TestSoftmax2() {}
  virtual ~TestSoftmax2() {}

  void SetUp() {
    for (unsigned int i = 0; i < 32; ++i) {
      input[i] = i - 10;
    }
  }

  void TearDown() {}

 protected:
  int8_t input[32];
};

TEST_F(TestSoftmax2, Test2x16) {
  unsigned int group = 16;
  unsigned int size = 2 * group;
  float scale = 0.5;

  float output_gt[size];
  float output[size];

  softmax_c(input, scale, 2, group, output_gt);
  softmax2_neon(input, scale, group, output);

  for (unsigned int i = 0; i < size; ++i) {
    EXPECT_NEAR(output_gt[i], output[i], 1e-2 * output_gt[i]);
  }
}

TEST_F(TestSoftmax2, Test2x13) {
  unsigned int group = 13;
  unsigned int size = 2 * group;
  float scale = 0.5;

  float output_gt[size];
  float output[size];

  softmax_c(input, scale, 2, group, output_gt);
  softmax2_neon(input, scale, group, output);

  for (unsigned int i = 0; i < size; ++i) {
    EXPECT_NEAR(output_gt[i], output[i], 1e-2 * output_gt[i]);
  }
}

TEST_F(TestSoftmax2, Test2x7) {
  unsigned int group = 7;
  unsigned int size = 2 * group;
  float scale = 0.5;

  float output_gt[size];
  float output[size];

  softmax_c(input, scale, 2, group, output_gt);
  softmax2_neon(input, scale, group, output);

  for (unsigned int i = 0; i < size; ++i) {
    EXPECT_NEAR(output_gt[i], output[i], 1e-2 * output_gt[i]);
  }
}

TEST_F(TestSoftmax2, Test2x16_o1) {
  unsigned int group = 16;
  unsigned int size = 2 * group;
  float scale = 0.5;

  float output_gt[size];
  float output[size];

  softmax_c(input, scale, 2, group, output_gt);
  softmax2_o1_neon(input, scale, group, output);

  for (unsigned int i = 0; i < group; ++i) {
    EXPECT_NEAR(output_gt[2 * i + 1], output[i], 1e-2 * output_gt[i]);
  }
}

TEST_F(TestSoftmax2, Test_v2_o1) {
  unsigned int group = 8;
  unsigned int size = 2 * group;
  float scale = 0.5;

  float output_gt[size];
  float output[size];

  int8_t v0[group];
  int8_t v1[group];

  softmax2_o1_neon(input, scale, group, output_gt);
  for (unsigned i = 0; i < group; i++) {
    v0[i] = input[2 * i];
    v1[i] = input[2 * i + 1];
  }

  int8x8_t d0 = vld1_s8(v0);
  int8x8_t d1 = vld1_s8(v1);

  float32x4x2_t r = softmax2_vector_o1_neon(d0, d1, scale);
  vst1q_f32(output, r.val[0]);
  vst1q_f32(output + 4, r.val[1]);

  for (unsigned int i = 0; i < group; ++i) {
    EXPECT_EQ(output_gt[i], output[i]);
  }
}

TEST_F(TestSoftmax2, Test2x13_o1) {
  unsigned int group = 13;
  unsigned int size = 2 * group;
  float scale = 0.5;

  float output_gt[size];
  float output[size];

  softmax_c(input, scale, 2, group, output_gt);
  softmax2_o1_neon(input, scale, group, output);

  for (unsigned int i = 0; i < group; ++i) {
    EXPECT_NEAR(output_gt[2 * i + 1], output[i], 1e-2 * output_gt[i]);
  }
}

class TestSoftmax4 : public ::testing::Test {
 public:
  TestSoftmax4() {}
  virtual ~TestSoftmax4() {}

  void SetUp() {
    for (unsigned int i = 0; i < 64; ++i) {
      input[i] = i - 20;
    }
  }

  void TearDown() {}

 protected:
  int8_t input[64];
};

TEST_F(TestSoftmax4, Test4x16) {
  unsigned int group = 16;
  unsigned int size = 4 * group;
  float scale = 0.5;

  float output_gt[size];
  float output[size];

  softmax_c(input, scale, 4, group, output_gt);
  softmax4_neon(input, scale, group, output);

  for (unsigned int i = 0; i < size; ++i) {
    EXPECT_NEAR(output_gt[i], output[i], 1e-2 * output_gt[i]);
  }
}

TEST_F(TestSoftmax4, Test4x13) {
  unsigned int group = 13;
  unsigned int size = 4 * group;
  float scale = 0.5;

  float output_gt[size];
  float output[size];

  softmax_c(input, scale, 4, group, output_gt);
  softmax4_neon(input, scale, group, output);

  for (unsigned int i = 0; i < size; ++i) {
    EXPECT_NEAR(output_gt[i], output[i], 1e-2 * output_gt[i]);
  }
}

TEST_F(TestSoftmax4, Test4x7) {
  unsigned int group = 7;
  unsigned int size = 4 * group;
  float scale = 0.5;

  float output_gt[size];
  float output[size];

  softmax_c(input, scale, 4, group, output_gt);
  softmax4_neon(input, scale, group, output);

  for (unsigned int i = 0; i < size; ++i) {
    EXPECT_NEAR(output_gt[i], output[i], 1e-2 * output_gt[i]);
  }
}
#endif
