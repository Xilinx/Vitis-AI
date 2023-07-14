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

#include "util.hpp"

using namespace vitis::ai::facerecog;

TEST(TestUtil, TestRotate) {
  vector<float> points{0.34375, 0.46875, 0.6875, 0.328125,
      0.578125, 0.648438, 0.492188, 0.773438, 0.828125, 0.671875};
  float width = 87;
  float height = 113;

  int i = 0;
  while (i < 10) {
    points[i] = points[i] * width;
    ++i;
    points[i] = points[i] * height;
    ++i;
  }

  Eigen::MatrixXf m_rotate = get_rotate_matrix(points);

  vector<float> results{0.96058, -0.37110, 23.303, 0.37110, 0.96058, -9.7696};
  EXPECT_NEAR(results[0], m_rotate(0, 0), 1e-4*results[0]);
  EXPECT_NEAR(results[1], m_rotate(0, 1), 1e-4*(-results[1]));
  EXPECT_NEAR(results[2], m_rotate(0, 2), 1e-4*results[2]);
  EXPECT_NEAR(results[3], m_rotate(1, 0), 1e-4*results[3]);
  EXPECT_NEAR(results[4], m_rotate(1, 1), 1e-4*results[4]);
  EXPECT_NEAR(results[5], m_rotate(1, 2), 1e-4*(-results[5]));
}
