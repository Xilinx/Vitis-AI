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

#include "util.hpp"

using std::pair;
using std::tuple;
using std::make_tuple;
using std::get;
using Eigen::MatrixXf;
using Eigen::Matrix3f;
using Eigen::VectorXf;
using Eigen::Map;

namespace vitis {
namespace ai {

static float ref_points[] = {
    30.29459953, 65.53179932, 48.02519989, 33.54930115, 62.72990036,
    51.69630051, 51.50139999, 71.73660278, 92.3655014, 92.20410156};

static float ref_matrix[][40] = {{
    30.2946, 65.5318, 48.0252, 33.5493, 62.7299,
    51.6963, 51.5014, 71.7366, 92.3655, 92.2041,
    51.6963, 51.5014, 71.7366, 92.3655, 92.2041,
    -30.2946, -65.5318, -48.0252, -33.5493, -62.7299,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1},
    {-30.2946, -65.5318, -48.0252, -33.5493, -62.7299,
    51.6963, 51.5014, 71.7366, 92.3655, 92.2041,
    51.6963, 51.5014, 71.7366, 92.3655, 92.2041,
    30.2946, 65.5318, 48.0252, 33.5493, 62.7299,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1}};

Matrix3f solve_rotate(MatrixXf A, VectorXf b) {
  VectorXf r = A.colPivHouseholderQr().solve(b);
  Matrix3f t_inv;
  t_inv << r(0), -r(1), 0, r(1), r(0), 0, r(2), r(3), 1;
  Matrix3f t = t_inv.inverse();
  t(0, 2) = 0;
  t(1, 2) = 0;
  t(2, 2) = 1;
  return t;
}

MatrixXf get_rotate_matrix(const vector<float>& points) {

  Map<MatrixXf> m_points(const_cast<float *>(points.data()), 2, 5);
  MatrixXf m_points_t = m_points.transpose();
  Map<VectorXf> v_points(m_points_t.data(), m_points_t.size());

  Map<MatrixXf> m_ref0(ref_matrix[0], 10, 4);
  Map<MatrixXf> m_ref1(ref_matrix[1], 10, 4);

  Matrix3f m_t0 = solve_rotate(m_ref0, v_points);
  Matrix3f m_t1 = solve_rotate(m_ref1, v_points);
  m_t1 << -m_t1.leftCols(1), m_t1.rightCols(2);

  MatrixXf m_points_ext(5, 3);
  m_points_ext << m_points_t, MatrixXf::Ones(5, 1);

  MatrixXf m_pt0 = m_points_ext * m_t0;
  MatrixXf m_pt1 = m_points_ext * m_t1;

  Map<MatrixXf> m_ref_points(ref_points, 5, 2);

  float norm0 = (m_pt0.block(0, 0, 5, 2) - m_ref_points).norm();
  float norm1 = (m_pt1.block(0, 0, 5, 2) - m_ref_points).norm();

  MatrixXf m_rotate = (norm0 <= norm1) ?
      m_t0.block(0, 0, 3, 2).transpose() : m_t1.block(0, 0, 3, 2).transpose();

  return m_rotate;
}

}
}

