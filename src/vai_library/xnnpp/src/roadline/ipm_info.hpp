/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License,
 * Version 2.0 (the "License");
 * you may not use this file except in
 * compliance with the License.
 * You may obtain a copy of the License at
 *
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by
 * applicable law or agreed to in writing, software
 * distributed under the
 * License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the
 * specific language governing permissions and
 * limitations under the
 * License.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace std;

namespace vitis {
namespace nnpp {
namespace roadline {

class IpmInfo {
 public:
  IpmInfo(int ratio, float ipm_width, float ipm_height, float ipm_left,
          float ipm_right, float ipm_top, float ipm_bottom,
          float ipm_interpolation, float ipm_vp_portion, float focal_length_x,
          float focal_length_y, float optical_center_x, float optical_center_y,
          float camera_height, float pitch, float yaw);
  ~IpmInfo();

  void IPM(const vector<int>& img, vector<int>& outImage);
  void Recover(const vector<int>& img, vector<int>& outImage);
  int initialize_ipm();

  const float PI = 3.1415926536f;

 private:
  vector<vector<float>> TransformImage2Ground(const vector<vector<float>>& iv);
  vector<vector<float>> GetVanishingPoint();
  vector<vector<float>> TransformGround2Image(const vector<vector<float>>& iv);
  template <typename T>
  vector<vector<T>> dot(const vector<vector<T>>& iv1, T i2);
  template <typename T>
  vector<vector<T>> dot(const vector<vector<T>>& iv1,
                        const vector<vector<T>>& iv2);

  inline float veci(int i) { return float(i / (int)ipm_width_); }
  inline float vecj(int i) { return float(i % (int)ipm_width_); }

  // IpmInfo
  // int ratio_;
  float ipm_width_;
  float ipm_height_;
  float ipm_left_;
  float ipm_right_;
  float ipm_top_;
  float ipm_bottom_;
  // float ipm_interpolation_;
  // float ipm_vp_portion_;
  // camera Info
  float focal_length_x_;
  float focal_length_y_;
  float optical_center_x_;
  float optical_center_y_;
  float camera_height_;
  float pitch_;
  float yaw_;

  static vector<int> vx1;
  static vector<int> vy1;
};

}  // namespace roadline
}  // namespace nnpp
}  // namespace vitis
