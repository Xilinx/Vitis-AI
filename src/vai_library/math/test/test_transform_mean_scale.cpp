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
#include <iostream>

#include "../src/image_util.cpp"
using namespace vitis::ai;
int main() {
  std::vector<uint8_t> src(3 * 256);
  std::vector<float> mean = {127, 114, 130};
  std::vector<float> scale = {1, 0.25, 0.0625};
  for (int i = 0; i <= 255; i++) {
    src[i * 3] = i;
    src[i * 3 + 1] = i;
    src[i * 3 + 2] = i;
  }
  std::vector<int8_t> data_0(src.size());
  std::vector<int8_t> data_1(src.size());
  for (int each_mean = 0; each_mean < 25500; each_mean += 1)
    for (int each_scale = -8; each_scale <= 0; each_scale += 1) {
      scale = std::vector<float>(3, pow(2, each_scale));
      mean = std::vector<float>(3, float(each_mean) / 100);
      std::vector<int> scale_int;
      if (!calc_scale(scale, scale_int)) continue;
      transform_rgb(256, 1, src.data(), data_0.data(), mean[0], scale[0],
                    mean[1], scale[1], mean[2], scale[2]);
      transform_mean_scale(256, 1, src.data(), data_1.data(), mean, scale_int,
                           false);
      for (size_t i = 0; i < data_0.size(); i++) {
        if (data_0[i] != data_1[i]) {
          std::cout << "check transform rgb fail " << i << ": (" << int(src[i])
                    << " - " << mean[2 - i % 3] << ") *  " << scale[2 - i % 3]
                    << " = old: " << int(data_0[i]) << ", (" << int(src[i])
                    << " - " << mean[2 - i % 3] << ") << "
                    << scale_int[2 - i % 3] << " = new: " << int(data_1[i])
                    << std::endl;
        }
      }
      transform_bgr(256, 1, src.data(), data_0.data(), mean[0], scale[0],
                    mean[1], scale[1], mean[2], scale[2]);
      transform_mean_scale(256, 1, src.data(), data_1.data(), mean, scale_int,
                           true);
      for (size_t i = 0; i < data_0.size(); i++) {
        if (data_0[i] != data_1[i]) {
          std::cout << "check transform bgr fail " << i << ": (" << int(src[i])
                    << " - " << mean[i % 3] << ") *  " << scale[i % 3]
                    << " = old: " << int(data_0[i]) << ", (" << int(src[i])
                    << " - " << mean[i % 3] << ") << " << scale_int[i % 3]
                    << " = new: " << int(data_1[i]) << std::endl;
        }
      }
    }
  return 0;
}
