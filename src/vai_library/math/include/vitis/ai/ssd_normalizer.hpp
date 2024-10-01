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
#ifndef DPMATH_SSD_NORMALIZER_HPP_
#define DPMATH_SSD_NORMALIZER_HPP_

#include <cstdint>

namespace vitis {
namespace ai {

class SSDNormalizer {
 public:
  SSDNormalizer(bool across_spatial, bool channel_shared, int height, int width,
                int channel, int output_fix_pos, float eps);
  SSDNormalizer(bool across_spatial, bool channel_shared, int height, int width,
                int channel, int output_fix_pos);

  virtual ~SSDNormalizer();

  void loadScaleParam(const int8_t* scale, int scale_fix_pos);
  void loadScaleParam(const float* scale);

  template <typename T>
  void normalize(const int8_t* input, T* output);

  int normalize_neon(const int8_t* input, int8_t* output);

 protected:
  static constexpr float EPS = 1e-8;

  bool across_spatial_;
  bool channel_shared_;

  int height_;
  int width_;
  int channel_;

  int output_fix_pos_;

  float eps_;

  int spatial_dim_;
  int num_;

  float* scale_;  // initialization only once
  float* norm_buf_;
  float* scale_buf_;
};

}  // namespace ai
}  // namespace vitis

#endif
