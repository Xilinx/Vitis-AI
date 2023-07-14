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
#include <cstring>
#include <typeinfo>

#include "vitis/ai/ssd_normalizer.hpp"

using std::memset;
using std::pow;
using std::round;
using std::sqrt;

namespace vitis {
namespace ai {

SSDNormalizer::SSDNormalizer(bool across_spatial, bool channel_shared,
                             int height, int width, int channel,
                             int output_fix_pos, float eps)
    : across_spatial_(across_spatial),
      channel_shared_(channel_shared),
      height_(height),
      width_(width),
      channel_(channel),
      output_fix_pos_(output_fix_pos),
      eps_(eps),
      scale_(nullptr),
      norm_buf_(nullptr),
      scale_buf_(nullptr) {
  spatial_dim_ = height_ * width_;
  num_ = spatial_dim_ * channel_;
  /*!
   *  norm / scale          | across_spatial (true) | across_spatial (false)
   * channel_shared (true)  |         1 / 1         |    spatial_dim / 1
   * channel_shared (false) |      1 / channel      | spatial_dim / channel
   */
  scale_ = new float[channel_shared_ ? 1 : channel_];
  norm_buf_ = new float[spatial_dim_];
  scale_buf_ = new float[channel_shared_ ? 1 : channel_];
}

SSDNormalizer::SSDNormalizer(bool across_spatial, bool channel_shared,
                             int height, int width, int channel,
                             int output_fix_pos)
    : SSDNormalizer(across_spatial, channel_shared, height, width, channel,
                    output_fix_pos, EPS) {}

SSDNormalizer::~SSDNormalizer() {
  if (norm_buf_ != nullptr) {
    delete[] norm_buf_;
    norm_buf_ = nullptr;
  }
  if (scale_ != nullptr) {
    delete[] scale_;
    scale_ = nullptr;
  }
  if (scale_buf_ != nullptr) {
    delete[] scale_buf_;
    scale_buf_ = nullptr;
  }
}

void SSDNormalizer::loadScaleParam(const int8_t *scale, int scale_fix_pos) {
  float shift = pow(2, output_fix_pos_ - scale_fix_pos);
  for (int i = 0; i < (channel_shared_ ? 1 : channel_); ++i) {
    scale_[i] = shift * scale[i];
  }
}

void SSDNormalizer::loadScaleParam(const float *scale) {
  float shift = pow(2, output_fix_pos_);
  for (int i = 0; i < (channel_shared_ ? 1 : channel_); ++i) {
    scale_[i] = shift * scale[i];
  }
}

template <typename T>
void SSDNormalizer::normalize(const int8_t *input, T *output) {
  if (across_spatial_) {
    float norm = 0.f;
    for (int i = 0; i < num_; ++i) {
      norm += input[i] * input[i];
    }
    norm = sqrt(norm + eps_);
    if (channel_shared_) {
      norm = scale_[0] / norm;
      for (int i = 0; i < num_; ++i) {
        if (typeid(T) == typeid(int8_t)) {
          output[i] = round(norm * input[i]);
        } else {
          output[i] = norm * input[i];
        }
      }
    } else {
      for (int c = 0; c < channel_; ++c) {
        scale_buf_[c] = scale_[c] / norm;
      }
      for (int i = 0; i < spatial_dim_; ++i) {
        for (int c = 0; c < channel_; ++c) {
          if (typeid(T) == typeid(int8_t)) {
            output[c] = round(scale_buf_[c] * input[c]);
          } else {
            output[c] = scale_buf_[c] * input[c];
          }
        }
        input += channel_;
        output += channel_;
      }
    }
  } else {
    const int8_t *pi = input;
    memset(norm_buf_, 0, spatial_dim_ * sizeof(float));
    for (int i = 0; i < spatial_dim_; ++i) {
      for (int c = 0; c < channel_; ++c) {
        norm_buf_[i] += pi[c] * pi[c];
      }
      pi += channel_;
    }
    for (int i = 0; i < spatial_dim_; ++i) {
      norm_buf_[i] = 1.f / sqrt(norm_buf_[i] + eps_);
    }
    if (channel_shared_) {
      for (int i = 0; i < spatial_dim_; ++i) {
        norm_buf_[i] = scale_[0] * norm_buf_[i];
      }
      for (int i = 0; i < spatial_dim_; ++i) {
        for (int c = 0; c < channel_; ++c) {
          if (typeid(T) == typeid(int8_t)) {
            output[c] = round(norm_buf_[i] * input[c]);
          } else {
            output[c] = norm_buf_[i] * input[c];
          }
        }
        input += channel_;
        output += channel_;
      }
    } else {
      for (int i = 0; i < spatial_dim_; ++i) {
        for (int c = 0; c < channel_; ++c) {
          if (typeid(T) == typeid(int8_t)) {
            output[c] = round(scale_[c] * norm_buf_[i] * input[c]);
          } else {
            output[c] = scale_[c] * norm_buf_[i] * input[c];
          }
        }
        input += channel_;
        output += channel_;
      }
    }
  }
}

template void SSDNormalizer::normalize(const int8_t *input, int8_t *output);
template void SSDNormalizer::normalize(const int8_t *input, float *output);

}  // namespace ai
}  // namespace vitis
