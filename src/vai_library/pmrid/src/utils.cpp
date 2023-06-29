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

#include "utils.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>
using namespace std;

// used to KSigma transform
float K(const float iso) {
  float k_coeff[2] = {0.0005995267, 0.00868861};
  return k_coeff[0] * iso + k_coeff[1];
}

// used to KSigma transform
float Sigma(const float iso) {
  float s_coeff[3] = {7.11772e-7, 6.514934e-4, 0.11492713};
  return s_coeff[0] * iso * iso + s_coeff[1] * iso + s_coeff[2];
}

// KSigma transform
float KSigma(const float iso, const float value, const float scale) {
  float ret;
  float anchor = 1600;
  float v = 959.0;
  auto k = K(iso);
  auto sigma = Sigma(iso);
  auto k_a = K(anchor);
  auto sigma_a = Sigma(anchor);
  auto cvt_k = k_a / k;
  auto cvt_b = (sigma / (k * k) - sigma_a / (k_a * k_a)) * k_a;
  ret = (value * v * cvt_k + cvt_b) / v * scale;
  return ret;
}
// bayer to rggb and KSigma transform, combine to 1 operator to improve
// performance
void bayer2rggb_KSigma(const unsigned short* input,
                       std::vector<signed char>& rggb, const float iso,
                       const int rows, const int cols, const int channels,
                       const float scale, const float input_scale) {
  float anchor = 1600;
  float v = 959.0;
  auto k = K(iso);
  auto sigma = Sigma(iso);
  auto k_a = K(anchor);
  auto sigma_a = Sigma(anchor);
  auto cvt_k = k_a / k;
  auto cvt_b = (sigma / (k * k) - sigma_a / (k_a * k_a)) * k_a;

  CHECK((rows % 2) || (cols % 2) == 0) << "rows is not 2N, or cols is not 2N";
  auto new_rows = rows / 2;
  auto new_cols = cols / 2;
  // Normalized    input / 65535
  // KSigma        (input * v * cvt_k + cvt_b) / v
  // Normalized    input * scale
  // fixed         int(input * input_scale + 0.5f),  +0.5f because rounding mode
  // bayer2rggb    R G R G   ->RGGBRGGB
  //               G B G B
  float a = 1.0f / 65535 * v * cvt_k / v * scale * input_scale;
  float b = cvt_b / v * scale * input_scale + 0.5f;
  for (auto h = 0; h < new_rows; ++h) {
    for (auto w = 0; w < new_cols; ++w) {
      auto idx = h * new_cols * channels + w * channels;

      rggb[idx] =
          (int)std::min((float)input[h * 2 * cols + w * 2 + 0] * a + b, 127.0f);
      rggb[idx + 1] =
          (int)std::min((float)input[h * 2 * cols + w * 2 + 1] * a + b, 127.0f);
      rggb[idx + 2] = (int)std::min(
          (float)input[(h * 2 + 1) * cols + w * 2 + 0] * a + b, 127.0f);
      rggb[idx + 3] = (int)std::min(
          (float)input[(h * 2 + 1) * cols + w * 2 + 1] * a + b, 127.0f);
    }
  }
}

// pad operator, pad_value use KSigma and fixed value of 0.0f
void pad(const std::vector<signed char>& input, const int rows, const int cols,
         const int channels, const int ph, const int pw,
         const signed char pad_value, void* data) {
  //  __TIC__(pad)
  auto dst_rows = rows + ph * 2;
  auto dst_cols = cols + pw * 2;
  auto src_row_size = cols * channels;
  auto dst_row_size = dst_cols * channels;
  auto src = input.data();
  signed char* dst = (signed char*)data;
  // pad top and bottom
  std::fill_n(dst, ph * dst_row_size, pad_value);
  std::fill_n(dst + (dst_rows - ph) * dst_row_size, ph * dst_row_size,
              pad_value);
  // pad left and right
  for (auto h = ph; h < dst_rows - ph; h++) {
    auto offset = h * dst_row_size;
    std::fill_n(dst + offset, pw * channels, pad_value);
  }
  for (auto h = ph; h < dst_rows - ph; h++) {
    auto offset = h * dst_row_size + (dst_cols - pw) * channels;
    std::fill_n(dst + offset, pw * channels, pad_value);
  }
  // copy source data
  for (auto h = ph; h < dst_rows - ph; h++) {
    auto src_offset = (h - ph) * src_row_size;
    auto dst_offset = h * dst_row_size + pw * channels;
    std::copy_n(src + src_offset, src_row_size, dst + dst_offset);
  }
  //  __TOC__(pad)
}

void set_input(const cv::Mat& raw, const float iso,
               vitis::ai::library::InputTensor& tensor, int batch_idx) {
  constexpr int height = 3000;
  constexpr int width = 4000;
  constexpr int ph = (32 - ((height / 2) % 32)) / 2;
  constexpr int pw = (32 - ((width / 2) % 32)) / 2;
  auto data = tensor.get_data(batch_idx);
  auto fix_scale = vitis::ai::library::tensor_scale(tensor);
  // reverse order all the data
  cv::Mat raw_flip;
  cv::flip(raw, raw_flip, -1);

  // Normalized, bayer2rggb, KSigma, fixed all the raw data
  auto rggb = std::vector<signed char>(height * width);
  bayer2rggb_KSigma(raw_flip.ptr<ushort>(0), rggb, iso, raw_flip.rows,
                    raw_flip.cols, tensor.channel, 256.0f, fix_scale);

  // pad_value use KSigma and fixed value of 0.0f
  signed char pad_value = (int)std::max(
      std::min(KSigma(iso, 0.0f, 256.0f) * fix_scale + 0.5f, 127.0f), -128.0f);
  pad(rggb, height / 2, width / 2, tensor.channel, ph, pw, pad_value,
      (void*)data);
}

// improve performance
std::vector<float> invKSigma_unpad_rggb2bayer(
    void* output_data, void* input_data, const float output_scale,
    const float input_scale, const int rows, const int cols,
    const int input_width, const int channels, const int ph, const int pw,
    const float iso, const float scale) {
  auto ret = std::vector<float>(rows * cols);
  float anchor = 1600;
  float v = 959.0;
  auto k = K(iso);
  auto sigma = Sigma(iso);
  auto k_a = K(anchor);
  auto sigma_a = Sigma(anchor);
  auto cvt_k = k_a / k;
  auto cvt_b = (sigma / (k * k) - sigma_a / (k_a * k_a)) * k_a;

  signed char* output = (signed char*)output_data;
  signed char* input = (signed char*)input_data;

  // output+input in float    output * output_scale + input * input_scale
  // Normalized               output / scale
  // inverse KSigma           (output * v - cvt_b) / cvt_k / v
  // rggb2bayer               RGGBRGGB  ->  R G R G
  //                                        G B G B

  float a = input_scale / scale * v / cvt_k / v;
  float b = output_scale / scale * v / cvt_k / v;
  float c = cvt_b / cvt_k / v;

  //  float a = input_scale / scale * v;
  //  float b = output_scale / scale * v;

  for (auto h = 0; h < rows; h = h + 2) {
    for (auto w = 0; w < cols; w = w + 2) {
      auto idx =
          input_width * channels * (ph + h / 2) + channels * (pw + w / 2);

      ret[h * cols + w] = (float)input[idx] * a + (float)output[idx] * b - c;
      ret[h * cols + w + 1] =
          (float)input[idx + 1] * a + (float)output[idx + 1] * b - c;
      ret[(h + 1) * cols + w] =
          (float)input[idx + 2] * a + (float)output[idx + 2] * b - c;
      ret[(h + 1) * cols + w + 1] =
          (float)input[idx + 3] * a + (float)output[idx + 3] * b - c;
    }
  }
  return ret;
}
