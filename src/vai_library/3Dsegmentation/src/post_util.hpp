/*
 * Copyright 2019 xilinx Inc.
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
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <string.h>
#include "float.h"
#include <sstream>
#include <algorithm>

namespace vitis {
namespace ai {
namespace Segmentation3DPost {
using namespace std;

template<typename T>
std::vector<int> argmax(const std::vector<T>& input, std::vector<int> dim) {
  auto H = dim[1];
  auto W = dim[2];
	std::vector<int> index(H);
  //cout << H << " " << W << endl;
	for (auto i = 0; i < H; i++) {
		auto offset = i*W;
    auto start = input.begin() + (size_t)offset + 1u;
    auto end = input.begin() + (size_t)offset + W - 1u;
    std::vector<int> temp(start, end);
    auto cur = std::max_element(temp.begin(), temp.end());
    auto arg = std::distance(temp.begin(), cur);
    index[i] = arg + 1u;
	}	
	return index;	
}


template<typename T1, typename T2, typename T3>
void scatter_add_(std::vector<T1>& self, const std::vector<T2>& index, const std::vector<T3>& other, std::vector<int> shape) {
  auto C = shape[0];
  auto H = shape[1];
  auto W = shape[2];
  for (auto c = 0; c < C; c++) {
    for (auto h = 0; h < H; h++) {
      for (auto w = 0; w < W; w++) {
        auto index_chw = c*H*W + h*W + w;
        auto self_chw = c*H*W + index[index_chw]*W + w;
        self[self_chw] += other[index_chw];
      }
    }
  }
}


template <typename T1, typename T2>
std::vector<T1> gather_1(std::vector<T1>& input, std::vector<T2>& index, int index_channel, int index_height, int index_width) {
  std::vector<T1> dst;
  dst.resize(index.size());
  //cout << index_channel << "/" << index_height << "/" << index_width << endl;
  for (auto i = 0; i < index_channel; i++) {
    for (auto j = 0; j < index_height; j++) {
      for (auto k = 0; k < index_width; k++) {
        auto index_ijk = i*index_height*index_width + j*index_width + k;
        dst[index_ijk] = input[i*index_height*index_width + index[index_ijk]*index_width + k];
      }
    }
  }
  return dst;
}


template <typename T>
std::vector<T> permute(const std::vector<T>& input, size_t C, size_t H, size_t W) {
  std::vector<T> output(input.size());
  //cout << "permute " << input.size() << " " << C << " " << H << " " << W << endl;
  for (auto c = 0u; c < C; c++) {
    for (auto h = 0u; h < H; h++) {
      for (auto w = 0u; w < W; w++) {
        output[c*H*W + w*H + h] = input[c*H*W + h*W + w];
        //cout << input[c*H*W + h*W + w] << " " << w << " " << h << " " << c*H*W + h*W + w << " " << c*H*W + w*H + h << endl;
      }
    }
  }
  return output;
}

std::pair<std::vector<float>, std::vector<uint32_t>> topK_distance(const std::vector<float> &scores, uint32_t k) {
  auto min_size = k > scores.size() ? scores.size() : k;
  std::vector<uint32_t> indices(min_size);
  std::vector<float> scalars(min_size);
  std::vector<std::pair<uint32_t, float>> scores_with_index(scores.size());
  for (auto i = 0u; i < scores.size(); ++i) {
    scores_with_index[i] = std::make_pair(i, scores[i]); 
  }
  auto compare = [&](std::pair<uint32_t, float> a, std::pair<uint32_t, float> b) {return a.second > b.second;};
  std::make_heap(scores_with_index.begin(), scores_with_index.end(), compare);

  for (auto n = 0u; n < min_size; ++n) {
    std::pop_heap(scores_with_index.begin(), scores_with_index.end() - n, compare); 
    indices[n] = (scores_with_index.end() - n - 1)->first;
    scalars[n] = (scores_with_index.end() - n - 1)->second;
  }
  return std::make_pair(scalars,indices);
}


/*
void readfile(string& filename, vector<float>& data) {
  ifstream input_file(filename);
  std::string line;
  while (std::getline(input_file, line)) {
    istringstream ss(line);
    float num;
    ss >> num;
    data.push_back(num);
  }
  cout << filename << " " << data.size() << endl;
}
*/

template <typename T>
static void im2col(
  const T *data_im,
  const int64_t channels,
  const int64_t height,
  const int64_t width,
  const int64_t output_height,
  const int64_t output_width,
  const int64_t kernel_h,
  const int64_t kernel_w,
  const int64_t pad_h,
  const int64_t pad_w,
  const int64_t stride_h,
  const int64_t stride_w,
  const int64_t dilation_h,
  const int64_t dilation_w,
  T *data_col) {
    const int64_t height_col = output_height;
    const int64_t width_col = output_width;
    const int64_t channels_col = channels * kernel_h * kernel_w;
    for (int64_t c_col = 0; c_col < channels_col; ++c_col)
    {
        int64_t w_offset = c_col % kernel_w;
        int64_t h_offset = (c_col / kernel_w) % kernel_h;
        int64_t c_im = c_col / kernel_h / kernel_w;
        for (int64_t h_col = 0; h_col < height_col; ++h_col)
        {
            int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
            for (int64_t w_col = 0; w_col < width_col; ++w_col)
            {
                int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                data_col[(c_col * height_col + h_col) * width_col + w_col] =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                        ? data_im[(c_im * height + h_im) * width + w_im]
                        : static_cast<T>(0);
            }
        }
    }
}



template <typename T>
void unfold(
  vector<T>& input, 
  vector<int> input_size,
  vector<T>& output,
  std::pair<int, int> kernel_size, 
  std::pair<int, int> dilation,
  std::pair<int, int> padding,
  std::pair<int, int> stride) {
  int kernel_height = kernel_size.first;
  int kernel_width = kernel_size.second;
  int dilation_height = dilation.first;
  int dilation_width = dilation.second;
  int pad_height = padding.first;
  int pad_width = padding.second;
  int stride_height = stride.first;
  int stride_width = stride.second;
  int batch_size = input_size[0];
  int n_input_plane = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];
  int output_height = (input_height + 2 * pad_height -
    (dilation_height * (kernel_height - 1) + 1)) /
    stride_height + 1;
  int output_width = (input_width + 2 * pad_width -
    (dilation_width * (kernel_width - 1) + 1)) /
    stride_width + 1;
  //cout << n_input_plane << " " << output_width << " " << output_height << endl;
  int n_output_plane = n_input_plane * kernel_width * kernel_height;
  int output_length = output_height * output_width;
  output.resize(batch_size * n_output_plane * output_length);
  for (int ind = 0; ind < batch_size; ind++) {
    T* input_n = input.data() + ind * (n_input_plane * input_width * input_height);
    T* output_n = output.data() + ind * (n_output_plane * output_length);
    im2col(input_n, n_input_plane, input_height, input_width,
           output_height, output_width, kernel_height, kernel_width,
           pad_height, pad_width, stride_height, stride_width, 
           dilation_height, dilation_width, output_n);
  }


}




} //namespace
}
}
