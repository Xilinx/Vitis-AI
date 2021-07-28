/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <vector>

using namespace std;
using std::exp;

void getResult(const vector<vector<float>>& box,
    const vector<int>& keep,
    vector<vector<float>>& results) {

  results.clear();
  results.reserve(keep.size());
  for (auto i = 0u; i < keep.size(); ++i) {
    auto b = box[keep[i]];
    b[2] -= b[0];
    b[3] -= b[1];
    results.emplace_back(b);
  }
}

void NMS(const float nms_threshold,
    const vector<vector<float>>& box,
    vector<vector<float>>& results) {

  auto count = box.size();
  vector<pair<size_t, float>> order(count);
  for (auto i = 0u; i < count; ++i) {
    order[i].first = i;
    order[i].second = box[i][4];
  }

  sort(order.begin(), order.end(),
      [](const pair<int, float>& ls, const pair<int, float>& rs) {
      return ls.second > rs.second;
      }
      );

  vector<int> keep;
  vector<bool> exist_box(count, true);
  for (auto i = 0u; i < count; ++i) {
    auto idx = order[i].first;
    if (!exist_box[idx]) continue;
    keep.emplace_back(idx);
    for (auto j = i + 1; j < count; ++j) {
      auto kept_idx = order[j].first;
      if (!exist_box[kept_idx]) continue;
      auto x1 = max(box[idx][0], box[kept_idx][0]);
      auto y1 = max(box[idx][1], box[kept_idx][1]);
      auto x2 = min(box[idx][2], box[kept_idx][2]);
      auto y2 = min(box[idx][3], box[kept_idx][3]);
      auto intersect = max(0.f, x2 - x1 + 1) * max(0.f, y2 - y1 + 1);
      auto sum_area =
        (box[idx][2] - box[idx][0] + 1) * (box[idx][3] - box[idx][1] + 1) +
        (box[kept_idx][2] - box[kept_idx][0] + 1) *
        (box[kept_idx][3] - box[kept_idx][1] + 1);
      auto overlap = intersect / (sum_area - intersect);
      if (overlap >= nms_threshold) exist_box[kept_idx] = false;
    }
  }
  getResult(box, keep, results);
}

vector<vector<float>> FilterBox(
  const float bb_out_scale, const float det_threshold,
  float* bbout, int w, int h, float* pred) {

  vector<vector<float>> boxes;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int position = i * w + j;
      vector<float> box;
      if (pred[position * 2 + 1] > det_threshold) {
        box.push_back(bbout[position * 4 + 0] * bb_out_scale + j * 4);
        box.push_back(bbout[position * 4 + 1] * bb_out_scale + i * 4);
        box.push_back(bbout[position * 4 + 2] * bb_out_scale + j * 4);
        box.push_back(bbout[position * 4 + 3] * bb_out_scale + i * 4);
        box.push_back(pred[position * 2 + 1]);
        boxes.push_back(box);
      }
    }
  }
  return boxes;
}

void softmax_c(float* input, float scale,
    unsigned int cls, float* output) {
  float sum = 0.f;
  float max = input[0];
  for (unsigned int i = 1; i < cls; ++i) {
    if (input[i] > max) max = input[i];
  }
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] = exp((input[i]-max) * scale);
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) output[i] /= sum;
}

void softmax_c(float* input, float scale, unsigned int cls,
    unsigned int group, float* output) {
  for (unsigned int i = 0; i < group; ++i) {
    softmax_c(input, scale, cls, output);
    input += cls;
    output += cls;
  }
}

//# Softmax method with float input data
void softmax(float* input, float scale, unsigned int cls,
    unsigned int group, float* output) {
  softmax_c(input, scale, cls, group, output);
}

void GSTilingLayer(float *in_data,int input_batch,
    int input_channels, int input_height,
    int input_width, int output_channels,
    int stride, int output_height,
    int output_width,  float *out)
{

  int ox, oy, oc, oi;
  int n, ic, iy, ix;
  int count_per_output_map = output_width * output_height * output_channels;
  int count_per_input_map = input_width * input_height * input_channels;
  float *tmp = new float[output_channels * output_height * output_width];

  int k=0;

  for (n = 0; n < input_batch; ++n) {
    int ii = 0;
    for (iy = 0; iy < input_height; ++iy) {
      for (ix = 0; ix < input_width; ++ix) {
        for (ic = 0; ic < input_channels; ++ic) {
          int off = ic / output_channels;
          ox = ix * stride + off % stride;
          oy = iy * stride + off / stride;
          oc = ic % output_channels;
          oi = (oc * output_height + oy) * output_width + ox;
          if (oi < output_channels * output_height * output_width)
            tmp[oi] = in_data[ii];
          else
            std::cout<<oi<<std::endl;

          ++ii;
        }
      }
    }
  }

  int cnt = 0;
  for (int h = 0; h < output_height; h++) {
    for (int w = 0; w < output_width; w++) {
      for (int c = 0; c < output_channels; c++) {
        out[cnt++] =
          tmp[c * (output_height * output_width) + h * (output_width) + w];
      }
    }
  }
  delete[] tmp;
}

void displayResults (std::vector<std::vector<float>>& results,
  std::string image_path, int batch) {
  /// Display Detected Boxes and overlay on images
  std::cout << "-- Batch Index: " << batch << std::endl;
  cv::Mat image;
  cv::Mat ml = cv::imread (image_path.c_str());
  cv::resize(ml, image, cv::Size {320, 320});

  for (auto result: results) {
    std::cout << "  ";
    for (auto r: result) {
      std::cout << r << " ";
    }
    std::cout << std::endl;
    int x  = int(result[0]);
    int y  = int(result[1]);
    int sx = int(result[2]);
    int sy = int(result[3]);
    cv::rectangle(image, cv::Rect{cv::Point(x, y), cv::Size{sx, sy}}, 0xff);
  }
  std::cout << std::endl << std::endl;
  std::string outpath =
    std::string("outputs/") + std::to_string(batch) + std::string(".jpg");
  cv::imwrite(outpath.c_str(), image);
}
