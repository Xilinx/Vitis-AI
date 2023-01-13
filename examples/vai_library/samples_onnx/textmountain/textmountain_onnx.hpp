/*
 * Copyright 2019 Xilinx Inc.
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

#pragma once
#include <assert.h>
#include <glog/logging.h>

#include <opencv2/imgproc/imgproc_c.h>
#include <algorithm>  // std::generate
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/onnx_task.hpp"
#include "vitis/ai/profiling.hpp"

using namespace std;
using namespace cv;
using namespace vitis::ai;

float g_mean[3] = {102.9801, 115.9465, 122.7717};
float g_scales[3] = {1.0, 1.0, 1.0};

// return value
typedef cv::Point2d arr4_point2d[4];
struct OnnxTextMountainResult {
  int width = 0;
  int height = 0;
  struct tmitem {
    tmitem(arr4_point2d& inbox, float inscore) : box(inbox), score(inscore) {}
    arr4_point2d box;
    float score;
  };
  std::vector<tmitem> res;
};

// model class
class OnnxTextMountain : public OnnxTask {
 public:
  static std::unique_ptr<OnnxTextMountain> create(
      const std::string& model_name) {
    return std::unique_ptr<OnnxTextMountain>(new OnnxTextMountain(model_name));
  }

 protected:
  explicit OnnxTextMountain(const std::string& model_name);
  OnnxTextMountain(const OnnxTextMountain&) = delete;

 public:
  virtual ~OnnxTextMountain() {}
  virtual std::vector<OnnxTextMountainResult> run(
      const std::vector<cv::Mat>& imgs);

 private:
  cv::Mat preprocess(cv::Mat image, int idx);
  void preprocess(const std::vector<cv::Mat>& image);
  OnnxTextMountainResult postprocess_textmountain(int idx);
  std::vector<OnnxTextMountainResult> postprocess_textmountain();
  void fix_scale(Point2f* vertices, Point2d* dest, int idx);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;
  std::vector<float> scale_h;
  std::vector<float> scale_w;
  int real_batch;
  int batch_size;
  std::vector<int> total_number_elements;
  std::vector<float*> output_tensor_ptr;
};

static void set_input_image(const cv::Mat& image, float* data) {
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto c_t = abs(c - 2);  // BRG to RGB
        auto image_data =
            (image.at<cv::Vec3b>(h, w)[c_t] - g_mean[c_t]) * g_scales[c_t];
        data[c * image.rows * image.cols + h * image.cols + w] =
            (float)image_data;
      }
    }
  }
}

static int round2nearest_multiple(int x, int p) {
  float tmp = (float)x / p;
  return std::max(p, int(std::round(tmp) * p));
}

cv::Mat OnnxTextMountain::preprocess(cv::Mat image, int idx) {
  cv::Mat imgout =
      cv::Mat::zeros(input_shapes_[0][2], input_shapes_[0][3], CV_8UC3);
  float scale_resize =
      float(input_shapes_[0][3]) / std::max(image.cols, image.rows);
  float w_resize = round2nearest_multiple(scale_resize * image.cols, 32);
  float h_resize = round2nearest_multiple(scale_resize * image.rows, 32);
  // std::cout <<"wh _resize : " << w_resize << " " << h_resize <<"\n"; // 704
  // 960
  cv::Mat img(imgout, cv::Rect(0, 0, w_resize, h_resize));
  cv::resize(image, img, cv::Size(w_resize, h_resize), 0, 0, cv::INTER_LINEAR);
  // std::cout << "img resize size : " << img.cols << " " << img.rows << "\n";
  scale_h[idx] = float(h_resize * output_shapes_[0][2] /
                       (input_shapes_[0][2] * image.rows));
  scale_w[idx] = float(w_resize * output_shapes_[0][3] /
                       (input_shapes_[0][3] * image.cols));
  //
  return imgout;
}
// textmountain preprocess
void OnnxTextMountain::preprocess(const std::vector<cv::Mat>& images) {
  real_batch = std::min((int)input_shapes_[0][0], (int)images.size());
  for (auto index = 0; index < real_batch; ++index) {
    auto resize_image = preprocess(images[index], index);
    set_input_image(resize_image,
                    input_tensor_values.data() + batch_size * index);
  }
}

// hwc & chw issue
void maxpool(float* input, std::vector<int>& output, cv::Mat& shrink_scores,
             int output_h, int output_w) {
  const int kernel_size = 3;
  float thres_center = 0.6;
  float max_v, tmp_v;
  int max_y, max_x;
  for (auto i = 0; i < output_h; ++i) {
    for (auto j = 0; j < output_w; ++j) {
      max_y = max_x = 0;
      max_v = -10000;
      for (int di = 0; di < kernel_size; di++) {
        for (int dj = 0; dj < kernel_size; dj++) {
          auto input_h_idx = ((i - 1) + di);
          auto input_w_idx = ((j - 1) + dj);
          if (input_w_idx < 0 || input_h_idx < 0 || input_h_idx >= output_h ||
              input_w_idx >= output_w) {
            continue;
          }
          tmp_v = input[input_h_idx * output_w + input_w_idx];
          if (max_v < tmp_v) {
            max_v = tmp_v;
            max_x = input_w_idx;
            max_y = input_h_idx;
          }
        }
      }
      output[(i * output_w + j) * 2 + 0] = max_x;
      output[(i * output_w + j) * 2 + 1] = max_y;
      // std::cout <<"maxpool : " << max_x << "  " << max_y << "\n";
      if (input[i * output_w + j] > thres_center) {
        shrink_scores.ptr<uchar>(i)[j] = 255;
        // std::cout <<i << " " << j << "\n";
      }
    }
  }
  // cv::imwrite("textmount_shrinkscore.jpg", shrink_scores );
}

void get_cc_eval(cv::Mat& img, float* pred0_sft, std::vector<bool>& valid,
                 cv::Mat& score_i, std::vector<float>& score_mean, int output_h,
                 int output_w) {
  /*
     groupSoftmax() -->   score[:1:], score_i
       // score is pred0 which is  mpool's second part;  score_i is label.
     CropAndResizeFunction.forward --> image, boxID_ptr
     crop_and_resize_gpu.forward()--> image, group_sum, groupNumsum, boxID_ptr
     crop_and_resize_gpu_forward() --> same as above
     CropAndResizeLaucher()--> image.data<float>, batch_size, image_height,
     image_width depth  group_sum.data<float>  groupNumsum.data<float>,
     boxID_ptr.data<int> cropAndResizeKernel() --> total_count,  image_ptr,
     group_sum, groupNumsum, boxID_ptr, batch, image_height image_width
  */

  float score_thres = 0.75;
  int label_num = cv::connectedComponents(img, score_i, 4, CV_32S);
  // std::cout <<"label_num : " << label_num <<"\n";
  std::vector<float> group_sum(label_num, 0.0);
  std::vector<float> groupNumsum(label_num, 0.0);

  int boxID = 0;
  for (int i = 0; i < output_h; i++) {
    for (int j = 0; j < output_w; j++) {
      boxID = score_i.ptr<int32_t>(i)[j];
      if (boxID != 0) {
        groupNumsum[boxID]++;
        group_sum[boxID] += pred0_sft[i * output_w + j];
      }
    }
  }
  // score_sum is group_sum, score_num is groupNumsum
  // std::cout <<"score_mean valid " << "\n";
  for (int i = 1; i < (int)group_sum.size(); i++) {
    score_mean.emplace_back((float)group_sum[i] /
                            (float)std::max(groupNumsum[i], float(1e-10)));
    valid.emplace_back(score_mean[i - 1] > score_thres && groupNumsum[i] > 5);
    // std::cout <<" mean valid : " <<  group_sum[i] << " " <<  groupNumsum[i]
    // << " "  << score_mean[i-1] << " " << valid[i-1] << "\n";
  }
}

void groupSearch(
    std::vector<std::pair<int, int>>& points_ptr,
    const std::vector<int>& next_ptr,  // next is indices_2d also is mpool
    cv::Mat& instance_ptr,             // instance_ptr is score_i
    float* pred0_sft,                  // prob_ptr is pred0_sft
    int output_h, int output_w) {
  int next_x;
  int next_y;
  int instance_idx;
  int next_xx;
  int next_yy;
  int points_num = points_ptr.size();
  std::vector<bool> circle_ptr(output_h * output_w, true);
  float score_thres_pixel = 0.6;

  for (int i = 0; i < points_num; i++) {
    int num_search = 1;
    int x = points_ptr[i].first;
    int y = points_ptr[i].second;
    next_x = next_ptr[(y * output_w + x) * 2 + 0];
    next_y = next_ptr[(y * output_w + x) * 2 + 1];

    instance_idx = instance_ptr.ptr<int32_t>(next_y)[next_x];

    while (instance_idx == 0) {
      if (num_search > (points_num + 3)) {
        circle_ptr[y * output_w + x] = false;
        circle_ptr[next_y * output_w + next_x] = false;
        break;
      }
      num_search = num_search + 1;

      if (circle_ptr[next_y * output_w + next_x] == false ||
          (next_x == x && next_y == y) ||
          pred0_sft[next_y * output_w + next_x] <= score_thres_pixel) {
        circle_ptr[y * output_w + x] = false;
        break;
      }
      next_xx = next_x;
      next_yy = next_y;
      next_x = next_ptr[(next_yy * output_w + next_xx) * 2 + 0];
      next_y = next_ptr[(next_yy * output_w + next_xx) * 2 + 1];

      instance_idx = instance_ptr.ptr<int32_t>(next_y)[next_x];
    }
    if (instance_idx > 0) {
      instance_ptr.ptr<int32_t>(y)[x] = instance_idx;
    }
  }
}

void OnnxTextMountain::fix_scale(Point2f* vertices, Point2d* dest, int idx) {
  for (int i = 0; i < 4; i++) {
    dest[i].x = int(vertices[i].x / scale_w[idx]);
    dest[i].y = int(vertices[i].y / scale_h[idx]);
  }
}

bool filtered(Point2d& in1, Point2d& in2) {
  return std::sqrt(std::pow(in1.x - in2.x, 2) + std::pow(in1.y - in2.y, 2)) <
         5.0;
}

// textmountain postprocess
OnnxTextMountainResult OnnxTextMountain::postprocess_textmountain(int idx) {
  // 3. postprocess
  // Output Node Name/Shape (3):
  //         1606 : 1x2x240x240  --->softmax
  //         1612 : 1x1x240x240  --->sigmoid
  //         1617 : 1x2x240x240

  // 3.1 maxpool  && shrink_scores
  int output_h = output_shapes_[0][2];
  int output_w = output_shapes_[0][3];
  cv::Mat shrink_scores(output_h, output_w, CV_8UC1, cvScalar(0));
  std::vector<int> mpool(output_w * output_h * 2, 0);
  __TIC__(maxpool)
  maxpool(output_tensor_ptr[1] + idx * (total_number_elements[1] / g_batchnum),
          mpool, shrink_scores, output_h, output_w);
  __TOC__(maxpool)

  // 3.2 get_cc_eval & groupmean
  cv::Mat score_i;
  std::vector<bool> valid;
  std::vector<float> score_mean;
  float* pred0_sft = output_tensor_ptr[0] +
                     idx * (total_number_elements[0] / g_batchnum) +
                     output_h * output_w;  // channel is first
  __TIC__(get_cc_eval)
  get_cc_eval(shrink_scores, pred0_sft, valid, score_i, score_mean, output_h,
              output_w);
  __TOC__(get_cc_eval)

  __TIC__(points_ptr)
  std::vector<std::pair<int, int>> points_ptr;
  points_ptr.reserve(20000);
  float score_thres_pixel = 0.6;
  for (int i = 0; i < output_h; i++) {
    for (int j = 0; j < output_w; j++) {
      if (pred0_sft[i * output_w + j] > score_thres_pixel &&
          score_i.ptr<int32_t>(i)[j] == 0) {
        points_ptr.emplace_back(std::pair(j, i));
        // std::cout <<"points : " << j << " " << i << "\n";
      }
    }
  }
  __TOC__(points_ptr)

  if (points_ptr.size() == 0) {
    return OnnxTextMountainResult{};
  }

  // 3.3 groupSearch()
  __TIC__(groupsearch)
  groupSearch(points_ptr, mpool, score_i, pred0_sft, output_h, output_w);
  __TOC__(groupsearch)

  // 3.4 image_idx_Tobox
  __TIC__(image_idx_tobox)
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  cv::Mat score_dst;
  score_i.convertTo(score_dst, CV_8U);

  cv::Mat mat1, mat2;
  RotatedRect box;
  Point2f vertices[4];
  Point2d vert_dest[4];

  cv::Mat img;
  OnnxTextMountainResult result;
  for (int i = 0; i < (int)valid.size(); i++) {
    if (!valid[i]) continue;
    cv::threshold(score_dst, mat1, i + 1, 0, THRESH_TOZERO_INV);
    cv::threshold(mat1, mat2, i, 0, THRESH_TOZERO);

    cv::findContours(mat2, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);
    // std::cout <<"contours.size : " << i << "  " << contours.size() <<"\n";
    box = cv::minAreaRect(contours[0]);
    box.points(vertices);
    fix_scale(vertices, vert_dest, idx);
    if ((filtered(vert_dest[0], vert_dest[1]) ||
         filtered(vert_dest[3], vert_dest[0]))) {
      continue;
    }
    //  for(nt j=0; j<4; j++) std::cout << " vertices " << vertices[j].x << "  "
    //  << vertices[j].y << "\n";
    result.res.push_back(
        OnnxTextMountainResult::tmitem(vert_dest, score_mean[i]));
    // print_result(vert_dest, score_mean[i]);
  }
  __TOC__(image_idx_tobox)
  return result;
}

std::vector<OnnxTextMountainResult>
OnnxTextMountain::postprocess_textmountain() {
  std::vector<OnnxTextMountainResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index) {
    ret.emplace_back(postprocess_textmountain(index));
  }
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

OnnxTextMountain::OnnxTextMountain(const std::string& model_name)
    : OnnxTask(model_name) {
  auto input_shape = input_shapes_[0];
  int total_number_elements_in = calculate_product(input_shape);
  std::vector<float> input_tensor_values_(total_number_elements_in);
  input_tensor_values_.swap(input_tensor_values);

  auto batch = input_shapes_[0][0];
  auto channel = input_shapes_[0][1];
  auto height = input_shapes_[0][2];
  auto width = input_shapes_[0][3];
  batch_size = channel * height * width;

  scale_h.resize(batch);
  scale_w.resize(batch);
  output_tensor_ptr.resize(3);
  total_number_elements.resize(3, 0);
}

std::vector<OnnxTextMountainResult> OnnxTextMountain::run(
    const std::vector<cv::Mat>& img) {
  __TIC__(total)
  __TIC__(preprocess)
  preprocess(img);

  if (input_tensors.size()) {
    input_tensors[0] = Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]);
  } else {
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]));
  }
  __TOC__(preprocess)

  __TIC__(session_run)
  run_task(input_tensors, output_tensors);
  for (int i = 0; i < 2; i++) {
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }
  if (total_number_elements[0] == 0) {
    total_number_elements[0] =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    total_number_elements[1] =
        output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
  }
  __TOC__(session_run)

  __TIC__(postprocess)
  std::vector<OnnxTextMountainResult> ret = postprocess_textmountain();
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}

