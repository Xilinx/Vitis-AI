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
#include "vitis/ai/profiling.hpp"

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/nnpp/multitaskv3.hpp>
#include "./prior_boxes.hpp"
#include "./ssd_detector.hpp"
#include "vitis/ai/onnx_task.hpp"

DEF_ENV_PARAM(MTNET_ONNX_ACC, "0")

using namespace std;
using namespace cv;
using namespace vitis::ai;

namespace onnx_multitaskv3 {

static void sigmoid_n(float* src, float* dst, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    auto tmp = src[i];
    dst[i] = (1. / (1. + exp(-tmp)));
  }
}

static void get_depth_ori(float* src, float* dst, size_t length, float scale) {
  for (size_t i = 0; i < length; ++i) {
    auto tmp = (float)src[i] * scale;
    dst[i] = tmp;
  }
}

static vector<shared_ptr<vector<float>>> CreatePriors(int image_width,
                                                      int image_height) {
  vector<onnx_multitaskv3::PriorBoxes> prior_boxes;
  prior_boxes.emplace_back(onnx_multitaskv3::PriorBoxes{image_width,
                                                        image_height,
                                                        128,
                                                        80,
                                                        {0.1, 0.1, 0.2, 0.2},
                                                        {10.0},
                                                        {30.0},
                                                        {},
                                                        0.5,
                                                        4.0,
                                                        4.0,
                                                        false,
                                                        false});
  prior_boxes.emplace_back(onnx_multitaskv3::PriorBoxes{image_width,
                                                        image_height,
                                                        64,
                                                        40,
                                                        {0.1, 0.1, 0.2, 0.2},
                                                        {30.0},
                                                        {60.0},
                                                        {},
                                                        0.5,
                                                        8.0,
                                                        8.0,
                                                        false,
                                                        false});
  prior_boxes.emplace_back(onnx_multitaskv3::PriorBoxes{image_width,
                                                        image_height,
                                                        32,
                                                        20,
                                                        {0.1, 0.1, 0.2, 0.2},
                                                        {60.0},
                                                        {100.0},
                                                        {},
                                                        0.5,
                                                        16.0,
                                                        16.0,
                                                        false,
                                                        false});
  prior_boxes.emplace_back(onnx_multitaskv3::PriorBoxes{image_width,
                                                        image_height,
                                                        16,
                                                        10,
                                                        {0.1, 0.1, 0.2, 0.2},
                                                        {100.0},
                                                        {160.0},
                                                        {},
                                                        0.5,
                                                        32.0,
                                                        32.0,
                                                        false,
                                                        false});
  prior_boxes.emplace_back(onnx_multitaskv3::PriorBoxes{image_width,
                                                        image_height,
                                                        8,
                                                        5,
                                                        {0.1, 0.1, 0.2, 0.2},
                                                        {160.0},
                                                        {220.0},
                                                        {},
                                                        0.5,
                                                        64.0,
                                                        64.0,
                                                        false,
                                                        false});
  prior_boxes.emplace_back(onnx_multitaskv3::PriorBoxes{image_width,
                                                        image_height,
                                                        6,
                                                        3,
                                                        {0.1, 0.1, 0.2, 0.2},
                                                        {220.0},
                                                        {280.0},
                                                        {},
                                                        0.5,
                                                        128.0,
                                                        128.0,
                                                        false,
                                                        false});
  prior_boxes.emplace_back(onnx_multitaskv3::PriorBoxes{image_width,
                                                        image_height,
                                                        4,
                                                        1,
                                                        {0.1, 0.1, 0.2, 0.2},
                                                        {280.0},
                                                        {340.0},
                                                        {},
                                                        0.5,
                                                        256.0,
                                                        256.0,
                                                        false,
                                                        false});
  int num_priors = 0;
  for (auto& p : prior_boxes) {
    num_priors += p.priors().size();
  }

  auto priors = vector<shared_ptr<vector<float>>>{};
  priors.reserve(num_priors);
  for (auto i = 0U; i < prior_boxes.size(); ++i) {
    priors.insert(priors.end(), prior_boxes[i].priors().begin(),
                  prior_boxes[i].priors().end());
  }
  return priors;
}

class OnnxMultiTaskv3 : public OnnxTask {
 public:
  static std::unique_ptr<OnnxMultiTaskv3> create(
      const std::string& model_name) {
    return std::unique_ptr<OnnxMultiTaskv3>(new OnnxMultiTaskv3(model_name));
  }

 protected:
  explicit OnnxMultiTaskv3(const std::string& model_name);
  OnnxMultiTaskv3(const OnnxMultiTaskv3&) = delete;

 public:
  virtual ~OnnxMultiTaskv3() {}
  virtual std::vector<MultiTaskv3Result> run(const std::vector<cv::Mat>& image);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  int real_batch;
  int batch_size;
  std::vector<float*> output_tensor_ptr;

  // mutlitask v3
  int num_detection_classes_;
  int num_segmention_classes_;
  std::vector<std::vector<onnx_multitaskv3::SSDOutputInfo>> all_loc_infos_;
  std::vector<std::vector<onnx_multitaskv3::SSDOutputInfo>> all_conf_infos_;
  std::vector<std::vector<onnx_multitaskv3::SSDOutputInfo>>
      all_centerness_infos_;
  std::vector<float> conf_result;
  std::vector<float> centerness_result;
  std::unique_ptr<onnx_multitaskv3::SSDdetector> detector_;
  std::vector<std::vector<float>> loc_infos_;

  std::vector<size_t> loc_size;
  std::vector<size_t> conf_size;
  std::vector<size_t> centerness_size;

  std::vector<uint8_t> color_c1;
  std::vector<uint8_t> color_c2;
  std::vector<uint8_t> color_c3;
  cv::Mat process_seg_visualization_c(size_t tensor_ind, int jdx);
  cv::Mat process_depth_ori(size_t tensor_ind, int jdx);
};

template <class T>
void max_index_c(T* d, int c, int g, uint8_t* results) {
  for (int i = 0; i < g; ++i) {
    auto it = std::max_element(d, d + c);
    results[i] = it - d;
    d += c;
  }
}

template <typename T>
std::vector<T> permute(const T* input, size_t C, size_t H, size_t W) {
  std::vector<T> output(C * H * W);
  for (auto c = 0u; c < C; c++) {
    for (auto h = 0u; h < H; h++) {
      for (auto w = 0u; w < W; w++) {
        output[h * W * C + w * C + c] = input[c * H * W + h * W + w];
      }
    }
  }
  return output;
}

template <typename T>
void permute(const T* input, size_t C, size_t H, size_t W,
             std::vector<float>& output) {
  for (auto c = 0u; c < C; c++) {
    for (auto h = 0u; h < H; h++) {
      for (auto w = 0u; w < W; w++) {
        output[h * W * C + w * C + c] = input[c * H * W + h * W + w];
      }
    }
  }
}

static void set_input_image(const cv::Mat& image, float* data) {
  float mean[3] = {104.f, 117.f, 123.f};
  float scales[3] = {0.00390625f, 0.00390625f, 0.00390625};
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto c_t = abs(c - 2);  // BRG to RGB
        auto image_data =
            (image.at<cv::Vec3b>(h, w)[c_t] - mean[c_t]) * scales[c_t];
        data[c * image.rows * image.cols + h * image.cols + w] =
            (float)image_data;
      }
    }
  }
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

cv::Mat OnnxMultiTaskv3::process_seg_visualization_c(size_t tensor_ind,
                                                     int jdx) {
  __TIC__(MULTITASK_SEG_VISUALIZATION)
  cv::Mat segmat(output_shapes_[tensor_ind][2], output_shapes_[tensor_ind][3],
                 CV_8UC3);
  unsigned int col_ind = 0;
  unsigned int row_ind = 0;
  auto chw = permute(
      output_tensor_ptr[tensor_ind] + jdx * output_shapes_[tensor_ind][1] *
                                          output_shapes_[tensor_ind][2] *
                                          output_shapes_[tensor_ind][3],
      output_shapes_[tensor_ind][1], output_shapes_[tensor_ind][2],
      output_shapes_[tensor_ind][3]);
  for (long int i = 0u;
       i < output_shapes_[tensor_ind][2] * output_shapes_[tensor_ind][3] *
               output_shapes_[tensor_ind][1];
       i = i + output_shapes_[tensor_ind][1]) {
    auto max_ind = std::max_element(
        chw.data() + i, chw.data() + i + output_shapes_[tensor_ind][1]);
    uint8_t posit = std::distance(chw.data() + i, max_ind);
    segmat.at<cv::Vec3b>(row_ind, col_ind) =
        cv::Vec3b((uint8_t)color_c1[posit], (uint8_t)color_c2[posit],
                  (uint8_t)color_c3[posit]);
    col_ind++;
    if (col_ind > output_shapes_[tensor_ind][3] - 1) {
      row_ind++;
      col_ind = 0;
    }
  }
  __TOC__(MULTITASK_SEG_VISUALIZATION)
  return segmat;
}

cv::Mat OnnxMultiTaskv3::process_depth_ori(size_t tensor_ind, int jdx) {
  __TIC__(MULTITASK_DEPTH)
  auto size = output_shapes_[tensor_ind][1] * output_shapes_[tensor_ind][2] *
              output_shapes_[tensor_ind][3];
  vector<float> rs(size);
  auto chw = permute(
      output_tensor_ptr[tensor_ind] + jdx * output_shapes_[tensor_ind][1] *
                                          output_shapes_[tensor_ind][2] *
                                          output_shapes_[tensor_ind][3],
      output_shapes_[tensor_ind][1], output_shapes_[tensor_ind][2],
      output_shapes_[tensor_ind][3]);
  get_depth_ori(chw.data(), rs.data(), size, 1);
  cv::Mat depth_results(output_shapes_[tensor_ind][2],
                        output_shapes_[tensor_ind][3], CV_8UC1);
  for (long int i = 0; i < size; i++) {
    depth_results.data[i] = (uint8_t)rs[i];
  }
  __TOC__(MULTITASK_DEPTH)
  return depth_results;
}

OnnxMultiTaskv3::OnnxMultiTaskv3(const std::string& model_name)
    : OnnxTask(model_name) {
  num_detection_classes_ = 3;
  auto batch = get_input_batch();
  auto os = 0u;
  for (auto i = 0u; i < 7; i++) {
    auto each_layer_size = batch * output_shapes_[i][1] * output_shapes_[i][2] *
                           output_shapes_[i][3];
    loc_size.push_back(each_layer_size);
    os += each_layer_size;
  }

  os = 0u;
  for (auto i = 7u; i < 14; i++) {
    auto each_layer_size = batch * output_shapes_[i][1] * output_shapes_[i][2] *
                           output_shapes_[i][3];
    conf_size.push_back(each_layer_size);
    os += each_layer_size;
  }

  os = 0u;
  for (auto i = 14u; i < 21; i++) {
    auto each_layer_size = batch * output_shapes_[i][1] * output_shapes_[i][2] *
                           output_shapes_[i][3];
    centerness_size.push_back(each_layer_size);
    os += each_layer_size;
  }

  auto input_shape = input_shapes_[0];
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  auto height = input_shapes_[0][2];
  auto width = input_shapes_[0][3];
  batch_size = input_shapes_[0][0];

  all_loc_infos_.resize(batch_size);
  auto batch_idx = 0;
  for (auto& loc_infos : all_loc_infos_) {
    loc_infos.reserve(7);
    loc_infos.assign(7, onnx_multitaskv3::SSDOutputInfo{});
    auto bbox_index = 0u;
    for (auto k = 0u; k < 7; ++k) {
      loc_infos[k].index_begin = bbox_index;
      loc_infos[k].bbox_single_size = 4;
      loc_infos[k].index_size =
          loc_size[k] / batch_size / loc_infos[k].bbox_single_size;
      bbox_index += loc_infos[k].index_size;
      loc_infos[k].size = loc_size[k] / batch_size;
    }
    batch_idx++;
  }

  all_conf_infos_.resize(batch_size);
  batch_idx = 0;
  for (auto& conf_infos : all_conf_infos_) {
    auto score_index = 0u;
    conf_infos.reserve(7);
    conf_infos.assign(7, onnx_multitaskv3::SSDOutputInfo{});
    for (auto k = 0u; k < 7; ++k) {
      conf_infos[k].index_begin = score_index;
      conf_infos[k].index_size =
          conf_size[k] / batch_size / num_detection_classes_;
      score_index += conf_infos[k].index_size;
      conf_infos[k].size = conf_size[k] / batch_size;
    }
    batch_idx++;
  }

  batch_idx = 0;
  all_centerness_infos_.resize(batch_size);
  for (auto& center_infos : all_centerness_infos_) {
    auto center_index = 0u;
    center_infos.reserve(7);
    center_infos.assign(7, onnx_multitaskv3::SSDOutputInfo{});
    for (auto k = 0u; k < 7; ++k) {
      center_infos[k].index_begin = center_index;
      center_infos[k].index_size = centerness_size[k] / batch_size;
      center_index += center_infos[k].index_size;
      center_infos[k].size = centerness_size[k] / batch_size;
    }
    batch_idx++;
  }

  output_tensor_ptr.resize(output_shapes_.size());
  auto priors = CreatePriors(width, height);
  conf_result.resize(priors.size() * num_detection_classes_);
  centerness_result.resize(priors.size());
  loc_infos_.resize(7);
  for (auto i = 0u; i < 7; i++) {
    loc_infos_[i].resize(all_loc_infos_[0][i].size);
  }
  std::vector<float> th_conf;
  if (ENV_PARAM(MTNET_ONNX_ACC) == 0)
    th_conf = {0.3f, 0.25f, 0.25f};
  else
    th_conf = {0.01, 0.01, 0.01};
  detector_ = std::make_unique<onnx_multitaskv3::SSDdetector>(
      num_detection_classes_,
      onnx_multitaskv3::SSDdetector::CodeType::CENTER_SIZE,
      false,    //
      200,      //
      th_conf,  //
      400,      //
      0.45,     //
      1.0, priors, 1);

  color_c1 = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
              180, 60,  0,  142, 7,   100, 100, 230, 32, 178};
  color_c2 = {64,  35, 150, 102, 153, 153, 170, 220, 142, 251,
              130, 20, 0,   0,   0,   60,  80,  0,   11,  43};
  color_c3 = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
              70,  220, 255, 0,   0,   0,   0,   0,   119, 255};
}

std::vector<MultiTaskv3Result> OnnxMultiTaskv3::run(
    const std::vector<cv::Mat>& image) {
  __TIC__(total)
  __TIC__(preprocess)
  cv::Mat resize_image;
  auto height = input_shapes_[0][2];
  auto width = input_shapes_[0][3];
  auto size = cv::Size((int)width, (int)height);

  auto batch = get_input_batch();
  real_batch = std::min((int)batch, int(image.size()));
  for (int k = 0; k < real_batch; k++) {
    cv::resize(image[k], resize_image, size);
    set_input_image(resize_image,
                    input_tensor_values.data() + k * input_shapes_[0][1] *
                                                     input_shapes_[0][2] *
                                                     input_shapes_[0][3]);
  }
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
  for (auto i = 0u; i < output_tensors.size(); i++) {
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }
  __TOC__(session_run)

  __TIC__(postprocess)
  CHECK_EQ(all_loc_infos_[0].size(), all_conf_infos_[0].size());

  std::vector<MultiTaskv3Result> ret;

  for (int j = 0; j < real_batch; j++) {
    std::vector<float> cs(conf_result.size());
    for (auto k = 0u; k < all_conf_infos_[0].size(); k++) {
      auto offset = all_conf_infos_[0][k].index_begin * num_detection_classes_;
      auto conf_hwc =
          permute(output_tensor_ptr[k + 7] + j * output_shapes_[k + 7][1] *
                                                 output_shapes_[k + 7][2] *
                                                 output_shapes_[k + 7][3],
                  output_shapes_[k + 7][1], output_shapes_[k + 7][2],
                  output_shapes_[k + 7][3]);
      sigmoid_n(conf_hwc.data(), conf_result.data() + offset,
                all_conf_infos_[0][k].index_size * 3);
      auto centerness_hwc =
          permute(output_tensor_ptr[k + 14] + j * output_shapes_[k + 14][1] *
                                                  output_shapes_[k + 14][2] *
                                                  output_shapes_[k + 14][3],
                  output_shapes_[k + 14][1], output_shapes_[k + 14][2],
                  output_shapes_[k + 14][3]);
      sigmoid_n(
          centerness_hwc.data(),
          centerness_result.data() + all_centerness_infos_[0][k].index_begin,
          all_centerness_infos_[0][k].index_size);
      for (auto i = 0u; i < centerness_result.size(); i++) {
        cs[i * 3 + 0] = conf_result[i * 3 + 0] * centerness_result[i];
        cs[i * 3 + 1] = conf_result[i * 3 + 1] * centerness_result[i];
        cs[i * 3 + 2] = conf_result[i * 3 + 2] * centerness_result[i];
      }
      permute(output_tensor_ptr[k] + j * output_shapes_[k][1] *
                                         output_shapes_[k][2] *
                                         output_shapes_[k][3],
              output_shapes_[k][1], output_shapes_[k][2], output_shapes_[k][3],
              loc_infos_[k]);
      all_loc_infos_[0][k].ptr = loc_infos_[k].data();
    }
    vector<Vehiclev3Result> v_result;

    std::map<uint32_t, onnx_multitaskv3::SSDOutputInfo> bbox_layer_infos;
    for (auto i = 0u; i < all_loc_infos_[0].size(); ++i) {
      bbox_layer_infos.emplace(std::make_pair(i, all_loc_infos_[0][i]));
    }
    detector_->Detect(bbox_layer_infos, cs.data(), v_result);
    auto seg = process_seg_visualization_c(21, j);
    auto drivable = process_seg_visualization_c(22, j);
    auto lane = process_seg_visualization_c(24, j);
    auto depth = process_depth_ori(23, j);

    ret.emplace_back(
        MultiTaskv3Result{512, 320, v_result, seg, lane, drivable, depth});
  }
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}

}  // namespace onnx_multitaskv3

