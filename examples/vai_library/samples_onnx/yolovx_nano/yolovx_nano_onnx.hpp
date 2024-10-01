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

#pragma once
#include <assert.h>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc_c.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "vitis/ai/onnx_task.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(ENABLE_YOLO_DEBUG, "0");

using namespace std;
using namespace cv;

static float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

static float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

static void applyNMS(const vector<vector<float>>& boxes,
                     const vector<float>& scores, const float nms,
                     const float conf, vector<size_t>& res) {
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  sort(order.begin(), order.end(),
       [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
         return ls.first > rs.first;
       });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });
  vector<bool> exist_box(count, true);

  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];
    if (!exist_box[i]) continue;
    if (scores[i] < conf) {
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(i);
    // cout << "nms push "<< i<<endl;
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j]) continue;
      float ovr = 0.0;
      ovr = cal_iou(boxes[j], boxes[i]);
      if (ovr >= nms) exist_box[j] = false;
    }
  }
}

static void letterbox(const cv::Mat& im, int w, int h, cv::Mat& om,
                      float& scale) {
  scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);
  cv::Mat img_res;
  if (im.size() != cv::Size(w, h)) {
    cv::resize(im, img_res, cv::Size(im.cols * scale, im.rows * scale), 0, 0,
               cv::INTER_LINEAR);
    auto dw = w - img_res.cols;
    auto dh = h - img_res.rows;
    if (dw > 0 || dh > 0) {
      om = cv::Mat(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
      copyMakeBorder(img_res, om, 0, dh, 0, dw, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
    } else {
      om = img_res;
    }
  } else {
    om = im;
    scale = 1.0;
  }
}

// return value
struct YolovxnanoOnnxResult {
  /**
   *@struct BoundingBox
   *@brief Struct of detection result with an object.
   */
  struct BoundingBox {
    /// Classification.
    int label;
    /// Confidence. The value ranges from 0 to 1.
    float score;
    /// (x0,y0,x1,y1). x0, x1 Range from 0 to the input image columns.
    /// y0,y1. Range from 0 to the input image rows.
    std::vector<float> box;
  };
  /// All objects, The vector of BoundingBox.
  std::vector<BoundingBox> bboxes;
};

// model class
class YolovxnanoOnnx : public OnnxTask {
 public:
  static std::unique_ptr<YolovxnanoOnnx> create(const std::string& model_name,
                                                const float conf_thresh_) {
    return std::unique_ptr<YolovxnanoOnnx>(
        new YolovxnanoOnnx(model_name, conf_thresh_));
  }

 protected:
  explicit YolovxnanoOnnx(const std::string& model_name,
                          const float conf_thresh_);
  YolovxnanoOnnx(const YolovxnanoOnnx&) = delete;

 public:
  virtual ~YolovxnanoOnnx() {}
  virtual std::vector<YolovxnanoOnnxResult> run(
      const std::vector<cv::Mat>& mats);
  virtual YolovxnanoOnnxResult run(const cv::Mat& mats);

 private:
  std::vector<YolovxnanoOnnxResult> postprocess();
  YolovxnanoOnnxResult postprocess(int idx);
  void preprocess(const cv::Mat& image, int idx, float& scale);
  void preprocess(const std::vector<cv::Mat>& mats);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  int real_batch;
  int batch_size;
  std::vector<float*> input_tensor_ptr;
  std::vector<float*> output_tensor_ptr;
  int output_tensor_size = 3;
  int channel = 0;
  int sHeight = 0;
  int sWidth = 0;
  float stride[3] = {8, 16, 32};
  float conf_thresh = 0.f;
  float conf_desigmoid = 0.f;
  float nms_thresh = 0.65f;
  int num_classes = 80;
  int anchor_cnt = 1;
  vector<float> scales;
};

void YolovxnanoOnnx::preprocess(const cv::Mat& image, int idx, float& scale) {
  cv::Mat resized_image;
  letterbox(image, sWidth, sHeight, resized_image, scale);
  set_input_image_bgr(resized_image,
                      input_tensor_values.data() + batch_size * idx,
                      std::vector<float>{0, 0, 0}, std::vector<float>{1, 1, 1});
  return;
}

// preprocess
void YolovxnanoOnnx::preprocess(const std::vector<cv::Mat>& mats) {
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  scales.resize(real_batch);
  for (auto index = 0; index < real_batch; ++index) {
    preprocess(mats[index], index, scales[index]);
  }
  return;
}

inline float sigmoid(float src) { return (1.0f / (1.0f + exp(-src))); }

// postprocess
YolovxnanoOnnxResult YolovxnanoOnnx::postprocess(int idx) {
  vector<vector<float>> boxes;

  int conf_box = 5 + num_classes;

  for (int i = 0; i < output_tensor_size; i++) {
    // 3 output layers  // 85x52x52  85x26x26 85x13x13:
    int ca = output_shapes_[i][1];
    int ha = output_shapes_[i][2];
    int wa = output_shapes_[i][3];
    if (ENV_PARAM(ENABLE_YOLO_DEBUG)) {
      LOG(INFO) << "channel=" << ca << ", height=" << ha << ", width=" << wa
                << ", stride=" << stride[i] << ", conf=" << conf_thresh
                << ", idx=" << idx << endl;
    }
    boxes.reserve(boxes.size() + ha * wa);
#define POS(C) ((C)*ha * wa + h * wa + w)
    for (int h = 0; h < ha; ++h) {
      for (int w = 0; w < wa; ++w) {
        for (int c = 0; c < anchor_cnt; ++c) {
          float score =
              output_tensor_ptr[i][POS(c * conf_box + 4) + idx * ca * ha * wa];
          if (score < conf_desigmoid) continue;
          vector<float> box(6);
          vector<float> out(4);
          for (int index = 0; index < 4; index++) {
            out[index] = output_tensor_ptr[i][POS(c * conf_box + index) +
                                              idx * ca * ha * wa];
          }
          box[0] = (w + out[0]) * stride[i];
          box[1] = (h + out[1]) * stride[i];
          box[2] = exp(out[2]) * stride[i];
          box[3] = exp(out[3]) * stride[i];
          float obj_score = sigmoid(score);
          auto conf_class_desigmoid = -logf(obj_score / conf_thresh - 1.0f);
          int max_p = -1;
          box[0] = box[0] - box[2] * 0.5;
          box[1] = box[1] - box[3] * 0.5;
          for (int p = 0; p < num_classes; p++) {
            float cls_score = output_tensor_ptr[i][POS(c * conf_box + 5 + p) +
                                                   idx * ca * ha * wa];
            if (cls_score < conf_class_desigmoid) continue;
            max_p = p;
            conf_class_desigmoid = cls_score;
          }
          if (max_p != -1) {
            box[4] = max_p;
            box[5] = obj_score * sigmoid(conf_class_desigmoid);
            boxes.push_back(box);
          }
        }
      }
    }
  }
  /* Apply the computation for NMS */
  vector<vector<vector<float>>> boxes_for_nms(num_classes);
  vector<vector<float>> scores(num_classes);

  for (const auto& box : boxes) {
    boxes_for_nms[box[4]].push_back(box);
    scores[box[4]].push_back(box[5]);
  }

  vector<vector<float>> res;
  for (auto i = 0; i < num_classes; i++) {
    vector<size_t> result_k;
    applyNMS(boxes_for_nms[i], scores[i], nms_thresh, conf_thresh, result_k);
    res.reserve(res.size() + result_k.size());
    transform(result_k.begin(), result_k.end(), back_inserter(res),
              [&](auto& k) { return boxes_for_nms[i][k]; });
  }

  vector<YolovxnanoOnnxResult::BoundingBox> results;
  for (const auto& r : res) {
    if (r[5] > conf_thresh) {
      YolovxnanoOnnxResult::BoundingBox result;
      result.score = r[5];
      result.label = r[4];
      result.box.resize(4);
      result.box[0] = r[0] / scales[idx];
      result.box[1] = r[1] / scales[idx];
      result.box[2] = result.box[0] + r[2] / scales[idx];
      result.box[3] = result.box[1] + r[3] / scales[idx];
      results.push_back(result);
    }
  }
  return YolovxnanoOnnxResult{results};
}

std::vector<YolovxnanoOnnxResult> YolovxnanoOnnx::postprocess() {
  std::vector<YolovxnanoOnnxResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index) {
    ret.emplace_back(postprocess(index));
  }
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

YolovxnanoOnnx::YolovxnanoOnnx(const std::string& model_name,
                               const float conf_thresh_)
    : OnnxTask(model_name) {
  int total_number_elements = calculate_product(input_shapes_[0]);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  channel = input_shapes_[0][1];
  sHeight = input_shapes_[0][2];
  sWidth = input_shapes_[0][3];
  batch_size = channel * sHeight * sWidth;
  input_tensor_ptr.resize(1);
  output_tensor_ptr.resize(output_tensor_size);
  conf_thresh = conf_thresh_;
  conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f);
}

YolovxnanoOnnxResult YolovxnanoOnnx::run(const cv::Mat& mats) {
  return run(vector<cv::Mat>(1, mats))[0];
}

std::vector<YolovxnanoOnnxResult> YolovxnanoOnnx::run(
    const std::vector<cv::Mat>& mats) {
  __TIC__(total)
  __TIC__(preprocess)
  preprocess(mats);
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
  for (int i = 0; i < output_tensor_size; i++) {
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }
  __TOC__(session_run)

  __TIC__(postprocess)
  std::vector<YolovxnanoOnnxResult> ret = postprocess();
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}

