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

#include "vitis/ai/nnpp/yolov8.hpp"

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <vector>
#include <numeric> //accumulate
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/math.hpp"
#include "vitis/ai/nnpp/apply_nms.hpp"

DEF_ENV_PARAM(ENABLE_YOLOv8_DEBUG, "0");

using namespace std;
namespace vitis {
namespace ai {

static vector<float> softmax(const std::vector<float>& input) {
  auto output = std::vector<float>(input.size());
  std::transform(input.begin(), input.end(), output.begin(), expf);
  auto sum = accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
  std::transform(output.begin(), output.end(), output.begin(),
                 [sum](float v) { return v / sum; });
  return output;
}

static vector<float> conv(const vector<vector<float>>& input) {
  // input size is 4 x 16
  // kernel is 16 x 1, value is 0,1,...,15
  vector<float> output(4, 0.0f);

  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 16; col++) {
      output[row] += input[row][col] * col;
    }
  }
  return output;
}

static vector<vector<float>> make_anchors(int w, int h) {
  vector<vector<float>> anchor_points;
  anchor_points.reserve(w * h);
  for (int i = 0; i < w; ++i) {
    float sy = i + 0.5f;
    for (int j = 0; j < h; ++j) {
      float sx = j + 0.5f;
      vector<float> anchor(2);
      anchor[0] = sx;
      anchor[1] = sy;
      anchor_points.emplace_back(anchor);
    }
  }
  return anchor_points;
}

static vector<float> dist2bbox(const vector<float>& distance,
                               const vector<float>& point, const float stride) {
  vector<float> box;
  box.resize(4);
  float x1 = point[0] - distance[0];
  float y1 = point[1] - distance[1];
  float x2 = point[0] + distance[2];
  float y2 = point[1] + distance[3];
  box[0] = (x1 + x2) / 2.0f * stride;  // x_c
  box[1] = (y1 + y2) / 2.0f * stride;  // y_c
  box[2] = (x2 - x1) * stride;         // width
  box[3] = (y2 - y1) * stride;         // height
  return box;
}

std::vector<YOLOv8Result> yolov8_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>&
        output_tensors_unsorted,
    const vitis::ai::proto::DpuModelParam& config,
    const std::vector<float>& scales, const std::vector<int>& left_padding,
    const std::vector<int>& top_padding) {
  auto& yolov8_params = config.yolo_v8_param();
  auto conf_thresh = yolov8_params.conf_thresh();
  auto max_boxes_num = yolov8_params.max_boxes_num();
  auto nms_thresh = yolov8_params.nms_threshold();
  auto max_nms_num = yolov8_params.max_nms_num();
  auto num_classes = yolov8_params.num_classes();

  std::vector<float> stride(yolov8_params.stride().begin(),
                            yolov8_params.stride().end());

  std::vector<std::string> detect_layer_name(
      yolov8_params.detect_layer_name().begin(),
      yolov8_params.detect_layer_name().end());

  std::vector<vitis::ai::library::OutputTensor> detect_output_tensors;

  for (auto i = 0u; i < detect_layer_name.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted.size(); j++) {
      auto pos = output_tensors_unsorted[j].name.find(detect_layer_name[i]);
      if (pos != std::string::npos &&
          pos + detect_layer_name[i].size() ==
              output_tensors_unsorted[j].name.size()) {
        detect_output_tensors.emplace_back(output_tensors_unsorted[j]);
        break;
      }
    }
  }

  std::vector<YOLOv8Result> res_vec;

  int batch_size = (scales.size() > detect_output_tensors[0].batch)
                       ? detect_output_tensors[0].batch
                       : scales.size();

  auto output_dim = 144;  // 80 + 16 * 4

  for (int k = 0; k < batch_size; k++) {
    vector<vector<float>> boxes;
    int out_num = detect_output_tensors.size();
    int count = 0;
    vector<vector<vector<float>>> pre_output;

    for (int i = 0; i < out_num; i++) {
      int width = detect_output_tensors[i].width;
      int height = detect_output_tensors[i].height;
      if (ENV_PARAM(ENABLE_YOLOv8_DEBUG)) {
        LOG(INFO) << "width=" << width << ", height=" << height;
      }
      auto anchor_points = make_anchors(height, width);
      int sizeOut = width * height;
      boxes.reserve(boxes.size() + sizeOut);
      int8_t* det_out = (int8_t*)detect_output_tensors[i].get_data(k);
      float det_scale =
          vitis::ai::library::tensor_scale(detect_output_tensors[i]);
      int8_t conf_thresh_inverse =
          -std::log(1.0f / conf_thresh - 1) / det_scale;
      pre_output.reserve(pre_output.size() + sizeOut);

      for (auto n = 0; n < sizeOut; ++n) {
        vector<vector<float>> pre_output_unit;
        pre_output_unit.resize(4);

        for (auto t = 0; t < 4; t++) {
          vector<float> softmax_;
          softmax_.reserve(16);
          for (auto m = 0; m < 16; m++) {
            float value = det_out[n * output_dim + t * 16 + m] * det_scale;
            softmax_.emplace_back(value);
          }

          pre_output_unit[t] = softmax(softmax_);
        }

        auto distance = conv(pre_output_unit);
        auto dbox = dist2bbox(distance, anchor_points[n], stride[i]);
        for (auto m = 0; m < num_classes; ++m) {
          if (det_out[n * output_dim + 64 + m] > conf_thresh_inverse) {
            count++;
            vector<float> box(6);
            for (int j = 0; j < 4; j++) {
              box[j] = dbox[j];
            }
            float cls_score =
                1.0 /
                (1 + exp(-1.0f * det_out[n * output_dim + 64 + m] * det_scale));
            box[4] = m;
            box[5] = cls_score;
            boxes.emplace_back(box);
          }
        }
      }
    }
    __TIC__(YOLOV8_SORT)
    auto compare = [=](vector<float>& lhs, vector<float>& rhs) {
      return lhs[5] > rhs[5];
    };
    if (ENV_PARAM(ENABLE_YOLOv8_DEBUG)) {
      LOG(INFO) << "boxes_total_size=" << boxes.size();
    }
    if (boxes.size() > max_boxes_num) {
      std::partial_sort(boxes.begin(), boxes.begin() + max_boxes_num,
                        boxes.end(), compare);
      boxes.resize(max_boxes_num);
    } else {
      std::sort(boxes.begin(), boxes.end(), compare);
    }
    __TOC__(YOLOV8_SORT)

    __TIC__(YOLOv8_NMS)
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
      if (ENV_PARAM(ENABLE_YOLOv8_DEBUG)) {
        LOG(INFO) << "class: " << i << " size:" << boxes_for_nms[i].size();
      }
      applyNMS(boxes_for_nms[i], scores[i], nms_thresh, 0, result_k, true);
      res.reserve(res.size() + result_k.size());
      transform(result_k.begin(), result_k.end(), back_inserter(res),
                [&](auto& k) { return boxes_for_nms[i][k]; });
    }
    if (ENV_PARAM(ENABLE_YOLOv8_DEBUG)) {
      LOG(INFO) << "res size: " << res.size();
      for (auto i = 0u; i < res.size(); ++i) {
        LOG(INFO) << "i = " << i << ", res size:" << res[i].size() << " ["
                  << res[i][0] << ", " << res[i][1] << ", " << res[i][2] << ", "
                  << res[i][3] << "], label:" << res[i][4]
                  << ", score: " << res[i][5];
      }
    }

    if (res.size() > max_nms_num) {
      std::partial_sort(res.begin(), res.begin() + max_nms_num, res.end(),
                        compare);
      res.resize(max_nms_num);
    } else {
      std::sort(res.begin(), res.end(), compare);
    }

    /* Restore the correct coordinate frame of the original image */
    vector<YOLOv8Result::BoundingBox> results;
    for (const auto& r : res) {
      YOLOv8Result::BoundingBox result;
      result.score = r[5];
      result.label = r[4];
      result.box.resize(4);

      //---for (xc, yc, w, h)---
      result.box[0] = (r[0] - r[2] / 2.0f - left_padding[k]) / scales[k];
      result.box[1] = (r[1] - r[3] / 2.0f - top_padding[k]) / scales[k];
      result.box[2] = result.box[0] + r[2] / scales[k];
      result.box[3] = result.box[1] + r[3] / scales[k];
      results.push_back(result);
    }
    res_vec.push_back(YOLOv8Result{results});
    __TOC__(YOLOv8_NMS)
  }
  return res_vec;
}
}  // namespace ai
}  // namespace vitis

