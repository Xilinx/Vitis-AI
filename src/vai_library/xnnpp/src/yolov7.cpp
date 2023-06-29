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

#include "vitis/ai/nnpp/yolov7.hpp"

#include <algorithm>
#include <fstream>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/math.hpp"
#include "vitis/ai/nnpp/apply_nms.hpp"

DEF_ENV_PARAM(ENABLE_YOLOv7_DEBUG, "0");

using namespace std;
namespace vitis {
namespace ai {

static float sigmoid(float p) { return 1.0 / (1 + exp(-p * 1.0)); }

static void correct_region_boxes(vector<vector<float>>& boxes, int n, int w,
                                 int h, int netw, int neth) {
  int new_w = 0;
  int new_h = 0;

  if (((float)netw / w) < ((float)neth / h)) {
    new_w = netw;
    new_h = (h * netw) / w;
  } else {
    new_h = neth;
    new_w = (w * neth) / h;
  }
  for (int i = 0; i < n; ++i) {
    boxes[i][0] = (boxes[i][0] - (netw - new_w) / 2. / netw) /
                  ((float)new_w / (float)netw);
    boxes[i][1] = (boxes[i][1] - (neth - new_h) / 2. / neth) /
                  ((float)new_h / (float)neth);
    boxes[i][2] *= (float)netw / new_w;
    boxes[i][3] *= (float)neth / new_h;
  }
}

std::vector<YOLOv7Result> yolov7_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>&
        output_tensors_unsorted,
    const vitis::ai::proto::DpuModelParam& config, const std::vector<int>& ws,
    const std::vector<int>& hs) {
  int sWidth = input_tensors[0].width;
  int sHeight = input_tensors[0].height;
  auto& yolo_params = config.yolo_v7_param();
  auto conf_thresh = yolo_params.conf_thresh();
  auto nms_thresh = yolo_params.nms_threshold();
  auto num_classes = yolo_params.num_classes();
  auto biases = std::vector<float>(yolo_params.biases().begin(),
                                   yolo_params.biases().end());
  std::vector<std::string> detect_layer_name(
      yolo_params.detect_layer_name().begin(),
      yolo_params.detect_layer_name().end());

  std::vector<vitis::ai::library::OutputTensor> detect_output_tensors;

  for (auto i = 0u; i < detect_layer_name.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted.size(); j++) {
      auto pos = output_tensors_unsorted[j].name.find(detect_layer_name[i]);
      if (pos != std::string::npos) {
        detect_output_tensors.emplace_back(output_tensors_unsorted[j]);
        if (ENV_PARAM(ENABLE_YOLOv7_DEBUG)) {
          LOG(INFO) << "pos:" << pos
                    << " find bbox layer:" << output_tensors_unsorted[j];
        }
        break;
      }
    }
  }

  std::vector<YOLOv7Result> res_vec;

  int batch_size = (ws.size() > detect_output_tensors[0].batch)
                       ? detect_output_tensors[0].batch
                       : ws.size();
  // channel is 255 which  is 3*(80+5)
  int anchor_cnt = 3;

  for (int k = 0; k < batch_size; k++) {
    vector<vector<float>> boxes;
    int out_num = detect_output_tensors.size();

    for (int i = 0; i < out_num; i++) {
      int width = detect_output_tensors[i].width;
      int height = detect_output_tensors[i].height;
      int sizeOut = detect_output_tensors[i].size;
      if (ENV_PARAM(ENABLE_YOLOv7_DEBUG)) {
        LOG(INFO) << "width=" << width << ", height=" << height
                  << ", sizeOut=" << sizeOut;
      }

      boxes.reserve(sizeOut);
      int8_t* det_out = (int8_t*)detect_output_tensors[i].get_data(k);
      float det_scale =
          vitis::ai::library::tensor_scale(detect_output_tensors[i]);
      int8_t conf_thresh_inverse =
          -std::log(1.0f / conf_thresh - 1) / det_scale;

      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          for (auto cnt = 0; cnt < anchor_cnt; cnt++) {
            int idx = 255 * (h * width + w) + cnt * 85;
            if (det_out[idx + 4] > conf_thresh_inverse) {
              vector<float> box;
              float obj_score = sigmoid(det_out[idx + 4] * det_scale);
              box.push_back((sigmoid(det_out[idx] * det_scale) * 2 - 0.5 + w) /
                            width);
              box.push_back(
                  (sigmoid(det_out[idx + 1] * det_scale) * 2 - 0.5 + h) /
                  height);
              box.push_back(pow(sigmoid(det_out[idx + 2] * det_scale) * 2, 2) *
                            biases[2 * cnt + 2 * anchor_cnt * i] /
                            (float)(sWidth));
              box.push_back(pow(sigmoid(det_out[idx + 3] * det_scale) * 2, 2) *
                            biases[2 * cnt + 2 * anchor_cnt * i + 1] /
                            (float)(sHeight));
              box.push_back(-1);
              box.push_back(obj_score);
              for (int p = 0; p < num_classes; p++) {
                box.push_back(obj_score *
                              sigmoid(det_out[idx + 5 + p] * det_scale));
              }
              boxes.push_back(box);
            }
          }
        }
      }
    }
    correct_region_boxes(boxes, boxes.size(), ws[k], hs[k], sWidth, sHeight);

    if (ENV_PARAM(ENABLE_YOLOv7_DEBUG)) {
      LOG(INFO) << "boxes_total_size=" << boxes.size();
    }

    __TIC__(YOLOv7_NMS)
    /* Apply the computation for NMS */
    vector<vector<float>> res;
    vector<float> scores(boxes.size());
    for (int k = 0; k < num_classes; k++) {
      transform(boxes.begin(), boxes.end(), scores.begin(), [k](auto& box) {
        box[4] = k;
        return box[6 + k];
      });
      vector<size_t> result_k;
      applyNMS(boxes, scores, nms_thresh, conf_thresh, result_k);
      transform(result_k.begin(), result_k.end(), back_inserter(res),
                [&boxes](auto& k) { return boxes[k]; });
    }

    if (ENV_PARAM(ENABLE_YOLOv7_DEBUG)) {
      LOG(INFO) << "res size: " << res.size();
      for (auto i = 0u; i < res.size(); ++i) {
        LOG(INFO) << "i = " << i << ", res size:" << res[i].size() << " ["
                  << res[i][0] << ", " << res[i][1] << ", " << res[i][2] << ", "
                  << res[i][3] << "], label:" << res[i][4]
                  << ", score: " << res[i][5];
      }
    }

    /* Restore the correct coordinate frame of the original image */
    vector<YOLOv7Result::BoundingBox> results;
    for (const auto& r : res) {
      auto score = r[r[4] + 6];
      if (score > conf_thresh) {
        YOLOv7Result::BoundingBox result;
        result.score = score;
        result.label = r[4];

        result.x = (r[0] - r[2] / 2.0f) ;
        result.y = (r[1] - r[3] / 2.0f ) ;
        result.width = r[2];
        result.height = r[3];
        results.push_back(result);
      }
    }
    res_vec.push_back(YOLOv7Result{sWidth, sHeight, results});
    __TOC__(YOLOv7_NMS)
  }
  return res_vec;
}
}  // namespace ai
}  // namespace vitis

