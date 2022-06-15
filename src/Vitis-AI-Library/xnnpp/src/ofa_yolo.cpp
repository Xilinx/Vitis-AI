/*
 * Copyright 2022 xilinx Inc.
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

#include "vitis/ai/nnpp/ofa_yolo.hpp"

#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/nnpp/apply_nms.hpp"
DEF_ENV_PARAM(ENABLE_OFAYOLO_DEBUG, "0");

using namespace std;
namespace vitis {
namespace ai {

static float sigmoid(float p) { return 1.0 / (1 + exp(-p * 1.0)); }

static void correct_region_boxes(vector<vector<float>>& boxes, int w, int h,
                                 int netw, int neth, int relative = 0) {
  float scale = min((float)netw / (float)w, (float)neth / (float)h);

  auto dw = float(netw - round(w * scale)) / 2.0f;
  auto dh = float(neth - round(h * scale)) / 2.0f;

  for (auto& box : boxes) {
    box[0] = (box[0] - dw) / scale / w * netw;
    box[1] = (box[1] - dh) / scale / h * neth;
    box[2] = box[2] / scale / w * netw;
    box[3] = box[3] / scale / h * neth;
  }
}

template <class T>
void print(vector<T> date) {
  for (auto d : date) {
    cout << d << " ";
  }
  cout << endl;
}
template <class T>
void print(vector<vector<T>> date) {
  int i = 0;
  for (auto d : date) {
    if (i++ < 10) print(d);
  }
}
static void detect(vector<vector<float>>& boxes, int8_t* result, int height,
                   int width, int num, float scale,
                   const vitis::ai::proto::DpuModelParam& config) {
  LOG_IF(INFO, ENV_PARAM(ENABLE_OFAYOLO_DEBUG))
      << " height " << height << " width " << width << " num " << num
      << " scale " << scale;
  auto& ofa_yolo_params = config.yolo_v5_param();
  auto& yolo_params = ofa_yolo_params.yolo_param();

  auto num_classes = yolo_params.num_classes();
  auto anchor_cnt = yolo_params.anchorcnt();
  auto conf_thresh = yolo_params.conf_threshold();
  auto biases = std::vector<float>(yolo_params.biases().begin(),
                                   yolo_params.biases().end());
  auto conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f) / scale;

  auto stride = std::vector<float>(ofa_yolo_params.stride().begin(),
                                   ofa_yolo_params.stride().end());
  int conf_box = 5 + num_classes;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < anchor_cnt; ++c) {
        auto idx = ((h * width + w) * anchor_cnt + c) * conf_box;
        if (result[idx + 4] < conf_desigmoid) continue;
        vector<float> box(6);
        box[0] = (w + sigmoid(result[idx] * scale) * 2.0f - 0.5f) * stride[num];

        box[1] =
            (h + sigmoid(result[idx + 1] * scale) * 2.0f - 0.5f) * stride[num];
        box[2] = pow(sigmoid(result[idx + 2] * scale) * 2, 2) *
                 biases[2 * c + 2 * anchor_cnt * num];
        box[3] = pow(sigmoid(result[idx + 3] * scale) * 2, 2) *
                 biases[2 * c + 2 * anchor_cnt * num + 1];
        float obj_score = sigmoid(result[idx + 4] * scale);
        auto conf_class_desigmoid =
            -logf(obj_score / conf_thresh - 1.0f) / scale;

        for (int p = 0; p < num_classes; p++) {
          if (result[idx + 5 + p] < conf_class_desigmoid) continue;
          box[4] = p;
          box[5] = obj_score * sigmoid(result[idx + 5 + p] * scale);
          boxes.push_back(box);
        }
      }
    }
  }
  LOG_IF(INFO, ENV_PARAM(ENABLE_OFAYOLO_DEBUG)) << boxes.size();
}

OFAYOLOResult ofa_yolo_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const int w, const int h) {
  return ofa_yolo_post_process(input_tensors, output_tensors, config,
                               vector<int>(1, w), vector<int>(1, h))[0];
}

std::vector<OFAYOLOResult> ofa_yolo_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>&
        output_tensors_unsorted,
    const vitis::ai::proto::DpuModelParam& config, const std::vector<int>& ws,
    const std::vector<int>& hs) {
  auto& ofa_yolo_params = config.yolo_v5_param();
  auto& yolo_params = ofa_yolo_params.yolo_param();
  auto max_boxes_num = ofa_yolo_params.max_boxes_num();
  auto max_nms_num = ofa_yolo_params.max_nms_num();

  auto num_classes = yolo_params.num_classes();
  auto conf_thresh = yolo_params.conf_threshold();
  auto nms_thresh = yolo_params.nms_threshold();
  auto layername = std::vector<std::string>(yolo_params.layer_name().begin(),
                                            yolo_params.layer_name().end());

  std::vector<vitis::ai::library::OutputTensor> output_tensors;
  for (auto i = 0u; i < layername.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted.size(); j++) {
      if (output_tensors_unsorted[j].name.find(layername[i]) !=
          std::string::npos) {
        output_tensors.push_back(output_tensors_unsorted[j]);
        break;
      }
    }
  }

  std::vector<OFAYOLOResult> res_vec;
  int batch_size = (ws.size() > output_tensors[0].batch)
                       ? output_tensors[0].batch
                       : ws.size();
  for (int k = 0; k < batch_size; k++) {
    /* four output nodes of YOLO-v3 */
    vector<vector<float>> boxes;
    int out_num = output_tensors.size();
    int j = 0;
    for (int i = (out_num - 1); i >= 0; i--) {
      int width = output_tensors[i].width;
      int height = output_tensors[i].height;

      int sizeOut = output_tensors[i].size;
      int8_t* dpuOut = (int8_t*)output_tensors[i].get_data(k);
      float scale = vitis::ai::library::tensor_scale(output_tensors[i]);
      boxes.reserve(boxes.size() + sizeOut);

      /* Store the object detection frames as coordinate information  */
      detect(boxes, dpuOut, height, width, j++, scale, config);
    }

    /* Apply the computation for NMS */
    if (boxes.size() > max_boxes_num) {
      stable_sort(boxes.begin(), boxes.end(),
                  [](vector<float> l, vector<float> r) { return l[5] > r[5]; });
      boxes.resize(max_boxes_num);
    }

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

    stable_sort(res.begin(), res.end(),
                [](vector<float> l, vector<float> r) { return l[5] > r[5]; });
    if (res.size() > max_nms_num) {
      res.resize(max_nms_num);
    }

    int sWidth = input_tensors[0].width;
    int sHeight = input_tensors[0].height;
    /* Restore the correct coordinate frame of the original image */
    correct_region_boxes(res, ws[k], hs[k], sWidth, sHeight);
    vector<OFAYOLOResult::BoundingBox> results;
    for (const auto& r : res) {
      if (r[5] > conf_thresh) {
        OFAYOLOResult::BoundingBox yolo_res;
        yolo_res.score = r[5];
        yolo_res.label = r[4];
        yolo_res.x = (r[0] - r[2] / 2.0) / sWidth;
        yolo_res.y = (r[1] - r[3] / 2.0) / sHeight;
        yolo_res.width = r[2] / sWidth;
        yolo_res.height = r[3] / sHeight;
        results.push_back(yolo_res);
      }
    }
    res_vec.push_back(OFAYOLOResult{sWidth, sHeight, results});
  }
  return res_vec;
}

}  // namespace ai
}  // namespace vitis
