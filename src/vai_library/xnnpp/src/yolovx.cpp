/*
 * Copyright 2021 Xilinx Inc.
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

#include "vitis/ai/nnpp/yolovx.hpp"

#include <bits/stdc++.h>

#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/nnpp/apply_nms.hpp"
DEF_ENV_PARAM(ENABLE_YOLOV5_DEBUG, "0");

using namespace std;
namespace vitis {
namespace ai {

static float sigmoid(float p) { return 1.0f / (1.0f + exp(-p)); }

static void detect(vector<vector<float>>& boxes, int8_t* result, int height,
                   int width, float scale, float stride, float conf_thresh,
                   int anchor_cnt, int num_classes, int yolo_type = 0) {
  auto conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f) / scale;
  int conf_box = 5 + num_classes;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < anchor_cnt; ++c) {
        auto idx = ((h * width + w) * anchor_cnt + c) * conf_box;
        if (float(result[idx + 4]) < conf_desigmoid) continue;
        vector<float> box(6);
        box[0] = (w + result[idx] * scale) * stride;
        box[1] = (h + result[idx + 1] * scale) * stride;
        box[2] = std::exp(result[idx + 2] * scale) * stride;
        box[3] = std::exp(result[idx + 3] * scale) * stride;
        float obj_score = sigmoid(result[idx + 4] * scale);
        auto conf_class_desigmoid =
            -logf(obj_score / conf_thresh - 1.0f) / scale;
        int max_p = -1;

        if (yolo_type == 3) {  // YOLOX_NANO
          box[0] = box[0] - box[2] * 0.5;
          box[1] = box[1] - box[3] * 0.5;
        }
        for (int p = 0; p < num_classes; p++) {
          if (float(result[idx + 5 + p]) < conf_class_desigmoid) continue;
          max_p = p;
          conf_class_desigmoid = result[idx + 5 + p];
        }
        if (max_p != -1) {
          box[4] = max_p;
          box[5] = obj_score * sigmoid(conf_class_desigmoid * scale);
          boxes.push_back(box);
        }
      }
    }
  }
}

std::vector<YOLOvXResult> yolovx_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>&
        output_tensors_unsorted,
    const vitis::ai::proto::DpuModelParam& config,
    const std::vector<float>& img_scale) {
  auto& yolo_v5_params = config.yolo_v5_param();
  auto& yolo_params = yolo_v5_params.yolo_param();
  auto yolo_type = yolo_params.type();

  auto num_classes = yolo_params.num_classes();
  auto anchor_cnt = yolo_params.anchorcnt();
  auto nms_thresh = yolo_params.nms_threshold();
  auto conf_thresh = yolo_params.conf_threshold();
  std::vector<float> stride(yolo_v5_params.stride().begin(),
                            yolo_v5_params.stride().end());

  std::vector<std::string> layername(yolo_params.layer_name().begin(),
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

  std::vector<YOLOvXResult> res_vec;
  int batch_size = (img_scale.size() > output_tensors[0].batch)
                       ? output_tensors[0].batch
                       : img_scale.size();
  for (int k = 0; k < batch_size; k++) {
    /* four output nodes of YOLO-v3 */
    vector<vector<float>> boxes;
    int out_num = output_tensors.size();
    for (int i = (out_num - 1); i >= 0; i--) {
      int width = output_tensors[i].width;
      int height = output_tensors[i].height;

      int sizeOut = output_tensors[i].size;
      int8_t* dpuOut = (int8_t*)output_tensors[i].get_data(k);
      float scale = vitis::ai::library::tensor_scale(output_tensors[i]);
      boxes.reserve(boxes.size() + sizeOut);

      /* Store the object detection frames as coordinate information  */
      if (ENV_PARAM(ENABLE_YOLOV5_DEBUG)) {
        LOG(INFO) << "i:" << i << "anchor:" << anchor_cnt << ", w:" << width
                  << ", h:" << height << "s:" << stride[i]
                  << ", c:" << output_tensors[i].channel;
      }
      detect(boxes, dpuOut, height, width, scale, stride[i], conf_thresh,
             anchor_cnt, num_classes, yolo_type);
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
      applyNMS(boxes_for_nms[i], scores[i], nms_thresh, conf_thresh, result_k,
               false);
      res.reserve(res.size() + result_k.size());
      transform(result_k.begin(), result_k.end(), back_inserter(res),
                [&](auto& k) { return boxes_for_nms[i][k]; });
    }

    /* Restore the correct coordinate frame of the original image */
    vector<YOLOvXResult::BoundingBox> results;
    for (const auto& r : res) {
      if (r[5] > conf_thresh) {
        YOLOvXResult::BoundingBox yolo_res;
        yolo_res.score = r[5];
        yolo_res.label = r[4];
        yolo_res.box.resize(4);

        if (yolo_type == 3) {  // 3: YOLOV5_NANO
          yolo_res.box[0] = r[0] / img_scale[k];
          yolo_res.box[1] = r[1] / img_scale[k];
          yolo_res.box[2] = yolo_res.box[0] + r[2] / img_scale[k];
          yolo_res.box[3] = yolo_res.box[1] + r[3] / img_scale[k];
        } else {
          yolo_res.box[0] = (r[0] - r[2] / 2.0) / img_scale[k];
          yolo_res.box[1] = (r[1] - r[3] / 2.0) / img_scale[k];
          yolo_res.box[2] = yolo_res.box[0] + r[2] / img_scale[k];
          yolo_res.box[3] = yolo_res.box[1] + r[3] / img_scale[k];
        }
        results.push_back(yolo_res);
      }
    }
    res_vec.push_back(YOLOvXResult{results});
  }
  return res_vec;
}

}  // namespace ai
}  // namespace vitis
