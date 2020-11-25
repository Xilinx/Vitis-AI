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

#include "vitis/ai/nnpp/yolov2.hpp"

#include <vector>

#include "vitis/ai/nnpp/apply_nms.hpp"

using namespace std;
namespace vitis {
namespace ai {

static float sigmoid(float p) { return 1.0 / (1 + exp(-p * 1.0)); }

static void correct_region_boxes(vector<vector<float>>& boxes, int n, int w,
                                 int h, int netw, int neth, int relative = 0) {
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

static void detect(vector<vector<float>>& boxes, int8_t* result, int height,
                   int width, int num, int sHeight, int sWidth, float scale,
                   const vitis::ai::proto::DpuModelParam& config) {
  auto& yolo_params = config.yolo_v3_param();

  auto num_classes = yolo_params.num_classes();
  auto anchor_cnt = yolo_params.anchorcnt();
  auto conf_thresh = yolo_params.conf_threshold();
  auto biases = std::vector<float>(yolo_params.biases().begin(),
                                   yolo_params.biases().end());
  auto conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f);
  int conf_box = 5 + num_classes;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < anchor_cnt; ++c) {
        int idx = ((h * width + w) * anchor_cnt + c) * conf_box;
        if (result[idx + 4] * scale <= conf_desigmoid) continue;
        float obj_score = sigmoid(result[idx + 4] * scale);
        vector<float> cls;
        float s = 0.0;
        for (int i = 0; i < num_classes; ++i)
          cls.push_back(result[idx + 5 + i] * scale);
        float large = *max_element(cls.begin(), cls.end());
        for (size_t i = 0; i < cls.size(); ++i) {
          cls[i] = exp(cls[i] - large);
          s += cls[i];
        }
        vector<float>::iterator biggest = max_element(cls.begin(), cls.end());
        large = *biggest;
        for (size_t i = 0; i < cls.size(); ++i) cls[i] = cls[i] * 1.0 / s;
        vector<float> box;

        box.push_back((w + sigmoid(result[idx] * scale)) / float(width));
        box.push_back((h + sigmoid(result[idx + 1] * scale)) / float(height));
        box.push_back(exp(result[idx + 2] * scale) *
                      biases[2 * c + 2 * anchor_cnt * num] / float(width));
        box.push_back(exp(result[idx + 3] * scale) *
                      biases[2 * c + 2 * anchor_cnt * num + 1] / float(height));
        box.push_back(-1);
        box.push_back(obj_score);
        for (int p = 0; p < num_classes; p++) {
          box.push_back(obj_score * cls[p]);
        }
        boxes.push_back(box);
      }
    }
  }
}

YOLOv2Result yolov2_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const int w, const int h) {
  std::vector<int> ws{w};
  std::vector<int> hs{h};
  return yolov2_post_process(input_tensors, output_tensors, config, ws, hs)[0];
}

std::vector<YOLOv2Result> yolov2_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const std::vector<int>& w,
    const std::vector<int>& h) {
  int sWidth = input_tensors[0].width;
  int sHeight = input_tensors[0].height;

  auto num_classes = config.yolo_v3_param().num_classes();
  auto nms_thresh = config.yolo_v3_param().nms_threshold();
  auto conf_thresh = config.yolo_v3_param().conf_threshold();
  auto mAP = config.yolo_v3_param().test_map();

  std::vector<YOLOv2Result> res_vec;
  int batch_size =
      (w.size() > output_tensors[0].batch) ? output_tensors[0].batch : w.size();
  for (int j = 0; j < batch_size; j++) {
    /* four output nodes of YOLO-v2 */
    vector<vector<float>> boxes;
    int out_num = output_tensors.size();
    for (int i = 0; i < out_num; i++) {
      int width = output_tensors[i].width;
      int height = output_tensors[i].height;

      int sizeOut = output_tensors[i].size;
      int8_t* dpuOut = (int8_t*)output_tensors[i].get_data(j);
      float scale = vitis::ai::library::tensor_scale(output_tensors[i]);
      boxes.reserve(sizeOut);
      /* Store the object detection frames as coordinate information  */
      detect(boxes, dpuOut, height, width, i, sHeight, sWidth, scale, config);
    }
    /* Restore the correct coordinate frame of the original image */
    if (mAP) {
      correct_region_boxes(boxes, boxes.size(), w[j], h[j], sWidth, sHeight);
    }

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

    vector<YOLOv2Result::BoundingBox> results;
    for (size_t i = 0; i < res.size(); ++i) {
      int label = res[i][4];
      if (res[i][label + 6] > conf_thresh) {
        YOLOv2Result::BoundingBox yolo_res;
        yolo_res.score = res[i][label + 6];
        yolo_res.label = res[i][4];
        yolo_res.x = res[i][0] - res[i][2] / 2.0;
        yolo_res.y = res[i][1] - res[i][3] / 2.0;
        if (yolo_res.x < 0) yolo_res.x = 0;
        if (yolo_res.y < 0) yolo_res.y = 0;
        yolo_res.width = res[i][2];
        yolo_res.height = res[i][3];
        results.push_back(yolo_res);
      }
    }
    res_vec.push_back(YOLOv2Result{sWidth, sHeight, results});
  }
  return res_vec;
}

}  // namespace ai
}  // namespace vitis
