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

#include "vitis/ai/nnpp/yolov6.hpp"

#include <bits/stdc++.h>

#include <algorithm>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/math.hpp"
#include "vitis/ai/nnpp/apply_nms.hpp"
DEF_ENV_PARAM(ENABLE_YOLOV6_DEBUG, "0");
DEF_ENV_PARAM(DEBUG_YOLOV6_WODFL, "0");

using namespace std;
namespace vitis {
namespace ai {

static void detect_wodfl_int8(vector<vector<float>>& boxes,
                              int8_t* bbox_layer_ptr, int8_t* cls_layer_ptr,
                              int height, int width, float stride,
                              float conf_thresh, int anchor_cnt,
                              int num_classes, float bbox_scale,
                              float cls_scale) {
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < anchor_cnt; ++c) {
        auto idx = ((h * width + w) * anchor_cnt + c);
        auto cls_idx = idx * num_classes;
        vector<float> box(6);

        int max_p = -1;
        int8_t conf = -128;  // the range of int8_t is -128->127
        int8_t conf_thresh_inverse = -std::log(1.0f / conf_thresh - 1)/cls_scale;
        for (int p = 0; p < num_classes; p++) {
          if (cls_layer_ptr[cls_idx + p] < conf_thresh_inverse)
            continue;
          if (cls_layer_ptr[cls_idx + p] < conf) continue;
          max_p = p;
          conf = cls_layer_ptr[cls_idx + p];
        }
        if (max_p != -1) {
          box[4] = max_p;
          box[5] = 1.0f / (1.0f + std::exp(-conf * cls_scale));
          float x1y1_x = w + 0.5 - bbox_layer_ptr[idx * 4] * bbox_scale;
          float x1y1_y = h + 0.5 - bbox_layer_ptr[idx * 4 + 1] * bbox_scale;
          float x2y2_w = w + 0.5 + bbox_layer_ptr[idx * 4 + 2] * bbox_scale;
          float x2y2_h = h + 0.5 + bbox_layer_ptr[idx * 4 + 3] * bbox_scale;
          box[0] = (x1y1_x + x2y2_w) / 2 * stride;
          box[1] = (x1y1_y + x2y2_h) / 2 * stride;
          box[2] = (x2y2_w - x1y1_x) * stride;
          box[3] = (x2y2_h - x1y1_y) * stride;
          box[0] -= box[2] / 2;
          box[1] -= box[3] / 2;

          boxes.push_back(box);
        }
      }
    }
  }
}

static void detect_int8(vector<vector<float>>& boxes, int8_t* bbox_layer_ptr,
                        int8_t* cls_layer_ptr, int height, int width,
                        float stride, float conf_thresh, int anchor_cnt,
                        int num_classes, float bbox_scale, float cls_scale) {
  int size = height * width * anchor_cnt;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < anchor_cnt; ++c) {
        auto idx = ((h * width + w) * anchor_cnt + c);
        auto cls_idx = idx * num_classes;
        vector<float> box(6);
        int max_p = -1;
        int8_t conf = -128;  // the range of int8_t is -128->127
        int8_t conf_thresh_inverse = -std::log(1.0f / conf_thresh - 1)/cls_scale;
        for (int p = 0; p < num_classes; p++) {
          if (cls_layer_ptr[cls_idx + p] < conf_thresh_inverse)
            continue;
          if (cls_layer_ptr[cls_idx + p] * 1.0f < conf) continue;
          max_p = p;
          conf = cls_layer_ptr[cls_idx + p] ;
        }
        if (max_p != -1) {
          box[4] = max_p;
          box[5] = 1.0f / (1.0f + std::exp(-conf * cls_scale));
          float x1y1_x = w + 0.5 - bbox_layer_ptr[idx] * bbox_scale;
          float x1y1_y = h + 0.5 - bbox_layer_ptr[idx + size * 1] * bbox_scale;
          float x2y2_w = w + 0.5 + bbox_layer_ptr[idx + size * 2] * bbox_scale;
          float x2y2_h = h + 0.5 + bbox_layer_ptr[idx + size * 3] * bbox_scale;
          box[0] = (x1y1_x + x2y2_w) / 2 * stride;
          box[1] = (x1y1_y + x2y2_h) / 2 * stride;
          box[2] = (x2y2_w - x1y1_x) * stride;
          box[3] = (x2y2_h - x1y1_y) * stride;
          box[0] -= box[2] / 2;
          box[1] -= box[3] / 2;
          boxes.push_back(box);
        }
      }
    }
  }
}

static void detect_wodfl(vector<vector<float>>& boxes, float* bbox_layer_ptr,
                         float* cls_layer_ptr, int height, int width,
                         float stride, float conf_thresh, int anchor_cnt,
                         int num_classes) {
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < anchor_cnt; ++c) {
        auto idx = ((h * width + w) * anchor_cnt + c);
        auto cls_idx = idx * num_classes;
        vector<float> box(6);

        float x1y1_x = w + 0.5 - bbox_layer_ptr[idx * 4];
        float x1y1_y = h + 0.5 - bbox_layer_ptr[idx * 4 + 1];
        float x2y2_w = w + 0.5 + bbox_layer_ptr[idx * 4 + 2];
        float x2y2_h = h + 0.5 + bbox_layer_ptr[idx * 4 + 3];
        box[0] = (x1y1_x + x2y2_w) / 2 * stride;
        box[1] = (x1y1_y + x2y2_h) / 2 * stride;
        box[2] = (x2y2_w - x1y1_x) * stride;
        box[3] = (x2y2_h - x1y1_y) * stride;
        box[0] -= box[2] / 2;
        box[1] -= box[3] / 2;

        int max_p = -1;
        float conf = 0.f;
        for (int p = 0; p < num_classes; p++) {
          if (cls_layer_ptr[cls_idx + p] < conf_thresh) continue;
          if (cls_layer_ptr[cls_idx + p] < conf) continue;
          max_p = p;
          conf = cls_layer_ptr[cls_idx + p];
        }
        if (max_p != -1) {
          box[4] = max_p;
          box[5] = conf;
          boxes.push_back(box);
        }
      }
    }
  }
}

static void detect(vector<vector<float>>& boxes, float* bbox_layer_ptr,
                   float* cls_layer_ptr, int height, int width, float stride,
                   float conf_thresh, int anchor_cnt, int num_classes) {
  int size = height * width * anchor_cnt;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < anchor_cnt; ++c) {
        auto idx = ((h * width + w) * anchor_cnt + c);
        auto cls_idx = idx * num_classes;
        vector<float> box(6);

        float x1y1_x = w + 0.5 - bbox_layer_ptr[idx];
        float x1y1_y = h + 0.5 - bbox_layer_ptr[idx + size * 1];
        float x2y2_w = w + 0.5 + bbox_layer_ptr[idx + size * 2];
        float x2y2_h = h + 0.5 + bbox_layer_ptr[idx + size * 3];
        box[0] = (x1y1_x + x2y2_w) / 2 * stride;
        box[1] = (x1y1_y + x2y2_h) / 2 * stride;
        box[2] = (x2y2_w - x1y1_x) * stride;
        box[3] = (x2y2_h - x1y1_y) * stride;
        box[0] -= box[2] / 2;
        box[1] -= box[3] / 2;

        // box[0] = (w - bbox_layer_ptr[idx]) * stride;
        // box[1] = (h - bbox_layer_ptr[idx + size * 1]) * stride;
        // box[2] = (w + bbox_layer_ptr[idx + size * 2]) * stride;
        // box[3] = (h + bbox_layer_ptr[idx + size * 3]) * stride;
        // box[2] = box[2] - box[0];
        // box[3] = box[3] - box[1];

        int max_p = -1;
        float conf = 0.f;
        for (int p = 0; p < num_classes; p++) {
          if (cls_layer_ptr[cls_idx + p] < conf_thresh) continue;
          if (cls_layer_ptr[cls_idx + p] < conf) continue;
          max_p = p;
          conf = cls_layer_ptr[cls_idx + p];
        }
        if (max_p != -1) {
          box[4] = max_p;
          box[5] = conf;
          boxes.push_back(box);
        }
      }
    }
  }
}

void yolov6_middle_process(
    const vitis::ai::library::InputTensor& input_tensor,
    const vitis::ai::library::OutputTensor& output_tensor) {
  if (ENV_PARAM(ENABLE_YOLOV6_DEBUG)) {
    LOG(INFO) << " middle output layer:" << output_tensor;
    LOG(INFO) << " middle input layer:" << input_tensor;
  }
  auto batch = output_tensor.batch;
  auto o_w = output_tensor.width;
  auto o_h = output_tensor.height;
  auto o_c = output_tensor.channel;
  auto inner_c = o_c / 4;
  auto HW = o_w * o_h;

  float in_scale = vitis::ai::library::tensor_scale(input_tensor);
  float out_scale = vitis::ai::library::tensor_scale(output_tensor);
  auto size = input_tensor.size / batch;
  std::vector<float> tmp(size);

  if (ENV_PARAM(ENABLE_YOLOV6_DEBUG)) {
    LOG(INFO) << "size:" << size;
  }

  for (auto b = 0u; b < batch; ++b) {
    int8_t* out_ptr = (int8_t*)output_tensor.get_data(b);
    int8_t* in_ptr = (int8_t*)input_tensor.get_data(b);
    for (auto hw = 0u; hw < HW; hw++) {
      for (auto i = 0u; i < 4; i++) {
        std::memcpy(in_ptr + (i * HW + hw) * inner_c,
                    out_ptr + hw * o_c + i * inner_c, inner_c);
      }
    }

    auto group = o_w * o_h * 4;
    vitis::ai::softmax(in_ptr, out_scale, inner_c, group, tmp.data());

    std::transform(tmp.begin(), tmp.end(), in_ptr, [=](const float& a) {
      return (int8_t)(std::round(a * in_scale));
    });
  }
}

std::vector<YOLOv6Result> yolov6_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>&
        output_tensors_unsorted,
    const vitis::ai::proto::DpuModelParam& config,
    const std::vector<float>& img_scale, const std::vector<int>& left,
    const std::vector<int>& top) {
  auto use_graph_runner = config.use_graph_runner();
  auto& yolo_v6_params = config.yolo_v6_param();

  auto num_classes = yolo_v6_params.num_classes();
  auto anchor_cnt = yolo_v6_params.anchorcnt();
  auto nms_thresh = yolo_v6_params.nms_threshold();
  auto conf_thresh = yolo_v6_params.conf_threshold();
  auto max_nms_num = yolo_v6_params.max_nms_num();
  auto without_dfl = yolo_v6_params.without_dfl();

  std::vector<float> stride(yolo_v6_params.stride().begin(),
                            yolo_v6_params.stride().end());

  std::vector<std::string> bbox_layername(
      yolo_v6_params.bbox_layer_name().begin(),
      yolo_v6_params.bbox_layer_name().end());

  std::vector<std::string> cls_layername(
      yolo_v6_params.cls_layer_name().begin(),
      yolo_v6_params.cls_layer_name().end());

  std::vector<vitis::ai::library::OutputTensor> bbox_output_tensors;
  std::vector<vitis::ai::library::OutputTensor> cls_output_tensors;
  for (auto i = 0u; i < bbox_layername.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted.size(); j++) {
      auto pos = output_tensors_unsorted[j].name.find(bbox_layername[i]);
      if (pos != std::string::npos &&
          pos + bbox_layername[i].size() ==
              output_tensors_unsorted[j].name.size()) {
        // if (output_tensors_unsorted[j].name.find(bbox_layername[i]) !=
        //    std::string::npos) {
        bbox_output_tensors.push_back(output_tensors_unsorted[j]);
        if (ENV_PARAM(ENABLE_YOLOV6_DEBUG)) {
          LOG(INFO) << "pos:" << pos
                    << " find bbox layer:" << output_tensors_unsorted[j];
        }
        break;
      }
    }
  }

  for (auto i = 0u; i < cls_layername.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted.size(); j++) {
      auto pos = output_tensors_unsorted[j].name.find(cls_layername[i]);
      if (pos != std::string::npos &&
          pos + cls_layername[i].size() ==
              output_tensors_unsorted[j].name.size()) {
        // if (output_tensors_unsorted[j].name.find(cls_layername[i]) !=
        //    std::string::npos) {
        cls_output_tensors.push_back(output_tensors_unsorted[j]);
        if (ENV_PARAM(ENABLE_YOLOV6_DEBUG)) {
          LOG(INFO) << "pos:" << pos
                    << "find cls layer:" << output_tensors_unsorted[j];
        }
        break;
      }
    }
  }

  std::vector<YOLOv6Result> res_vec;

  int batch = bbox_output_tensors[0].batch;

  int batch_size = (img_scale.size() > bbox_output_tensors[0].batch)
                       ? bbox_output_tensors[0].batch
                       : img_scale.size();

  for (int k = 0; k < batch_size; k++) {
    vector<vector<float>> boxes;
    int out_num = bbox_output_tensors.size();
    for (int i = (out_num - 1); i >= 0; i--) {
      int width = cls_output_tensors[i].width;
      int height = cls_output_tensors[i].height;

      int sizeOut;
      if (use_graph_runner) {
        float* bbox_out = (float*)bbox_output_tensors[i].get_data(k);
        float* cls_out = (float*)cls_output_tensors[i].get_data(k);

        sizeOut = bbox_output_tensors[i].size / batch / sizeof(float);
        boxes.reserve(boxes.size() + sizeOut);
        if (without_dfl) {
          detect_wodfl(boxes, bbox_out, cls_out, height, width, stride[i],
                       conf_thresh, anchor_cnt, num_classes);
        } else {
          detect(boxes, bbox_out, cls_out, height, width, stride[i],
                 conf_thresh, anchor_cnt, num_classes);
        }
      } else {
        sizeOut = bbox_output_tensors[i].size / batch;
        float bbox_scale =
            vitis::ai::library::tensor_scale(bbox_output_tensors[i]);
        float cls_scale =
            vitis::ai::library::tensor_scale(cls_output_tensors[i]);
        int8_t* bbox_out = (int8_t*)bbox_output_tensors[i].get_data(k);
        int8_t* cls_out = (int8_t*)cls_output_tensors[i].get_data(k);
        boxes.reserve(boxes.size() + sizeOut);
        if (without_dfl) {
          detect_wodfl_int8(boxes, bbox_out, cls_out, height, width, stride[i],
                            conf_thresh, anchor_cnt, num_classes, bbox_scale,
                            cls_scale);
        } else {
          detect_int8(boxes, bbox_out, cls_out, height, width, stride[i],
                      conf_thresh, anchor_cnt, num_classes, bbox_scale,
                      cls_scale);
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
      if (ENV_PARAM(ENABLE_YOLOV6_DEBUG)) {
        LOG(INFO) << "class: " << i << " size:" << boxes_for_nms[i].size();
      }
      applyNMS(boxes_for_nms[i], scores[i], nms_thresh, conf_thresh, result_k,
               false);
      if (result_k.size() > max_nms_num) {
        result_k.resize(max_nms_num);
      }
      res.reserve(res.size() + result_k.size());
      transform(result_k.begin(), result_k.end(), back_inserter(res),
                [&](auto& k) { return boxes_for_nms[i][k]; });
    }

    if (ENV_PARAM(ENABLE_YOLOV6_DEBUG)) {
      LOG(INFO) << "res size: " << res.size();
      for (auto i = 0u; i < res.size(); ++i) {
        LOG(INFO) << "i = " << i << ", res size:" << res[i].size() << "["
                  << res[i][0] << ", " << res[i][1] << ", " << res[i][2] << ", "
                  << res[i][3] << "]" << res[i][4] << " ,score: " << res[i][5];
      }
    }
    auto compare = [=](vector<float>& lhs, vector<float>& rhs) {
      // LOG(ERROR) << "lhs = " << lhs.size()
      //           << ", rhs: " << rhs.size();
      // LOG(ERROR) << "lhs = " << lhs[0] << " " << lhs[1] <<
      // " "
      //           << lhs[2] << " " << lhs[3] << " " << lhs[4]
      //           << ", rhs: " << rhs[0] << " " << rhs[1];

      return lhs[5] > rhs[5];
    };
    if (res.size() > max_nms_num) {
      std::partial_sort(res.begin(), res.begin() + max_nms_num, res.end(),
                        compare);
      res.resize(max_nms_num);
    } else {
      std::sort(res.begin(), res.end(), compare);
    }

    /* Restore the correct coordinate frame of the original image */
    vector<YOLOv6Result::BoundingBox> results;
    for (const auto& r : res) {
      if (r[5] > conf_thresh) {
        YOLOv6Result::BoundingBox yolo_res;
        yolo_res.score = r[5];
        yolo_res.label = r[4];
        yolo_res.box.resize(4);

        yolo_res.box[0] = (r[0] - left[k]) / img_scale[k];
        yolo_res.box[1] = (r[1] - top[k]) / img_scale[k];
        // yolo_res.box[0] = r[0] / img_scale[k];
        // yolo_res.box[1] = r[1] / img_scale[k];
        yolo_res.box[2] = yolo_res.box[0] + r[2] / img_scale[k];
        yolo_res.box[3] = yolo_res.box[1] + r[3] / img_scale[k];
        results.push_back(yolo_res);
      }
    }
    res_vec.push_back(YOLOv6Result{results});
  }
  return res_vec;
}

}  // namespace ai
}  // namespace vitis

