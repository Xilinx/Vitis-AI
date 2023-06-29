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

#include "vitis/ai/nnpp/fcos.hpp"

#include <algorithm>
#include <cmath>  // sqrt()
#include <fstream>
#include <numeric>  // iota()
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/math.hpp"
#include "vitis/ai/nnpp/apply_nms.hpp"
DEF_ENV_PARAM(ENABLE_FCOS_DEBUG, "0");

using namespace std;
namespace vitis {
namespace ai {

static void compute_location(vector<vector<float>>& locations, const int width,
                             const int height, const int stride) {
  vector<float> shifts_x, shifts_y;

  for (int i = 0; i < width; i++) {
    shifts_x.emplace_back(i * stride);
  }

  for (int i = 0; i < height; i++) {
    shifts_y.emplace_back(i * stride);
  }

  for (auto n = 0; n < height; ++n) {
    for (auto m = 0; m < width; ++m) {
      locations[n * width + m][0] = shifts_x[m] + stride / 2;
      locations[n * width + m][1] = shifts_y[n] + stride / 2;
    }
  }
}

std::vector<FCOSResult> fcos_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>&
        output_tensors_unsorted,
    const vitis::ai::proto::DpuModelParam& config, const std::vector<int>& ws,
    const std::vector<int>& hs) {
  int sWidth = input_tensors[0].width;
  int sHeight = input_tensors[0].height;

  auto& fcos_params = config.fcos_param();
  auto conf_thresh = fcos_params.pre_nms_thresh();
  auto pre_nms_top_n = fcos_params.pre_nms_top_n();
  auto nms_thresh = fcos_params.nms_threshold();
  auto max_nms_num = fcos_params.fpn_post_nms_top_n();
  auto num_classes = fcos_params.num_classes();

  std::vector<float> stride(fcos_params.stride().begin(),
                            fcos_params.stride().end());

  std::vector<std::string> cls_layername(fcos_params.cls_layer_name().begin(),
                                         fcos_params.cls_layer_name().end());
  // layer which store (left, top, right, bottom)
  std::vector<std::string> offset_layername(
      fcos_params.offset_layer_name().begin(),
      fcos_params.offset_layer_name().end());
  std::vector<std::string> center_layername(
      fcos_params.center_layer_name().begin(),
      fcos_params.center_layer_name().end());

  std::vector<vitis::ai::library::OutputTensor> cls_output_tensors;
  std::vector<vitis::ai::library::OutputTensor> offset_output_tensors;
  std::vector<vitis::ai::library::OutputTensor> center_output_tensors;

  for (auto i = 0u; i < cls_layername.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted.size(); j++) {
      auto pos = output_tensors_unsorted[j].name.find(cls_layername[i]);
      if (pos != std::string::npos &&
          pos + cls_layername[i].size() ==
              output_tensors_unsorted[j].name.size()) {
        cls_output_tensors.push_back(output_tensors_unsorted[j]);
        break;
      }
    }
  }

  for (auto i = 0u; i < offset_layername.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted.size(); j++) {
      auto pos = output_tensors_unsorted[j].name.find(offset_layername[i]);
      if (pos != std::string::npos &&
          pos + offset_layername[i].size() ==
              output_tensors_unsorted[j].name.size()) {
        offset_output_tensors.push_back(output_tensors_unsorted[j]);
        break;
      }
    }
  }

  for (auto i = 0u; i < center_layername.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted.size(); j++) {
      auto pos = output_tensors_unsorted[j].name.find(center_layername[i]);
      if (pos != std::string::npos &&
          pos + center_layername[i].size() ==
              output_tensors_unsorted[j].name.size()) {
        center_output_tensors.push_back(output_tensors_unsorted[j]);
        break;
      }
    }
  }

  std::vector<FCOSResult> res_vec;

  int batch_size = (ws.size() > cls_output_tensors[0].batch)
                       ? cls_output_tensors[0].batch
                       : ws.size();

  for (int k = 0; k < batch_size; k++) {
    vector<vector<float>> boxes;
    int out_num = cls_output_tensors.size();

    for (int i = 0; i < out_num; i++) {
      int width = cls_output_tensors[i].width;
      int height = cls_output_tensors[i].height;

      float cls_scale = vitis::ai::library::tensor_scale(cls_output_tensors[i]);
      float offset_scale =
          vitis::ai::library::tensor_scale(offset_output_tensors[i]);
      float center_scale =
          vitis::ai::library::tensor_scale(center_output_tensors[i]);
      int8_t* cls_out = (int8_t*)cls_output_tensors[i].get_data(k);
      int8_t* offset_out = (int8_t*)offset_output_tensors[i].get_data(k);
      int8_t* center_out = (int8_t*)center_output_tensors[i].get_data(k);

      int sizeOut = width * height;

      std::vector<vector<float>> location_out_float(sizeOut);

      for (auto n = 0; n < sizeOut; ++n) location_out_float[n].resize(2);

      __TIC__(COMPUTE_LOCATION)
      compute_location(location_out_float, width, height, stride[i]);
      __TOC__(COMPUTE_LOCATION)

      std::vector<vector<float>> pre_offset;
      std::vector<vector<float>> pre_locations;
      std::vector<int> pre_class;
      std::vector<float> pre_class_value;

      __TIC__(FCOS_STORE)
      int count = 0;
      int8_t conf_thresh_inverse =
          -std::log(1.0f / conf_thresh - 1) / cls_scale;
      for (auto n = 0; n < sizeOut; ++n) {
        for (auto m = 0; m < num_classes; ++m) {
          if (cls_out[n * num_classes + m] > conf_thresh_inverse) {
            count++;
            vector<float> offset_out_float(4);
            for (auto t = 0; t < 4; t++) {
              offset_out_float[t] =
                  offset_out[n * 4 + t] * offset_scale * stride[i];
            }

            float cls_out_float =
                1.0 /
                (1 + exp(-1.0f * cls_out[n * num_classes + m] * cls_scale));
            float center_out_float =
                1.0 / (1 + exp(-1.0f * center_out[n] * center_scale));

            pre_offset.emplace_back(offset_out_float);
            pre_class.emplace_back(m);
            pre_class_value.emplace_back(cls_out_float * center_out_float);
            pre_locations.emplace_back(location_out_float[n]);
          }
        }
      }
      __TOC__(FCOS_STORE)

      __TIC__(FCOS_SORT)
      vector<int> top_k_indices;
      if (count > pre_nms_top_n) {
        vector<int> indices(pre_class_value.size());
        std::iota(indices.begin(), indices.end(), 0);
        // Sorting the indices based on the scores
        sort(indices.begin(), indices.end(), [&](const int& a, const int& b) {
          return pre_class_value[a] > pre_class_value[b];
        });
        // Getting the top k indices
        top_k_indices =
            vector<int>(indices.begin(), indices.begin() + pre_nms_top_n);
      } else {
        top_k_indices.resize(count);
        std::iota(top_k_indices.begin(), top_k_indices.end(), 0);
      }
      __TOC__(FCOS_SORT)

      __TIC__(FCOS_COOR)
      // cout << "top_k_size=" << top_k_indices.size() << endl;
      for (auto index : top_k_indices) {
        vector<float> box(6);
        int TO_REMOVE = 1;
        box[0] = std::max(
            0.0f, (std::min(pre_locations[index][0] - pre_offset[index][0],
                            (sWidth - TO_REMOVE) * 1.0f)));
        box[1] = std::max(
            0.0f, (std::min(pre_locations[index][1] - pre_offset[index][1],
                            (sHeight - TO_REMOVE) * 1.0f)));
        box[2] = std::max(
            0.0f, (std::min(pre_locations[index][0] + pre_offset[index][2],
                            (sWidth - TO_REMOVE) * 1.0f)));
        box[3] = std::max(
            0.0f, (std::min(pre_locations[index][1] + pre_offset[index][3],
                            (sHeight - TO_REMOVE) * 1.0f)));
        box[2] = box[2] - box[0] + TO_REMOVE;
        box[3] = box[3] - box[1] + TO_REMOVE;
        box[0] = box[0] + box[2] / 2.0f;
        box[1] = box[1] + box[3] / 2.0f;
        box[4] = pre_class[index];
        box[5] = sqrt(pre_class_value[index]);

        if (box[2] >= 0 && box[3] >= 0) {
          boxes.emplace_back(box);
        }
      }
      __TOC__(FCOS_COOR)
    }
    __TIC__(FCOS_NMS)
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
      if (ENV_PARAM(ENABLE_FCOS_DEBUG)) {
        LOG(INFO) << "class: " << i << " size:" << boxes_for_nms[i].size();
      }
      applyNMS(boxes_for_nms[i], scores[i], nms_thresh, 0, result_k, false);
      res.reserve(res.size() + result_k.size());
      transform(result_k.begin(), result_k.end(), back_inserter(res),
                [&](auto& k) { return boxes_for_nms[i][k]; });
    }
    if (ENV_PARAM(ENABLE_FCOS_DEBUG)) {
      LOG(INFO) << "res size: " << res.size();
      for (auto i = 0u; i < res.size(); ++i) {
        LOG(INFO) << "i = " << i << ", res size:" << res[i].size() << " ["
                  << res[i][0] << ", " << res[i][1] << ", " << res[i][2] << ", "
                  << res[i][3] << "], label:" << res[i][4]
                  << ", score: " << res[i][5];
      }
    }
    auto compare = [=](vector<float>& lhs, vector<float>& rhs) {
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
    vector<FCOSResult::BoundingBox> results;
    for (const auto& r : res) {
      FCOSResult::BoundingBox result;
      result.score = r[5];
      result.label = r[4];
      result.box.resize(4);

      //---for (xc, yc, w, h)---
      result.box[0] = (r[0] - r[2] / 2.0) * ws[k] / sWidth;
      result.box[1] = (r[1] - r[3] / 2.0) * hs[k] / sHeight;
      result.box[2] = result.box[0] + r[2] * ws[k] / sWidth;
      result.box[3] = result.box[1] + r[3] * hs[k] / sHeight;

      results.push_back(result);
    }
    res_vec.push_back(FCOSResult{results});
    __TOC__(FCOS_NMS)
  }
  return res_vec;
}

}  // namespace ai
}  // namespace vitis
