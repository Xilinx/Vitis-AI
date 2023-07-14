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
#include <glog/logging.h>

#include <memory>
#include <vector>

#include "vart/runner_helper.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/xmodel_postprocessor.hpp"
#include "xir/graph/graph.hpp"

namespace {
static std::vector<std::vector<float>> FilterBox(const float det_threshold, float* bbout,
                                       int w, int h, float* pred) {
  std::vector<std::vector<float>> boxes;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int position = i * w + j;
      std::vector<float> box;
      if (pred[position * 2 + 1] > det_threshold) {
        box.push_back(bbout[position * 4 + 0] + j * 4);
        box.push_back(bbout[position * 4 + 1] + i * 4);
        box.push_back(bbout[position * 4 + 2] + j * 4);
        box.push_back(bbout[position * 4 + 3] + i * 4);
        box.push_back(pred[position * 2 + 1]);
        boxes.push_back(box);
      }
    }
  }
  return boxes;
}
static void getResult(const std::vector<std::vector<float>>& box, const std::vector<int>& keep,
                      std::vector<std::vector<float>>& results) {
  results.clear();
  results.reserve(keep.size());
  for (auto i = 0u; i < keep.size(); ++i) {
    auto b = box[keep[i]];
    b[2] -= b[0];
    b[3] -= b[1];
    results.emplace_back(b);
  }
}

static void NMS(const float nms_threshold, const std::vector<std::vector<float>>& box,
                std::vector<std::vector<float>>& results) {
  auto count = box.size();
  std::vector<std::pair<size_t, float>> order(count);
  for (auto i = 0u; i < count; ++i) {
    order[i].first = i;
    order[i].second = box[i][4];
  }
  sort(order.begin(), order.end(),
       [](const std::pair<int, float>& ls, const std::pair<int, float>& rs) {
         return ls.second > rs.second;
       });

  std::vector<int> keep;
  std::vector<bool> exist_box(count, true);
  for (auto i = 0u; i < count; ++i) {
    auto idx = order[i].first;
    if (!exist_box[idx]) continue;
    keep.emplace_back(idx);
    for (auto j = i + 1; j < count; ++j) {
      auto kept_idx = order[j].first;
      if (!exist_box[kept_idx]) continue;
      auto x1 = std::max(box[idx][0], box[kept_idx][0]);
      auto y1 = std::max(box[idx][1], box[kept_idx][1]);
      auto x2 = std::min(box[idx][2], box[kept_idx][2]);
      auto y2 = std::min(box[idx][3], box[kept_idx][3]);
      auto intersect = std::max(0.f, x2 - x1 + 1) * std::max(0.f, y2 - y1 + 1);
      auto sum_area =
          (box[idx][2] - box[idx][0] + 1) * (box[idx][3] - box[idx][1] + 1) +
          (box[kept_idx][2] - box[kept_idx][0] + 1) *
              (box[kept_idx][3] - box[kept_idx][1] + 1);
      auto overlap = intersect / (sum_area - intersect);
      if (overlap >= nms_threshold) exist_box[kept_idx] = false;
    }
  }
  getResult(box, keep, results);
}

static float my_div(float a, size_t b) { return a / static_cast<float>(b); };

static vitis::ai::proto::BoundingBox build_bbobx(float x, float y, float w, float h,
                                          float score, float label = 0) {
  auto ret = vitis::ai::proto::BoundingBox();
  auto index = (unsigned int)label;
  ret.mutable_label()->set_index(index);
  ret.mutable_label()->set_score(score);
  ret.mutable_size()->set_width(w);
  ret.mutable_size()->set_height(h);
  ret.mutable_top_left()->set_x(x);
  ret.mutable_top_left()->set_y(y);
  return ret;
}

struct DenseBox {
 public:
  static xir::OpDef get_op_def() {
    return xir::OpDef("densebox")  //
        .add_input_arg(xir::OpArgDef{
            "bbox", xir::OpArgDef::REQUIRED, xir::DataType::Type::FLOAT,
            "bounding box tensor "
            "`[batch, in_height, in_width, in_channels]`."})
        .add_input_arg(xir::OpArgDef{
            "conf", xir::OpArgDef::REQUIRED, xir::DataType::Type::FLOAT,
            "confidence. "
            "`[batch, in_height, in_width, in_channels]`."})
        .set_annotation("postprocessor for densebox");
  }

  explicit DenseBox(vitis::ai::XmodelPostprocessorInitializationArgs&& args) {
    auto input_shape = args.graph_input_tensor->get_shape();
    CHECK_EQ(input_shape.size(), 4u);
    height_ = input_shape[1];
    width_ = input_shape[2];
    // python does not support float
    det_threshold_ = (float)args.graph->get_attr<double>("det_threshold");
    nms_threshold_ = (float)args.graph->get_attr<double>("nms_threshold");
  }
  vitis::ai::proto::DpuModelResult process(
      const vart::simple_tensor_buffer_t<float>& bbobx,
      const vart::simple_tensor_buffer_t<float>& conf);

 private:
  int width_;
  int height_;
  float det_threshold_;
  float nms_threshold_;
};

vitis::ai::proto::DpuModelResult DenseBox::process(
    const vart::simple_tensor_buffer_t<float>& bb,
    const vart::simple_tensor_buffer_t<float>& conf) {
  auto bb_shape = bb.tensor->get_shape();
  auto conf_shape = conf.tensor->get_shape();
  CHECK_EQ(bb_shape.size(), 4u);
  CHECK_EQ(conf_shape.size(), 4u);
  CHECK_EQ(bb_shape[0], conf_shape[0]);
  CHECK_EQ(bb_shape[1], conf_shape[1]);
  CHECK_EQ(bb_shape[2], conf_shape[2]);
  CHECK_EQ(bb_shape[3], 4);
  CHECK_EQ(conf_shape[3], 2);
  int h = bb_shape[1];
  int w = bb_shape[2];
  auto input_width = width_;
  auto input_height = height_;
  auto ret = vitis::ai::proto::DpuModelResult();
  std::vector<std::vector<float>> boxes =
      FilterBox(det_threshold_, bb.data, w, h, conf.data);
  std::vector<std::vector<float>> results;
  NMS(nms_threshold_, boxes, results);
  auto dr = ret.mutable_detect_result();
  for (auto& x : results) {
    *(dr->mutable_bounding_box()->Add()) =
        build_bbobx(my_div(x[0], input_width),   //
                    my_div(x[1], input_height),  //
                    my_div(x[2], input_width),   //
                    my_div(x[3], input_height),  //
                    x[4]                         //
        );
  }
  return ret;
}
}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<DenseBox>>();
}
