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
#include <memory>
#include <vector>
#include <cmath>

#include "vart/runner_helper.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/xmodel_postprocessor.hpp"
#include "xir/graph/graph.hpp"
#include "vitis/ai/nnpp/apply_nms.hpp"

namespace {

static float sigmoid(float p) { return 1.0 / (1 + exp(-p * 1.0)); }

static void detect(vector<vector<float>>& boxes, const float* result, const int height,
                   const int width, const int num, const int sHeight, const int sWidth, 
		   const int num_classes, const int anchor_cnt, const float conf_thresh, 
		   const vector<float>& biases) {
  auto conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f);

  int conf_box = 5 + num_classes;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < anchor_cnt; ++c) {
        int idx = ((h * width + w) * anchor_cnt + c) * conf_box;
        if (result[idx + 4] < conf_desigmoid) continue;
        vector<float> box;

        float obj_score = sigmoid(result[idx + 4]);
        box.push_back((w + sigmoid(result[idx])) / width);
        box.push_back((h + sigmoid(result[idx + 1])) / height);
        box.push_back(exp(result[idx + 2]) *
                      biases[2 * c + 2 * anchor_cnt * num] / float(sWidth));
        box.push_back(exp(result[idx + 3]) *
                      biases[2 * c + 2 * anchor_cnt * num + 1] /
                      float(sHeight));
        box.push_back(-1);
        box.push_back(obj_score);
        for (int p = 0; p < num_classes; p++) {
          box.push_back(obj_score * sigmoid(result[idx + 5 + p]));
        }
        boxes.push_back(box);
      }
    }
  }
}

static vitis::ai::proto::BoundingBox build_bbobx(float x, float y, float w, float h,
                                          float score, float label = 0) {
  vitis::ai::proto::BoundingBox ret = vitis::ai::proto::BoundingBox();
  unsigned int index = (unsigned int)label;
  ret.mutable_label()->set_index(index);
  ret.mutable_label()->set_score(score);
  ret.mutable_size()->set_width(w);
  ret.mutable_size()->set_height(h);
  ret.mutable_top_left()->set_x(x);
  ret.mutable_top_left()->set_y(y);
  return ret;
}

struct YoloV3 {
  static xir::OpDef get_op_def() {
    return xir::OpDef("yolov3")  //
        .add_input_arg(
            xir::OpArgDef{"input", xir::OpArgDef::REQUIRED_AND_REPEATED,
                          xir::DataType::Type::FLOAT, "input layers"})
        .set_annotation("postprocessor for yolov3");
  }

  explicit YoloV3(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {
    auto input_shape = args.graph_input_tensor->get_shape();
    CHECK_EQ(input_shape.size(), 4u);
    sHeight_ = input_shape[1];
    sWidth_ = input_shape[2];
    num_classes_ = args.graph->get_attr<int>("num_classes");
    anchorCnt_ = args.graph->get_attr<int>("anchorCnt");
    // python does not support float
    conf_threshold_ = (float)args.graph->get_attr<double>("conf_threshold");
    nms_threshold_ = (float)args.graph->get_attr<double>("nms_threshold");
    biases_ = vitis::ai::vec_map(args.graph->get_attr<std::vector<double>>("biases"),
                             [](const double& x) { return (float)x; });
  };

  vitis::ai::proto::DpuModelResult process(
      const std::vector<vart::simple_tensor_buffer_t<float>>&
          tensor_buffers) {
    if (tensor_buffers.empty()) {
      return {};
    }
    vector<vector<float>> boxes;
    int out_num = tensor_buffers.size();
    int j = 0;
    for (int i = (out_num - 1); i >= 0; i--) {
      auto output_shape = tensor_buffers[i].tensor->get_shape();
      int height = output_shape[1];
      int width = output_shape[2];

      int sizeOut = height * width * anchorCnt_;
      boxes.reserve(sizeOut);
      /* Store the object detection frames as coordinate information  */
      detect(boxes, tensor_buffers[i].data, height, width, j++, sHeight_, 
             sWidth_, num_classes_, anchorCnt_, conf_threshold_, biases_);
    }

    /* Apply the computation for NMS */
    vector<vector<float>> res;
    vector<float> scores(boxes.size());
    for (int k = 0; k < num_classes_; k++) {
      transform(boxes.begin(), boxes.end(), scores.begin(), [k](auto& box) {
        box[4] = k;
        return box[6 + k];
      });
      vector<size_t> result_k;
      applyNMS(boxes, scores, nms_threshold_, conf_threshold_, result_k);
      transform(result_k.begin(), result_k.end(), back_inserter(res),
                [&boxes](auto& k) { return boxes[k]; });
    }
    
    auto ret = vitis::ai::proto::DpuModelResult();
    auto r = ret.mutable_detect_result();
    for (size_t i = 0; i < res.size(); ++i) {
      if (res[i][res[i][4] + 6] > conf_threshold_) {
        *(r->mutable_bounding_box()->Add()) =
          build_bbobx(res[i][0] - res[i][2] / 2.0,  //x
		      res[i][1] - res[i][3] / 2.0,  //y
		      res[i][2],                    //width
		      res[i][3],                    //height
		      res[i][res[i][4] + 6],        //score
		      res[i][4]                     //label
          );
      }
    }
    return ret;
  }

 private:
  int sWidth_;
  int sHeight_;
  int num_classes_;
  int anchorCnt_;
  float conf_threshold_;
  float nms_threshold_;
  std::vector<float> biases_;
};
}  // namespace
extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<YoloV3>>();
}
