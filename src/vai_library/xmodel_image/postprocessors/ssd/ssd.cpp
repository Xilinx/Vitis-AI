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
#include "./prior_boxes.hpp"
#include "./ssd_detector.hpp"

namespace {

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

struct Ssd {
  static xir::OpDef get_op_def() {
    return xir::OpDef("ssd")  //
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

  explicit Ssd(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {
    auto input_shape = args.graph_input_tensor->get_shape();
    CHECK_EQ(input_shape.size(), 4u);
    sHeight_ = input_shape[1];
    sWidth_ = input_shape[2];
    is_tf_ = false;
    if (args.graph->has_attr("is_tf")) {
      is_tf_ = args.graph->get_attr<bool>("is_tf");
    }
    is_mlperf_ = false;
    if (args.graph->has_attr("is_mlperf")) {
      is_mlperf_ = args.graph->get_attr<bool>("is_mlperf"); 
    }
    num_classes_ = args.graph->get_attr<int>("num_classes");
    keep_top_k_ = args.graph->get_attr<int>("keep_top_k");
    top_k_ = args.graph->get_attr<int>("top_k");
    // python does not support float
    nms_threshold_ = (float)args.graph->get_attr<double>("nms_threshold");
    conf_threshold_ = vitis::ai::vec_map(args.graph->get_attr<std::vector<double>>("conf_threshold"),
		                                 [](const double& x) { return (float)x; });
    json_prior_box_param_ = args.graph->get_attr<std::string>("prior_box_param");
    priors_ = vitis::ai::dpssd::CreatePriors(sWidth_, sHeight_, is_tf_, is_mlperf_, json_prior_box_param_);
  };

  vitis::ai::proto::DpuModelResult process(
      const vart::simple_tensor_buffer_t<float>& bb,
      const vart::simple_tensor_buffer_t<float>& conf) {
    
    auto valid_output_tensors_size = 2;
    bbox_layer_indexes_.emplace(0);
    conf_layer_indexes_.emplace(1);
    output_layer_infos_.reserve(valid_output_tensors_size);
    output_layer_infos_.assign(valid_output_tensors_size, vitis::ai::dpssd::SSDOutputInfo{});
    output_layer_infos_[0].output_tensor_index = 0;
    output_layer_infos_[0].type = 2;  // conf:1, bbox:2
    output_layer_infos_[0].order = 0;
    output_layer_infos_[0].base_ptr = bb.data;
    output_layer_infos_[0].ptr = output_layer_infos_[0].base_ptr;
    output_layer_infos_[0].index_begin = 0;
    output_layer_infos_[0].bbox_single_size = 4;
    output_layer_infos_[0].index_size =
          bb.mem_size / sizeof(float) /
          output_layer_infos_[0].bbox_single_size;
    output_layer_infos_[0].scale = 1.0;
    output_layer_infos_[0].size = bb.mem_size;

    output_layer_infos_[1].output_tensor_index = 1;
    output_layer_infos_[1].type = 1;  // conf:1, bbox:2
    output_layer_infos_[1].order = 0;
    output_layer_infos_[1].base_ptr = conf.data;
    output_layer_infos_[1].ptr = output_layer_infos_[1].base_ptr;
    output_layer_infos_[1].index_begin = 0;
    output_layer_infos_[1].index_size =
          conf.mem_size / sizeof(float) / num_classes_;
    output_layer_infos_[1].scale = 1.0;
    output_layer_infos_[1].size = conf.mem_size;
    detector_ = vitis::ai::dpssd::CreateSSDUniform(priors_, num_classes_, 
		    nms_threshold_, conf_threshold_, keep_top_k_, top_k_, 
		    is_tf_, is_mlperf_);
    vector<vector<float>> res;
    std::map<uint32_t, vitis::ai::dpssd::SSDOutputInfo> bbox_layer_infos;
    for (auto i = 0u; i < output_layer_infos_.size(); ++i) {
      if (output_layer_infos_[i].type == 2) {  // bbox
        bbox_layer_infos.emplace(std::make_pair(i, output_layer_infos_[i]));
      }
    }
    detector_->detect(bbox_layer_infos, conf.data, &res);

    auto ret = vitis::ai::proto::DpuModelResult();
    auto r = ret.mutable_detect_result();
    for (size_t i = 0; i < res.size(); ++i) {
      *(r->mutable_bounding_box()->Add()) =
        build_bbobx(
	      res[i][0],      //x
      	      res[i][1],      //y
      	      res[i][2],      //width
      	      res[i][3],      //height
      	      res[i][4],      //score
      	      res[i][5]       //label
        );
    }
    return ret;
  }

 private:
  int sWidth_;
  int sHeight_;
  int num_classes_;
  int keep_top_k_;
  int top_k_;
  float nms_threshold_;
  std::vector<float> conf_threshold_;
  bool is_tf_;
  bool is_mlperf_;
  std::string json_prior_box_param_;
  std::set<int> bbox_layer_indexes_;
  std::set<int> conf_layer_indexes_;
  std::vector<vitis::ai::dpssd::SSDOutputInfo> output_layer_infos_;
  std::vector<std::shared_ptr<std::vector<float>>> priors_;
  std::unique_ptr<vitis::ai::dpssd::SSDdetector> detector_;
};
}  // namespace
extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<Ssd>>();
}
