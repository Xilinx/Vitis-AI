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
#include "tfssd_post.hpp"

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/text_format.h>

#include <future>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>

#include "./anchor/flexiblegrid_anchor.hpp"
#include "./anchor/grid_anchor.hpp"
#include "./anchor/multiscale_anchor.hpp"
#include "./anchor/ssd_anchor.hpp"
#include "object_detection/protos/pipeline.pb.h"

using namespace std;
namespace vitis {
namespace ai {

static vector<shared_ptr<vector<float>>> CreatePriors(
    const object_detection::protos::TrainEvalPipelineConfig& tfcfg,
    const vitis::ai::proto::DpuModelParam& config, int image_width,
    int image_height) {
  auto anchor_gen = tfcfg.model().ssd().anchor_generator();

  if (anchor_gen.has_ssd_anchor_generator()) {
    vitis::ai::dptfssd::SSDAnchor pbx{
        tfcfg.model()
            .ssd()
            .anchor_generator()
            .ssd_anchor_generator()
            .num_layers(),
        tfcfg.model()
            .ssd()
            .anchor_generator()
            .ssd_anchor_generator()
            .reduce_boxes_in_lowest_layer(),
        tfcfg.model()
            .ssd()
            .anchor_generator()
            .ssd_anchor_generator()
            .min_scale(),
        tfcfg.model()
            .ssd()
            .anchor_generator()
            .ssd_anchor_generator()
            .max_scale(),
        tfcfg.model()
            .ssd()
            .anchor_generator()
            .ssd_anchor_generator()
            .interpolated_scale_aspect_ratio(),
        std::vector<int>(config.tfssd_param().feature_map_list().begin(),
                         config.tfssd_param().feature_map_list().end()),
        std::vector<float>(tfcfg.model()
                               .ssd()
                               .anchor_generator()
                               .ssd_anchor_generator()
                               .aspect_ratios()
                               .begin(),
                           tfcfg.model()
                               .ssd()
                               .anchor_generator()
                               .ssd_anchor_generator()
                               .aspect_ratios()
                               .end()),
        image_width,
        image_height};
    return vector<shared_ptr<vector<float>>>{pbx.priors()};
  }
  if (anchor_gen.has_multiscale_anchor_generator()) {
    vitis::ai::dptfssd::MultiscaleAnchor pbx{
        tfcfg.model()
            .ssd()
            .anchor_generator()
            .multiscale_anchor_generator()
            .min_level(),
        tfcfg.model()
            .ssd()
            .anchor_generator()
            .multiscale_anchor_generator()
            .max_level(),
        tfcfg.model()
            .ssd()
            .anchor_generator()
            .multiscale_anchor_generator()
            .anchor_scale(),
        tfcfg.model()
            .ssd()
            .anchor_generator()
            .multiscale_anchor_generator()
            .scales_per_octave(),
        std::vector<int>(config.tfssd_param().feature_map_list().begin(),
                         config.tfssd_param().feature_map_list().end()),
        std::vector<float>(tfcfg.model()
                               .ssd()
                               .anchor_generator()
                               .multiscale_anchor_generator()
                               .aspect_ratios()
                               .begin(),
                           tfcfg.model()
                               .ssd()
                               .anchor_generator()
                               .multiscale_anchor_generator()
                               .aspect_ratios()
                               .end()),
        image_width,
        image_height};
    return vector<shared_ptr<vector<float>>>{pbx.priors()};
  }
  return vector<shared_ptr<vector<float>>>{};
}

static std::unique_ptr<vitis::ai::dptfssd::TFSSDdetector> createSSD(
    const vector<shared_ptr<vector<float>>>& priors, float scale_score,
    float scale_loc, SCORE_CONVERTER score_converter,
    const object_detection::protos::TrainEvalPipelineConfig& tfcfg) {
  const int num_classes = tfcfg.model().ssd().num_classes() + 1;
  const float NMS_THRESHOLD = tfcfg.model()
                                  .ssd()
                                  .post_processing()
                                  .batch_non_max_suppression()
                                  .iou_threshold();
  vector<float> th_conf(num_classes, tfcfg.model()
                                         .ssd()
                                         .post_processing()
                                         .batch_non_max_suppression()
                                         .score_threshold());
  th_conf[0] = 0.0;
  const int KEEP_TOP_K = tfcfg.model()
                             .ssd()
                             .post_processing()
                             .batch_non_max_suppression()
                             .max_total_detections();
  const int TOP_K = tfcfg.model()
                        .ssd()
                        .post_processing()
                        .batch_non_max_suppression()
                        .max_detections_per_class();
  float y_scale =
      tfcfg.model().ssd().box_coder().faster_rcnn_box_coder().y_scale();
  float x_scale =
      tfcfg.model().ssd().box_coder().faster_rcnn_box_coder().x_scale();
  float height_scale =
      tfcfg.model().ssd().box_coder().faster_rcnn_box_coder().height_scale();
  float width_scale =
      tfcfg.model().ssd().box_coder().faster_rcnn_box_coder().width_scale();

  // std::cout <<"createSSD:" << num_classes << " " << NMS_THRESHOLD << " " <<
  // th_conf.size() << " " << KEEP_TOP_K << " "
  //        << TOP_K << " " << y_scale << " " << x_scale << " " << height_scale
  //        << " " << width_scale << std::endl;

  return std::unique_ptr<vitis::ai::dptfssd::TFSSDdetector>(
      new vitis::ai::dptfssd::TFSSDdetector(
          num_classes, vitis::ai::dptfssd::TFSSDdetector::CodeType::CENTER_SIZE,
          false, KEEP_TOP_K, th_conf, TOP_K, NMS_THRESHOLD, 1.0, priors,
          y_scale, x_scale, height_scale, width_scale, score_converter,
          scale_score, scale_loc));
}

TFSSDPost::~TFSSDPost() {}

static std::string slurp(const char* filename) {
  std::ifstream in;
  std::stringstream sstr;
  try {
    in.open(filename, std::ifstream::in);
    sstr << in.rdbuf();
    in.close();
    if (sstr.str().empty()) {
      throw -1;
    }
  } catch (...) {
     std::cerr << "failed to open file " << filename <<"\n";
  }
  return sstr.str();
}

TFSSDPost::TFSSDPost(
    const std::string& model_name,
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, 
    const std::string& dirname,
    int& real_batch_sizex )
    : input_tensors_(input_tensors), 
      output_tensors_(output_tensors),
      real_batch_size(real_batch_sizex) {
  // read official tensorflow ssd configure file
  std::string model_namex(model_name);
  if (model_name.size() > 7 && model_name.substr( model_name.size()-7, 7) == ".xmodel") {
     size_t pos = 0;
     if ((pos = model_name.rfind("/")) != std::string::npos) {
        model_namex = model_name.substr(pos+1, model_name.size()-7-(pos+1) );
     } else {
        model_namex = model_name.substr(0, model_name.size()-7);
     }
  }
  std::string cfgpath = dirname + "/" + model_namex + "_officialcfg.prototxt";

  auto text = slurp(cfgpath.c_str());
  if(text.empty()) {
     LOG(FATAL) << "parse error for tensorflow offical config file: " << cfgpath;
  }

  google::protobuf::LogSilencer* s1 = new google::protobuf::LogSilencer;
  if (0) {
    std::cerr << "suppress warning of unused variable " << s1 << std::endl;
  }

  object_detection::protos::TrainEvalPipelineConfig tfcfg;
  auto ok = google::protobuf::TextFormat::ParseFromString(text, &tfcfg);
  if (!ok) {
    LOG(FATAL) << "parse error for tensorflow offical config file: " << cfgpath;
  }
  delete s1;

  num_classes_ = (tfcfg.model().ssd().num_classes() + 1);
  score_converter_ =
      SCORE_CONVERTER(tfcfg.model().ssd().post_processing().score_converter());

  for (auto it = config.tfssd_param().output_info().begin();
       it != config.tfssd_param().output_info().end(); it++) {
    for (auto i = 0u; i < output_tensors.size(); i++) {
      if (output_tensors[i].name.find(it->name()) != std::string::npos) {
        if (it->type() == 1) {
          CONF_IDX = i;
          break;
        } else if (it->type() == 2) {
          LOC_IDX = i;
          break;
        }
      }
    }
  }

  scale_conf_ = vitis::ai::library::tensor_scale(output_tensors_[CONF_IDX]);
  scale_loc_ = vitis::ai::library::tensor_scale(output_tensors_[LOC_IDX]);

  __TIC__(PRIORBOX)

  priors_ = vitis::ai::CreatePriors(tfcfg, config, (int)input_tensors[0].width,
                                    (int)input_tensors[0].height);

  __TOC__(PRIORBOX)
  __TIC__(CREATESSD)
  detector_ =
      createSSD(priors_, scale_conf_, scale_loc_, score_converter_, tfcfg);

  __TOC__(CREATESSD)
}

std::vector<vitis::ai::TFSSDResult> TFSSDPost::ssd_post_process() {

  auto ret = std::vector<vitis::ai::TFSSDResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(ssd_post_process(i));
  }

  return ret;
}

vitis::ai::TFSSDResult TFSSDPost::ssd_post_process(unsigned int idx) {
  __TIC__(SSD_total)

  int8_t* conf = (int8_t*)(output_tensors_[CONF_IDX].get_data(idx));

  std::vector<float> softmax_data_;
  softmax_data_.reserve(priors_.size() * num_classes_);

  __TIC__(SSD_softmax)
  if (score_converter_ == SOFTMAX) {
    vitis::ai::softmax((int8_t*)conf, scale_conf_, num_classes_, priors_.size(),
                       softmax_data_.data());
  // } else if (score_converter_ == SIGMOID) {
  //   sigmoid_c((int8_t*)conf, scale_conf_, num_classes_, priors_.size(),
  //             softmax_data_.data());
  }

  __TOC__(SSD_softmax)

  int8_t* box_c = (int8_t*)(output_tensors_[LOC_IDX].get_data(idx));
  (void)box_c;

  std::vector<TFSSDResult::BoundingBox> bboxes;

  TFSSDResult results{(int)input_tensors_[0].width,
                      (int)input_tensors_[0].height, bboxes};
  // detector_->Detect((int8_t*)box_c, softmax_data_.data(), &results);
  if (score_converter_ == SOFTMAX) {
    detector_->Detect((int8_t*)box_c, softmax_data_.data(), &results);
  } else if (score_converter_ == SIGMOID) {
    detector_->Detect((int8_t*)box_c, conf, &results);
  }
  return results;
}

}  // namespace ai
}  // namespace vitis

