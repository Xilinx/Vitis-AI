/*
 * Copyright 2019 Xilinx Inc.
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
 
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/text_format.h>
#include <fstream>
#include "./anchor/flexiblegrid_anchor.hpp"
#include "./anchor/grid_anchor.hpp"
#include "./anchor/multiscale_anchor.hpp"
#include "./anchor/ssd_anchor.hpp"
#include "object_detection/protos/pipeline.pb.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>
#include "tfssd_post.hpp"
#include "sort_wrapper.h"
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
		const vector<shared_ptr<vector<float>>>& priors, const short* &fx_priors, float scale_score,
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
					false, KEEP_TOP_K, th_conf, TOP_K, NMS_THRESHOLD, 1.0, priors, fx_priors,
					y_scale, x_scale, height_scale, width_scale, score_converter,
					scale_score, scale_loc));
}

TFSSDPost::~TFSSDPost() {}

static std::string slurp(const char* filename) {
	std::ifstream in;
	in.open(filename, std::ifstream::in);
	std::stringstream sstr;
	sstr << in.rdbuf();
	in.close();
	return sstr.str();
}
//int inWidth=300;
//int idHeight = 300;
int inBatch = 1;
TFSSDPost::TFSSDPost(int inWidth, int inHeight, float conf_scale, float loc_scale,
		const vitis::ai::proto::DpuModelParam& config)//,
				: inWidth_(inWidth), inHeight_(inHeight),scale_conf_(conf_scale), scale_loc_(loc_scale)
				  //    : input_tensors_(input_tensors), output_tensors_(output_tensors) {
{
	// read official tensorflow ssd configure file
	//auto path = dpumeta.dirname + "/" + config.tfssd_param().official_cfg();
	//string path = "/scratch/gaoyue/mlperf_cc/model/" + config.tfssd_param().official_cfg();
	string path = "./ssd_mobilenet_v1_coco_tf/ssd_mobilenet_v1_coco_tf_officialcfg.prototxt";
	//  auto text = slurp(path.c_str());


	std::ifstream in;
	in.open(path.c_str(), std::ifstream::in);
	std::stringstream sstr;
	sstr << in.rdbuf();
	in.close();
	auto text = sstr.str();

	google::protobuf::LogSilencer* s1 = new google::protobuf::LogSilencer;
	if (0) {
		std::cerr << "suppress warning of unused variable " << s1 << std::endl;
	}

	object_detection::protos::TrainEvalPipelineConfig tfcfg;
	auto ok = google::protobuf::TextFormat::ParseFromString(text, &tfcfg);
	if (!ok) {
		LOG(FATAL) << "parse error for tensorflow offical config file: " << path;
	}

	num_classes_ = (tfcfg.model().ssd().num_classes() + 1);
	score_converter_ =
			SCORE_CONVERTER(tfcfg.model().ssd().post_processing().score_converter());
	__TIC__(PRIORBOX)

	priors_ = vitis::ai::CreatePriors(tfcfg, config, inWidth, inHeight);
	short *nfx_priors_ = new short[1917*4];
	const int PRIOR_BW_VAL = 4096.0;
	const int PRIOR_FBITS =  12;
	short idx = 0;
	for (auto pr : priors_) {
		//      std::cout << "priors size: " << (int)priors_.size() << "\n";
		auto& prior_bbox = *pr;
		for (int it = 0; it < 4; it++) {
			nfx_priors_[idx++] = floor((short)(prior_bbox[it] * (1 << PRIOR_FBITS)) + 0.5);
		}
	}
	fixed_priors_ = &nfx_priors_[0];

	__TOC__(PRIORBOX)
	__TIC__(CREATESSD)
	
	detector_ =
			createSSD(priors_, fx_priors_, scale_conf_, scale_loc_, score_converter_, tfcfg);

	__TOC__(CREATESSD)
}

std::vector<vitis::ai::TFSSDResult> TFSSDPost::ssd_post_process(int8_t* conf, int8_t* loc, bool en_swpost, AcceleratorHandle* postproc_handle) {
	auto batch_size = inBatch;
	auto ret = std::vector<vitis::ai::TFSSDResult>{};
	ret.reserve(batch_size);
	for (auto i = 0u; i < batch_size; ++i) {
		ret.emplace_back(ssd_post_process(conf,loc,i,en_swpost,postproc_handle));
	}
	return ret;
}

vitis::ai::TFSSDResult TFSSDPost::ssd_post_process(int8_t* conf,int8_t* box_c, unsigned int idx, bool en_swpost, AcceleratorHandle* postproc_handle) {
	__TIC__(SSD_total)

		//  std::cout <<  "########### Post Proc Start ########################" << std::endl;

		 (void)box_c;

	std::vector<TFSSDResult::BoundingBox> bboxes;

	TFSSDResult results{inWidth_,
		inHeight_, bboxes};

	detector_->Detect<int8_t>(box_c, conf, &results, en_swpost,postproc_handle);

	//detector_->Detect<int8_t>(&results, box_c);

	__TOC__(SSD_total)
	return results;
}

}  // namespace ai
}  // namespace vitis
