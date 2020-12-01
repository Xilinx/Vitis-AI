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
#include "./tfssd_detector.hpp"

#include <cmath>
#include <algorithm>
#include <functional>
#include <tuple>
#include <thread>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "vitis/ai/profiling.hpp"

#include "./sort_xrt/sort_wrapper.h"
extern PPHandle* pphandle;

using namespace cv;
using namespace std;
using vitis::ai::TFSSDResult;
namespace vitis {
namespace ai {
namespace dptfssd {

TFSSDdetector::TFSSDdetector(unsigned int num_classes, 
		CodeType code_type, bool variance_encoded_in_target,
		unsigned int keep_top_k,
		const vector<float>& confidence_threshold,
		unsigned int nms_top_k, float nms_threshold, float eta,
		const std::vector<std::shared_ptr<std::vector<float> > >& priors,
		const short* &fx_priors,
		float y_scale,
		float x_scale,
		float height_scale,
		float width_scale,
		SCORE_CONVERTER score_converter,
		float scale_score, 
		float scale_loc, 
		bool clip)
: num_classes_(num_classes),
  code_type_(code_type),
  variance_encoded_in_target_(false),
  keep_top_k_(keep_top_k),
  confidence_threshold_(confidence_threshold),
  nms_top_k_(nms_top_k),
  nms_threshold_(nms_threshold),
  eta_(eta),
  priors_(priors),
  fx_priors_(fx_priors),
  y_scale_(y_scale),
  x_scale_(x_scale),
  height_scale_(height_scale),
  width_scale_(width_scale),
  score_converter_(score_converter),
  scale_score_(scale_score),
  scale_loc_(scale_loc),
  clip_(clip) {
	num_priors_ = priors_.size();
	nms_confidence_ = *std::min_element(confidence_threshold_.begin() + 1,
			confidence_threshold_.end());
	if (score_converter_ == SIGMOID) {
		// 1/(1+exp(-x*scale))==y;  -->  x=-ln(1/y-1)/scale
		// std::cout << "nms_confidence_:" << nms_confidence_ ;
		nms_confidence_ = (-log(1.0/nms_confidence_ -1.0 ))/scale_score_;
		// std::cout << "   new nms_confidence_:" << nms_confidence_ << std::endl;
		// also need fix confidence_threshold_
		for (auto i=1u; i< confidence_threshold_.size(); i++){
			confidence_threshold_[i] =  (-log(1.0/confidence_threshold_[i] -1.0))/scale_score_;
		}
	}
}


template <typename T>
void TFSSDdetector::Detect(TFSSDResult* result, int8_t *loc) {

	short *nms_out;

	__TIC__(Sort)
	//  std::cout << "nms sort kernel.....\n"; 
	runHWSort(pphandle, loc, nms_out);
	//std::cout << "nms sort kernel done\n"; 
	__TOC__(Sort)

	const int MAX_BOX_PER_CLASS = 32;
	const int out_scale = 16384.0;
	int off = 0;
	for (auto label = 1u; label < num_classes_; ++label) {

		short *perclass_nms_out = nms_out+(MAX_BOX_PER_CLASS*(label-1)*8);
		//std::cout << "res of c: " << label << ", cnt: " << (int)perclass_nms_out[0] << "\n";

		int otid = 8;
		for (auto idx = 0; idx < (int)perclass_nms_out[0]; idx++)
		{
			float x1 = (float)perclass_nms_out[otid]/out_scale;
			float y1 = (float)perclass_nms_out[otid+1]/out_scale;
			float w1 = (float)perclass_nms_out[otid+2]/out_scale;
			float h1 =  (float)perclass_nms_out[otid+3]/out_scale;
			int8_t score = (int8_t)perclass_nms_out[otid+4];

			float xc = x1 - w1/2;
			float yc = y1 - h1/2;
			float x = std::max(std::min(xc, 1.f), 0.f);
			float y = std::max(std::min(yc, 1.f), 0.f);
			float w = std::max(std::min(w1, 1.f), 0.f);
			float h = std::max(std::min(h1, 1.f), 0.f);

			TFSSDResult::BoundingBox res;
			res.label = label;
			// res.score = score;
			//  =  1.0/(1.0+exp(-1.0*input[i]*scale ));
			res.score = (score_converter_ == SIGMOID) ? 1.0/(1.0+exp(-1.0*score*scale_score_ )) : score;

			res.x = x;
			res.y = y;
			res.width = w;
			res.height = h;
			otid += 8;
			//std::cout << "----- res.x:" <<  res.x << " res.y" << res.y << " width: " << res.width << " height:" << res.height<< std::endl;

			//std::cout <<"Detect(): Rect: bbox: " << x << " " << y << " " << w << " " << h 
			//           << "----- res.x:" <<  res.x << " res.y" << res.y << " width: " << res.width << " height:" << res.height<< std::endl;

			result->bboxes.emplace_back(res);
		}
	}
}

template void TFSSDdetector::Detect<int8_t>(TFSSDResult* result, int8_t* loc);

}  // namespace dpssd
}  // namespace ai
}  // namespace vitis
