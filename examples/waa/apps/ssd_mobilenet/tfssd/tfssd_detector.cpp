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

#include "sort_wrapper.h"


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
void TFSSDdetector::Detect(const T* loc_data, const T* conf_data, TFSSDResult* result, bool en_swpost, AcceleratorHandle* postproc_handle) {
  //# Hardware accelerated post process call
  if(!en_swpost) {
    short *nms_out;
    __TIC__(Sort)
    //  std::cout << "nms sort kernel.....\n"; 
    runHWSort(postproc_handle, nms_out);
    //std::cout << "nms sort kernel done\n"; 
    __TOC__(Sort)

    const int MAX_BOX_PER_CLASS = 32;
    const int out_scale = 16384.0;
    int off = 0;
    for (auto label = 1u; label < num_classes_; ++label) {

      short *perclass_nms_out = nms_out+(MAX_BOX_PER_CLASS*(label-1)*8);
     // std::cout << "res of c: " << label << ", cnt: " << (int)perclass_nms_out[0] << "\n";

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
  else {
    decoded_bboxes_.clear();
    const T(*bboxes)[4] = (const T(*)[4])loc_data;

    unsigned int num_det = 0;
    vector<vector<pair<float,int>>> indices(num_classes_);
    vector<vector<pair<float, int>>> score_index_vec(num_classes_);

    // Get top_k scores (with corresponding indices).
    GetMultiClassMaxScoreIndexMT(conf_data, 1, num_classes_ - 1, &score_index_vec, 4);
    
    __TIC__(NMS)
    for (unsigned int c = 1; c < num_classes_; ++c) {
      // Perform NMS for one class
      ApplyOneClassNMS(bboxes,  c, score_index_vec[c], &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > 0 && num_det > keep_top_k_) {
      vector<tuple<float, int, int>> score_index_tuples;
      for (auto label = 0u; label < num_classes_; ++label) {
        const vector<pair<float,int>>& label_indices = indices[label];
        for (auto j = 0u; j < label_indices.size(); ++j) {
          auto idx = label_indices[j].second;
          //auto score = conf_data[idx * num_classes_ + label];
          auto score = label_indices[j].first;
          score_index_tuples.emplace_back(score, label, idx);
        }
      }

      // Keep top k results per image.
      std::sort(score_index_tuples.begin(), score_index_tuples.end(),
                [](const tuple<float, int, int>& lhs,
                  const tuple<float, int, int>& rhs) {
        return get<0>(lhs) > get<0>(rhs);
      });
      score_index_tuples.resize(keep_top_k_);

      indices.clear();
      indices.resize(num_classes_);
      for (auto& item : score_index_tuples) {
        indices[get<1>(item)].push_back(make_pair(get<0>(item),get<2>(item)));
      }
    }
    __TOC__(NMS)

    __TIC__(Box)

    for (auto label = 1u; label < indices.size(); ++label) {
      for (auto idx : indices[label]) {
  //      auto score = conf_data[idx * num_classes_ + label];
        auto score = idx.first;
        if (score < confidence_threshold_[label]) {
          continue;
        }
        auto& bbox = decoded_bboxes_[idx.second];
        bbox[0] = std::max(std::min(bbox[0], 1.f), 0.f);
        bbox[1] = std::max(std::min(bbox[1], 1.f), 0.f);
        bbox[2] = std::max(std::min(bbox[2], 1.f), 0.f);
        bbox[3] = std::max(std::min(bbox[3], 1.f), 0.f);

        auto box_rect =
            Rect_<float>(Point2f(bbox[0], bbox[1]), Point2f(bbox[2], bbox[3]));
        TFSSDResult::BoundingBox res;
        res.label = label;
        // res.score = score;
        //  =  1.0/(1.0+exp(-1.0*input[i]*scale ));
        res.score = (score_converter_ == SIGMOID) ? 1.0/(1.0+exp(-1.0*score*scale_score_ )) : score;
        
        res.x = box_rect.x;
        res.y = box_rect.y;
        res.width = box_rect.width;
        res.height = box_rect.height;
  //	  std::cout << "----- res.x:" <<  res.x << " res.y" << res.y << " width: " << res.width << " height:" << res.height<< std::endl;
  //    std::cout <<"Detect(): Rect: bbox: " << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3] 
  //                << "----- res.x:" <<  res.x << " res.y" << res.y << " width: " << res.width << " height:" << res.height<< std::endl;
        result->bboxes.emplace_back(res);
      }
    }
    __TOC__(Box)
  }
}

template void TFSSDdetector::Detect<int8_t>(const int8_t* loc, const int8_t* conf, TFSSDResult* result, bool en_swpost, AcceleratorHandle* postproc_handle);

template <typename T>
void TFSSDdetector::ApplyOneClassNMS(
    const T (*bboxes)[4], int label,
    const vector<pair<float, int>>& score_index_vec, vector<pair<float,int>>* indices) {
  // Get top_k scores (with corresponding indices).
  // vector<pair<float, int> > score_index_vec;
  // GetOneClassMaxScoreIndex(conf_data, label, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold_;
  indices->clear();
  unsigned int i = 0;
  while (i < score_index_vec.size()) {
    //__TIC__(Decode)
    const int idx = score_index_vec[i].second;
    const float score = score_index_vec[i].first;

    if (decoded_bboxes_.find(idx) == decoded_bboxes_.end()) {
      DecodeBBox(bboxes, idx, true);
    }
    //__TOC__(Decode)
    //__TIC__(OVERLAP)
    bool keep = true;
    for (auto k = 0u; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k].second;
        float overlap = JaccardOverlap(bboxes, idx, kept_idx);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(make_pair(score,idx));
    }
    ++i;
    if (keep && eta_ < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta_;
    }
    //__TOC__(OVERLAP)
  }
}

template void TFSSDdetector::ApplyOneClassNMS(
    const int (*bboxes)[4],  int label,
    const vector<pair<float, int>>& score_index_vec, vector<pair<float,int>>* indices);
template void TFSSDdetector::ApplyOneClassNMS(
    const int8_t (*bboxes)[4], int label,
    const vector<pair<float, int>>& score_index_vec, vector<pair<float,int>>* indices);

void TFSSDdetector::GetOneClassMaxScoreIndex(
    const int8_t* conf_data, int label,
    vector<pair<float, int>>* score_index_vec) {
  __TIC__(PUSH2)
  conf_data += label;
  for (int i = 0; i < num_priors_; ++i) {
    auto score = *conf_data;
  //  cout << "compare " << (int)score <<" vs "<< nms_confidence_<< "\n";
    if (score > nms_confidence_) {
      score_index_vec->emplace_back(score, i);
    }
    conf_data += num_classes_;
  }

  __TOC__(PUSH2)
  __TIC__(SORT2)
  std::stable_sort(
      score_index_vec->begin(), score_index_vec->end(),
      [](const pair<float, int>& lhs,
         const pair<float, int>& rhs) { return lhs.first > rhs.first; });
  __TOC__(SORT2)

  if (nms_top_k_ < score_index_vec->size()) {
    score_index_vec->resize(nms_top_k_);
  }
}

void TFSSDdetector::GetMultiClassMaxScoreIndex(
    const int8_t* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec) {
  for (auto i = start_label; i < start_label + num_classes; ++i) {
    GetOneClassMaxScoreIndex(conf_data, i, &((*score_index_vec)[i]));
  }
}

void TFSSDdetector::GetMultiClassMaxScoreIndexMT(
    const int8_t* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec, int threads) {
  // CHECK_GT(threads, 0);
  int thread_classes = num_classes / threads;
  int last_thread_classes = num_classes % threads + thread_classes;

  vector<std::thread> workers;

  auto c = start_label;
  for (auto i = 0; i < threads - 1; ++i) {
    workers.emplace_back(&TFSSDdetector::GetMultiClassMaxScoreIndex, this,
                         conf_data, c, thread_classes, score_index_vec);
    c += thread_classes;
  }
  workers.emplace_back(&TFSSDdetector::GetMultiClassMaxScoreIndex, this,
                       conf_data, c, last_thread_classes, score_index_vec);

  for (auto& worker : workers)
    if (worker.joinable()) worker.join();
}

void BBoxSize(vector<float>& bbox, bool normalized) {
  float width = bbox[2] - bbox[0];
  float height = bbox[3] - bbox[1];
  if (width > 0 && height > 0) {
    if (normalized) {
      bbox[4] = width * height;
    } else {
      bbox[4] = (width + 1) * (height + 1);
    }
  } else {
    bbox[4] = 0.f;
  }
}

float IntersectBBoxSize(const vector<float>& bbox1, const vector<float>& bbox2,
                        bool normalized) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] || bbox2[1] > bbox1[3] ||
      bbox2[3] < bbox1[1]) {
    // Return 0 if there is no intersection.
    return 0.f;
  }

  vector<float> intersect_bbox(5);
  intersect_bbox[0] = max(bbox1[0], bbox2[0]);
  intersect_bbox[1] = max(bbox1[1], bbox2[1]);
  intersect_bbox[2] = min(bbox1[2], bbox2[2]);
  intersect_bbox[3] = min(bbox1[3], bbox2[3]);
  BBoxSize(intersect_bbox, normalized);
  return intersect_bbox[4];
}

template <typename T>
float TFSSDdetector::JaccardOverlap(const T (*bboxes)[4], int idx, int kept_idx,
                                  bool normalized) {
  const vector<float>& bbox1 = decoded_bboxes_[idx];
  const vector<float>& bbox2 = decoded_bboxes_[kept_idx];
  float intersect_size = IntersectBBoxSize(bbox1, bbox2, normalized);
  return intersect_size <= 0 ? 0 : intersect_size / (bbox1[4] + bbox2[4] - intersect_size);
}

template float TFSSDdetector::JaccardOverlap(const int (*bboxes)[4], int idx, int kept_idx, bool normalized);
template float TFSSDdetector::JaccardOverlap(const int8_t (*bboxes)[4], int idx, int kept_idx, bool normalized);

template <typename T>
void TFSSDdetector::DecodeBBox(const T (*bboxes)[4], int idx, bool normalized) {
  // in tfconcat_decode, we get the center and size directly in prior_bbox, so we don't need decode center/size here.
  // corresponding logic is in  box_coders/faster_rcnn_box_coder.py: _decode()
  // scale bboxes

  auto& prior_bbox = *priors_[idx];
  auto y_center_a = prior_bbox[0];
  auto x_center_a  = prior_bbox[1];
  auto ha = prior_bbox[2];
  auto wa = prior_bbox[3];

  auto ty = bboxes[idx][0]*scale_loc_/y_scale_;
  auto tx = bboxes[idx][1]*scale_loc_/x_scale_;
  auto th = bboxes[idx][2]*scale_loc_/height_scale_;
  auto tw = bboxes[idx][3]*scale_loc_/width_scale_;

  auto w =  exp(tw ) * wa;
  auto h =  exp(th ) * ha;
  auto ycenter = ty * ha + y_center_a;
  auto xcenter = tx * wa + x_center_a;


#if 0
  std::cout <<"DecodeBox:  bboxes[idx] " << float( bboxes[idx][0]) << " " << float( bboxes[idx][1] ) << " " 
            <<  float (bboxes[idx][2]) << " " << float( bboxes[idx][3] ) << std::endl;

  std::cout << "DecodeBox: ha wa:" << ha << " " << wa 
            << " ty tx th tw:" << ty << " " << tx << " " << th << " " << tw 
            << "    yxcenter:" << ycenter << " " << xcenter << std::endl;
#endif

  vector<float> bbox(5, 0); 

  // seems x, y changed ? Yes. test and found. 

  bbox[0] = xcenter - w/2.0;
  bbox[1] = ycenter - h/2.0; 
  bbox[2] = xcenter + w/2.0;
  bbox[3] = ycenter + h/2.0;

  BBoxSize(bbox, normalized);
  decoded_bboxes_.emplace(idx, std::move(bbox));
}

template void TFSSDdetector::DecodeBBox(const int (*bboxes)[4], int idx,
                                      bool normalized);
template void TFSSDdetector::DecodeBBox(const int8_t (*bboxes)[4], int idx,
                                      bool normalized);
}  // namespace dpssd
}  // namespace ai
}  // namespace vitis
