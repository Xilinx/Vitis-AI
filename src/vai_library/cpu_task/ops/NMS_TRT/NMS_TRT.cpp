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

#include <cmath>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <thread>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include <vitis/ai/env_config.hpp>

// confidence thresh for ssd; set to 0.0 if wanting to use the value in model (which is 0.005)
DEF_ENV_PARAM_2(CONF_THRESH, "0.3", float)

using namespace std;

namespace {

// part 1: nms;

static float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

static float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

void applyNMS(const vector<vector<float>>& boxes, const vector<float>& scores,
              const float nms, const float conf, vector<size_t>& res) {
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  stable_sort(order.begin(), order.end(),
              [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
                return ls.first > rs.first;
              });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });
  vector<bool> exist_box(count, true);

  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];
    if (!exist_box[i]) continue;
    if (scores[i] < conf) {
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(i);
    // cout << "nms push "<< i<<endl;
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j]) continue;
      float ovr = 0.0;
      ovr = cal_iou(boxes[j], boxes[i]);
      if (ovr >= nms) exist_box[j] = false;
    }
  }
}

// part 2: tfssd part
enum SCORE_CONVERTER { SOFTMAX = 0, SIGMOID = 1 };
struct TFSSDResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /**
   * @struct BoundingBox
   * @brief Struct of an object coordinate, confidence, classification.
   */
  struct BoundingBox {
    /// Classification
    int label;
    /// Confidence
    float score;
    /// x-coordinate. x is normalized relative to the input image columns.
    /// Range from 0 to 1.
    float x;
    /// y-coordinate. y is normalized relative to the input image rows.
    /// Range from 0 to 1.
    float y;
    /// Width. Width is normalized relative to the input image columns,
    /// Range from 0 to 1.
    float width;
    /// Height. Heigth is normalized relative to the input image rows,
    /// Range from 0 to 1.
    float height;
  };
  /// All objects, a vector of BoundingBox
  std::vector<BoundingBox> bboxes;
};

class TFSSDdetector {
 public:
  enum CodeType { CORNER, CENTER_SIZE, CORNER_SIZE };

  TFSSDdetector(
      unsigned int num_classes, 
      CodeType code_type,
      bool variance_encoded_in_target, 
      unsigned int keep_top_k,
      const std::vector<float>& confidence_threshold, 
      unsigned int nms_top_k,
      float nms_threshold, 
      float eta,
      float y_scale, 
      float x_scale, 
      float height_scale, 
      float width_scale,
      SCORE_CONVERTER score_converter, 
      float scale_score = 1.0,
      float scale_loc = 1.0, 
      bool clip = false);

  void set_priors(std::vector<float>& in_priors) { 
      priors_.swap(in_priors); 
      num_priors_ = priors_.size()/4;
  }
  template <typename T>
  void Detect(const T* loc_data, const float* conf_data, TFSSDResult* result);

  template <typename T>
  void ApplyOneClassNMS(
      const T (*bboxes)[4], const float* conf_data, int label,
      const std::vector<std::pair<float, int> >& score_index_vec,
      std::vector<int>* indices);

  void GetOneClassMaxScoreIndex(
      const float* conf_data, int label,
      std::vector<std::pair<float, int> >* score_index_vec);

  void GetMultiClassMaxScoreIndex(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int> > >* score_index_vec);

  void GetMultiClassMaxScoreIndexMT(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int> > >* score_index_vec,
      int threads = 2);

  template <typename T>
  void DecodeBBox(const T (*bboxes)[4], int idx, bool normalized);

  std::map<int, std::vector<float> > decoded_bboxes_;

  const unsigned int num_classes_=0;
  CodeType code_type_;
  bool variance_encoded_in_target_=false;
  unsigned int keep_top_k_=0;
  std::vector<float> confidence_threshold_;
  float nms_confidence_=0.0;
  unsigned int nms_top_k_=0;
  float nms_threshold_=0.0;
  float eta_=0.0;

  std::vector<float> priors_;
  float y_scale_=0.0;
  float x_scale_=0.0;
  float height_scale_=0.0;
  float width_scale_=0.0;

  SCORE_CONVERTER score_converter_=SIGMOID;
  float scale_score_=0.0;
  float scale_loc_=0.0;

  bool clip_=false;
  int num_priors_=0;
};

TFSSDdetector::TFSSDdetector(unsigned int num_classes, 
                             CodeType code_type,
                             bool variance_encoded_in_target,
                             unsigned int keep_top_k,
                             const vector<float>& confidence_threshold,
                             unsigned int nms_top_k, 
                             float nms_threshold,
                             float eta,
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
      y_scale_(y_scale),
      x_scale_(x_scale),
      height_scale_(height_scale),
      width_scale_(width_scale),
      score_converter_(score_converter),
      scale_score_(scale_score),
      scale_loc_(scale_loc),
      clip_(clip) 
{
  if (score_converter_ == SIGMOID) {
    // 1/(1+exp(-x*scale))==y;  -->  x=-ln(1/y-1)/scale
    // also need fix confidence_threshold_
    for (auto i = 1u; i < confidence_threshold_.size(); i++) {
      confidence_threshold_[i] =
          (-log(1.0 / confidence_threshold_[i] - 1.0)) / scale_score_;
    }
  }
  nms_confidence_ = *std::min_element(confidence_threshold_.begin() + 1,
                                      confidence_threshold_.end());
}

template <typename T>
void TFSSDdetector::Detect(const T* loc_data, const float* conf_data, TFSSDResult* result) {
  decoded_bboxes_.clear();
  const T(*bboxes)[4] = (const T(*)[4])loc_data;

  unsigned int num_det = 0;
  vector<vector<int>> indices(num_classes_);
  vector<vector<pair<float, int>>> score_index_vec(num_classes_);

  // Get top_k scores (with corresponding indices).
  GetMultiClassMaxScoreIndexMT(conf_data, 1, num_classes_ - 1, &score_index_vec);

  for (unsigned int c = 1; c < num_classes_; ++c) {
    // Perform NMS for one class
    ApplyOneClassNMS(bboxes, conf_data, c, score_index_vec[c], &(indices[c]));
    num_det += indices[c].size();
  }

  if (keep_top_k_ > 0 && num_det > keep_top_k_) {
    vector<tuple<float, int, int>> score_index_tuples;
    for (auto label = 0u; label < num_classes_; ++label) {
      const vector<int>& label_indices = indices[label];
      for (auto j = 0u; j < label_indices.size(); ++j) {
        auto idx = label_indices[j];
        auto score = conf_data[idx * num_classes_ + label];
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
      indices[get<1>(item)].push_back(get<2>(item));
    }
  }
  for (auto label = 1u; label < indices.size(); ++label) {
    for (auto idx : indices[label]) {
      auto score = conf_data[idx * num_classes_ + label];

      auto& bbox = decoded_bboxes_[idx];
      TFSSDResult::BoundingBox res;
      res.label = label;
      res.score =  1.0 / (1.0 + exp(-1.0 * score * scale_score_));

      res.x = bbox[0] - bbox[2] / 2.0;
      res.y = bbox[1] - bbox[3] / 2.0;

      res.width = bbox[2];
      res.height = bbox[3];
      result->bboxes.emplace_back(res);
    }
  }
}

template void TFSSDdetector::Detect(const float* loc_data, const float* conf_data, TFSSDResult* result);

template <typename T>
void TFSSDdetector::ApplyOneClassNMS(
    const T (*bboxes)[4], const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices) {
  vector<size_t> results;
  vector<vector<float>> boxes;
  vector<float> scores;
  map<size_t, int> resultmap;
  int i = 0;
  for (auto& sc : score_index_vec) {
    const int idx = sc.second;
    if (decoded_bboxes_.find(idx) == decoded_bboxes_.end()) {
      DecodeBBox(bboxes, idx, true);
    }
    boxes.push_back(decoded_bboxes_[idx]);
    scores.push_back(sc.first);
    resultmap[i++] = idx;
  }
  applyNMS(boxes, scores, nms_threshold_, confidence_threshold_[label], results);
  for (auto& r : results) {
    indices->push_back(resultmap[r]);
  }
}

template void TFSSDdetector::ApplyOneClassNMS(
    const float (*bboxes)[4], const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices);

void TFSSDdetector::GetOneClassMaxScoreIndex(
    const float* conf_data, int label,
    vector<pair<float, int>>* score_index_vec) {
  conf_data += label;

  for (int i = 0; i < num_priors_; ++i) {
    auto score = *conf_data;
    if (score > nms_confidence_) {
      score_index_vec->emplace_back(score, i);
    }
    conf_data += num_classes_;
  }

  std::stable_sort(
      score_index_vec->begin(), score_index_vec->end(),
      [](const pair<float, int>& lhs, const pair<float, int>& rhs) {
        return lhs.first > rhs.first;
      });

  if (nms_top_k_ < score_index_vec->size()) {
    score_index_vec->resize(nms_top_k_);
  }
}

void TFSSDdetector::GetMultiClassMaxScoreIndex(
    const float* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec) {
  for (auto i = start_label; i < start_label + num_classes; ++i) {
    GetOneClassMaxScoreIndex(conf_data, i, &((*score_index_vec)[i]));
  }
}

void TFSSDdetector::GetMultiClassMaxScoreIndexMT(
    const float* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec, int threads) {
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
     worker.join();
}

template <typename T>
void TFSSDdetector::DecodeBBox(const T (*bboxes)[4], int idx, bool normalized) {
  // in tfconcat_decode, we get the center and size directly in prior_bbox, so
  // we don't need decode center/size here. corresponding logic is in
  // box_coders/faster_rcnn_box_coder.py: _decode() scale bboxes

  auto y_center_a = (priors_[idx*4+0] + priors_[idx*4+2])/2.0;
  auto x_center_a = (priors_[idx*4+1] + priors_[idx*4+3])/2.0;
  auto ha = priors_[idx*4+2]-priors_[idx*4+0];
  auto wa = priors_[idx*4+3]-priors_[idx*4+1];

  auto ty = bboxes[idx][0] * scale_loc_ / y_scale_;
  auto tx = bboxes[idx][1] * scale_loc_ / x_scale_;
  auto th = bboxes[idx][2] * scale_loc_ / height_scale_;
  auto tw = bboxes[idx][3] * scale_loc_ / width_scale_;

  auto w = exp(tw) * wa;
  auto h = exp(th) * ha;
  auto ycenter = ty * ha + y_center_a;
  auto xcenter = tx * wa + x_center_a;

#if 0
  std::cout <<"DecodeBox:  bboxes[idx] " << float( bboxes[idx][0]) << " " << float( bboxes[idx][1] ) << " "
            <<  float (bboxes[idx][2]) << " " << float( bboxes[idx][3] ) << std::endl;

  std::cout << "DecodeBox: ha wa:" << ha << " " << wa
            << " ty tx th tw:" << ty << " " << tx << " " << th << " " << tw
            << "    yxcenter:" << ycenter << " " << xcenter << std::endl;
#endif

  vector<float> bbox(4, 0);

  // seems x, y changed ? Yes. test and found.
  // bbox x,y,w,h
  bbox[0] = xcenter;
  bbox[1] = ycenter;
  bbox[2] = w;
  bbox[3] = h;

  decoded_bboxes_.emplace(idx, std::move(bbox));
}

template void TFSSDdetector::DecodeBBox(const float (*bboxes)[4], int idx, bool normalized);

// part 3: op part
template<typename T>
static void printv(const T& t) {
  std::cout <<"Vec : ";  for(auto &i: t) { std::cout << i << " " ; }  std::cout <<"\n";
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs}{

    NMS_THRESHOLD = op->get_attr<double>("iou_threshold_u_float");
    TOP_K = op->get_attr<int>("max_detections_per_class_u_int");
    KEEP_TOP_K = op->get_attr<int>("max_total_detections_u_int");
    conf_att = ENV_PARAM(CONF_THRESH );
    if (abs(conf_att) < 0.00001) {
       conf_att = op->get_attr<double>("score_threshold_u_float");
    }
  }

  int calculate(vart::simple_tensor_buffer_t<void> output, std::vector<vart::simple_tensor_buffer_t<void>> input) {
    CHECK_EQ( input.size(), 3);
    input_shape_0 = input[0].tensor->get_shape(); //
    input_shape_1 = input[1].tensor->get_shape(); //
    input_shape_2 = input[2].tensor->get_shape(); //
      // printv(input_shape_0);  //  1917 4
      // printv(input_shape_1);  // 1 1917 91
      // printv(input_shape_2);  // 1 1917 4
    CHECK_GT(input[1].mem_size, input[0].mem_size);
    CHECK_EQ(input_shape_1.size(), 3);
    num_classes = input_shape_1[2];   

    th_conf.resize(num_classes, conf_att);
    th_conf[0] = 0.0;
    det = std::unique_ptr<TFSSDdetector>(
      new TFSSDdetector(
          num_classes, 
          TFSSDdetector::CodeType::CENTER_SIZE,
          false, 
          KEEP_TOP_K, 
          th_conf, 
          TOP_K, 
          NMS_THRESHOLD, 
          1.0, 
          y_scale, 
          x_scale, 
          height_scale, 
          width_scale, 
          score_converter,
          scale_score, 
          scale_loc));

    float* in0 = (float*)input[0].data;  (void)in0;
    float* in1 = (float*)input[1].data;  (void)in1;
    float* in2 = (float*)input[2].data;  (void)in2;

    std::vector<float> anchor( input[0].mem_size/sizeof(float) );
    memcpy(anchor.data(), in0,  input[0].mem_size);
      // printv( anchor);
      // std::vector<float> tmpv1( input[1].mem_size/sizeof(float) );
      // memcpy(tmpv1.data(), in1,  input[1].mem_size);
      //    printv(tmpv1);
    float* outlayer = (float*)output.data; 
    memset(outlayer, 0, output.mem_size );

    det->set_priors(anchor);
    TFSSDResult result;
    det->Detect(in2, in1, &result);

    unsigned int i=0;
    for( i=0; i< std::min(result.bboxes.size(), output.mem_size/(6*sizeof(float))); i++) {
      outlayer[6*i+0] = result.bboxes[i].label;
      outlayer[6*i+1] = result.bboxes[i].y;
      outlayer[6*i+2] = result.bboxes[i].x;
      outlayer[6*i+3] = result.bboxes[i].height+result.bboxes[i].y;
      outlayer[6*i+4] = result.bboxes[i].width+result.bboxes[i].x;
      outlayer[6*i+5] = result.bboxes[i].score;

      if (0)
        std::cout << outlayer[6*i+0] << " " 
                  << outlayer[6*i+1] << " " 
                  << outlayer[6*i+2] << " " 
                  << outlayer[6*i+3] << " " 
                  << outlayer[6*i+4] << " " 
                  << outlayer[6*i+5] << "\n";
    }
    for(; i<output.mem_size/(sizeof(float)*6); i++) {
        outlayer[6*i+0] = 1;
    }
    return 0;
  }

private:
  std::vector<std::int32_t> input_shape_0, input_shape_1, input_shape_2;
  std::unique_ptr<TFSSDdetector> det;

  int num_classes = 91;
  double conf_att = 0.0;
  int KEEP_TOP_K = 100;
  std::vector<float> th_conf;
  int TOP_K = 100;
  float NMS_THRESHOLD = 0.6;
  float y_scale = 10.0;
  float x_scale = 10.0;
  float height_scale = 5.0;
  float width_scale = 5.0;
  SCORE_CONVERTER score_converter = SIGMOID;
  float scale_score = 1.0;
  float scale_loc = 1.0;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
