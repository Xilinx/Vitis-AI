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

#ifndef CAFFE_DENSEBOX_MAP_EVALUATION_LAYER_HPP_
#define CAFFE_DENSEBOX_MAP_EVALUATION_LAYER_HPP_

#include <vector>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
/**
 * @brief Computes the detection map accuracy of densebox model.
 */
typedef struct bbox {
  int xmin;
  int ymin;
  int xmax;
  int ymax;
  int id;
}BBox;

template <typename Dtype>
class DenseboxMapEvaluationLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides DenseboxMapEvaluationParameter accuracy_param,
   *     with DenseboxMapEvaluationLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank @f$ k @f$ at which a prediction is considered
   *     correct.  For example, if @f$ k = 5 @f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  explicit DenseboxMapEvaluationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DenseboxMapEvaluation"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  // If there are two top blobs, then the second blob will contain average
  // percision per class.
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlos() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- DenseboxMapEvaluationLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  
  virtual void ApplyNms(const std::vector<std::pair<BBox, float>> & pred_bbox_conf,  
      std::vector<std::pair<BBox, float>>* nms_bbox_conf);

  static bool cmp(const std::pair<BBox, float>& a_bbx, const std::pair<BBox, float>& b_bbx) {
    return a_bbx.second > b_bbx.second;  
  } 

  int channel_axes_;
  int outer_num_, inner_num_;
  float nms_threshold_, conf_threshold_, iou_threshold_;
};

}  // namespace caffe

#endif // CAFFE_DENSEBOX_MAP_EVALUATION_LAYER_HPP_
