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

#ifndef CAFFE_SCALE_FIXED_LAYER_HPP_
#define CAFFE_SCALE_FIXED_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/scale_layer.hpp"

namespace caffe {

/*
 * This layer directly inherits ScaleLayer
*/
template <typename Dtype> class ScaleFixedLayer : public ScaleLayer<Dtype> {
public:
  explicit ScaleFixedLayer(const LayerParameter &param)
      : ScaleLayer<Dtype>(param) {}
  void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                  const vector<Blob<Dtype> *> &top) override;
  void Reshape(const vector<Blob<Dtype> *> &bottom,
               const vector<Blob<Dtype> *> &top) override;

  inline const char *type() const override { return "ScaleFixed"; }

protected:
  void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override;
  void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override;
  void Backward_cpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override;
  void Backward_gpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override;

  bool enable_fix_;
  FixedParameter_FixedMethod fixed_method_;
  int bit_width_;
  int weight_dec_pos_;
  int bias_dec_pos_;

  shared_ptr<Layer<Dtype>> fixed_forward_scale_layer_;
  bool bias_term_;
};

#endif

} // namespace caffe
