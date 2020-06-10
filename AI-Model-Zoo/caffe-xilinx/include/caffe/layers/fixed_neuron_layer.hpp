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

#ifndef CAFFE_FIXED_NEURON_LAYER_HPP_
#define CAFFE_FIXED_NEURON_LAYER_HPP_

#include <vector>
#include <random>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class FixedNeuronLayer : public Layer<Dtype> {
 public:
  explicit FixedNeuronLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
        // gen_((std::random_device())()) {}
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

  inline const char* type() const override { return "FixedNeuron"; }
  inline int ExactNumBottomBlobs() const override { return 1; }
  inline int ExactNumTopBlobs() const override { return 1; }

 protected:
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) override;

  bool enable_fix_;
  bool follow_data_layer;
  FixedParameter_FixedMethod fixed_method_;
  int bit_width_;
  int dec_pos_;
  // Dtype moving_average_fraction_;
  map<int,int> pos_hist_;
  // Dtype data_fraction_;
  // std::mt19937 gen_;
  // std::uniform_real_distribution<> dist_;
};

}  // namespace caffe

#endif  // CAFFE_FIXED_NEURON_LAYER_HPP_
