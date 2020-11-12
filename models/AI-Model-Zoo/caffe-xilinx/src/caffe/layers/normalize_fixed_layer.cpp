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

#include <algorithm>
#include <vector>

#include "caffe/layers/normalize_fixed_layer.hpp"
#include "caffe/util/quantize.hpp"

namespace caffe {

/**
 */
template <typename Dtype>
void NormalizeFixedLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // Set up normalize layer
  NormalizeLayer<Dtype>::LayerSetUp(bottom, top);

  // Set up fixed parameters
  FixedParameter fixed_param = this->layer_param_.fixed_param();
  enable_fix_ = fixed_param.enable();
  fixed_method_ = fixed_param.fixed_method();
  bit_width_ = fixed_param.bit_width();
  weight_dec_pos_ = 0;

  // Set up normalize layer for fixed forward
  LayerParameter layer_param(this->layer_param_);
  layer_param.set_type("Normalize");
  fixed_forward_normalize_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
  fixed_forward_normalize_layer_->SetUp(bottom, top);
}

template <typename Dtype>
void NormalizeFixedLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  NormalizeLayer<Dtype>::Reshape(bottom, top);
  fixed_forward_normalize_layer_->Reshape(bottom, top);
}

template <typename Dtype>
void NormalizeFixedLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  auto fixed_blobs = fixed_forward_normalize_layer_->blobs();
  if (!enable_fix_) {
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
               fixed_blobs[0]->mutable_cpu_data());
    fixed_forward_normalize_layer_->Forward(bottom, top);
    return;
  }

  // Fix weights
  if (this->phase_ == TRAIN || this->iter() == 0) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {
      this->weight_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_overflow(
          this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
          this->bit_width_));
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S) {
      this->weight_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_diffs(
          this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
          this->bit_width_));
    } else {
      LOG(FATAL) << "Unknown fixed method: " << this->fixed_method_;
    }

    caffe_cpu_fix(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
                  fixed_blobs[0]->mutable_cpu_data(), this->bit_width_,
                  this->weight_dec_pos_);
  } else if (this->phase_ == TEST) {
  }

  fixed_forward_normalize_layer_->Forward(bottom, top);
}

template <typename Dtype>
void NormalizeFixedLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

  NormalizeLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(NormalizeFixedLayer);
#endif

INSTANTIATE_CLASS(NormalizeFixedLayer);
REGISTER_LAYER_CLASS(NormalizeFixed);

} // namespace caffe
