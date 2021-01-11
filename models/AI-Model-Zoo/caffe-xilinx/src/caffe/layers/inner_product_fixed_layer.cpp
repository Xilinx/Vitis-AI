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

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_fixed_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/quantize.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductFixedLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  InnerProductLayer<Dtype>::LayerSetUp(bottom, top);

  FixedParameter fixed_param = this->layer_param_.fixed_param();
  enable_fix_ = fixed_param.enable();
  fixed_method_ = fixed_param.fixed_method();
  bit_width_ = fixed_param.bit_width();
  weight_dec_pos_ = 0;
  bias_dec_pos_ = 0;

  LayerParameter layer_param(this->layer_param_);
  layer_param.set_type("InnerProduct");
  fixed_forward_inner_product_layer_ =
      LayerRegistry<Dtype>::CreateLayer(layer_param);
  fixed_forward_inner_product_layer_->SetUp(bottom, top);

}

template <typename Dtype>
void InnerProductFixedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::Reshape(bottom, top);
  fixed_forward_inner_product_layer_->Reshape(bottom, top);
}

template <typename Dtype>
void InnerProductFixedLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  auto fixed_blobs = fixed_forward_inner_product_layer_->blobs();
  if (!enable_fix_) {
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
               fixed_blobs[0]->mutable_cpu_data());
    if (this->bias_term_) {
      caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
                 fixed_blobs[1]->mutable_cpu_data());
    }
    fixed_forward_inner_product_layer_->Forward(bottom, top);
    return;
  }

  //if (this->phase_ == TRAIN || this->iter() == 0) {
  if ( !fixed_ || this->iter() == 0 ) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {
      this->weight_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_overflow(
          this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
          this->bit_width_));
      if (this->bias_term_) {
        this->bias_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_overflow(
            this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
            this->bit_width_));
      }
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S) {
      this->weight_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_diffs(
          this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
          this->bit_width_));
      if (this->bias_term_) {
        this->bias_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_diffs(
            this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
            this->bit_width_));
      }
    } else {
      LOG(FATAL) << "Unknown fixed method: " << this->fixed_method_;
    }

    caffe_cpu_fix(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
                  fixed_blobs[0]->mutable_cpu_data(), this->bit_width_,
                  this->weight_dec_pos_);
    if (this->bias_term_) {
      caffe_cpu_fix(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
                    fixed_blobs[1]->mutable_cpu_data(), this->bit_width_,
                    this->bias_dec_pos_);
    }
    fixed_ = true;
  } else if (this->phase_ == TEST) {
  }

  fixed_forward_inner_product_layer_->Forward(bottom, top);
}

template <typename Dtype>
void InnerProductFixedLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  fixed_ = false;
  InnerProductLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductFixedLayer);
#endif

INSTANTIATE_CLASS(InnerProductFixedLayer);
REGISTER_LAYER_CLASS(InnerProductFixed);

}  // namespace caffe
