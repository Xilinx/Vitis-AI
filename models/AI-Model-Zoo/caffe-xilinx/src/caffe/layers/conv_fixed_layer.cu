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
//#include <float.h>

#include "caffe/layers/conv_fixed_layer.hpp"
#include "caffe/util/quantize.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionFixedLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  auto fixed_blobs = fixed_forward_conv_layer_->blobs();
  if (!enable_fix_) {
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
               fixed_blobs[0]->mutable_gpu_data());
    if (this->bias_term_) {
      caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->gpu_data(),
                 fixed_blobs[1]->mutable_gpu_data());
    }
    fixed_forward_conv_layer_->Forward(bottom, top);
    return;
  }

  // Fix weights and bias
  if (this->phase_ == TRAIN || this->iter() == 0) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_overflow(
          this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
          this->bit_width_));
      if (this->bias_term_) {
        this->bias_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_overflow(
            this->blobs_[1]->count(), this->blobs_[1]->gpu_data(),
            this->bit_width_));
      }
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S) {
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_diffs(
          this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
          this->bit_width_));
      if (this->bias_term_) {
        this->bias_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_diffs(
            this->blobs_[1]->count(), this->blobs_[1]->gpu_data(),
            this->bit_width_));
      }
    } else {
      LOG(FATAL) << "Unknown fixed method: " << this->fixed_method_;
    }

    caffe_gpu_fix(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
                  fixed_blobs[0]->mutable_gpu_data(), this->bit_width_,
                  this->weight_dec_pos_);
    if (this->bias_term_) {
      caffe_gpu_fix(this->blobs_[1]->count(), this->blobs_[1]->gpu_data(),
                    fixed_blobs[1]->mutable_gpu_data(), this->bit_width_,
                    this->bias_dec_pos_);
    }
  } else if (this->phase_ == TEST) {
  }

  fixed_forward_conv_layer_->Forward(bottom, top);
}

template <typename Dtype>
void ConvolutionFixedLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

  ConvolutionLayer<Dtype>::Backward_gpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionFixedLayer);

} // namespace caffe
