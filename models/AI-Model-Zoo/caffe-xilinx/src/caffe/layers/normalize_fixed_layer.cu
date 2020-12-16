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

#include "caffe/layers/normalize_fixed_layer.hpp"
#include "caffe/util/quantize.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeFixedLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  auto fixed_blobs = fixed_forward_normalize_layer_->blobs();
  if (!enable_fix_) {
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
               fixed_blobs[0]->mutable_gpu_data());
    fixed_forward_normalize_layer_->Forward(bottom, top);
    return;
  }

  // Fix weights
  if (this->phase_ == TRAIN || this->iter() == 0) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_overflow(
          this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
          this->bit_width_));
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S) {
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_diffs(
          this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
          this->bit_width_));
    } else {
      LOG(FATAL) << "Unknown fixed method: " << this->fixed_method_;
    }

    caffe_gpu_fix(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
                  fixed_blobs[0]->mutable_gpu_data(), this->bit_width_,
                  this->weight_dec_pos_);
  } else if (this->phase_ == TEST) {
  }

  fixed_forward_normalize_layer_->Forward(bottom, top);
}

template <typename Dtype>
void NormalizeFixedLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

  NormalizeLayer<Dtype>::Backward_gpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeFixedLayer);

} // namespace caffe
