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

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/prelu_fixed_layer.hpp"
#include "caffe/util/quantize.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void PReLUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* slope_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void PReLUBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* slope_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[c]);
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void PReLUParamBackward(const int n,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0);
    for ( int k = 1; k < rows; k++ ) {
        out_diff[index] += in_diff[index + k*rowPitch]
           * in_data[index + k*rowPitch] * (in_data[index + k*rowPitch] <= 0);
    }
  }
}

template <typename Dtype>
void PReLUFixedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // Fix slope_data
  Blob<Dtype> weight_fixed(this->blobs_[0]->shape());
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
                  weight_fixed.mutable_gpu_data(), this->bit_width_,
                  this->weight_dec_pos_);
  } else if (this->phase_ == TEST) {
  }

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  PReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, weight_fixed.gpu_data(), div_factor);//div_factor);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void PReLUFixedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      bottom_data ,
      backward_buff_.mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
       multiplier_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), slope_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        slope_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data,
        div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PReLUFixedLayer);


}  // namespace caffe
