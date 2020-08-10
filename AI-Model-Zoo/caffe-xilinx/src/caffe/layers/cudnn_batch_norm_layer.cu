#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  const Dtype* bias_data = this->blobs_[1]->gpu_data();

  Dtype* top_data = (top[0] == bottom[0]) ?
      private_top_->mutable_gpu_data() : top[0]->mutable_gpu_data();

  Dtype* save_mean = save_mean_.mutable_gpu_data();
  Dtype* save_inv_var = save_inv_var_.mutable_gpu_data();
  double epsilon = max(this->eps_, CUDNN_BN_MIN_EPSILON);

  if (this->phase_ == TRAIN) {
    double factor = (this->iter() == 0) ?
                    1.0 : (1. - this->moving_average_fraction_);
    // Call Batch normalization forward
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      Caffe::cudnn_handle(), mode_,
      cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
      bottom_desc_, bottom_data,
      top_desc_, top_data,
      scale_bias_mean_var_desc_, scale_data, bias_data,
      factor,
      this->blobs_[2]->mutable_gpu_data(),  // mean
      this->blobs_[3]->mutable_gpu_data(),  // variance
      epsilon, save_mean, save_inv_var));
  } else if (this->phase_ == TEST) {
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      Caffe::cudnn_handle(), mode_,
      cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
      bottom_desc_, bottom_data,
      top_desc_, top_data,
      scale_bias_mean_var_desc_, scale_data, bias_data,
      this->blobs_[2]->gpu_data(),  // mean
      this->blobs_[3]->gpu_data(),  // variance
      epsilon));
  } else {
    LOG(FATAL) << "Unknown phase";
  }

  if (top[0] == bottom[0]) {
    private_bottom_->CopyFrom(*bottom[0]);
    top[0]->CopyFrom(*private_top_);
  }
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (top[0] == bottom[0]) {
    private_top_->CopyFrom(*top[0], true);
    top_diff = private_top_->gpu_diff();
    bottom_data = private_bottom_->gpu_data();
  }

  const Dtype* save_mean = save_mean_.gpu_data();
  const Dtype* save_inv_var = save_inv_var_.gpu_data();

  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();

  double epsilon = max(this->eps_, CUDNN_BN_MIN_EPSILON);

  // call Batch Normalization Backward
  CUDNN_CHECK(cudnnBatchNormalizationBackward(
      Caffe::cudnn_handle(), mode_,
      cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
#if CUDNN_VERSION >= 4005
      cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::one,
#endif
      bottom_desc_, bottom_data,
      bottom_desc_, top_diff,
      bottom_desc_, bottom_diff,
      scale_bias_mean_var_desc_,
      scale_data, scale_diff, bias_diff,
      epsilon, save_mean, save_inv_var));
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBatchNormLayer);

}  // namespace caffe

#endif
