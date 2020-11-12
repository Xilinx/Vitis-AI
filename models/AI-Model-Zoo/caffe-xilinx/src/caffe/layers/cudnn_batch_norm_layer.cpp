#ifdef USE_CUDNN

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::LayerSetUp(bottom, top);

  BatchNormParameter param = this->layer_param_.batch_norm_param();

  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createTensor4dDesc<Dtype>(&scale_bias_mean_var_desc_);

  // currently only SPATIAL mode is supported (most commonly used mode)
  // If there's enough demand we can implement CUDNN_BATCHNORM_PER_ACTIVATION
  // though it's not currently implemented for the CPU layer
  mode_ = CUDNN_BATCHNORM_SPATIAL;
  int channels = bottom[0]->channels();

  if (this->blobs_.size() != 5) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(5);
    this->blobs_[0].reset(new Blob<Dtype>(1, channels, 1, 1, true));
    this->blobs_[1].reset(new Blob<Dtype>(1, channels, 1, 1, true));
    this->blobs_[2].reset(new Blob<Dtype>(1, channels, 1, 1, true));
    this->blobs_[3].reset(new Blob<Dtype>(1, channels, 1, 1, true));
    this->blobs_[4].reset(new Blob<Dtype>(1, 1, 1, 1, true));

    if (param.has_scale_filler()) {
      shared_ptr<Filler<Dtype> > scale_filler(
          GetFiller<Dtype>(param.scale_filler()));
      scale_filler->Fill(this->blobs_[0].get());
    } else {
      caffe_set(this->blobs_[0]->count(), Dtype(1),
                this->blobs_[0]->mutable_cpu_data());
    }

    if (param.has_bias_filler()) {
      shared_ptr<Filler<Dtype> > bias_filler(
          GetFiller<Dtype>(param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    } else {
      caffe_set(this->blobs_[1]->count(), Dtype(0),
                this->blobs_[1]->mutable_cpu_data());
    }

    for (int i = 2; i < 5; i++) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
    for (int i = 0; i < 5; i++) {
      caffe_set(this->blobs_[i]->count(), Dtype(1),
                this->blobs_[i]->mutable_cpu_mask());
    }
  }
  handles_setup_ = true;

  if (bottom == top) { // CUDNN_BN not support in-place
    private_top_ = make_shared<Blob<Dtype>>(top[0]->shape());
    private_bottom_ = make_shared<Blob<Dtype>>(bottom[0]->shape());
  }
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::Reshape(bottom, top);

  // set up main tensors
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());

  // aux tensors for caching mean & invVar from fwd to bwd pass
  int C = bottom[0]->channels();
  int H = bottom[0]->height();
  int W = bottom[0]->width();
  if (mode_ == CUDNN_BATCHNORM_SPATIAL) {
    save_mean_.Reshape(1, C, 1, 1);
    save_inv_var_.Reshape(1, C, 1, 1);
  } else if (mode_ == CUDNN_BATCHNORM_PER_ACTIVATION) {
    save_mean_.Reshape(1, C, H, W);
    save_inv_var_.Reshape(1, C, H, W);
  } else {
    LOG(FATAL) << "Unknown cudnnBatchNormMode_t";
  }
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc_,
      bottom_desc_, mode_));

  if (top[0] == bottom[0]) {
    if (!private_top_) {
      private_top_ = make_shared<Blob<Dtype>>(top[0]->shape());
    } else {
      private_top_->ReshapeLike(*top[0]);
    }
    if (!private_bottom_) {
      private_bottom_ = make_shared<Blob<Dtype>>(bottom[0]->shape());
    } else {
      private_bottom_->ReshapeLike(*bottom[0]);
    }
  }
}

template <typename Dtype>
CuDNNBatchNormLayer<Dtype>::~CuDNNBatchNormLayer() {
  if (!handles_setup_) return;
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_);
}

INSTANTIATE_CLASS(CuDNNBatchNormLayer);

}  // namespace caffe

#endif
