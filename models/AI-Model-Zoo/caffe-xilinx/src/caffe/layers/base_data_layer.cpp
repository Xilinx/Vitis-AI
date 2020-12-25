#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  is_vpgnet = false;
  output_lanemaps_ = false;
  if (top.size() == 1) {
    output_labels_ = false;
    output_segmaps_ = false;
  } else if (top.size() == 2) {
    output_labels_ = true;
    output_segmaps_ = false;
  } else  {
    output_labels_ = true;
    if (top.size() == 3){
      if (strcmp(this->type(), "DriveData") == 0) {
	is_vpgnet = true;
	output_segmaps_ = false;
      } else {
	output_segmaps_ = true;
	is_vpgnet = false;
      }
    }
    if (top.size() == 5) {
      output_lanemaps_ = true;
    }
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
      if(top.size() == 3 && this->is_vpgnet){
        prefetch_[i].label_1_.mutable_cpu_data();
      }
    }
    if (this->output_segmaps_) {
      prefetch_[i].segmap_.mutable_cpu_data();
    }
    if (this->output_lanemaps_) {
      prefetch_[i].lanelabel_.mutable_cpu_data();
      prefetch_[i].lanemap_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
        if(top.size() == 3 && this->is_vpgnet){
          prefetch_[i].label_1_.mutable_gpu_data();
        }
      }
      if (this->output_segmaps_) {
        prefetch_[i].segmap_.mutable_gpu_data();
      }
      if (this->output_lanemaps_) {
	prefetch_[i].lanelabel_.mutable_gpu_data();
	prefetch_[i].lanemap_.mutable_gpu_data();
      }
      CUDA_CHECK(cudaEventCreate(&prefetch_[i].copied_));
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data()->async_gpu_push();
        if (this->output_labels_) {
            batch->label_.data()->async_gpu_push();
            if(this->is_vpgnet){
                batch->label_1_.data()->async_gpu_push();
            }
            
        }
        if (this->output_segmaps_) {
            batch->segmap_.data()->async_gpu_push();
        }
	if (this->output_lanemaps_) {
	  batch->lanelabel_.data()->async_gpu_push();
	  batch->lanemap_.data()->async_gpu_push();
	}
        cudaStream_t stream = batch->data_.data()->stream();
        CUDA_CHECK(cudaEventRecord(batch->copied_, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(), top[1]->mutable_cpu_data());
    if(this->is_vpgnet){
        top[2]->ReshapeLike(batch->label_1_);
        caffe_copy(batch->label_1_.count(), batch->label_1_.cpu_data(), top[2]->mutable_cpu_data());
    }
  }
  if (this->output_segmaps_) {
    top[2]->ReshapeLike(batch->segmap_);
    caffe_copy(batch->segmap_.count(), batch->segmap_.cpu_data(),
        top[2]->mutable_cpu_data());
  }
  if (this->output_lanemaps_) {
    top[3]->ReshapeLike(batch->lanelabel_);
    caffe_copy(batch->lanelabel_.count(), batch->lanelabel_.cpu_data(),
	       top[3]->mutable_cpu_data()); // label map
    top[4]->ReshapeLike(batch->lanemap_);
    caffe_copy(batch->lanemap_.count(), batch->lanemap_.cpu_data(),
	       top[4]->mutable_cpu_data()); // type map
  }
  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
