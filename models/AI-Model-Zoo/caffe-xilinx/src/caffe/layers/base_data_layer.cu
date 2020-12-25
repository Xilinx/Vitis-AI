#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // check batch has finished copying to the device
  CUDA_CHECK(cudaStreamWaitEvent(cudaStreamDefault, batch->copied_, 0));

  // Reshape to loaded data.
  if (this->transform_param_.use_gpu_transform()) {
    // instead of copy, perform out-of-place transform(!)
    this->data_transformer_->TransformGPU(top[0]->num(),
                                       top[0]->channels(),
                                       batch->data_.height(),
                                       batch->data_.width(),
                                       batch->data_.gpu_data(),
                                       top[0]->mutable_gpu_data(),
                                       batch->random_vec_.mutable_gpu_data());
  }  else {
    // Copy the data
    // Reshape to loaded data.
    top[0]->ReshapeLike(batch->data_);
    caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
               top[0]->mutable_gpu_data());
  }

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
    if(top.size() == 3 && this->is_vpgnet){
        top[2]->ReshapeLike(batch->label_1_);
        caffe_copy(batch->label_1_.count(), batch->label_1_.gpu_data(), top[2]->mutable_gpu_data());
    }
  }
  
  if (this->output_segmaps_) {
    top[2]->ReshapeLike(batch->segmap_);
    caffe_copy(batch->segmap_.count(), batch->segmap_.gpu_data(),
        top[2]->mutable_gpu_data());
  }
  if (this->output_lanemaps_) {
    top[3]->ReshapeLike(batch->lanelabel_);
    caffe_copy(batch->lanelabel_.count(), batch->lanelabel_.gpu_data(),
	       top[3]->mutable_gpu_data()); // label map
    top[4]->ReshapeLike(batch->lanemap_);
    caffe_copy(batch->lanemap_.count(), batch->lanemap_.gpu_data(),
	       top[4]->mutable_gpu_data()); // type map
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
