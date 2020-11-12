#include "caffe/layers/operate_segmap_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void OperatorForward(const int count, const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, count) {
	if (bottom_data[index] >= 200) { 
		top_data[index] = 250;
	} else if (bottom_data[index] >18 && bottom_data[index] < 30) {
		top_data[index] = 18;
	} else {
		top_data[index] = bottom_data[index];
	}  
    
  }
}

template <typename Dtype>
void OperateSegLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int count = top[0]->count();
  OperatorForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);  
}


template <typename Dtype>
__global__ void OperatorBackward(const int count, const Dtype* top_diff, Dtype* bottom_diff) {
   CUDA_KERNEL_LOOP(index, count) {
    bottom_diff[index] = top_diff[index];
  }
}

template <typename Dtype>
void OperateSegLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  if (propagate_down[0]) {
    OperatorBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_diff); 
  }
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(OperateSegLayer);

}  // namespace caffe
