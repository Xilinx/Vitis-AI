#ifdef USE_CUDNN
#include <vector>

#include "caffe/util/quantize.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnPoolingForward(Caffe::cudnn_handle(), pooling_desc_,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data));

#ifdef DPU_ACCURACY
  // simulate DPU average pooling, which convert dividing to bit shifting
  float scale = 0.f;
  bool needScaling = false;
  if (this->kernel_h_ == 3 && this->kernel_w_ == 3) {
    needScaling = true;
    scale = 9.0 * 7.f / 64.f;
  } else if (this->kernel_h_ == 5 && this->kernel_w_ == 5) {
    needScaling = true;
    scale = 25.0 * 10.f / 256.f;
  } else if (this->kernel_h_ == 6 && this->kernel_w_ == 6) {
    needScaling = true;
    scale = 36.0 * 7.f / 256.f;
  } else if (this->kernel_h_ == 7 && this->kernel_w_ == 7){
    needScaling = true;
    scale = 49.0 * 21.f / 1024.f;
  } else if (this->kernel_h_ == 14 && this->kernel_w_ == 14){
    needScaling = true;
    scale = 196.0 * 21.f / 4096.f;
  } 

  if ( needScaling )
    caffe_pooling_scale( top[0]->count(), 
                         top[0]->gpu_data(),
                         top[0]->mutable_gpu_data(),
                         scale );
#endif // DPU_ACCURACY
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  CUDNN_CHECK(cudnnPoolingBackward(Caffe::cudnn_handle(), pooling_desc_,
        cudnn::dataType<Dtype>::one,
        top_desc_, top_data, top_desc_, top_diff,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff));
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNPoolingLayer);

}  // namespace caffe
#endif
