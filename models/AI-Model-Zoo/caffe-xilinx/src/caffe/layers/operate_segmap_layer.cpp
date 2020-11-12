#include "caffe/layers/operate_segmap_layer.hpp"

namespace caffe {
		

template <typename Dtype>
void OperateSegLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
}

template <typename Dtype>
void OperateSegLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
    const vector<Blob<Dtype>*>& top) { 
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int count = top[0]->count();
  for ( int i = 0; i < count; i++ ) {
	if (bottom_data[count] >= 200) { 
		top_data[count] = 250;
	} else if (bottom_data[count] >18 && bottom_data[count] < 30) {
		top_data[count] = 18;
	} else {
		top_data[count] = bottom_data[count];
	}  
  }	  
}

template <typename Dtype>
void OperateSegLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	for ( int i = 0; i < count; i++ ) {
		bottom_diff[count] = top_diff[count];
	}
  }
}

#ifdef CPU_ONLY
STUB_GPU(OperateSegLayer);
#endif

INSTANTIATE_CLASS(OperateSegLayer);
REGISTER_LAYER_CLASS(OperateSeg);

} // namespace
