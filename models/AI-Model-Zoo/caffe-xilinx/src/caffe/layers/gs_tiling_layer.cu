#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gs_tiling_layer.hpp"

namespace caffe {

    template <typename Dtype>
    __global__ void tiling_kernel(const Dtype *x, int wi, int hi, int ci, int wo, int ho, int co, int batch, int stride, int forward, Dtype *out)
    {
        int size = batch*ci*hi*wi;
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i >= size) return;
        int iw = i % wi;
        i = i / wi;
        int ih = i % hi;
        i = i / hi;
        int ic = i % ci;
        i = i / ci;
        int ib = i % batch;
        int ow = iw / stride;
        int oh = ih / stride;
        int oc = ((ih % stride) * stride + (iw % stride)) * ci + ic;
        int oi = ((ib * co + oc) * ho + oh) * wo + ow;
        int ii = ((ib * ci + ic) * hi + ih) * wi + iw;


        if(forward)
        {
            out[oi] = x[ii];
        }         
        else
        {
            out[ii] = x[oi];
        }
    }


template <typename Dtype>
void GSTilingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
        int count = bottom[0]->count();
        Dtype *top_data = top[0]->mutable_gpu_data();
	int batch_num_ = bottom[0]->num();
        if(reverse_)
        {
	        tiling_kernel<Dtype>
	         <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(bottom_data, output_width_, output_height_,
	                  output_channels_, input_width_, input_height_, input_channels_, batch_num_, stride_, !reverse_, top_data);

        }else
	    {
	        tiling_kernel<Dtype>
	         <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(bottom_data, input_width_, input_height_,
	                  input_channels_, output_width_, output_height_, output_channels_, batch_num_, stride_, !reverse_, top_data);
	    }
}

template <typename Dtype>
void GSTilingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if(!propagate_down[0]){
            return;
        }
        int count = top[0]->count();
	int batch_num_ = bottom[0]->num();
        const Dtype *top_diff = top[0]->gpu_diff();
        Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
        if(reverse_)
        {
	        tiling_kernel<Dtype>
	         <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(top_diff, output_width_, output_height_,
	                  output_channels_, input_width_, input_height_, input_channels_, batch_num_, stride_, reverse_, bottom_diff);

        }else
	    {
	        tiling_kernel<Dtype>
	         <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(top_diff, input_width_, input_height_,
	                  input_channels_, output_width_, output_height_, output_channels_, batch_num_, stride_, reverse_, bottom_diff);
	    }
}

INSTANTIATE_LAYER_GPU_FUNCS(GSTilingLayer);

}  // namespace caffe
