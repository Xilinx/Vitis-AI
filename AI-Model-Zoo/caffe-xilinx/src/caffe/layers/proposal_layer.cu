#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/layers/proposal_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace std;

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

namespace caffe {

template <typename Dtype>
__global__ void CalcBBoxesGPU(int num, int width, int height, int anchor_num, Dtype stride_h,
		Dtype stride_w, const Dtype* img_info, const Dtype* reg, const Dtype* anchor, Dtype* bbox) {
	CUDA_KERNEL_LOOP(i, num) {
		int map=height*width;
		int anchor_index=(i/(map))%anchor_num;
		Dtype* bbox_with_offset=bbox+4*i;
		const Dtype* reg_with_offset=reg+(i/map)*4*map+i%map;
		int img_index=i/map/anchor_num;
		Dtype img_height=img_info[img_index*3+0];
		Dtype img_width=img_info[img_index*3+1];

		Dtype x(i%width);
		Dtype y((i/width)%height);
		bbox_with_offset[0]=stride_w*x+anchor[anchor_index*4+0]-Dtype(1);
		bbox_with_offset[1]=stride_h*y+anchor[anchor_index*4+1]-Dtype(1);
		bbox_with_offset[2]=stride_w*x+anchor[anchor_index*4+2]-Dtype(1);
		bbox_with_offset[3]=stride_h*y+anchor[anchor_index*4+3]-Dtype(1);

		Dtype bbox_width=bbox_with_offset[2]-bbox_with_offset[0]+Dtype(1);
		Dtype bbox_height=bbox_with_offset[3]-bbox_with_offset[1]+Dtype(1);
		Dtype bbox_center_x=bbox_with_offset[0]+Dtype(0.5)*bbox_width;
		Dtype bbox_center_y=bbox_with_offset[1]+Dtype(0.5)*bbox_height;

		Dtype bbox_center_x_new=reg_with_offset[map*0]*bbox_width+bbox_center_x;
		Dtype bbox_center_y_new=reg_with_offset[map*1]*bbox_height+bbox_center_y;
		Dtype bbox_width_new=exp(reg_with_offset[map*2])*bbox_width;
		Dtype bbox_height_new=exp(reg_with_offset[map*3])*bbox_height;

		bbox_with_offset[0]=bbox_center_x_new-Dtype(0.5)*bbox_width_new;
		bbox_with_offset[1]=bbox_center_y_new-Dtype(0.5)*bbox_height_new;
		bbox_with_offset[2]=bbox_center_x_new+Dtype(0.5)*bbox_width_new;
		bbox_with_offset[3]=bbox_center_y_new+Dtype(0.5)*bbox_height_new;

		bbox_with_offset[0]=max(min(bbox_with_offset[0], img_width-Dtype(1)), Dtype(0));
		bbox_with_offset[1]=max(min(bbox_with_offset[1], img_height-Dtype(1)), Dtype(0));
		bbox_with_offset[2]=max(min(bbox_with_offset[2], img_width-Dtype(1)), Dtype(0));
		bbox_with_offset[3]=max(min(bbox_with_offset[3], img_height-Dtype(1)), Dtype(0));
	}
}

static const int ele_count=32;
static const int ele_bitnum=sizeof(unsigned int)*8;
static const int bitnum=ele_bitnum*ele_count;

template <typename Dtype>
__global__ void FilterBBoxGPU(int num, Dtype min_width, Dtype min_height, const Dtype* bbox, const int* index, unsigned int* exist) {
	int i=blockIdx.x * blockDim.x + threadIdx.x;
	if(i<DIVUP(num, ele_bitnum)) {
		for(int j=0; j<ele_bitnum; j++) {
			if(i*ele_bitnum+j<num) {
				const Dtype* bbox_offset=bbox+4*index[i*ele_bitnum+j];
				if(bbox_offset[2]-bbox_offset[0]>min_width && bbox_offset[3]-bbox_offset[1]>min_height) {
					exist[i] |= 1U<<j;
				}
				else {
					exist[i] &= ~(1U<<j);
				}
			}
		}
	}
}

template <typename Dtype>
__device__ static inline Dtype GetIoU(const Dtype* rect1, const Dtype* rect2) {
	Dtype left=rect1[0]>rect2[0]?rect1[0]:rect2[0];
	Dtype top=rect1[1]>rect2[1]?rect1[1]:rect2[1];
	Dtype right=rect1[2]<rect2[2]?rect1[2]:rect2[2];
	Dtype bottom=rect1[3]<rect2[3]?rect1[3]:rect2[3];
	if(right<left || bottom<top) {
		return Dtype(0);
	}
	else {
		Dtype inter=(right-left)*(bottom-top);
		return inter/((rect1[2]-rect1[0])*(rect1[3]-rect1[1])+(rect2[2]-rect2[0])*(rect2[3]-rect2[1])-inter);
	}
}

template <typename Dtype>
__global__ void BBoxOverlapGPU(int bbox_num, Dtype overlap_th, const Dtype* bbox, const int* index, unsigned int* overlap) {
	int thread=threadIdx.x;

	int col_size=min(bbox_num-blockIdx.x*bitnum, bitnum);
	int row_size=min(bbox_num-blockIdx.y*bitnum, bitnum);

	__shared__ Dtype bbox_block[bitnum*4];
	if(thread<col_size) {
		bbox_block[thread*4+0]=bbox[index[blockIdx.x*bitnum+thread]*4+0];
		bbox_block[thread*4+1]=bbox[index[blockIdx.x*bitnum+thread]*4+1];
		bbox_block[thread*4+2]=bbox[index[blockIdx.x*bitnum+thread]*4+2];
		bbox_block[thread*4+3]=bbox[index[blockIdx.x*bitnum+thread]*4+3];
	}
	__syncthreads();

	if(thread<row_size) {
		int row=blockIdx.y*bitnum+thread;
		unsigned int res[ele_count];
		for(int i=0; i<ele_count; i++) {
			res[i]=0U;
		}
		for(int i=0; i<col_size; i++) {
			if(GetIoU(bbox+index[row]*4, bbox_block+i*4)>overlap_th) {
				res[i/ele_bitnum] |= 1U<<(i%ele_bitnum);
			}
		}
		for(int i=0; i<ele_count; i++) {
			overlap[DIVUP(bbox_num, bitnum)*ele_count*row+blockIdx.x*ele_count+i]=res[i];
		}
	}
}

template <typename Dtype>
vector<int> ProposalLayer<Dtype>::NmsGPU(int num, Dtype img_scale, const Dtype* bbox, const Dtype* score) {
	vector<pair<Dtype, int>> score_index(num);
	for(size_t i=0; i<score_index.size(); i++) {
		score_index[i].first=score[i+num];
		score_index[i].second=i;
	}
	sort(score_index.begin(), score_index.end(),
			[](const pair<Dtype, int>& lhs, const pair<Dtype, int>& rhs){return lhs.first>rhs.first;});
	if(num>nms_input_num) {
		score_index.resize(nms_input_num);
	}
	int bbox_num=score_index.size();

	index.Reshape({bbox_num});
	int* index_data=index.mutable_cpu_data();
	for(int i=0; i<bbox_num; i++) {
		index_data[i]=score_index[i].second;
	}

	exist.Reshape({DIVUP(bbox_num, ele_bitnum)});
	FilterBBoxGPU<Dtype><<<CAFFE_GET_BLOCKS(exist.count()), CAFFE_CUDA_NUM_THREADS>>>(bbox_num, min_width*img_scale, min_height*img_scale,
			bbox, index.gpu_data(), exist.mutable_gpu_data());

	overlap.Reshape({DIVUP(bbox_num, bitnum)*ele_count*bbox_num});
	BBoxOverlapGPU<Dtype><<<dim3(DIVUP(bbox_num, bitnum), DIVUP(bbox_num, bitnum)), bitnum>>>(bbox_num, nms_iou_th, bbox, index.gpu_data(),
			overlap.mutable_gpu_data());

	unsigned int* exist_data=exist.mutable_cpu_data();
	const unsigned int* overlap_data=overlap.cpu_data();
	for(int i=0; i<bbox_num; i++) {
		if((exist_data[i/ele_bitnum] & (1U<<(i%ele_bitnum)))>0) {
			const unsigned int* overlap_data_offset=overlap_data+i*DIVUP(bbox_num, bitnum)*ele_count;
			if(bbox_num-i-1<ele_bitnum*2) {
				for(int j=i+1; j<bbox_num; j++) {
					exist_data[j/ele_bitnum] &= ~(overlap_data_offset[j/ele_bitnum] & (1U<<(j%ele_bitnum)));
				}
			}
			else {
				int end=bbox_num-bbox_num%ele_bitnum;
				int front=i+1+(ele_bitnum-(i+1)%ele_bitnum)%ele_bitnum;
				for(int j=i+1; j<front; j++) {
					exist_data[j/ele_bitnum] &= ~(overlap_data_offset[j/ele_bitnum] & (1U<<(j%ele_bitnum)));
				}
				for(int j=front; j<end; j+=ele_bitnum) {
					exist_data[j/ele_bitnum] &= ~overlap_data_offset[j/ele_bitnum];
				}
				for(int j=end; j<bbox_num; j++) {
					exist_data[j/ele_bitnum] &= ~(overlap_data_offset[j/ele_bitnum] & (1U<<(j%ele_bitnum)));
				}
			}
		}
	}

	vector<int> result;
	result.reserve(bbox_num);
	for(int i=0; i<bbox_num; i++) {
		if((exist_data[i/ele_bitnum] & (1U<<(i%ele_bitnum)))>0) {
			result.push_back(index_data[i]);
		}
	}
	if(result.size()>nms_output_num) {
		result.resize(nms_output_num);
	}
	return result;
}

template <typename Dtype>
__global__ void GenerateOutputGPU(int num, const Dtype* in, const int* result_batch_index, Dtype* out) {
	CUDA_KERNEL_LOOP(i, num) {
		const Dtype* bbox=in+result_batch_index[i*2]*4;
		out[i*5+0]=result_batch_index[i*2+1];
		out[i*5+1]=bbox[0];
		out[i*5+2]=bbox[1];
		out[i*5+3]=bbox[2];
		out[i*5+4]=bbox[3];
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::NmsGPU(int img_num, int num_per_img, const Dtype* img_info, const Dtype* bbox, const Dtype* score, Blob<Dtype>& output) {
	vector<vector<int>> result(img_num);
	int total_num=0;
	for(size_t i=0; i<result.size(); i++) {
		result[i]=NmsGPU(num_per_img, img_info[3*i+2], bbox+4*i*num_per_img, score+2*i*num_per_img);
		total_num+=result[i].size();
	}
	if(total_num==0) {
		// workaround for no valid bbox
		output.Reshape({1, 5});
		output.mutable_cpu_data()[0]=0;
		output.mutable_cpu_data()[1]=0;
		output.mutable_cpu_data()[2]=0;
		output.mutable_cpu_data()[3]=1;
		output.mutable_cpu_data()[4]=1;
	}
	else {
		Blob<int> total_result_batch_index({total_num*2});
		int* total_result_batch_index_data=total_result_batch_index.mutable_cpu_data();
		for(size_t i=0; i<result.size(); i++) {
			for(size_t j=0; j<result[i].size(); j++) {
				total_result_batch_index_data[0]=result[i][j]+i*num_per_img;
				total_result_batch_index_data[1]=i;
				total_result_batch_index_data+=2;
			}
		}

		output.Reshape({total_num, 5});
		GenerateOutputGPU<Dtype><<<CAFFE_GET_BLOCKS(total_num), CAFFE_CUDA_NUM_THREADS>>>(total_num, bbox,
				total_result_batch_index.gpu_data(), output.mutable_gpu_data());
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	int bbox_num=bottom[0]->count()/2;
	CalcBBoxesGPU<Dtype><<<CAFFE_GET_BLOCKS(bbox_num), CAFFE_CUDA_NUM_THREADS>>>(bbox_num, bottom[0]->width(), bottom[0]->height(), anchor.num(), stride_h,
			stride_w, bottom[2]->gpu_data(), bottom[1]->gpu_data(), anchor.gpu_data(), bboxes.mutable_gpu_data());
	int bbox_num_per_img=bbox_num/bottom[0]->num();
	NmsGPU(bottom[0]->num(), bbox_num_per_img, bottom[2]->cpu_data(), bboxes.gpu_data(), bottom[0]->cpu_data(), *top[0]);
}

template <typename Dtype>
void ProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(ProposalLayer);

}  // namespace caffe
