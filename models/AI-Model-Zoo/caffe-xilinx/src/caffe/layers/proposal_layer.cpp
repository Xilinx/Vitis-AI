#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/layers/proposal_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace std;

namespace caffe {

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top) {
	const ProposalParameter& param = this->layer_param_.proposal_param();

	CHECK_GT(param.stride_h(), 0);
	CHECK_GT(param.stride_w(), 0);
	CHECK_GE(param.nms_iou_th(), 0.0f);
	CHECK_LE(param.nms_iou_th(), 1.0f);
	CHECK_GT(param.anchor_size(), 0);

	stride_h=param.stride_h();
	stride_w=param.stride_w();
	nms_iou_th=param.nms_iou_th();
	nms_input_num=param.nms_input_num();
	nms_output_num=param.nms_output_num();
	min_width=param.min_width();
	min_height=param.min_height();
	anchor.Reshape({param.anchor_size(), 4});
	for(size_t i=0; i<param.anchor_size(); i++) {
		anchor.mutable_cpu_data()[i*4+0]=param.anchor(i).left();
		anchor.mutable_cpu_data()[i*4+1]=param.anchor(i).top();
		anchor.mutable_cpu_data()[i*4+2]=param.anchor(i).right();
		anchor.mutable_cpu_data()[i*4+3]=param.anchor(i).bottom();
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->shape().size(), 4);
	CHECK_EQ(bottom[1]->shape().size(), 4);
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	CHECK_EQ(bottom[0]->channels(), anchor.num()*2);
	CHECK_EQ(bottom[1]->channels(), anchor.num()*4);
	CHECK_EQ(bottom[0]->height(), bottom[1]->height());
	CHECK_EQ(bottom[0]->width(), bottom[1]->width());
	bboxes.ReshapeLike(*bottom[1]);
	// top will be reshaped again in forward
	top[0]->Reshape({1, 5});
}

template <typename Dtype>
void ProposalLayer<Dtype>::CalcBBoxes(int num, int width, int height, int anchor_num, Dtype stride_h,
		Dtype stride_w, const Dtype* img_info, const Dtype* reg, const Dtype* anchor, Dtype* bbox) {
	for(int i=0; i<num; i++) {
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

template <typename Dtype>
static inline Dtype GetIoU(const Dtype* rect1, const Dtype* rect2) {
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
vector<Dtype> ProposalLayer<Dtype>::Nms(int num, Dtype img_scale, const Dtype* bbox, const Dtype* score) {
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

	vector<Dtype> bbox_sorted(score_index.size()*4);
	for(size_t i=0; i<score_index.size(); i++) {
		bbox_sorted[i*4+0]=bbox[score_index[i].second*4+0];
		bbox_sorted[i*4+1]=bbox[score_index[i].second*4+1];
		bbox_sorted[i*4+2]=bbox[score_index[i].second*4+2];
		bbox_sorted[i*4+3]=bbox[score_index[i].second*4+3];
	}

	Dtype min_width_this=min_width*img_scale;
	Dtype min_height_this=min_height*img_scale;
	vector<bool> exist(score_index.size(), true);
	for(size_t i=0; i<exist.size(); i++) {
		exist[i]=(bbox_sorted[i*4+2]-bbox_sorted[i*4+0]>min_width_this)
				&& (bbox_sorted[i*4+3]-bbox_sorted[i*4+1]>min_height_this);
	}

	const Dtype* bbox_sorted_data=bbox_sorted.data();
	for(size_t i=0; i<exist.size(); i++) {
		if(exist[i]) {
			for(size_t j=i+1; j<exist.size(); j++) {
				if(exist[j]) {
					exist[j]=GetIoU(bbox_sorted_data+4*i, bbox_sorted_data+4*j)<nms_iou_th;
				}
			}
		}
	}

	vector<Dtype> result;
	result.reserve(4*exist.size());
	for(size_t i=0; i<exist.size(); i++) {
		if(exist[i]) {
			result.push_back(bbox_sorted[4*i+0]);
			result.push_back(bbox_sorted[4*i+1]);
			result.push_back(bbox_sorted[4*i+2]);
			result.push_back(bbox_sorted[4*i+3]);
		}
	}
	if(result.size()>4*nms_output_num) {
		result.resize(4*nms_output_num);
	}
	return result;
}

template <typename Dtype>
vector<Dtype> ProposalLayer<Dtype>::Nms(int img_num, int num_per_img, const Dtype* img_info, const Dtype* bbox, const Dtype* score) {
	vector<vector<Dtype>> result(img_num);
	size_t total_num=0;
	for(size_t i=0; i<result.size(); i++) {
		result[i]=Nms(num_per_img, img_info[3*i+2], bbox+4*i*num_per_img, score+2*i*num_per_img);
		total_num+=result[i].size()/4;
	}
	vector<Dtype> roi_result;
	if(total_num==0) {
		// workaround for no valid bbox
		roi_result={0, 0, 0, 1, 1};
	}
	else {
		roi_result.reserve(5*total_num);
		for(size_t i=0; i<result.size(); i++) {
			for(size_t j=0; j<result[i].size(); j+=4) {
				roi_result.push_back(i);
				roi_result.push_back(result[i][j+0]);
				roi_result.push_back(result[i][j+1]);
				roi_result.push_back(result[i][j+2]);
				roi_result.push_back(result[i][j+3]);
			}
		}
	}
	return roi_result;
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	int bbox_num=bottom[0]->count()/2;
	CalcBBoxes(bbox_num, bottom[0]->width(), bottom[0]->height(), anchor.num(), stride_h,
			stride_w, bottom[2]->cpu_data(), bottom[1]->cpu_data(), anchor.cpu_data(), bboxes.mutable_cpu_data());
	int bbox_num_per_img=bbox_num/bottom[0]->num();
	vector<Dtype> result=Nms(bottom[0]->num(), bbox_num_per_img, bottom[2]->cpu_data(), bboxes.cpu_data(), bottom[0]->cpu_data());
	top[0]->Reshape({static_cast<int>(result.size())/5, 5});
	caffe_copy(result.size(), result.data(), top[0]->mutable_cpu_data());
}

#ifdef CPU_ONLY
STUB_GPU(ProposalLayer);
#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}  // namespace caffe
