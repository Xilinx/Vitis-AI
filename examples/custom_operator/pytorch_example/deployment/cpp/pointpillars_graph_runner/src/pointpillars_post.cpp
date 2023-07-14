/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <queue>
#include <cmath>
#include <chrono>
#include <thread>
#include <numeric>

#include <opencv2/highgui.hpp>

#include <vitis/ai/profiling.hpp>

#include "./anchor.hpp"
#include "./helper.hpp"
#include "./pointpillars.hpp"
#include "./pointpillars_post.hpp"

using namespace std;
using namespace chrono;

DEF_ENV_PARAM(ANCHORMASK_MT, "2");
DEF_ENV_PARAM(ANCHORMASK_MAXSIZE, "60000");

namespace vitis { namespace ai { namespace pp { 

extern  std::vector<int> g_grid_size;
extern  G_ANCHOR g_anchor;

std::vector<int> topk(const V1F& scores, int k, V2F& bboxes_in, V2F& bboxes_out);
std::vector<int> non_max_suppression_cpu(
  V2F& bboxes_,
  const V1F& scores_,
  int pre_max_size_,
  int post_max_size_,
  float iou_threshold );
V3F corners_nd_2d(const V2F& dims);
V2F corner_to_standup_nd(const V3F& boxes_corner);
V2F center_to_corner_box2d_to_standup_nd(const V2F& );
V3F rotation_2d(const V3F& points, const V2F& angles);
V3F einsum(const V3F& points, const V3F& rot_mat_T);

int cumsum(V2F& v);

int get_max(float* in, int num){
   auto pos = std::max_element(in, in+num);
   return pos-in;
}

void PointPillarsPost::get_anchors_mask( const std::vector<std::shared_ptr<preout_dict>>& v_pre_dict_)
{
  for(int i=0; i<realbatchnum; i++) {
      __TIC__(inner_anchors_mask)
      __TIC__(dense_voxel_map)
      // sparse_sum_for_anchors_mask(  v_pre_dict_[i]->GetCoors(), v_pre_dict_[i]->GetSize() , dense_voxel_map );
      for(auto &it: dense_voxel_map) {
        it.assign(it.size() ,0);
      }
      for(int j=0; j< v_pre_dict_[i]->GetSize(); j++){
        // v2f[ coors[i].first  ][ coors[i].second ]+=1;  
        dense_voxel_map[ v_pre_dict_[i]->GetCoors()(j,0,2 ) ][  v_pre_dict_[i]->GetCoors()(j,0,3 ) ]+=1;  
      }
      __TOC__(dense_voxel_map)
      __TIC__(cumsum)    // 4ms
      cumsum(dense_voxel_map);
      __TOC__(cumsum)
    
      __TIC__(fused_get_anchors_area)
      fused_get_anchors_area(i );
      __TOC__(fused_get_anchors_area)
      __TOC__(inner_anchors_mask)
   }
}

std::vector<int> topk(const V1F& scores, int k, V2F& bboxes_in, V2F& bboxes_out)
{
  std::vector<int> vout(k);

  struct cmp1 {
    bool operator ()(const std::pair<int, float>& a, const std::pair<int, float>& b ) {
      return std::get<1>(a) >  std::get<1>(b);
    }
  };
  priority_queue<std::pair<int,float>, vector<std::pair<int, float>>,cmp1> minHeap;

  for(unsigned int i=0; i<scores.size(); i++) {
    if (i< (unsigned int)k) {
      minHeap.push(std::make_pair(i, scores[i]) );
      continue;
    }
    if (scores[i]<= std::get<1>(minHeap.top())) {
        continue;
    }
    if (scores[i]> std::get<1>(minHeap.top())) {
        minHeap.pop();
        minHeap.push(std::make_pair(i, scores[i]) );
    }
  }
  int pos = k-1;
  while(!minHeap.empty()){
    vout[pos] = std::get<0>(minHeap.top());
    bboxes_out[pos].swap(bboxes_in[ vout[pos] ]);
    minHeap.pop();
    pos--;
  }
  return vout;
}

std::vector<int> non_max_suppression_cpu(V2F& boxes_in,
           const V1F& scores_,
           int pre_max_size_,
           int post_max_size_,
           float iou_threshold)
{
  float eps=1.0;
  int bsize = std::min((int)scores_.size(), pre_max_size_);

  V2F boxes( bsize);
  V1I indices = topk(scores_, bsize, boxes_in,  boxes);

  std::vector<int> keep;
  keep.reserve(post_max_size_);
  std::vector<bool> suppressed_rw(bsize, false);
  std::vector<float> area_rw( bsize, 0.0);

  int i_idx, j_idx;
  float  xx1, xx2, w, h, inter, ovr;
  for(int i = 0; i < bsize; ++i){
    area_rw[i] = (boxes[i][2] - boxes[i][0] + eps) * (boxes[i][3] - boxes[i][1] + eps);
  }
  for(int i=0; i< bsize; i++) {
    i_idx = i;
    if( suppressed_rw[i_idx] == true) {
       continue;
    }
    if ((int)keep.size() < post_max_size_) {
        keep.emplace_back(indices[ i_idx] );
    } else {
        return keep;
    }
    for(int j=i+1; j<bsize; j++) {
      j_idx = j;
      if( suppressed_rw[j_idx] == true) {
          continue;
      }
      xx2 = std::min(boxes[i_idx][2], boxes[j_idx][2]);
      xx1 = std::max(boxes[i_idx][0], boxes[j_idx][0]);
      w = xx2 - xx1 + eps;
      if (w > 0){
        xx2 = std::min(boxes[i_idx][3], boxes[j_idx][3]);
        xx1 = std::max(boxes[i_idx][1], boxes[j_idx][1]);
        h = xx2 - xx1 + eps;
        if (h > 0){
          inter = w * h;
          ovr = inter / (area_rw[i_idx] + area_rw[j_idx] - inter);
          if(ovr >= iou_threshold) {
            suppressed_rw[j_idx] = true;
          }
        } // end of if(h>0)
      } // end of if(w>0)       
    }   // end of for(j
  }// end of for(i
  return keep;
}

// dims has 2 elements
V2I PointPillarsPost::unravel_index_2d(const V1I& index, const V1I& dims )
{
  V2I ret(index.size(), V1I(2,0));
  for(unsigned int i=0; i<index.size(); i++ ) {
    int x = index[i]/dims[1];
    int y = index[i] - x*dims[1];
    ret[i][0]=x;
    ret[i][1]=y;
  }
  ret[2].swap(ret[3]);
  return ret;
}

V3F PointPillarsPost::corners_nd_2d(const V2F& dims)
{
  auto dim_size_0 = dims.size();
  auto dim_size_1 = corners_norm.size();
  auto dim_size_2 = corners_norm[0].size();

  V3F corners( dim_size_0,
              V2F( dim_size_1,
                V1F(dim_size_2, 0 ) ) ) ;

  for(unsigned int i=0; i<dim_size_0; i++) {
    for(unsigned int j=0; j<dim_size_1; j++) {
      for(unsigned int k=0; k<dim_size_2; k++) {
        corners[i][j][k] = dims[i][k+3] * ( corners_norm[j][k]-0.5);
      }
    }
  }
  return corners;
}

V2F corner_to_standup_nd(const V3F& boxes_corner)
{
  if (boxes_corner.empty()) {
     return V2F{};
  }
  V2F ret(boxes_corner.size(), V1F(boxes_corner[0][0].size()*2, 0));
  auto cmp0 = [](const V1F&in1, const V1F&in2)->bool { return in1[0] < in2[0]; };
  auto cmp1 = [](const V1F&in1, const V1F&in2)->bool { return in1[1] < in2[1]; };

  for(unsigned int i=0; i<boxes_corner.size(); i++) {
    ret[i][0] = (*std::min_element(boxes_corner[i].begin(), boxes_corner[i].end(), cmp0))[0];
    ret[i][1] = (*std::min_element(boxes_corner[i].begin(), boxes_corner[i].end(), cmp1))[1];
    ret[i][2] = (*std::max_element(boxes_corner[i].begin(), boxes_corner[i].end(), cmp0))[0];
    ret[i][3] = (*std::max_element(boxes_corner[i].begin(), boxes_corner[i].end(), cmp1))[1];
  }
  return ret;
}

V2F PointPillarsPost::center_to_corner_box2d_to_standup_nd(const V2F& box)
{
  __TIC__(center_to_corner_box2d_1)
  V3F boxes_corner = rotation_2d(corners_nd_2d(box), box);
  if (boxes_corner.empty()) {
     return V2F{};
  }
  __TOC__(center_to_corner_box2d_1)
  __TIC__(center_to_corner_box2d_2)
  for(unsigned int i=0; i<boxes_corner.size(); i++) {
    for(unsigned int j=0; j<boxes_corner[0].size(); j++) {
      for(unsigned int k=0; k<boxes_corner[0][0].size(); k++) {
        boxes_corner[i][j][k]+= box[i][k];
      }
    }
  }
  __TOC__(center_to_corner_box2d_2)
  return corner_to_standup_nd(boxes_corner);
}

V3F rotation_2d(const V3F& points, const V2F& angles)
{
  V1F rot_sin(angles.size());
  V1F rot_cos(angles.size());
  for(unsigned int i=0; i<angles.size(); i++) {
    rot_sin[i] = sin(angles[i][6]);
    rot_cos[i] = cos(angles[i][6]);
  }
  V3F rot_mat_T(2, V2F(2, V1F(angles.size(), 0)));
  for(unsigned int i=0; i<angles.size(); i++) {
    rot_mat_T[0][0][i] = rot_cos[i];
    rot_mat_T[0][1][i] = -rot_sin[i];
    rot_mat_T[1][0][i] = rot_sin[i];
    rot_mat_T[1][1][i] = rot_cos[i];
  }
  return einsum(points, rot_mat_T);
}

V3F einsum(const V3F& points, const V3F& rot_mat_T)
{
  V3F ret(points.size(), V2F( points[0].size(), V1F(rot_mat_T[0].size(), 0 ) ));
  for(unsigned int a=0; a < points.size(); a++) {
    for(unsigned int i=0; i < points[0].size(); i++) {
      for(unsigned int k=0; k < rot_mat_T[0].size(); k++) {
        for(unsigned int j=0; j < points[0][0].size(); j++) {
           ret[a][i][k] += points[a][i][j]*rot_mat_T[j][k][a];
        }
      }
    }
  }
  return ret;
}


V1F PointPillarsPost::get_decode_box(int batchidx, int idx) {
  V1F o(7, 0);
  float za = g_anchor.anchors[ idx ][2] + g_anchor.anchors[  idx ][5]/2;
  float diagonal = sqrt( pow(g_anchor.anchors[  idx  ][4], 2.0) + pow(g_anchor.anchors[ idx  ][3], 2.0));
  o[0] = dpu_data.box_[batchidx][ idx *7+0] * diagonal + g_anchor.anchors[ idx  ][0];
  o[1] = dpu_data.box_[batchidx][ idx *7+1] * diagonal + g_anchor.anchors[ idx  ][1];
  o[2] = dpu_data.box_[batchidx][ idx *7+2] * g_anchor.anchors[  idx  ][5] + za;
  o[3] = exp(  dpu_data.box_[batchidx][ idx *7+3]) * g_anchor.anchors[ idx ][3];
  o[4] = exp(  dpu_data.box_[batchidx][ idx *7+4]) * g_anchor.anchors[ idx ][4];
  o[5] = exp(  dpu_data.box_[batchidx][ idx *7+5]) * g_anchor.anchors[ idx ][5];
  o[6] = dpu_data.box_[batchidx][ idx *7+6] + g_anchor.anchors[ idx ][6];
  o[2] = o[2] - o[5]/2;
  return o;
}

int cumsum(V2F& v)
{
    for(unsigned int i=1; i<v.size(); i++) {
      for(unsigned int j=0; j<v[0].size(); j++) {
        v[i][j]+=v[i-1][j];
      }
    }
    for(unsigned int i=0; i<v.size(); i++) {
      for(unsigned int j=1; j<v[0].size(); j++) {
        v[i][j]+=v[i][j-1];
      }
    }
    return 0;
}

void PointPillarsPost::fused_get_anchors_area_thread(int start, int len, V1I& anchors_maskx)
{
  for(int i=start; i<start+len; i++) {
    if (   dense_voxel_map[g_anchor.anchors_bv[i][3]][ g_anchor.anchors_bv[i][2]]
         - dense_voxel_map[g_anchor.anchors_bv[i][3]][ g_anchor.anchors_bv[i][0]]
         - dense_voxel_map[g_anchor.anchors_bv[i][1]][ g_anchor.anchors_bv[i][2]]
         + dense_voxel_map[g_anchor.anchors_bv[i][1]][ g_anchor.anchors_bv[i][0]]
         > 1) {
       anchors_maskx.emplace_back(i);
    }
  }
}

void PointPillarsPost::fused_get_anchors_area(int batchidx)
{
    std::vector<std::thread> vth;
    int start=0, len=0, size = g_anchor.anchors_bv.size();
 
    __TIC__(inner_fused_anchor_thread_total)
    for(int i=0; i<anchors_mask_thread_num; i++) {
       start = i * size/anchors_mask_thread_num;
       len = (i != anchors_mask_thread_num-1) ? size/anchors_mask_thread_num : (size- (size/anchors_mask_thread_num*(anchors_mask_thread_num-1))) ;
       anchors_maskx[i].clear();
       vth.emplace_back( std::thread( &PointPillarsPost::fused_get_anchors_area_thread, this, start, len, std::ref( anchors_maskx[i]) ));
    }
    for(int i=0; i<anchors_mask_thread_num; i++) {
       vth[i].join();
    }
    __TOC__(inner_fused_anchor_thread_total)

    __TIC__(fused_combine)
    int total_size=0;
    for(int i=0; i<anchors_mask_thread_num; i++) {
       total_size += anchors_maskx[i].size();
    } 
    anchors_mask[batchidx].resize(total_size);
    int*p_dst = anchors_mask[batchidx].data();
    for(int i=0; i<anchors_mask_thread_num; i++) { 
       memcpy(p_dst, anchors_maskx[i].data(), sizeof(int)*anchors_maskx[i].size());
       p_dst += anchors_maskx[i].size() ;
    }
    __TOC__(fused_combine)
}

std::unique_ptr<PointPillarsPost> PointPillarsPost::create(
    const std::vector<vart::TensorBuffer*>& output_tensors,
    int batchnum,
    int& realbatchnum
) {
  return std::unique_ptr<PointPillarsPost>(
      new PointPillarsPost( output_tensors, batchnum, realbatchnum ));
}

PointPillarsPost::~PointPillarsPost(){}

PointPillarsPost::PointPillarsPost(
    const std::vector<vart::TensorBuffer*>& output_tensors,
    int batchnumin,
    int& realbatchnumin ) :
      output_tensors_(output_tensors),
      batchnum(batchnumin),
      realbatchnum(realbatchnumin),
      anchors_mask(batchnum)
{
  corners_norm = unravel_index_2d(V1I({0,1,2,3}), V1I({2,2}));

  V2F tmpv2f( g_grid_size[1], V1F( g_grid_size[0], 0));
  dense_voxel_map.swap(tmpv2f);
  get_dpu_data();

  nms_confidence_ = -log(1.0/cfg_nms_score_threshold -1.0 );

  anchors_mask_thread_num = 1;

  if(ENV_PARAM(ANCHORMASK_MT) >= 1) {
    anchors_mask_thread_num = ENV_PARAM(ANCHORMASK_MT);
    anchors_maskx.resize( anchors_mask_thread_num );
    for(int i=0; i<anchors_mask_thread_num; i++) {
      anchors_maskx[i].reserve(  ENV_PARAM(ANCHORMASK_MAXSIZE)/2 );
    }
  }
  for(int i=0; i<batchnum; i++) {
     anchors_mask[i].initialize( ENV_PARAM(ANCHORMASK_MAXSIZE) );
  }
}

std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto ret = tensor->get_shape();
  std::fill(ret.begin(), ret.end(), 0);
  return ret;
}

void PointPillarsPost::get_dpu_data()
{
  int BOX_IDX=2, CLS_IDX=1, DIR_IDX=0;
  uint64_t data_out = 0u;
  size_t size_out = 0u;
  for(int i=0;i<3;i++){
    //   VoxelNet__VoxelNet_RPN_rpn__Conv2d_conv_dir_cls__514_fix_
    //   VoxelNet__VoxelNet_RPN_rpn__Conv2d_conv_cls__496_fix_
    //   VoxelNet__VoxelNet_RPN_rpn__Conv2d_conv_box__486_fix_
    std::string namei = output_tensors_[i]->get_tensor()->get_name();
    if (namei.find("dir") != std::string::npos){
       DIR_IDX = i;
       continue;
    }
    if (namei.find("box") != std::string::npos){
       BOX_IDX = i;
       continue;
    }
    if (namei.find("cls") != std::string::npos && namei.find("dir_cls") == std::string::npos ){
       CLS_IDX = i;
       continue;
    }
  }
 
  dpu_data.box_.resize(batchnum);
  dpu_data.cls_.resize(batchnum);
  dpu_data.dir_.resize(batchnum);

  std::vector<std::int32_t> idx;
  for(int i=0; i<(int)batchnum; i++) {  
    idx = get_index_zeros(output_tensors_[DIR_IDX]->get_tensor());
    idx[0] = i; 
    std::tie(data_out, size_out) = output_tensors_[DIR_IDX]->data(idx);  
    dpu_data.dir_[i] = (float*)data_out;

    idx = get_index_zeros(output_tensors_[CLS_IDX]->get_tensor());
    idx[0] = i; 
    std::tie(data_out, size_out) = output_tensors_[CLS_IDX]->data(idx);
    dpu_data.cls_[i] = (float*)data_out;

    idx = get_index_zeros(output_tensors_[BOX_IDX]->get_tensor());
    idx[0] = i; 
    std::tie(data_out, size_out) = output_tensors_[BOX_IDX]->data(idx);
    dpu_data.box_[i] = (float*)data_out;
  }
}

std::vector<PointPillarsResult> PointPillarsPost::post_process( )
{
   std::vector<PointPillarsResult> ret(realbatchnum);
   for(int i=0; i<realbatchnum; i++) {
      ret[i] = post_process(i);
   }
   return ret;
}

PointPillarsResult PointPillarsPost::post_process( int batchidx )
{
  __TIC__(do_predict_inner)

  __TIC__(select_greater_than_confidence)
  top_scores.clear();
  box_preds.clear();
  dir_labels.clear();
  top_labels.clear();

  int pos = 0;

  for(int i=0; i<anchors_mask[batchidx].size(); i++){
    pos = cfg_num_class == 1 ? 0 : get_max(  &dpu_data.cls_[batchidx][  anchors_mask[batchidx][i]*cfg_num_class ], cfg_num_class);
    if( dpu_data.cls_[batchidx][   anchors_mask[batchidx][i]*cfg_num_class +pos ] >= nms_confidence_) {
      top_scores.emplace_back( dpu_data.cls_[batchidx][   anchors_mask[batchidx][i]*cfg_num_class +pos  ] );
      box_preds.emplace_back( get_decode_box(batchidx,  anchors_mask[batchidx][i]) );
      dir_labels.emplace_back( 
        dpu_data.dir_[batchidx][anchors_mask[batchidx][i]*2+0 ] >= dpu_data.dir_[batchidx][anchors_mask[batchidx][i]*2+1 ]  ? 0 : 1
      );
      top_labels.emplace_back( pos ); 
    }
  }
  if (top_scores.empty() ) {
     return PointPillarsResult{};
  }
  __TOC__(select_greater_than_confidence)

  __TIC__(center_to_corner_box2d)
  V2F boxes_for_nms = center_to_corner_box2d_to_standup_nd(box_preds);
  __TOC__(center_to_corner_box2d)
 
  __TIC__(nms)
  V1I selected = non_max_suppression_cpu(
     boxes_for_nms,
     top_scores,
     cfg_nms_pre_max_size,
     cfg_nms_post_max_size,
     cfg_nms_iou_threshold );
  __TOC__(nms)

  __TIC__(selected_size)
  V2F selected_boxes(selected.size());
  V1I selected_dir_labels(selected.size());
  V1I selected_labels(selected.size());

  PointPillarsResult res;
  res.ppresult.final_scores.reserve( selected.size() );

  for(unsigned int i=0; i<selected.size(); i++) {
    selected_boxes[i].swap( box_preds[selected[i]]);
    selected_dir_labels[i] = dir_labels[selected[i]];
    selected_labels[i] = top_labels[selected[i]];
    res.ppresult.final_scores.emplace_back(  1.0/(1.0+exp(-1.0* top_scores[selected[i]]))   );
    // std::cout <<"final_score:  " << i << "  " <<  res.ppresult.final_scores[i] << "\n";
  }
  __TOC__(selected_size)
  box_preds.swap( selected_boxes );
  res.ppresult.label_preds.swap(selected_labels);
  dir_labels.swap(selected_dir_labels);

  auto size_0 = box_preds[0].size();
  for(unsigned int i=0; i<box_preds.size(); i++){
    if ( bool(box_preds[i][ size_0 -1 ] >0) ^ bool(dir_labels[i]) ) {
       box_preds[i][ size_0 -1 ] += 3.14159265 ;
    }
  }

  res.ppresult.final_box_preds.swap( box_preds);
  __TOC__(do_predict_inner)

  return res;
}

}}}

