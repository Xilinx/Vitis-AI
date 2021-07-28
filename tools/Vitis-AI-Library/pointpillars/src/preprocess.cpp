/*
 * Copyright 2019 Xilinx Inc.
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
#include <memory>
#include <thread>

#include <vitis/ai/profiling.hpp>
#include "second/protos/pipeline.pb.h"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#include "./helper.hpp"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "./preprocess.hpp"

#define THNUM 2
DEF_ENV_PARAM(XLNX_POINTPILLARS_PRE_MT, "2");

namespace vitis { namespace ai { 

extern std::vector<int> g_grid_size;
extern ::second::protos::TrainEvalPipelineConfig cfg;

PointPillarsPre::~PointPillarsPre() { }

PointPillarsPre::PointPillarsPre( 
    int8_t* in_addr1,  int in_scale1,  int in_width1,  int in_height1,  int in_channel1,
    int8_t* out_addr1, float out_scale1, int out_width1, int out_height1, int out_channel1,
    int8_t* in_addr2,  int in_scale2,  int in_width2,  int in_height2,  int in_channel2 )
  : in_addr1_(in_addr1), in_scale1_(in_scale1), in_height1_(in_height1),
    out_addr1_(out_addr1), out_scale1_(out_scale1),
    in_addr2_(in_addr2), in_scale2_(in_scale2), in_width2_(in_width2), in_height2_(in_height2), in_channel2_(in_channel2)
{
/*
    128      100 12000 4
    0.0625   1   12000 64
    16      432  496   64    
    std::cout << "  in1 out1 in2 : addr, scale, width height channel \n"
    << " " << in_addr1  << " " <<  in_scale1  << " " <<  in_width1  << " " <<  in_height1  << " " <<  in_channel1  << "\n"
    << " " << out_addr1 << " " <<  out_scale1 << " " <<  out_width1 << " " <<  out_height1 << " " <<  out_channel1 << "\n"
    << " " << in_addr2  << " " <<  in_scale2  << " " <<  in_width2  << " " <<  in_height2  << " " <<  in_channel2  << "\n";
*/
    pre_dict_ = std::make_shared<preout_dict>( in_addr1, in_height1, in_width1, in_channel1  );
    V1F point_cloud_range_;
    V1F pc_len;

    point_cloud_range_.assign(cfg.model().second().voxel_generator().point_cloud_range().begin(), cfg.model().second().voxel_generator().point_cloud_range().end());
    for(int i=0; i<3; i++) {
      pc_len.emplace_back(  point_cloud_range_[i+3] - point_cloud_range_[i]); 
    }
    cfg_voxel_size.assign(cfg.model().second().voxel_generator().voxel_size().begin(), cfg.model().second().voxel_generator().voxel_size().end());
    for ( int i = 0; i < 3; i++ ) {
       voxelmap_shape_[2-i] = round(( point_cloud_range_[3+i] - point_cloud_range_[i]) / cfg_voxel_size[i]);
    }
    coor_to_voxelidx.assign(voxelmap_shape_[1]*voxelmap_shape_[2], -1);
    cfg_max_number_of_points_per_voxel = (int)cfg.model().second().voxel_generator().max_number_of_points_per_voxel();
    cfg_max_number_of_voxels = (int)cfg.eval_input_reader().max_number_of_voxels();

    for(int i=0; i<3; i++) {
      scale_pcstartlen[i] =  in_scale1_*point_cloud_range_[i]/pc_len[i];
      scale_pclen[i] = in_scale1_/pc_len[i];
      point_range[i] =  point_cloud_range_[i]/cfg_voxel_size[i];
    }
    bDirect =  abs(in_scale2_*out_scale1_ -1) < 0.0001;

    process_net1_cleanmem();

    if(ENV_PARAM( XLNX_POINTPILLARS_PRE_MT) >= 1) {
       PRE_MT_NUM = ENV_PARAM( XLNX_POINTPILLARS_PRE_MT );
       if (PRE_MT_NUM >8) PRE_MT_NUM = 8;
    }

    vth0.reserve(PRE_MT_NUM);
}

inline bool PointPillarsPre::judge_op_same(int canvas_index, int idx)
{
  for(int i=0; i<PRE_MT_NUM; i++) {
    if ( i != idx && canvas_index_arr[i] == canvas_index ) {
       return true;
    }
  }
  return false;
}

void PointPillarsPre::process_net0( const float* points, int len_f )
{
   int start = 0, len = 0 , size = len_f/4;
   int voxel_num = 0;
   pre_dict_->clear();

   if(PRE_MT_NUM==1) {
      process_net0_thread(points, 0, 0, size, voxel_num);
   }
   else {
      for(int i=0; i<PRE_MT_NUM; i++) {
         start = i * size/PRE_MT_NUM;
         len = (i != PRE_MT_NUM-1) ? size/PRE_MT_NUM : (size- (size/PRE_MT_NUM*(PRE_MT_NUM-1))) ;
         vth0.emplace_back( std::thread( &PointPillarsPre::process_net0_thread, this,  points, i, start, len, std::ref(voxel_num) ) );
      }
      for(int i=0; i<PRE_MT_NUM; i++) {
         vth0[i].join();
      }
   }

   coor_to_voxelidx.assign(voxelmap_shape_[1]*voxelmap_shape_[2] ,-1);   // 185us
   pre_dict_->SetSize(voxel_num);
   vth0.clear();
}

void PointPillarsPre::process_net0_thread(const float* points, int idx, int start, int len, int& voxel_num)
{
    __TIC__(POINT_TO_VOXELX)
    std::array<int32_t, 3> coor;
    bool failed = false;
    int c = 0, i = 0, j = 0, num = 0;
    int32_t voxelidx = 0;

    for (i = start; i <  start+len; i++)  {
        failed = false;
        for (j = 0; j < 3 ; j++) {
            c = floor( (points[i*4+j]) / cfg_voxel_size[j] - point_range[j] );
            if ( c < 0 || c >= g_grid_size[j]) {
                failed = true;
                break;
            }
            coor[2 - j] = c;
        }
        if (failed){
            continue;
        }

        auto canvas_index = coor[1]*voxelmap_shape_[2] + coor[2];

        if(PRE_MT_NUM != 1) {
           while ( judge_op_same(canvas_index, idx ) ) {
             std::this_thread::sleep_for(std::chrono::microseconds(10));
           }
           canvas_index_arr[idx] = canvas_index;
   
           if ( (voxelidx = coor_to_voxelidx[canvas_index]) == -1 )  {
             mtx.lock();
             if (voxel_num == cfg_max_number_of_voxels) {
                mtx.unlock();
                canvas_index_arr[idx] = -1;
                continue;
             }
             coor_to_voxelidx [canvas_index] = voxel_num;
             pre_dict_->coorData[ voxel_num ] = std::make_pair( coor[1], coor[2]);
             voxelidx = voxel_num;
             voxel_num ++;
             mtx.unlock();
           }
        } else {
           canvas_index_arr[idx] = canvas_index;
           if ( (voxelidx = coor_to_voxelidx[canvas_index]) == -1 )  {
             if (voxel_num == cfg_max_number_of_voxels) {
                canvas_index_arr[idx] = -1;
                continue;
                // break;
             }
             coor_to_voxelidx [canvas_index] = voxel_num;
             pre_dict_->coorData[ voxel_num ] = std::make_pair( coor[1], coor[2]);
             voxelidx = voxel_num;
             voxel_num ++;
           }
        } 
   
        num = pre_dict_->GetNumPoints()[voxelidx];
        if (  num  < cfg_max_number_of_points_per_voxel ) {
            pre_dict_->GetVoxels()( voxelidx, num, 0) = int8_t(round( points[ i*4+0] *scale_pclen[0] - scale_pcstartlen[0]));
            pre_dict_->GetVoxels()( voxelidx, num, 1) = int8_t(round( points[ i*4+1] *scale_pclen[1] - scale_pcstartlen[1]));
            pre_dict_->GetVoxels()( voxelidx, num, 2) = int8_t(round( points[ i*4+2] *scale_pclen[2] - scale_pcstartlen[2]));
            pre_dict_->GetVoxels()( voxelidx, num, 3) = int8_t(round( points[ i*4+3] *in_scale1_));

            pre_dict_->GetNumPoints()[voxelidx] += 1;
        }
    
        canvas_index_arr[idx] = -1;

    }
    __TOC__(POINT_TO_VOXELX)

}

void PointPillarsPre::process_net1_cleanmem()
{
  memset(in_addr2_, 0, in_height2_ * in_width2_ * in_channel2_ );  
}

void PointPillarsPre::process_net1_thread( int start, int len)
{
  PointPillarsScatterOutDpuTensorMap canvas(in_addr2_, in_height2_,  in_width2_, in_channel2_);  // dest

  if (bDirect) {
     for (auto iCoor = start; iCoor < start+len; iCoor++) {
         auto& vv=pre_dict_->coorData[iCoor];
         memcpy( in_addr2_ + (int32_t)   vv.first  * in_width2_*in_channel2_ + (int32_t) vv.second * in_channel2_  ,
                  out_addr1_ + iCoor*in_channel2_ ,
                  in_channel2_
                );
     }
  } else {
     for (auto iCoor = start; iCoor < start+len; iCoor++) {
          auto& vv=pre_dict_->coorData[iCoor];
          std::transform(  &out_addr1_[iCoor*in_channel2_+0],
                           &out_addr1_[iCoor*in_channel2_+in_channel2_],
                           &canvas(  (int32_t)vv.first, (int32_t)vv.second , 0 ) ,
                           [&](int8_t x){ return in_scale2_ * out_scale1_ * x; }
                        );
     }
  }
}

void PointPillarsPre::process_net1()
{
   std::vector<std::thread> vth;
   int start=0, len=0, size =  pre_dict_->GetSize();

   for(int i=0; i<THNUM; i++) {
      start = i * size/THNUM;
      len = (i != THNUM-1) ? size/THNUM : (size- (size/THNUM*(THNUM-1))) ;
      vth.emplace_back( std::thread( &PointPillarsPre::process_net1_thread, this, start, len) );
   }
   for(int i=0; i<THNUM; i++) {
      vth[i].join();
   }
}

}}

