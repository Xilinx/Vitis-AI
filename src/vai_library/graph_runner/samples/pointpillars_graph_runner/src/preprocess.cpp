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
#include <memory>
#include <thread>

#include <vitis/ai/profiling.hpp>
#include "./helper.hpp"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "./preprocess.hpp"

DEF_ENV_PARAM(XLNX_POINTPILLARS_PRE_MT, "1");
DEF_ENV_PARAM(XLNX_POINTPILLARS_MIDDLE_MT, "2");

namespace vitis { namespace ai {  namespace pp {

int XLNX_POINTPILLARS_MIDDLE_MT = 2;
extern std::vector<int> g_grid_size;

PointPillarsPre::~PointPillarsPre() { }

PointPillarsPre::PointPillarsPre( 
    std::vector<int8_t*>& in_addr0,  int in_scale0,  int in_width0,  int in_height0, int in_channel0,  
    std::vector<float*>& in_addr1,   int in_width1,  int in_height1, 
    int batchnumin, int& realbatchnumin )
  : in_addr0_(in_addr0), in_scale0_(in_scale0), 
    in_addr1_(in_addr1), in_width1_(in_width1), in_height1_(in_height1),
    batchnum(batchnumin), realbatchnum(realbatchnumin)
{
    for(int i=0; i<batchnum; i++) {
       pre_dict_.emplace_back(std::make_shared<preout_dict>( 
            in_addr0[i], in_height0, in_width0, in_channel0,
            in_addr1[i], in_height1, in_width1, 1));
    }
    V1F pc_len;

    for(int i=0; i<3; i++) {
      pc_len.emplace_back( cfg_point_cloud_range[i+3] - cfg_point_cloud_range[i]); 
    }
    for ( int i = 0; i < 3; i++ ) {
       voxelmap_shape_[2-i] = round(( cfg_point_cloud_range[3+i] - cfg_point_cloud_range[i]) / cfg_voxel_size[i]);
    }
    coor_to_voxelidx.assign(voxelmap_shape_[1]*voxelmap_shape_[2], -1);

    for(int i=0; i<3; i++) {
      scale_pcstartlen[i] =  in_scale0_*cfg_point_cloud_range[i]/pc_len[i];
      scale_pclen[i] = in_scale0_/pc_len[i];
      point_range[i] =  cfg_point_cloud_range[i]/cfg_voxel_size[i];
    }

    if(ENV_PARAM( XLNX_POINTPILLARS_PRE_MT) >= 1) {
       PRE_MT_NUM = ENV_PARAM( XLNX_POINTPILLARS_PRE_MT );
       if (PRE_MT_NUM >8) PRE_MT_NUM = 8;
    }
    if(ENV_PARAM( XLNX_POINTPILLARS_MIDDLE_MT) >= 1) {
       XLNX_POINTPILLARS_MIDDLE_MT = ENV_PARAM( XLNX_POINTPILLARS_MIDDLE_MT );
       if ( XLNX_POINTPILLARS_MIDDLE_MT>2)  XLNX_POINTPILLARS_MIDDLE_MT = 2;
    }

    vth0.reserve(PRE_MT_NUM);
}

inline bool PointPillarsPre::judge_op_same(int canvas_index, int threadidx)
{
  for(int i=0; i<PRE_MT_NUM; i++) {
    if ( i != threadidx && canvas_index_arr[i] == canvas_index ) {
       return true;
    }
  }
  return false;
}

void PointPillarsPre::process_net0( const float* points, int len_f , int batchidx)
{
   int start = 0, len = 0 , size = len_f/4;
   int voxel_num = 0;
   pre_dict_[batchidx]->clear();

   if(PRE_MT_NUM==1) {
      process_net0_thread(points, 0, 0, size, voxel_num, batchidx);
   }
   else {
      for(int i=0; i<PRE_MT_NUM; i++) {
         start = i * size/PRE_MT_NUM;
         len = (i != PRE_MT_NUM-1) ? size/PRE_MT_NUM : (size- (size/PRE_MT_NUM*(PRE_MT_NUM-1))) ;
         vth0.emplace_back( std::thread( &PointPillarsPre::process_net0_thread, this,  points, i, start, len, std::ref(voxel_num), batchidx));
      }
      for(int i=0; i<PRE_MT_NUM; i++) {
         vth0[i].join();
      }
   }

   coor_to_voxelidx.assign(voxelmap_shape_[1]*voxelmap_shape_[2] ,-1);   // 185us
   pre_dict_[batchidx]->SetSize(voxel_num);
   vth0.clear();
}

void PointPillarsPre::process_net0_thread(const float* points, int threadidx, int start, int len, int& voxel_num, int batchidx)
{
    // assume [12000 4] is [1 12000 4] // batch height width no_channel
    
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
           while ( judge_op_same(canvas_index, threadidx ) ) {
             std::this_thread::sleep_for(std::chrono::microseconds(10));
           }
           canvas_index_arr[threadidx] = canvas_index;
   
           if ( (voxelidx = coor_to_voxelidx[canvas_index]) == -1 )  {
             mtx.lock();
             if (voxel_num == cfg_max_number_of_voxels) {
                mtx.unlock();
                canvas_index_arr[threadidx] = -1;
                continue;
             }
             coor_to_voxelidx [canvas_index] = voxel_num;
             // pre_dict_[batchidx]->coorData[ voxel_num ] = std::make_pair( coor[1], coor[2]);
             // assume [12000 4] is [1 12000 4]
             pre_dict_[batchidx]->GetCoors()( voxel_num, 0, 2) = coor[1]; 
             pre_dict_[batchidx]->GetCoors()( voxel_num, 0, 3) = coor[2];
             
             voxelidx = voxel_num;
             voxel_num ++;
             mtx.unlock();
           }
        } else {
           if ( (voxelidx = coor_to_voxelidx[canvas_index]) == -1 )  {
             if (voxel_num == cfg_max_number_of_voxels) {
                continue;
                // break;
             }
             coor_to_voxelidx [canvas_index] = voxel_num;
             // pre_dict_[batchidx]->coorData[ voxel_num ] = std::make_pair( coor[1], coor[2]);
             // assume [12000 4] is [1 12000 4]
             pre_dict_[batchidx]->GetCoors()( voxel_num, 0, 2) = coor[1]; 
             pre_dict_[batchidx]->GetCoors()( voxel_num, 0, 3) = coor[2];
             voxelidx = voxel_num;
             voxel_num ++;
           }
        } 
   
        num = pre_dict_[batchidx]->GetNumPoints()[voxelidx];
        if (  num  < cfg_max_number_of_points_per_voxel ) {
            pre_dict_[batchidx]->GetVoxels()( voxelidx, num, 0) = int8_t(round( points[ i*4+0] *scale_pclen[0] - scale_pcstartlen[0]));
            pre_dict_[batchidx]->GetVoxels()( voxelidx, num, 1) = int8_t(round( points[ i*4+1] *scale_pclen[1] - scale_pcstartlen[1]));
            pre_dict_[batchidx]->GetVoxels()( voxelidx, num, 2) = int8_t(round( points[ i*4+2] *scale_pclen[2] - scale_pcstartlen[2]));
            pre_dict_[batchidx]->GetVoxels()( voxelidx, num, 3) = int8_t(round( points[ i*4+3] *in_scale0_));
            pre_dict_[batchidx]->GetNumPoints()[voxelidx] += 1;
        }
    
        if(PRE_MT_NUM != 1) {
           canvas_index_arr[threadidx] = -1;
        }
    }
    // indicate the end
    pre_dict_[batchidx]->GetCoors()( voxel_num, 0, 2) = -1;
    pre_dict_[batchidx]->GetCoors()( voxel_num, 0, 3) = -1;
    __TOC__(POINT_TO_VOXELX)
}


}}}

