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

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/text_format.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <thread>

#include "second/protos/second.pb.h"
#include "second/protos/pipeline.pb.h"
#include "second/protos/model.pb.h"
#include <vitis/ai/profiling.hpp>
#include "./postprocess/anchor.hpp"
#include "./helper.hpp"

namespace vitis { namespace ai {

extern second::protos::TrainEvalPipelineConfig cfg;
extern std::vector<int> g_grid_size;
extern G_ANCHOR g_anchor;

//--------------------- supporting func -------
V4F submesh(std::vector<float>& in, std::vector<unsigned int>& v_dim, int pos);
V5F meshgrid(std::vector<float>& x_, std::vector<float>& y_, std::vector<float>& z_,
         std::vector<float>& rot, std::string indexing );
V6F concat_sizes(const V5F& v5fin, const std::vector<float>& sizes);
V6F make_transpose(const V6F& v6f);
V5F make_reshape_v6to5(V6F& v6f);
V5F concat_anchor( V6F& v6f);
V2F make_reshape_v5to2(V5F& v5f);
float limit_period(float val, float offset, float period);
V2F rbbox2d_to_near_bbox(const V2F& v2f, const std::vector<int>& idx);

V4F submesh(std::vector<float>& in, std::vector<unsigned int>& v_dim, int pos) 
{
   V4F v4f;
   V3F v3f;
   V2F v2f;
   V1F v1f;

   for (unsigned int i=0; i<v_dim[0]; i++) {
     for (unsigned int j=0; j<v_dim[1]; j++) {
       for (unsigned int k=0; k<v_dim[2]; k++) {
         for (unsigned int r=0; r<v_dim[3]; r++) {
             switch(pos) {
                 case 0:
                    v1f.emplace_back( in[i] );   break;
                 case 1:
                    v1f.emplace_back( in[j] );   break;
                 case 2:
                    v1f.emplace_back( in[k] );   break;
                 case 3:
                    v1f.emplace_back( in[r] );   break;
                 default:
                    std::cerr << " not supported " << std::endl;
                    exit(-1); 
             }
         }
         v2f.emplace_back(v1f);
         v1f.clear();
       }
       v3f.emplace_back(v2f);
       v2f.clear();
     }
     v4f.emplace_back(v3f);
     v3f.clear();
   }
   return v4f;
}

V5F meshgrid(std::vector<float>& x_, std::vector<float>& y_, std::vector<float>& z_, 
         std::vector<float>& rot, std::string indexing )
{
  if (!(indexing == "ij")) {
    std::cerr << " not supported " << std::endl;
    exit(-1);
  }
  V5F ret;
  std::vector<unsigned int> v_dim{(unsigned int)x_.size(), (unsigned int)y_.size(), (unsigned int)z_.size(), (unsigned int)rot.size()};
  V4F v4f;
  v4f = submesh(x_, v_dim, 0);   ret.emplace_back(v4f);
  v4f = submesh(y_, v_dim, 1);   ret.emplace_back(v4f);   
  v4f = submesh(z_, v_dim, 2);   ret.emplace_back(v4f);
  v4f = submesh(rot,v_dim, 3);   ret.emplace_back(v4f);
  return ret;
}

V6F concat_sizes(const V5F& v5fin, const std::vector<float>& sizes)
{
   /*
     new in: 4 216, 248, 1, 2
       inV:  4 216 248 1 1 2 1 
       outv:   216 248 1 1 2 7
            7 is 3 + 3 +1.  first 3 is the first 3 of 4 in the inV 
                            next 3 is the sizes
                            last 1 is the last 1 of 4 in the inV
    */
   V6F v6f;
   V5F v5f;
   V4F v4f;
   V3F v3f;
   V2F v2f;
   V1F v1f;

   unsigned int i0=0;
   for(unsigned int i1 = 0; i1 < v5fin[0].size(); i1++){                      // 216
     for(unsigned int i2 = 0; i2 < v5fin[0][0].size(); i2++){                 // 248
       for(unsigned int i4 = 0; i4 < v5fin[0][0][0][0].size(); i4++){         // 2
          for(i0 = 0; i0 < v5fin.size()-1; i0++){                             // 4-1==3
             v1f.emplace_back( v5fin[i0][i1][i2][0][i4]); 
          }
          for(unsigned int j=0; j<sizes.size(); j++) {
             v1f.emplace_back(sizes[j]); 
          }
          v1f.emplace_back( v5fin[ i0 ][i1][i2][0][i4]); // i0 is used here
          v2f.emplace_back(v1f);
          v1f.clear(); 
       } // end i4
       v3f.emplace_back(v2f); 
       v2f.clear();
       v4f.emplace_back(v3f);
       v3f.clear();
       v5f.emplace_back(v4f);
       v4f.clear();
     }// end i2
     v6f.emplace_back(v5f);
     v5f.clear();
   } // end i1
   // std::cout <<" v6f : " << v6f.size() << " " << v6f[0].size() << " " <<  v6f[0][0].size() << " "  <<  v6f[0][0][0].size() << " "  <<  v6f[0][0][0][0].size() << " "  <<  v6f[0][0][0][0][0].size() << " \n";

   return v6f;
}

// change the dim 0 & 2 of numpy.transpose. 0 & 2 is fixed. doesn't support other.
V6F make_transpose(const V6F& v6f)
{
  V6F v6fo( v6f[0][0].size(), V5F( v6f[0].size(), V4F( v6f.size())) );

  for (unsigned int i0=0; i0<v6f.size(); i0++ ){
    for (unsigned int i1=0; i1<v6f[0].size(); i1++ ){
      for (unsigned int i2=0; i2<v6f[0][0].size(); i2++ ){
         v6fo[i2][i1][i0]= v6f[i0][i1][i2]; 
      }
    }
  }  
  return v6fo;
}

// change (1, 248, 216, 1, 2, 7) --> (1, 248, 216, 2, 7)
V5F make_reshape_v6to5(V6F& v6f)
{
  V5F v5f( v6f.size(),  V4F( v6f[0].size(), V3F( v6f[0][0].size(), V2F(v6f[0][0][0][0].size() ) ) ));
  for (unsigned int i0=0; i0<v6f.size(); i0++ ){
    for (unsigned int i1=0; i1<v6f[0].size(); i1++ ){
      for (unsigned int i2=0; i2<v6f[0][0].size(); i2++ ){
        v5f[i0][i1][i2]= v6f[i0][i1][i2][0]; // 0 is hard coded here
        // v5f[i0][i1][i2].swap(v6f[i0][i1][i2][0]); // 0 is hard coded here
      }
    }
  }  
  return v5f;
}

// concat: change ( 3   1, 248, 216, 2, 7) --> (1, 248, 216, 6, 7). note the outside dim of v6f which is 3 for 3 anchors
V5F concat_anchor( V6F& v6f)
{
  // std::cout <<"in concat_anchor : size: " << v6f.size() << " " << v6f[0].size() << " " << v6f[0][0].size() << " " << v6f[0][0][0].size() << std::endl;
  V5F v5f( v6f[0].size() , 
           V4F( v6f[0][0].size(), 
                V3F(v6f[0][0][0].size(), 
                    V2F( v6f.size()* v6f[0][0][0][0].size()  ) ) ));

  for(unsigned int i1=0; i1<v6f[0].size(); i1++) {
    for(unsigned int i2=0; i2<v6f[0][0].size(); i2++) {
      for(unsigned int i3=0; i3<v6f[0][0][0].size(); i3++) {
        for(unsigned int i0=0; i0<v6f.size(); i0++) {  
            // v5f[i1][i2][i3].insert( v5f[i1][i2][i3].end(),  v6f[i0][i1][i2][i3].begin(), v6f[i0][i1][i2][i3].end() );
            v5f[i1][i2][i3][i0*2+0].swap( v6f[i0][i1][i2][i3][0] );
            v5f[i1][i2][i3][i0*2+1].swap( v6f[i0][i1][i2][i3][1] );
        }
      }
    }
  }
//  std::cout <<" in concat_anchor: v5f size:" << v5f.size() << " " <<  v5f[0].size() << " " << v5f[0][0].size() << " " << v5f[0][0][0].size() << " " << v5f[0][0][0][0].size() << "\n";

  return v5f;
}

// (1, 248, 216, 6, 7) ---> (?, 7)
V2F make_reshape_v5to2(V5F& v5f)
{
  V2F v2f(v5f.size()*v5f[0].size()*v5f[0][0].size()*v5f[0][0][0].size() );
  int j = 0;
  for(unsigned int i0=0; i0<v5f.size(); i0++) {  
    for(unsigned int i1=0; i1<v5f[0].size(); i1++) {
      for(unsigned int i2=0; i2<v5f[0][0].size(); i2++) {
        for(unsigned int i3=0; i3<v5f[0][0][0].size(); i3++) {
           // v2f.emplace_back(v5f[i0][i1][i2][i3]);
           v2f[j++].swap(v5f[i0][i1][i2][i3]);
        }
      }
    }
  }
  return v2f;
}

inline float limit_period(float val )
{
  return val - floor(val / 3.14159265 + 0.5 ) * 3.14159265;
}

V2F rbbox2d_to_near_bbox(const V2F& v2f, const std::vector<int>& idx)
{
  V2F v2fo(v2f.size());
  for (unsigned int i=0; i< v2f.size(); i++) {
    if (  abs( limit_period(  v2f[i][ idx[4] ] ) ) > 3.14159265/4.0  ) {
      V1F{ v2f[i][idx[0]] - v2f[i][idx[3]]/2,
           v2f[i][idx[1]] - v2f[i][idx[2]]/2,
           v2f[i][idx[0]] + v2f[i][idx[3]]/2,
           v2f[i][idx[1]] + v2f[i][idx[2]]/2
         }.swap( v2fo[i]);
    } else {
      V1F{ v2f[i][idx[0]] - v2f[i][idx[2]]/2,
           v2f[i][idx[1]] - v2f[i][idx[3]]/2,
           v2f[i][idx[0]] + v2f[i][idx[2]]/2,
           v2f[i][idx[1]] + v2f[i][idx[3]]/2
         }.swap( v2fo[i]);
    }
  }
  return v2fo;
}

anchor_stride::anchor_stride(const std::vector<float>& sizes,
                const std::vector<float>& strides,
                const std::vector<float>& offsets,
                const std::vector<float>& rotations,
                float matched_threshold,
                float unmatched_threshold,
                const std::vector<float>& point_cloud_range,
                const std::vector<float>& voxel_size,
                const std::vector<int>& grid_size,
                int out_size_factor) 
  : sizes_(sizes), 
    strides_(strides), 
    offsets_(offsets),
    rotations_(rotations),
    matched_threshold_(matched_threshold),
    unmatched_threshold_(unmatched_threshold),
    point_cloud_range_(point_cloud_range),
    voxel_size_(voxel_size),
    grid_size_(grid_size),
    out_size_factor_(out_size_factor)
{
}

anchor_stride::~anchor_stride()
{
}

void anchor_stride::generate_anchors()
{
#if 0	
    # voxel_builder.py: VoxelGenerator::__init__:  line 174 
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    # [0, -40, -3, 70.4, 40, 1]
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (
        point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    
     
    evaluate_single -> input_reader_builder.build      single_demo_v1.py
    build  -> build_kernel          ->          input_reader_builder.py
    KittiDataset.__init__           ->          dataset_tools.py
    TargetAssigner.generate_anchors ->          target_assigner_builder.py
         TargetAssigner:__init__ : prepare  anchor_generators
    AnchorGeneratorRange.generate   ->          anchor_generator_builder.py
    create_anchors_3d_range  
    # anchor_generator_builder.py  create_anchors_3d_stride

    # 1. feature_size (feature_map_size)
        # input_reader_builder.py  build_kernel  line 67:
        grid_size = voxel_generator.grid_size
        # [352, 400]
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]

        #voxel_builder.py  VoxelGenerator.__init__   line 174
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = ( 
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

    # 2. _size       : from cfg 
    # 3. anchor_strides:  from cfg
    # 4. anchor_offsets:  from cfg 
    # 4. _rotations  : from cfg
    # 5. self._dtype
#endif	
    // assert(point_cloud_range_.size() == voxel_size_.size() );
    std::vector<int> feature_map_size{1};
    for(int i=0; i<2; i++) {
       feature_map_size.emplace_back( int(grid_size_[1-i]/out_size_factor_ )); // [*feature_map_size, 1][::-1]
    }
    // create_anchors_3d_stride
    std::vector<float> z_centers, y_centers, x_centers;
    for(int i=0; i<feature_map_size[0]; i++){
       z_centers.emplace_back(i * strides_[2]+ offsets_[2]);
    } 
    for(int i=0; i<feature_map_size[1]; i++){
       y_centers.emplace_back(i * strides_[1]+ offsets_[1]);
    } 
    for(int i=0; i<feature_map_size[2]; i++){
       x_centers.emplace_back(i * strides_[0]+ offsets_[0]);
    } 
    //np.meshgrid(x_centers 36, y_centers 98, z_centers 1, rotations 2, indexing='ij')
    V5F v5fx = meshgrid(x_centers, y_centers, z_centers, rotations_, "ij");

    // sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    // do nothing, only expand to 1 1 1 1 1 3
    // sizes = np.tile(sizes, tile_size_shape) # tile_size_shape is  [216, 248, 1, 1, 2, 1]

    // rets.insert(3, sizes)  sizes's dim is [216, 248, 1, 1, 2, 1]
    //   this line is wrong, no use.: V6F v6f_sizes = make_tile_v6f(sizes_, v7f[0]);
    // ret = np.concatenate(rets, axis=-1)
    __TIC__(concat_size)
    V6F v6f = concat_sizes(v5fx, sizes_);
    __TOC__(concat_size)

    // return np.transpose(ret, [2, 1, 0, 3, 4, 5])
    // (216, 248, 1, 1, 2, 7) --> (1, 248, 216, 1, 2, 7)
    __TIC__(make_transpose)
    V6F v6fx = make_transpose(v6f);  // 0 2 is fixed integer.
    __TOC__(make_transpose)
    anchors_ = v6fx;
}

void anchor_stride::create_all_anchors_sub( V5F& anchors_v5) {
      __TIC__(generate_anchors)     
      generate_anchors(); 
      __TOC__(generate_anchors)     
      // here, change (1, 248, 216, 1, 2, 7) --> (1, 248, 216, 2, 7)
      // logic is in target_assigner_builder.py:generate_anchors: 
      // anchors = anchors.reshape([*anchors.shape[:3], -1, 7])
      __TIC__(make_reshape_6to5)
      anchors_v5 = make_reshape_v6to5(anchors_);
      __TOC__(make_reshape_6to5)
}

int anchor_stride::create_all_anchors( ) {
  V5F anchors5_all;
  V6F anchors6_all (  cfg.model().second().target_assigner().anchor_generators_size());

  // std::cout << "num_class_test:" << cfg.model().second().num_class() << "   size: " << cfg.model().second().target_assigner().anchor_generators_size() << "\n"; 
  std::vector<std::thread> vth;
  int inum = 0;
  std::unique_ptr<anchor_stride> ac[cfg.model().second().target_assigner().anchor_generators_size() ];
  for (const auto& ags: cfg.model().second().target_assigner().anchor_generators()) {
      ac[inum] = std::make_unique<anchor_stride>( 
         std::vector<float>( ags.anchor_generator_stride().sizes().begin(), 
                             ags.anchor_generator_stride().sizes().end()),
         std::vector<float>( ags.anchor_generator_stride().strides().begin(), 
                             ags.anchor_generator_stride().strides().end()),
         std::vector<float>( ags.anchor_generator_stride().offsets().begin(), 
                             ags.anchor_generator_stride().offsets().end()),
         std::vector<float>( ags.anchor_generator_stride().rotations().begin(), 
                             ags.anchor_generator_stride().rotations().end()),
         ags.anchor_generator_stride().matched_threshold(),
         ags.anchor_generator_stride().unmatched_threshold(),
         std::vector<float>( cfg.model().second().voxel_generator().point_cloud_range().begin(),
                             cfg.model().second().voxel_generator().point_cloud_range().end()),
         std::vector<float>( cfg.model().second().voxel_generator().voxel_size().begin(),
                             cfg.model().second().voxel_generator().voxel_size().end()),
         g_grid_size,
         cfg.model().second().rpn().layer_strides()[0]
      );
      vth.emplace_back(&anchor_stride::create_all_anchors_sub, ac[inum].get(), std::ref( anchors6_all[inum]) );
      inum++;
      /// no use . anchors5_all.insert(anchors5_all.end(), anchors_v5.begin(), anchors_v5.end() );
  }
  for(int i=0; i< cfg.model().second().target_assigner().anchor_generators_size(); i++) {
      vth[i].join();
  }
  // anchors = np.concatenate(anchors_list, axis=-2)
  // change ( 1, 248, 216, 2, 7) --> (1, 248, 216, 6, 7)
  __TIC__(concat_anchor)
  anchors5_all = concat_anchor( anchors6_all);
  __TOC__(concat_anchor)

  // anchors = anchors.reshape([-1, 7]) KittiDataset.__init__ 
  __TIC__(reshape_5to2)
  g_anchor.anchors = make_reshape_v5to2(anchors5_all);
  __TOC__(reshape_5to2)
  // printv("anchors:", g_anchor.anchors);

  // next: anchors_bv: in KittiDataset.__init__
  // anchors_bv = box_np_ops.rbbox2d_to_near_bbox( anchors[:, [0, 1, 3, 4, 6]] )
  __TIC__(rbbox2d_to_near_bbox)
  g_anchor.anchors_bv = rbbox2d_to_near_bbox( g_anchor.anchors , std::vector<int>{0, 1, 3, 4, 6});
  __TOC__(rbbox2d_to_near_bbox)

  // optimize: re-calculate anchors_bv for fast speed.
  float voxel_size0 = cfg.model().second().voxel_generator().voxel_size()[0];
  float voxel_size1 = cfg.model().second().voxel_generator().voxel_size()[1];
  float off_str0 = cfg.model().second().voxel_generator().point_cloud_range()[0]/voxel_size0;
  float off_str1 = cfg.model().second().voxel_generator().point_cloud_range()[1]/voxel_size1;
  for(unsigned int i=0; i<g_anchor.anchors_bv.size(); i++) {
    g_anchor.anchors_bv[i][0] = std::floor( g_anchor.anchors_bv[i][0] /voxel_size0 - off_str0 );
    g_anchor.anchors_bv[i][1] = std::floor( g_anchor.anchors_bv[i][1] /voxel_size1 - off_str1 );
    g_anchor.anchors_bv[i][2] = std::floor( g_anchor.anchors_bv[i][2] /voxel_size0 - off_str0 );
    g_anchor.anchors_bv[i][3] = std::floor( g_anchor.anchors_bv[i][3] /voxel_size1 - off_str1 );

    g_anchor.anchors_bv[i][0] = g_anchor.anchors_bv[i][0] < 0 ? 0 : g_anchor.anchors_bv[i][0];
    g_anchor.anchors_bv[i][1] = g_anchor.anchors_bv[i][1] < 0 ? 0 : g_anchor.anchors_bv[i][1];
    g_anchor.anchors_bv[i][2] = g_anchor.anchors_bv[i][2] < g_grid_size[0] -1 ?  g_anchor.anchors_bv[i][2] : g_grid_size[0] -1;
    g_anchor.anchors_bv[i][3] = g_anchor.anchors_bv[i][3] < g_grid_size[1] -1 ?  g_anchor.anchors_bv[i][3] : g_grid_size[1] -1;
  }

  return 0;
}

}}


