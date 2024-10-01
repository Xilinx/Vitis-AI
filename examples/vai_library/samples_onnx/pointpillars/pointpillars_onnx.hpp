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

#pragma once
#include <assert.h>
#include <glog/logging.h>

#include <opencv2/imgproc/imgproc_c.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>
#include <queue>
#include <map>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <sys/stat.h>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/onnx_task.hpp"
#include "vitis/ai/profiling.hpp"

using namespace std;
using namespace cv;
using namespace vitis::ai;

// -----  index  ----------
// part 1: type declaration
// part 2: return type declaration
// part 3: global var
// part 4: configure items
// part 5: inner function declaration
// part 6: anchor part
// part 6.1: anchor part : inner function declaration
// part 6.2: anchor part : inner function body
// part 6.3: anchor part : class function
// part 7: onnx model class
// part 8: preprocess
// part 9: postprocess part 
// part 10: model run body
// part 11: ui part


// part 1: type declaration
using V1F = std::vector<float>;
using V2F = std::vector<V1F>;
using V3F = std::vector<V2F>;
using V4F = std::vector<V3F>;
using V5F = std::vector<V4F>;
using V6F = std::vector<V5F>;

using V1I = std::vector<int>;
using V2I = std::vector<V1I>;
using V3I = std::vector<V2I>;

typedef struct{
 std::vector<float*> box_;
 std::vector<float*> cls_;
 std::vector<float*> dir_;
}DPU_DATA;

typedef struct {
  V2F anchors;
  V2F anchors_bv;
}G_ANCHOR;

struct lex_cmp{
  bool operator ()( const std::array<float,4>& in1 , const std::array<float,4>& in2) const {
     if ( in1[0] != in2[0])  return in1[0] < in2[0];
     if ( in1[1] != in2[1])  return in1[1] < in2[1];
     return in1[2] > in2[2];
  }
};

using lex_queue = std::priority_queue<std::array<float,4>, std::vector<std::array<float,4> >, lex_cmp>;

typedef Eigen::Tensor<float, 3, Eigen::RowMajor>          VoxelsTensor;
typedef Eigen::TensorMap<VoxelsTensor>                    VoxelsTensorMap;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor>          CoorTensor;
typedef Eigen::TensorMap<CoorTensor>                      CoorTensorMap;

class preout_dict
{
    public:
        preout_dict( float* dpu_in0, float* dpu_in1, int s0, int s1, int s2) :
                    // old hwc :   12000 100  4   |  new chw: 4 12000 100   # in1: 12000 4
            voxelsData (dpu_in0), coorData(dpu_in1)
        {
            memset(dpu_in0, 0, s0*s1*s2*sizeof(float));
            memset(dpu_in1, 0, s0*s1*sizeof(float));
            num_points.resize(s1, 0);
            cur_size = 0;
            voxelsShape[0] = s0;
            voxelsShape[1] = s1;
            voxelsShape[2] = s2;
        }
        void clear() { 
            memset(voxelsData, 0, voxelsShape[0]*voxelsShape[1]*voxelsShape[2]*sizeof(float)); 
            memset(coorData,   0, voxelsShape[0]*voxelsShape[1]*sizeof(float)); 
        }  
        void SetSize(int s)   {
           num_points.assign(s ,0);
           cur_size = s;
        }
        int  GetSize()        { return cur_size; }
        V1I& GetNumPoints()   { return num_points; }
        VoxelsTensorMap GetVoxels() { return VoxelsTensorMap( voxelsData, voxelsShape[0], voxelsShape[1], voxelsShape[2] ); }
        CoorTensorMap GetCoor()   { return CoorTensorMap( coorData, voxelsShape[1], voxelsShape[0] ); }
    private:
        int cur_size;
        V1I   num_points;
        float*     voxelsData;
        float*     coorData;
        std::array<int, 3> voxelsShape;
};

enum e_flag { E_RGB = 0x01, E_BEV = 0x02 };

struct DISPLAY_PARAM {
  V2F P2;
  V2F rect;
  V2F Trv2c;
  V2F p2rect;
};
struct ANNORET {
  /// Name of detected result in vector: such as Car Cylist Pedestrian.
  std::vector<std::string> name;
  /// Label of detected result.
  V1I label;
  /// Truncated information.
  V1F truncated;
  /// Occluded information.
  V1I occluded;
  /// Alpha information.
  V1F alpha;
  /// bbox information.
  V2I bbox;
  /// Dimensions information.
  V2F dimensions;
  /// Location information.
  V2F location;
  /// rotation_y information.
  V1F rotation_y;
  /// Score information.
  V1F score;
  /// box3d_camera information.
  V2F box3d_camera;
  /// box3d_lidar information.
  V2F box3d_lidar;
  /// Inner function to clear all fields.
  void clear() {
    name.clear();
    label.clear();
    truncated.clear();
    occluded.clear();
    alpha.clear();
    bbox.clear();
    dimensions.clear();
    location.clear();
    rotation_y.clear();
    score.clear();
    box3d_camera.clear();
    box3d_lidar.clear();
  }
};

// part 2: return type declaration
// return value
struct PPResult {
  /// Final box predicted.
  V2F final_box_preds;
  /// Final scores predicted.
  V1F final_scores;
  /// Final label predicted.
  V1I label_preds;
};
struct OnnxPointpillarsResult {
  /// Width of network input.
  int width = 0;
  /// Height of network input.
  int height = 0;
  /// Final result returned by the pointpillars neural network.
  PPResult ppresult;
};

// part 3: global var
std::vector<int> g_grid_size;
G_ANCHOR g_anchor;

// part 4: configure items
V1F cfg_voxel_size{0.16, 0.16, 4};
V1F cfg_point_cloud_range{0, -39.68, -3, 69.12, 39.68, 1};
V1I cfg_layer_strides{2, 2, 2};
V2F cfg_anchor_generator_stride_sizes{
 {1.6, 3.9, 1.56},
 {0.6, 1.76, 1.73},
 {0.6, 0.8, 1.73}    };
V2F cfg_anchor_generator_stride_strides{
 {0.32, 0.32, 0.0},
 {0.32, 0.32, 0.0},
 {0.32, 0.32, 0.0}  };
V2F cfg_anchor_generator_stride_offsets{
 {0.16, -39.52, -1.78},
 {0.16, -39.52, -1.465},
 {0.16, -39.52, -1.465}  };
V2F cfg_anchor_generator_stride_rotations{
 {0, 1.57},
 {0, 1.57},
 {0, 1.57} };
V1F cfg_anchor_generator_stride_matched_threshold{0.6, 0.5, 0.5};
V1F cfg_anchor_generator_stride_unmatched_threshold{0.45, 0.35, 0.35};
int cfg_max_number_of_points_per_voxel{100};
int cfg_max_number_of_voxels{12000};
int cfg_nms_pre_max_size{1000};
int cfg_nms_post_max_size{300};
float cfg_nms_iou_threshold{0.5};
float cfg_nms_score_threshold{0.5};
int cfg_num_class{3};
V1F cfg_post_center_limit_range{0, -39.68, -5, 69.12, 39.68, 5};
std::vector<std::string> cfg_class_names{ "Car", "Cyclist", "Pedestrian"};

// part 5: inner function declaration
void get_grid_size();
int getfloatfilelen(const std::string& file);
void myreadfile(float*dest, int size1, std::string filename);

void get_grid_size()
{
  for( int i=0; i<3; i++){
    g_grid_size.emplace_back( int((cfg_point_cloud_range[i+3] - cfg_point_cloud_range[i]) /cfg_voxel_size[i]));
  }
}

int getfloatfilelen(const std::string& file)
{
  struct stat statbuf;
  if(stat(file.c_str(), &statbuf)!=0){
    std::cerr << " bad file stat " << file << std::endl;
    exit(-1);
  }
  return statbuf.st_size/4;
}

void myreadfile(float* dest, int size1, std::string filename)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {
     cout<<"Can't open the file! " << filename << std::endl;
     return;
  }
  Tin.read( (char*)dest, size1*sizeof(float));
}


// part 6: anchor part
class anchor_stride{
public:
  static int create_all_anchors();

  anchor_stride(const std::vector<float>& sizes,
                const std::vector<float>& strides,
                const std::vector<float>& offsets,
                const std::vector<float>& rotations,
                float matched_threshold,
                float unmatched_threshold,
                const std::vector<float>& point_cloud_range,
                const std::vector<float>& voxel_size,
                const std::vector<int>&grid_size,
                int out_size_factor);

  anchor_stride(const anchor_stride& ) = delete;
  ~anchor_stride();
  void create_all_anchors_sub( V5F& anchors_v5);
  void generate_anchors();
public:
  V1F sizes_;
  V1F strides_;
  V1F offsets_;
  V1F rotations_;
  float matched_threshold_;
  float unmatched_threshold_;
  V1F point_cloud_range_;
  V1F voxel_size_;
  std::vector<int> grid_size_;
  int out_size_factor_;
  V6F anchors_;
};

// part 6.1: anchor part : inner function declaration
V4F submesh(std::vector<float>& in, std::vector<unsigned int>& v_dim, int pos);
V5F meshgrid(std::vector<float>& x_, std::vector<float>& y_, std::vector<float>& z_,
         std::vector<float>& rot, std::string indexing );
V6F concat_sizes(const V5F& v5fin, const std::vector<float>& sizes);
V6F make_transpose(const V6F& v6f);
V5F make_reshape_v6to5(V6F& v6f);
V5F concat_anchor( V6F& v6f);
V2F make_reshape_v5to2(V5F& v5f);
inline float limit_period(float val, float offset, float period);
V2F rbbox2d_to_near_bbox(const V2F& v2f, const std::vector<int>& idx);

// part 6.2: anchor part : inner function body
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

// part 6.3: anchor part : class function
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

anchor_stride::~anchor_stride() { }

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
  V6F anchors6_all ( 3);

  std::vector<std::thread> vth;
  std::unique_ptr<anchor_stride> ac[3 ];
  for (int i=0; i<3; i++) {
      ac[i] = std::make_unique<anchor_stride>(
         cfg_anchor_generator_stride_sizes[i],
         cfg_anchor_generator_stride_strides[i],
         cfg_anchor_generator_stride_offsets[i],
         cfg_anchor_generator_stride_rotations[i],
         cfg_anchor_generator_stride_matched_threshold[i],
         cfg_anchor_generator_stride_unmatched_threshold[i],
         cfg_point_cloud_range,
         cfg_voxel_size,
         g_grid_size,
         cfg_layer_strides[0]
      );
      vth.emplace_back(&anchor_stride::create_all_anchors_sub, ac[i].get(), std::ref( anchors6_all[i]) );
  }
  for(int i=0; i<3; i++) {
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
  float voxel_size0 = cfg_voxel_size[0];
  float voxel_size1 = cfg_voxel_size[1];
  float off_str0 = cfg_point_cloud_range[0]/voxel_size0;
  float off_str1 = cfg_point_cloud_range[1]/voxel_size1;

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
// end of anchor part

// part 7: onnx model class
class OnnxPointpillars : public OnnxTask {
 public:
  static std::unique_ptr<OnnxPointpillars> create(const std::string& model_name) {
    return std::unique_ptr<OnnxPointpillars>(new OnnxPointpillars(model_name));
  }

 protected:
  explicit OnnxPointpillars(const std::string& model_name);
  OnnxPointpillars(const OnnxPointpillars&) = delete;

 public:
  virtual ~OnnxPointpillars() {}
  virtual std::vector<OnnxPointpillarsResult> run(const std::vector<float*>& points, const std::vector<int>& vlen);
  void do_pointpillar_display(OnnxPointpillarsResult & res, int flag, DISPLAY_PARAM& g_test, cv::Mat& rgb_map, cv::Mat& bev_map, int imgwidth, int imgheight, ANNORET& annoret);

 private:
  void preprocess( float* points, int len_f , int batchidx);
  std::vector<OnnxPointpillarsResult> postprocess();
  OnnxPointpillarsResult postprocess(int idx);

  // from postprocess
  V3F corners_nd_2d(const V2F& dims);
  void get_dpu_data();
  void fused_get_anchors_area(int);
  V1F get_decode_box( int batchidx, int );
  void get_anchors_mask( const std::vector<std::shared_ptr<preout_dict>>& v_pre_dict_);
  V2F center_to_corner_box2d_to_standup_nd(const V2F& );
  // from parse_
  void predict_kitti_to_anno(V2F&, V2F&, V2F&, V1I&, V1F& ,  ANNORET& , int, int );
 private:
  std::vector<float> input_tensor_values0;
  std::vector<float> input_tensor_values1;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  DPU_DATA dpu_data;
  int real_batch;
  std::array<int, 3> voxelmap_shape_;
  V1I coor_to_voxelidx;
  std::vector<std::shared_ptr<preout_dict>> pre_dict_;
  std::vector<V1I> anchors_mask;

  V2I corners_norm;
  float nms_confidence_;
  V1F top_scores;
  V2F box_preds;
  V1I dir_labels;
  V1I top_labels;

  V2F dense_voxel_map;
  int batchnum = 0;
  std::array<float, 3> scale_pclen;
  std::array<float, 3> scale_pcstartlen;
  std::array<float, 3> point_range;
  
  std::vector<float*> output_tensor_ptr;
};

// part 8: preprocess
// input : 4 12000 100
void OnnxPointpillars::preprocess( float* points, int len_f, int batchidx) {
  std::array<int32_t, 3> coor;
  bool failed = false;
  int c = 0, i = 0, j = 0, num = 0, voxel_num = 0;
  int32_t voxelidx = 0;
  
  for (i = 0; i < len_f/4; i++)  {
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
      if ( (voxelidx = coor_to_voxelidx[canvas_index]) == -1 )  {
        if (voxel_num == cfg_max_number_of_voxels) {
           continue;
           // break;
        }
        coor_to_voxelidx [canvas_index] = voxel_num;

        pre_dict_[batchidx]->GetCoor()( voxel_num, 3) = coor[2];
        pre_dict_[batchidx]->GetCoor()( voxel_num, 2) = coor[1];

        voxelidx = voxel_num;
        voxel_num ++;
      }
      num = pre_dict_[batchidx]->GetNumPoints()[voxelidx];
      if (  num  < cfg_max_number_of_points_per_voxel ) {
          pre_dict_[batchidx]->GetVoxels()(0, voxelidx, num) =  points[i*4+0] * scale_pclen[0] - scale_pcstartlen[0];
          pre_dict_[batchidx]->GetVoxels()(1, voxelidx, num) =  points[i*4+1] * scale_pclen[1] - scale_pcstartlen[1];
          pre_dict_[batchidx]->GetVoxels()(2, voxelidx, num) =  points[i*4+2] * scale_pclen[2] - scale_pcstartlen[2];
          pre_dict_[batchidx]->GetVoxels()(3, voxelidx, num) =  points[i*4+3] ;
          pre_dict_[batchidx]->GetNumPoints()[voxelidx] += 1;

          // if (i<50 || i>len_f/4-10 ) {
          //   std::cout <<i << " : " ;
          //   for(int kk=0; kk<4; kk++) std::cout<<  pre_dict_[batchidx]->GetVoxels()(kk, voxelidx, num) << " ";
          //   std::cout << " \n " ;
          // }
      }
  }
  coor_to_voxelidx.assign(voxelmap_shape_[1]*voxelmap_shape_[2], -1);   
  pre_dict_[batchidx]->SetSize(voxel_num);

  return;
}

// part 9: postprocess part 

int get_max(float* in, int num);
std::vector<int> topk(const V1F& scores, int k, V2F& bboxes_in, V2F& bboxes_out);
std::vector<int> non_max_suppression_cpu(
  V2F& bboxes_,
  const V1F& scores_,
  int pre_max_size_,
  int post_max_size_,
  float iou_threshold );
V2F corner_to_standup_nd(const V3F& boxes_corner);
V3F rotation_2d(const V3F& points, const V2F& angles);
V3F einsum(const V3F& points, const V3F& rot_mat_T);
void sparse_sum_for_anchors_mask(CoorTensorMap coor, int size, V2F&  v2f);
int cumsum(V2F& v);
V2I unravel_index_2d(const V1I& index, const V1I& dims );
int get_max(float* in, int num){ return  std::max_element(in, in+num) - in; }

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
// std:cout << "keep.size:"<<keep.size() << "   " << bsize << "\n";
  return keep;
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

V3F OnnxPointpillars::corners_nd_2d(const V2F& dims)
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

V2F OnnxPointpillars::center_to_corner_box2d_to_standup_nd(const V2F& box)
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

// void sparse_sum_for_anchors_mask(const std::vector<std::pair<int,int>>& coors, int size, V2F&  v2f)
void sparse_sum_for_anchors_mask(CoorTensorMap coor, int size, V2F&  v2f)
{
  // std::cout << "mask : " << size <<"\n";
  for(auto &it: v2f) {
    it.assign(it.size() ,0);
  }
  for(int i=0; i< size; i++){
    // v2f[ coors[i].first  ][ coors[i].second ]+=1; 
    // change to original version: because I changed the definition of coors to 3 len
    v2f[ coor(i, 2) ][ coor(i, 3) ]+=1; 
    // if (i<10 || i > size-10)  std::cout <<"mask : " << i << " " << coor(i,2) << " " << coor(i,3) <<"\n"; 
  }
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

V2I unravel_index_2d(const V1I& index, const V1I& dims )
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

void OnnxPointpillars::fused_get_anchors_area(int batchidx)
{
  for(unsigned int i=0; i<g_anchor.anchors_bv.size(); i++) {
    if (   dense_voxel_map[g_anchor.anchors_bv[i][3]][ g_anchor.anchors_bv[i][2]]
         - dense_voxel_map[g_anchor.anchors_bv[i][3]][ g_anchor.anchors_bv[i][0]]
         - dense_voxel_map[g_anchor.anchors_bv[i][1]][ g_anchor.anchors_bv[i][2]]
         + dense_voxel_map[g_anchor.anchors_bv[i][1]][ g_anchor.anchors_bv[i][0]]
         > 1) {
       anchors_mask[batchidx].emplace_back(i);
    }
  }
}

void OnnxPointpillars::get_anchors_mask( const std::vector<std::shared_ptr<preout_dict>>& v_pre_dict_)
{
  for(int i=0; i<real_batch; i++) {
      sparse_sum_for_anchors_mask(  v_pre_dict_[i]->GetCoor(), v_pre_dict_[i]->GetSize(), dense_voxel_map );
      cumsum(dense_voxel_map);
      fused_get_anchors_area(i );
   }
}

V1F OnnxPointpillars::get_decode_box(int batchidx, int idx) {
  V1F o(7, 0);
  float za = g_anchor.anchors[ idx ][2] + g_anchor.anchors[  idx ][5]/2;
  float diagonal = sqrt( pow(g_anchor.anchors[  idx  ][4], 2.0) + pow(g_anchor.anchors[ idx  ][3], 2.0));
  o[0] = (dpu_data.box_[batchidx][ idx *7+0]) * diagonal + g_anchor.anchors[ idx ][0];
  o[1] = (dpu_data.box_[batchidx][ idx *7+1]) * diagonal + g_anchor.anchors[ idx ][1];
  o[2] = (dpu_data.box_[batchidx][ idx *7+2]) * g_anchor.anchors[  idx  ][5] + za;
  o[3] = exp(  (dpu_data.box_[batchidx][ idx *7+3])) * g_anchor.anchors[ idx ][3];
  o[4] = exp(  (dpu_data.box_[batchidx][ idx *7+4])) * g_anchor.anchors[ idx ][4];
  o[5] = exp(  (dpu_data.box_[batchidx][ idx *7+5])) * g_anchor.anchors[ idx ][5];
  o[6] = (dpu_data.box_[batchidx][ idx *7+6]) + g_anchor.anchors[ idx ][6];
  o[2] = o[2] - o[5]/2;
  return o;
}

OnnxPointpillarsResult OnnxPointpillars::postprocess(int batchidx) {
  OnnxPointpillarsResult res{(int)getInputWidth(), (int)getInputHeight()};
  top_scores.clear();
  box_preds.clear();
  dir_labels.clear();
  top_labels.clear();

  int pos = 0;

  for(unsigned int i=0; i<anchors_mask[batchidx].size(); i++){
    pos = get_max( &dpu_data.cls_[batchidx][ anchors_mask[batchidx][i]*cfg_num_class ], cfg_num_class);
    if( dpu_data.cls_[batchidx][   anchors_mask[batchidx][i]*cfg_num_class +pos ] >= nms_confidence_) {
      top_scores.emplace_back( dpu_data.cls_[batchidx][   anchors_mask[batchidx][i]*cfg_num_class +pos  ] );
      box_preds.emplace_back( get_decode_box(batchidx,  anchors_mask[batchidx][i]) );
      dir_labels.emplace_back(
        dpu_data.dir_[batchidx][anchors_mask[batchidx][i]*2+0 ] >= dpu_data.dir_[batchidx][anchors_mask[batchidx][i]*2+1 ]  ? 0 : 1
      );
      top_labels.emplace_back( pos );
      // std::cout << i << " " << pos <<"\n";
    }
  }
  if (top_scores.empty() ) {
     return res;
  }
  V2F boxes_for_nms = center_to_corner_box2d_to_standup_nd(box_preds);
  V1I selected = non_max_suppression_cpu(
     boxes_for_nms,
     top_scores,
     cfg_nms_pre_max_size,
     cfg_nms_post_max_size,
     cfg_nms_iou_threshold );
  V2F selected_boxes(selected.size());
  V1I selected_dir_labels(selected.size());
  V1I selected_labels(selected.size());

  res.ppresult.final_scores.reserve( selected.size() );

  for(unsigned int i=0; i<selected.size(); i++) {
    selected_boxes[i].swap( box_preds[selected[i]]);
    selected_dir_labels[i] = dir_labels[selected[i]];
    selected_labels[i] = top_labels[selected[i]];
    res.ppresult.final_scores.emplace_back(  1.0/(1.0+exp(-1.0* top_scores[selected[i]]))   );
    // std::cout <<"final_score:  " << i << "  " << selected[i] << "  " <<  res.ppresult.final_scores[i] << "\n";
  }
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
  return res;
}

std::vector<OnnxPointpillarsResult> OnnxPointpillars::postprocess() {
  std::vector<OnnxPointpillarsResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index) {
    ret.emplace_back(postprocess( index));
  }
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

OnnxPointpillars::OnnxPointpillars(const std::string& model_name) : OnnxTask(model_name) {
  int total_number_elements = calculate_product(input_shapes_[0]);
  input_tensor_values0.resize( total_number_elements );
  // never fix shape for input1 !!
  //  input_shapes_[1][3] = 0;
  //  input_shapes_[1][2] = input_shapes_[1][1];
  //  input_shapes_[1][1] = input_shapes_[1][0];
  //  input_shapes_[1][0] = input_shapes_[0][0];
  total_number_elements = input_shapes_[0][0] * input_shapes_[1][0] * input_shapes_[1][1];
  input_tensor_values1.resize( total_number_elements );
  auto channel = input_shapes_[0][1];
  auto height = input_shapes_[0][2];
  auto width = input_shapes_[0][3];

  // input shape : 6 4 12000 100 (nchw)
  // input2 shape : 12000 4 0 81  --> 6 12000 4
  // std::cout << " input shape : " << input_shapes_[0][0] << " " << input_shapes_[0][1] << " " << input_shapes_[0][2] << " " << input_shapes_[0][3] << " \n";
  // std::cout << " input2 shape : " << input_shapes_[1][0] << " " << input_shapes_[1][1] << " " << input_shapes_[1][2] << " " << input_shapes_[1][3] << " \n";

  output_tensor_ptr.resize(3);

  nms_confidence_ = -log(1.0/cfg_nms_score_threshold -1.0 );
  corners_norm = unravel_index_2d(V1I({0,1,2,3}), V1I({2,2}));
  get_grid_size();
  V2F tmpv2f( g_grid_size[1], V1F( g_grid_size[0], 0));
  dense_voxel_map.swap(tmpv2f);
  anchor_stride::create_all_anchors();

  batchnum = input_shapes_[0][0];
  anchors_mask.resize(batchnum);

  V1F pc_len;
  for(int i=0; i<3; i++) {
    pc_len.emplace_back(  cfg_point_cloud_range[i+3] - cfg_point_cloud_range[i]);
  }
  for (int i=0; i<3; i++) {
     voxelmap_shape_[2-i] = round(( cfg_point_cloud_range[3+i] - cfg_point_cloud_range[i]) / cfg_voxel_size[i]);
  }
  coor_to_voxelidx.assign(voxelmap_shape_[1]*voxelmap_shape_[2], -1);

  for(int i=0; i<3; i++) {
    scale_pcstartlen[i] = cfg_point_cloud_range[i]/pc_len[i];
    scale_pclen[i] = 1.0f/pc_len[i];
    point_range[i] =  cfg_point_cloud_range[i]/cfg_voxel_size[i];
  }

  for(int i=0; i<batchnum; i++) {
     // pre_dict_.emplace_back(std::make_shared<preout_dict>( in_addr1[i], in_height1, in_width1, in_channel1  ));
     pre_dict_.emplace_back(std::make_shared<preout_dict>( 
         input_tensor_values0.data() + i*(input_shapes_[0][1]*input_shapes_[0][2]*input_shapes_[0][3]), 
         input_tensor_values1.data() + i*(input_shapes_[1][1]*input_shapes_[1][2]), 
         channel, height, width ));
  }
  dpu_data.box_.resize(batchnum);
  dpu_data.cls_.resize(batchnum);
  dpu_data.dir_.resize(batchnum);
}

void OnnxPointpillars::get_dpu_data()
{
  // output shape : 1 248 216 42  :7 -->box
  // output shape : 1 248 216 18  :3 -->cls
  // output shape : 1 248 216 12  :2 -->dir
  // for(int i=0; i<3; i++) std::cout << " output shape : " << output_shapes_[i][0] << " " << output_shapes_[i][1] << " " << output_shapes_[i][2] << " " << output_shapes_[i][3] << " \n";
  for(int i=0; i<batchnum; i++) {
     dpu_data.box_[i] = output_tensor_ptr[0] + i*(output_shapes_[0][1]*output_shapes_[0][2]*output_shapes_[0][3]);
     dpu_data.cls_[i] = output_tensor_ptr[1] + i*(output_shapes_[1][1]*output_shapes_[1][2]*output_shapes_[1][3]);
     dpu_data.dir_[i] = output_tensor_ptr[2] + i*(output_shapes_[2][1]*output_shapes_[2][2]*output_shapes_[2][3]);
  }
}

// part 10: model run body
std::vector<OnnxPointpillarsResult> OnnxPointpillars::run(
    const std::vector<float*>& vpoints, const std::vector<int>& vlen) {

  if (input_tensors.size()) {
    input_tensors[0] = Ort::Experimental::Value::CreateTensor<float>( input_tensor_values0.data(), input_tensor_values0.size(), input_shapes_[0]);
    input_tensors[1] = Ort::Experimental::Value::CreateTensor<float>( input_tensor_values1.data(), input_tensor_values1.size(), input_shapes_[1]);
  } else {
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>( input_tensor_values0.data(), input_tensor_values0.size(), input_shapes_[0]));
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>( input_tensor_values1.data(), input_tensor_values1.size(), input_shapes_[1]));
  }

  real_batch = std::min((int)input_shapes_[0][0], (int)vlen.size());
  for(int i=0; i<real_batch; i++) {
    anchors_mask[i].clear();
  }

  input_tensor_values0.assign(input_tensor_values0.size(), 0);
  input_tensor_values1.assign(input_tensor_values1.size(), 0);

  for (auto index = 0; index < real_batch; ++index) {
    preprocess(vpoints[index], vlen[index], index);
  }
  
  __TIC__(session_run)
  run_task(input_tensors, output_tensors);
  for (int i = 0; i < 3; i++) {
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }

  get_dpu_data();
  __TOC__(session_run)

  get_anchors_mask(pre_dict_);

  __TIC__(postprocess)
  std::vector<OnnxPointpillarsResult> ret = postprocess();
  __TOC__(postprocess)
  return ret;
}

// part 11: ui part
int bevimage_width=1024;
int bevimage_height=512;

std::vector<cv::Scalar> vscalar = {
    cv::Scalar(0,   255, 255),
    cv::Scalar(255, 0,   255),
    cv::Scalar(255, 255, 0  ),
    cv::Scalar(0,   0,   255),
    cv::Scalar(255, 0,   0  ),
    cv::Scalar(0,   255, 0  )
};

// const variable here :
std::map<std::string, int>  BoundaryCond = {
    { "minX", 0},
    { "maxX", 80},
    { "minY", -40},
    { "maxY", 40},
    { "minZ", -3},
    { "maxZ", 3}
};
const float Resolution = 40.0/512;
static int BR_Width  = int( (BoundaryCond["maxY"] - BoundaryCond["minY"]) / Resolution + 1);

V3F project_to_image(V3F& points_3d, const V2F& proj_mat);
V3F rotation_3d_in_axis(const V3F& points, const V1F& angles);
V3F center_to_corner_box3d(const V2F& centers, const V2F& dims, const V1F& angles);
cv::Mat bev_preprocess(const V1F& PointCloud);
void bevpre_removePoints(const V1F& PointCloud, lex_queue& lex_q);
cv::Mat bevpre_makeBVFeature(lex_queue&);
std::tuple<V1I, V3F>  label2BevImage_v4( float score_thresh, ANNORET&);
void drawRotBboxes(cv::Mat& img, const V3F& bboxes, V1I&) ;
V2F matmul_2x2f(const V2F& in1, const V2F& in2);
V3F matmul_3x2f(const V3F& in1, const V2F& in2);
V2F box_lidar_to_camera(const V2F& data, const V2F& r_rect, const V2F& velo2cam);
V2F dim_transpose(const V2F& in);
V3F corners_nd_3d(const V2F& dims);
V2F unravel_index_3d(const V1I& index, const V1I& dims );


// below is new 3d draw logic;
static V2F roty(float   x ) ;
static V2F generate8CornersKitti( const V1F& ibox) ;
static void draw_proj_box(const V1F& ibox, cv::Mat& img, const V2F& p2rect, int);
static V2F project_3d_pts( V2F& corners, const V2F& p2rect);
static void draw_projected_box3d(cv::Mat& img, const V2F& qs, int label);
static void draw_rgb( V2F& fcbox, DISPLAY_PARAM& g_test,  cv::Mat& img , V1I&);

void OnnxPointpillars::do_pointpillar_display(OnnxPointpillarsResult & res, int flag, DISPLAY_PARAM& g_test, cv::Mat& rgb_map, cv::Mat& bev_map, int imgwidth, int imgheight, ANNORET& annoret)
{
  __TIC__(inner_do_pointpillar_display)    //78us

  __TIC__(box_lidar_to_camera)
  V2F final_box_preds_camera = box_lidar_to_camera(res.ppresult.final_box_preds, g_test.rect, g_test.Trv2c);
  __TOC__(box_lidar_to_camera)

  if (flag & E_RGB) {
    __TIC__(draw_rgb)   //347us
    draw_rgb(final_box_preds_camera, g_test, rgb_map, res.ppresult.label_preds);
    __TOC__(draw_rgb)

    if (!(flag & E_BEV)) {
       __TOC__(inner_do_pointpillar_display)
       return;
    }
  }

  __TIC__(fpbc)   // 3us
  V2F fbpc_locs(final_box_preds_camera.size(), V1F(3, 0) );
  V2F fbpc_dims(final_box_preds_camera.size(), V1F(3, 0) );
  V1F fbpc_angles(final_box_preds_camera.size(), 0 );

  for(unsigned int i=0; i<final_box_preds_camera.size(); i++) {
     fbpc_locs[i] = V1F{final_box_preds_camera[i][0],
                        final_box_preds_camera[i][1],
                        final_box_preds_camera[i][2] };
     fbpc_dims[i] = V1F{final_box_preds_camera[i][3],
                        final_box_preds_camera[i][4],
                        final_box_preds_camera[i][5] };
     fbpc_angles[i] = final_box_preds_camera[i][6];
  }
  __TOC__(fpbc)

  __TIC__(center_to_corner_box3d)    // 52us
  V3F box_corners = center_to_corner_box3d(fbpc_locs, fbpc_dims, fbpc_angles);
  __TOC__(center_to_corner_box3d)

  __TIC__(project_to_image)   // 32us
  V3F box_corners_in_image = project_to_image(box_corners, g_test.P2);
  __TOC__(project_to_image)

  __TIC__(corner_to_standup_nd2)   // 2us
  V2F box_2d_preds = corner_to_standup_nd(box_corners_in_image);
  __TOC__(corner_to_standup_nd2)

  __TIC__(predict_kitti_to_anno)   // 65us
  predict_kitti_to_anno(final_box_preds_camera, res.ppresult.final_box_preds, box_2d_preds, res.ppresult.label_preds, res.ppresult.final_scores, annoret, imgwidth, imgheight);
  __TOC__(predict_kitti_to_anno)

  if (!flag ) {
    return;
  }
  __TIC__(label2BevImage_v4)   //65us
  V3F bboxes;
  V1I vlab;
  std::tie(vlab, bboxes) = label2BevImage_v4( cfg_nms_score_threshold, annoret);
  __TOC__(label2BevImage_v4)

  __TIC__(drawRotBboxes)   // 6690us
  drawRotBboxes(bev_map, bboxes, vlab);
  cv::flip(bev_map, bev_map, -1);  //this line too slow : 6ms

  __TOC__(drawRotBboxes)
  __TOC__(inner_do_pointpillar_display)
}

V3F project_to_image(V3F& points_3d, const V2F& proj_mat) // [101 8 3][ 4 4 ]
{
  if (points_3d.empty()) {
     return V3F{};
  }

  // return [101 8 2 ]
  V1I points_shape{(int)points_3d.size(), (int)points_3d[0].size(), 1 };
  // points_4 = torch.cat( [points_3d, torch.zeros(*points_shape).type_as(points_3d)], dim=-1)
  for(unsigned int i=0; i<points_3d.size(); i++) {
    for(unsigned int j=0; j<points_3d[0].size(); j++) {
        points_3d[i][j].emplace_back(0.0);
    }
  }
  // point_2d = torch.matmul(points_4, proj_mat.t())
  V2F proj_mat_t( proj_mat[0].size(), V1F(proj_mat.size(), 0));
  for(unsigned int i=0; i<proj_mat.size(); i++) {
    for(unsigned int j=0; j<proj_mat[0].size(); j++) {
      proj_mat_t[j][i] = proj_mat[i][j];
    }
  }
  V3F v3f = matmul_3x2f(points_3d, proj_mat_t);
  // point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
  for(unsigned int i=0; i<v3f.size(); i++){
    for(unsigned int j=0; j<v3f[0].size(); j++){
      v3f[i][j][0]/=v3f[i][j][2];
      v3f[i][j][1]/=v3f[i][j][2];
      v3f[i][j].resize(2);
    }
  }
  return v3f;
}

V3F rotation_3d_in_axis(const V3F& points, const V1F& angles) // only axis == 1
{
  V1F rot_sin(angles.size(), 0);
  V1F rot_sin_n(angles.size(), 0);
  V1F rot_cos(angles.size(), 0);
  for(unsigned int i=0; i<angles.size(); i++) {
    rot_sin[i] = sin(angles[i]);
    rot_sin_n[i] = -rot_sin[i];
    rot_cos[i] = cos(angles[i]);
  }
  V3F rot_mat_T(3, V2F(3, V1F(points.size(), 0)));
  rot_mat_T[0][0] = rot_cos;
  rot_mat_T[0][1] = V1F(points.size() ,0);
  rot_mat_T[0][2] = rot_sin_n;
  rot_mat_T[1][0] = V1F(points.size(), 0);
  rot_mat_T[1][1] = V1F(points.size(), 1);
  rot_mat_T[1][2] = V1F(points.size(), 0);
  rot_mat_T[2][0] = rot_sin;
  rot_mat_T[2][1] = V1F(points.size(), 0);
  rot_mat_T[2][2] = rot_cos;

  return einsum(points, rot_mat_T);
}

V3F center_to_corner_box3d(const V2F& centers, const V2F& dims, const V1F& angles)
{
  if (dims.empty()) {
     return V3F{};
  }
  V3F corners = corners_nd_3d(dims);
  corners = rotation_3d_in_axis(corners, angles);

  for(unsigned int i=0; i< corners.size(); i++){
    for(unsigned int j=0; j< corners[0].size(); j++){
      for(unsigned int k=0; k< corners[0][0].size(); k++){
        corners[i][j][k] += centers[i][k];
      }
    }
  }
  return corners;
}

void OnnxPointpillars::predict_kitti_to_anno(V2F& final_box_preds_camera, V2F& final_box_preds, V2F& box_2d_preds, V1I& label_preds, V1F& final_scores, ANNORET& annoret, int image_width, int image_height)
{
/*
  python   box,                    box_lidar,        bbox,          score,       label
           box_preds,              box_preds_lidar,  box_2d_preds,  scores,      label_preds
           box3d_camera            box3d_lidar       bbox           scores       label_preds
  c++      final_box_preds_camera  final_box_preds   box_2d_preds   final_scores label_preds

     predictions_dict = {
         "bbox": box_2d_preds,
         "box3d_camera": final_box_preds_camera,
         "box3d_lidar": final_box_preds,
         "scores": final_scores,
         "label_preds": label_preds,
         "image_idx": img_idx,
     }

  typedef struct {
    V2F box_2d_preds;           ):
    V2F final_box_preds_camera
    V2F final_box_preds;
    V1F final_scores;
    V1I label_preds;
  }G_RET;

   box_2d_preds = preds_dict["bbox"]
   box_preds = preds_dict["box3d_camera"]
   scores = preds_dict["scores"]
   box_preds_lidar = preds_dict["box3d_lidar"]
   label_preds = preds_dict["label_preds"]
*/

  // std::cout <<" predict_kitti_to_anno -- :  box_2d_preds.size:  " << box_2d_preds.size() << " image with height : " << image_width << "  " << image_height << "\n";
  if(! box_2d_preds.size()) {
    return;
  }

  for(unsigned int i=0; i<box_2d_preds.size(); i++) {
    if(   box_2d_preds[i][0] > image_width
       || box_2d_preds[i][1] > image_height
       || box_2d_preds[i][2] < 0
       || box_2d_preds[i][3] < 0 ) {
      continue;
    }
    bool bfindsmall = false;
    for(int j=0; j<3; j++){
      if(   final_box_preds[i][j] < cfg_post_center_limit_range[j]
         || final_box_preds[i][j+3] > cfg_post_center_limit_range[j+3] ) {
        bfindsmall = true;
        break;
      }
    }
    if (bfindsmall) {
      continue;
    }
    V1F bbox = box_2d_preds[i];
    bbox[2] = std::min(bbox[2], float(image_width));
    bbox[3] = std::min(bbox[3], float(image_height));
    bbox[0] = std::max(bbox[0], 0.0f);
    bbox[1] = std::max(bbox[1], 0.0f);

    annoret.name.push_back( std::string(cfg_class_names[ int(label_preds[i]) ]  ));
    annoret.label.push_back( int(label_preds[i]) );
    annoret.truncated.emplace_back(0.0);
    annoret.occluded.emplace_back(0);
    float alpha = -atan2(-final_box_preds[i][1], final_box_preds[i][0] ) +  final_box_preds_camera[i][6];
    // std::cout << "alpha:  " << alpha << -final_box_preds[i][1] << " " << final_box_preds[i][0] << " "
    //     <<  final_box_preds_camera[i][6] << "  atan2:" << atan2(-final_box_preds[i][1], final_box_preds[i][0] ) << std::endl;

    annoret.alpha.emplace_back(alpha);
    annoret.bbox.emplace_back(V1I{ int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])  } );
    // std::cout << "bbox:" << int(bbox[0]) << " " << int(bbox[1]) << " " << int(bbox[2]) << " " <<  int(bbox[3]) << std::endl;
    annoret.dimensions.emplace_back(
       V1F{ final_box_preds_camera[i][4],
            final_box_preds_camera[i][5],
            final_box_preds_camera[i][3]      });
    // std::cout << " dimension:" << final_box_preds_camera[i][3] << " " << final_box_preds_camera[i][4] << " " << final_box_preds_camera[i][5] << "\n";
    annoret.location.emplace_back(
       V1F{ final_box_preds_camera[i][0],
            final_box_preds_camera[i][1],
            final_box_preds_camera[i][2]      });
    // std::cout << "location:" << final_box_preds_camera[i][0] << " " << final_box_preds_camera[i][1] << " " << final_box_preds_camera[i][2] << "\n";
    annoret.rotation_y.emplace_back( final_box_preds_camera[i][6]);
    annoret.box3d_camera.emplace_back(final_box_preds_camera[i]);
    annoret.box3d_lidar.emplace_back(final_box_preds[i]);
    annoret.score.emplace_back( final_scores[i] );
  }
}

cv::Mat bev_preprocess(const V1F& PointCloud)
{
    lex_queue lex_q;
    __TIC__(bev_preprocess_bevpre_removePoints)
    bevpre_removePoints(PointCloud, lex_q );
    __TOC__(bev_preprocess_bevpre_removePoints)
    __TIC__(bev_preprocess_bevpre_makeBVFeature)
    cv::Mat BEVMap = bevpre_makeBVFeature( lex_q );
    __TOC__(bev_preprocess_bevpre_makeBVFeature)
    return BEVMap;
}

void bevpre_removePoints(const V1F& PointCloud,  lex_queue& lex_q )
{
    unsigned int loopsize = PointCloud.size()/4;
    for(unsigned int i=0; i<loopsize; i++) {
      if (   PointCloud[i*4+0] >= BoundaryCond["minX"]
          && PointCloud[i*4+0] <= BoundaryCond["maxX"]
          && PointCloud[i*4+1] >= BoundaryCond["minY"]
          && PointCloud[i*4+1] <= BoundaryCond["maxY"]
          && PointCloud[i*4+2] >= BoundaryCond["minZ"]
          && PointCloud[i*4+2] <= BoundaryCond["maxZ"] )
      {
         lex_q.emplace( std::array<float,4>{
              std::floor(PointCloud[i*4+0]/Resolution),
              float(std::floor(PointCloud[i*4+1]/Resolution) + BR_Width/2),
              PointCloud[i*4+2],
              PointCloud[i*4+3] }  );
      }
    }
}

cv::Mat bevpre_makeBVFeature(lex_queue& lex_q)
{
  cv::Mat img(bevimage_height, bevimage_width, CV_8UC3, cv::Scalar(0, 0, 0));
  uchar* p;
  unsigned int loopx = 0;
  std::array<float,4> topv;

  float tmpv =  255.0 * BoundaryCond["minZ"] / (BoundaryCond["maxZ"] - BoundaryCond["minZ"]);
  float tmpv_1 = 255.0/(BoundaryCond["maxZ"] - BoundaryCond["minZ"]);
  float tmpv_2 = 255.0/log(64);

  while( ! lex_q.empty()) {
    loopx = 0;
    topv = std::move(lex_q.top());
    lex_q.pop();
    if(topv[0] <bevimage_height && topv[1] <bevimage_width) {
      while(  ! lex_q.empty()
             && lex_q.top()[0] == topv[0]
             && lex_q.top()[1] == topv[1] ) {
        lex_q.pop();
        loopx++;
      }
      p=img.ptr<uchar>( topv[0] );
      p[ int(topv[1]*3+0) ] = uchar(topv[3]*255);
      // p[ int(topv[1]*3+1) ] =  abs(topv[2])>0.0001 ? (topv[2] - BoundaryCond["minZ"]) / (BoundaryCond["maxZ"] - BoundaryCond["minZ"])*255 : 0 ;
      p[ int(topv[1]*3+1) ] = abs(topv[2])>0.0001 ? topv[2]/tmpv_1 - tmpv : 0 ;
      p[ int(topv[1]*3+2) ] = std::min(255.0, tmpv_2*log(loopx+1));
    }
  }
  return img;
}

std::tuple<V1I, V3F> label2BevImage_v4( float score_thresh, ANNORET& annoret)
{
  V3F ret;
  V1I retlab;
  for(unsigned int i=0; i<annoret.score.size(); i++) {
    // std::cout <<" label2BevImage_v4v:   " << annoret.score[i] << "  " << score_thresh <<" \n";
    if (annoret.score[i] < score_thresh) {
      continue;
    }
    retlab.emplace_back( annoret.label[i] );
    float box_x = annoret.box3d_lidar[i][0];
    float box_y = annoret.box3d_lidar[i][1];
    float box_h = annoret.box3d_lidar[i][3];
    float box_w = annoret.box3d_lidar[i][4];
    float rot_angle = annoret.box3d_lidar[i][6];
    int img_x = std::floor(box_x / Resolution);
    int img_y = std::floor(box_y / Resolution) + BR_Width/2;
    int img_w = std::floor(box_w / Resolution);
    int img_h = std::floor(box_h / Resolution);
    float angle = -rot_angle - 3.14159265/2;

    V2F offset={ {float(0.5*img_w), float(-0.5*img_w), float(-0.5*img_w), float(0.5*img_w) },
                 {float(0.5*img_h), float(0.5*img_h),  float(-0.5*img_h), float(-0.5*img_h) }};

    V2F rotZ_2d_angle{ { cos(angle), -sin(angle)},  {sin(angle), cos(angle) } };

    V2F offset_1(2, V1F(4,0));
    for(int ii=0 ;ii<2; ii++) {
      for(int j=0; j<4; j++) {
        for(int k=0; k<2; k++) {
          offset_1[ii][j] += rotZ_2d_angle[ii][k]*offset[k][j];
        }
      }
    }

    V2F offset_2={ {float(img_x), float(img_x), float(img_x), float(img_x)},
                   {float(img_y), float(img_y), float(img_y), float(img_y) }};
    V2F corners(2, V1F(4,0));
    for(int ii=0 ;ii<2; ii++) {
      for(int j=0; j<4; j++) {
        corners[ii][j] = offset_1[ii][j] + offset_2[ii][j];
      }
    }
    V2F corners_t(4, V1F(2, 0));
    for(int ii=0 ;ii<4; ii++) {
      for(int j=0; j<2; j++) {
        corners_t[ii][j] = corners[j][ii];
      }
    }
    ret.emplace_back(corners_t);
  }
  return std::make_tuple( retlab, ret);
}

void drawRotBboxes(cv::Mat& img, const V3F& bboxes, V1I& vlab) // [N 4 2]
{
  cv::Point pt1, pt2;
  for (unsigned int i=0; i<bboxes.size(); i++) {
    for(unsigned int j=0; j<bboxes[0].size(); j++) {
       pt1.x = bboxes[i][j][1];
       pt1.y = bboxes[i][j][0];
       pt2.x = bboxes[i][(j+1)%bboxes[0].size()][1];
       pt2.y = bboxes[i][(j+1)%bboxes[0].size()][0];
       cv::line(img, pt1, pt2,
                 vscalar[ vlab[i] ] );
    }
    pt1.x = (bboxes[i][0][1] + bboxes[i][2][1])/2.0 ;
    pt1.y = (bboxes[i][0][0] + bboxes[i][2][0])/2.0 ;
    pt2.x = (bboxes[i][0][1] + bboxes[i][3][1])/2.0 ;
    pt2.y = (bboxes[i][0][0] + bboxes[i][3][0])/2.0 ;
    cv::line(img, pt1, pt2, vscalar[ vlab[i]]);
  }
}

V2F matmul_2x2f(const V2F& in1, const V2F& in2)
{
  V2F ret(in1.size(), V1F(in2[0].size(), 0));

  for(unsigned int i=0; i<in1.size(); i++) {
    for( unsigned int j=0; j<in2[0].size(); j++) {
      for(unsigned int k=0; k<in1[0].size(); k++) {
        ret[i][j] += in1[i][k] * in2[k][j];
      }
    }
  }
  return ret;
}

V3F matmul_3x2f(const V3F& in1, const V2F& in2)
{
  V3F ret(in1.size(), V2F( in1[0].size(), V1F( in2[0].size(),0 )));
  for(unsigned int i=0; i<in1.size(); i++) {
    ret[i] = matmul_2x2f(in1[i], in2);
  }
  return ret;
}

V2F box_lidar_to_camera(const V2F& data, const V2F& r_rect, const V2F& velo2cam)
{
   // lidar_to_camera
   //   unsigned int num_points = data.size();
   // make points
   if (data.empty() ) {
       return V2F{};
   }
   V2F points(data.size(), V1F(4,1));
   for(unsigned int i=0; i<data.size(); i++) {
     for(unsigned int j=0; j<3; j++) {
       points[i][j]=data[i][j];
     }
   }

   V2F mid = matmul_2x2f(r_rect, velo2cam);
   V2F mid_t = dim_transpose(mid);
   V2F mid_2 = matmul_2x2f(points, mid_t);

   for(int unsigned i=0; i<mid_2.size(); i++) {
     mid_2[i] = V1F{mid_2[i][0], mid_2[i][1], mid_2[i][2], data[i][4], data[i][5], data[i][3], data[i][6]};
   }
   return mid_2;
}

V2F dim_transpose(const V2F& in)
{
  V2F ret(in[0].size(), V1F(in.size(), 0));
  for(unsigned int i=0; i<in.size(); i++) {
    for(unsigned int j=0; j<in[0].size(); j++) {
      ret[i][j] = in[j][i];
    }
  }
  return ret;
}

V3F corners_nd_3d(const V2F& dims)
{
  if (dims.empty()) {
    return V3F{};
  }

  static V2F corners_norm;
  if (corners_norm.empty()) {
      V1F origin{0.5, 1.0, 0.5};
      V1I var1{0,1,2,3,4,5,6,7};
      V1I var2{2,2,2};
      corners_norm = unravel_index_3d( var1, var2);

      corners_norm[2].swap(corners_norm[3]);
      corners_norm[6].swap(corners_norm[7]);

      for(unsigned int j=0; j<corners_norm.size(); j++) {
        for(unsigned int k=0; k<corners_norm[0].size(); k++) {
          corners_norm[j][k] = corners_norm[j][k] - origin[k];
        }
      }
  }
  V3F corners(dims.size(),
              V2F(corners_norm.size(),
                V1F(corners_norm[0].size(), 0 ) ) ) ;

  for(unsigned int i=0; i<corners.size(); i++) {
    for(unsigned int j=0; j<corners_norm.size(); j++) {
      for(unsigned int k=0; k<corners_norm[0].size(); k++) {
        corners[i][j][k] = dims[i][k] * corners_norm[j][k] ;
      }
    }
  }
  return corners;
}

// dims has 3 elements
V2F unravel_index_3d(const V1I& index, const V1I& dims )
{
  V2F ret(index.size(), V1F(3,0) );
  for(unsigned int i=0; i<index.size(); i++ ) {
    auto x = index[i]/(dims[1]*dims[2]);
    auto mid = index[i] - x*(dims[1]*dims[2]);
    auto y = mid/dims[2];
    auto z = mid - y*dims[2];
    ret[i][0]=x;
    ret[i][1]=y;
    ret[i][2]=z;
  }
  return ret;
}

//==========================================================================================

V2F roty(float  x )
{
  return V2F{
              { cos(x), 0, sin(x) },
              { 0,      1, 0 },
              {-sin(x), 0, cos(x) }
            };
}

// ibox:  xyz lhw a
V2F generate8CornersKitti( const V1F& ibox)
{
   V2F offset = {
      { 0.5,  0,  0.5 },
      {-0.5,  0,  0.5 },
      {-0.5, -1,  0.5 },
      { 0.5, -1,  0.5 },
      { 0.5,  0, -0.5 },
      {-0.5,  0, -0.5 },
      {-0.5, -1, -0.5 },
      { 0.5, -1, -0.5 }
   };
   for(unsigned int i=0; i<offset.size(); i++) {
      offset[i][0]*=ibox[3];
      offset[i][1]*=ibox[4];
      offset[i][2]*=ibox[5];
   }

   // dot () .T
   V2F roty_v = roty(ibox[6]) ;
   V2F corners(8, V1F(3, 0));

   for (int i=0; i<8; i++) {
     for (int j=0; j<3; j++) {
       for(int k=0; k<3; k++) {
          corners[i][j] += roty_v[j][k] * offset[i][k];
       }
       corners[i][j] += ibox[j];
       if (j==2 && corners[i][2]<0) {
          return V2F{};
       }
     }
   }

   return corners;
}

void draw_proj_box(const V1F& ibox, cv::Mat& img, const V2F& p2rect, int label)
{
  // std::cout <<"draw_proj_box :" << ibox[0] << " "  << ibox[1] << " " << ibox[2] << " " << ibox[3] << " " << ibox[4] << " " << ibox[5] << " " << ibox[6] << "\n";
  V2F corners = generate8CornersKitti(ibox);
  if (corners.empty()) {
     return;
  }
  for (int i=0; i<8; i++) {
     if (corners[i][2] < 0) {
        return;
     }
  }
  V2F proj_pts = project_3d_pts(corners, p2rect);
  draw_projected_box3d(img, proj_pts, label);

  // draw line for heading angle
  // ignore now ...
}

V2F project_3d_pts( V2F& corners, const V2F& p2rect)
{
   V2F proj_pts(8, V1F(4,0));

   for (int i=0; i<8; i++) {
     for (int j=0; j<4; j++) {
       for(int k=0; k<4; k++) {
          proj_pts[i][j] += p2rect[j][k]*( k==3? 1.0: corners[i][k]);
       }
     }
     proj_pts[i][0]/=proj_pts[i][2];
     proj_pts[i][1]/=proj_pts[i][2];
     proj_pts[i][2] =proj_pts[i][2];
   }
   return proj_pts;
}

void draw_projected_box3d(cv::Mat& img, const V2F& qs, int label)
{
   int thickness = 1;
   for (int k=0; k<4; k++) {
      std::vector<int> vi={k,        k+4,          k   };
      std::vector<int> vj={(k+1)%4,  (k+1)%4 + 4,  k+4 };
      for(int kk=0; kk<3; kk++) {
        cv::line(img,
                 cv::Point2f(int(qs[vi[kk]][0]), int(qs[vi[kk]][1])),
                 cv::Point2f(int(qs[vj[kk]][0]), int(qs[vj[kk]][1])),
                 // cv::Scalar(0,255,255),
                 vscalar[label],
                 thickness);
      }
   }
}

void draw_rgb( V2F& fcbox, DISPLAY_PARAM& g_test,  cv::Mat& img, V1I& label )
{
   for(unsigned int i=0; i<fcbox.size(); i++) {
      draw_proj_box(fcbox[i], img, g_test.p2rect, label[i]);
   }
}

