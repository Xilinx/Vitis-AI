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
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "./helper.hpp"
#include <vitis/ai/profiling.hpp>
#include "./pointpillars.hpp"
#include "./pointpillars_post.hpp"

using namespace std;
using namespace chrono;

namespace vitis { namespace ai { namespace pp {

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

// below is from another file;
V3F einsum(const V3F& points, const V3F& rot_mat_T);
V2F corner_to_standup_nd(const V3F& boxes_corner);

// below is new 3d draw logic;
static V2F roty(float 	x ) ;
static V2F generate8CornersKitti( const V1F& ibox) ;
static void draw_proj_box(const V1F& ibox, cv::Mat& img, const V2F& p2rect, int);
static V2F project_3d_pts( V2F& corners, const V2F& p2rect);
static void draw_projected_box3d(cv::Mat& img, const V2F& qs, int label);
static void draw_rgb( V2F& fcbox, DISPLAY_PARAM& g_test,  cv::Mat& img , V1I&);

void PointPillarsPost::do_pointpillar_display(PointPillarsResult & res, int flag, DISPLAY_PARAM& g_test, cv::Mat& rgb_map, cv::Mat& bev_map, int imgwidth, int imgheight, ANNORET& annoret)
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

void PointPillarsPost::predict_kitti_to_anno(V2F& final_box_preds_camera, V2F& final_box_preds, V2F& box_2d_preds, V1I& label_preds, V1F& final_scores, ANNORET& annoret, int image_width, int image_height)
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

V2F roty(float 	x ) 
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

}}}
