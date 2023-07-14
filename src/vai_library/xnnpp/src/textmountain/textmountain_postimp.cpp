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

#include <fstream>
#include <iostream>
#include <queue>
#include <sys/stat.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/math.hpp>
#include "textmountain_postimp.hpp"

// env parameter control: if filtering the result in postprocess or outside.
DEF_ENV_PARAM(XLNX_TEXTMOUNTAIN_FILTERIN, "1");
DEF_ENV_PARAM(XLNX_TEXTMOUNTAIN_IMPORT_POST, "0");

using namespace cv;
using namespace std;

namespace vitis {
namespace ai {

TextMountainPostImp::TextMountainPostImp(
    const vector<vitis::ai::library::InputTensor>& input_tensors,
    const vector<vitis::ai::library::OutputTensor>& output_tensors,
    int batch_size,
    int& real_batch_size,
    float* scale_h,
    float* scale_w
  ) :
      input_tensors_ (input_tensors),
      output_tensors_(output_tensors),
      batch_size_(batch_size),
      real_batch_size_(real_batch_size),
      scale_h_(scale_h),
      scale_w_(scale_w)
{
  XLNX_TEXTMOUNTAIN_FILTERIN = ENV_PARAM(XLNX_TEXTMOUNTAIN_FILTERIN);
  XLNX_TEXTMOUNTAIN_IMPORT_POST = ENV_PARAM(XLNX_TEXTMOUNTAIN_IMPORT_POST);
  scale_o =  tensor_scale(output_tensors_[0]);
  output_w = output_tensors_[0].width;   // 240
  output_h = output_tensors_[0].height;  // 240
  output_c = output_tensors_[0].channel; // 5

  thres_center = (-log(1.0/thres_center -1.0 ))/scale_o;
  std::vector<float> pred0_sft_tmp(output_w*output_h);
  pred0_sft_tmp.swap(pred0_sft);
}

TextMountainPostImp::~TextMountainPostImp() {}

vector<TextMountainResult> TextMountainPostImp::process() {
  auto ret = vector<vitis::ai::TextMountainResult>{};
  ret.reserve(real_batch_size_);
  for (auto i = 0; i < real_batch_size_; ++i) {
    ret.emplace_back(process(i));
  }
  return ret;
}

void TextMountainPostImp::maxpool( int8_t* input, 
            std::vector<int>& output , 
            cv::Mat& shrink_scores ){
  const int idx_pos = 2;
  int8_t max_v, tmp_v;
  int max_y, max_x;
  for (auto i = 0; i < output_h; ++i) {
    for (auto j = 0; j < output_w; ++j) {
      max_y = max_x = 0;
      max_v = -128;
      for (int di = 0; di < kernel_size; di++) {
        for (int dj = 0; dj < kernel_size; dj++) {
          auto input_h_idx = ((i - 1) + di);
          auto input_w_idx = ((j - 1) + dj);
          if (    input_w_idx < 0 
               || input_h_idx < 0 
               || input_h_idx >= output_h 
               || input_w_idx >= output_w) {
            continue;
          }
          tmp_v = input[input_h_idx * output_w * output_c  + input_w_idx*output_c + idx_pos ];
          if ( max_v < tmp_v ){
             max_v = tmp_v;
             max_x = input_w_idx;
             max_y = input_h_idx;
          }
        }
      }
      output[(i*output_w+j)*2 + 0] = max_x;
      output[(i*output_w+j)*2 + 1] = max_y;
      // std::cout <<"maxpool : " << max_x << "  " << max_y << "\n"; 
      if (input[i*output_w*output_c +j*output_c+idx_pos] > thres_center) {
         shrink_scores.ptr<uchar>(i)[j] = 255;
      }
    }
  }
  // cv::imwrite("textmount_shrinkscore.jpg", shrink_scores );
}

template <typename T>
static void my_softmax(T* input, float scale, unsigned int cls_real, unsigned int cls,
                      unsigned int group, float* output) {
  float sum = 0.0;
  float output_f[2];
  // std::cout <<"softmax:\n";
  for (unsigned int k = 0; k < group; ++k) {
    sum = 0.f;
    for (unsigned int i = 0; i < cls_real; ++i) {
      output_f[i] = exp(input[i] * scale);
      sum += output_f[i];
    }
    output[k] = output_f[1]/ sum;  // we don't care for background's softmax's value
    // std::cout << output[k] <<"\n";
    input += cls;
  }
}

void TextMountainPostImp::get_cc_eval(cv::Mat& img,   
                 std::vector<float>& pred0_sft,
                 std::vector<bool>& valid,
                 cv::Mat& score_i,
                 std::vector<float>& score_mean) {
  /*
     groupSoftmax() -->   score[:1:], score_i
       // score is pred0 which is  mpool's second part;  score_i is label.
     CropAndResizeFunction.forward --> image, boxID_ptr
     crop_and_resize_gpu.forward()--> image, group_sum, groupNumsum, boxID_ptr
     crop_and_resize_gpu_forward() --> same as above
     CropAndResizeLaucher()--> image.data<float>, batch_size, image_height, image_width
        depth  group_sum.data<float>  groupNumsum.data<float>, boxID_ptr.data<int>
     cropAndResizeKernel() --> total_count,  image_ptr, group_sum, groupNumsum, boxID_ptr,
        batch, image_height image_width
  */

  int label_num = cv::connectedComponents( img , score_i,  4, CV_32S ); 
  std::vector<float> group_sum(label_num, 0);
  std::vector<float> groupNumsum(label_num, 0);

  int  boxID=0;
  for(int i=0; i<output_h; i++) {
    for(int j=0; j<output_w; j++) {
      boxID = score_i.ptr<int32_t>(i)[j];
      if(boxID != 0) {
         groupNumsum[boxID]++;
         group_sum[boxID] += pred0_sft[ i*output_w + j ];
      }
    }
  }
  // score_sum is group_sum, score_num is groupNumsum
  // std::cout <<"score_mean valid " << "\n";
  for(int i=1; i<(int)group_sum.size(); i++) {
     score_mean.emplace_back( group_sum[i]/std::max( groupNumsum[i], float(1e-10) ) );
     valid.emplace_back( score_mean[i-1] > score_thres && groupNumsum[i] > 5 ) ;
     // std::cout << score_mean[i-1] << " " << valid[i-1] << "\n";
  }
}

void TextMountainPostImp::groupSearch(
           std::vector<std::pair<int,int>>& points_ptr ,
           std::vector<int>& next_ptr,  // next is indices_2d also is mpool
           cv::Mat & instance_ptr,       // instance_ptr is score_i
           std::vector<float> pred0_sft  // prob_ptr is pred0_sft
     ) {
  /*
   groupSearch --> (points_ptr, indices_2d., score_i.,pred_0_thres)
   CropAndResizeFunction.forward() -->  (points_ptr, indices_2d., score_i.,pred_0_thres,circle_ptr
   crop_and_resize_gpu_forward -->  same as above
       points_ptr,  next_ptr, instance_ptr, prob_ptr,  circle_ptr
   CropAndResizeLaucher  -->  above + batch_size, image_height, image_width,points_num
         const int *points_ptr,  const int *next_ptr, int *instance_ptr,const int *prob_ptr,int
         *circle_ptr,int batch, int image_height, int image_width,int points_num
   CropAndResizeKernel
  */

  int next_x;
  int next_y;
  int instance_idx;
  int next_xx;
  int next_yy;
  int points_num = points_ptr.size();
  std::vector<bool> circle_ptr(output_h*output_w , true);

  for(int i=0; i<points_num; i++) {
    int num_search=1;
    int x=points_ptr[i].first;
    int y=points_ptr[i].second;
    next_x=next_ptr[ (y*output_w+x )*2+0];
    next_y=next_ptr[ (y*output_w+x )*2+1];

    instance_idx=instance_ptr.ptr<int32_t>(next_y)[next_x];

    while(instance_idx==0){
        if (num_search>(points_num+3)){
            circle_ptr[y*output_w+x]=false;
            circle_ptr[next_y*output_w+next_x]=false;
            break;
        }
        num_search=num_search+1;

        if (   circle_ptr[next_y*output_w+next_x]==false
            || (next_x==x && next_y==y )
            || pred0_sft[next_y*output_w+next_x] <= score_thres_pixel){
            circle_ptr[y*output_w+x]=false;
            break;
        }
        next_xx=next_x;
        next_yy=next_y;
        next_x=next_ptr[ (next_yy*output_w+next_xx)*2 + 0 ];
        next_y=next_ptr[ (next_yy*output_w+next_xx)*2 + 1 ];

        instance_idx=instance_ptr.ptr<int32_t>(next_y)[next_x];
    }
    if (instance_idx>0){
        instance_ptr.ptr<int32_t>(y)[x] = instance_idx;
    }
  }
  // cv::imwrite("textmount_groupsearch.jpg", instance_ptr);
}

template<typename T>
void myreadfile(T* conf, int size1, std::string filename)
{ ifstream Tin;  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {      cout<<"Can't open the file!";      return ;  }
  Tin.read( (char*)conf, size1*sizeof(T));
}

bool filtered( Point2d& in1, Point2d& in2) {
   return std::sqrt(std::pow(in1.x-in2.x, 2)+ std::pow(in1.y-in2.y, 2)) <5.0;
}

void TextMountainPostImp::fix_scale(Point2f* vertices, Point2d* dest, int idx){
  for(int i=0;i<4;i++) {
     dest[i].x = int(vertices[i].x/scale_w_[idx]);
     dest[i].y = int(vertices[i].y/scale_h_[idx]);
  }
}

TextMountainResult TextMountainPostImp::process(int idx) {

  TextMountainResult result{int(input_tensors_[0].width), int(input_tensors_[0].height) };

  int8_t* p = (int8_t*)output_tensors_[0].get_data(idx);  (void)p;
  if (XLNX_TEXTMOUNTAIN_IMPORT_POST) {
     myreadfile( p , 288000, "/home/root/pp/ttm_output.bin");
  }
  // 1. softmax
  __TIC__(mysoftmax)
  my_softmax(p, scale_o, 2, output_c, output_w*output_h,  pred0_sft.data());
  __TOC__(mysoftmax)

  // 2. sigmoid  : ignore; optimized out;

  // 3. postprocess
  // 3.1 maxpool  && shrink_scores
  cv::Mat shrink_scores(output_h, output_w, CV_8UC1, cvScalar(0));
  std::vector<int> mpool(output_w*output_h*2, 0);
  __TIC__(maxpool)
  maxpool(p, mpool, shrink_scores);
  __TOC__(maxpool)

  // 3.2 get_cc_eval & groupmean
  cv::Mat score_i ;
  std::vector<bool> valid;
  std::vector<float> score_mean;
  __TIC__(get_cc_eval)
  get_cc_eval(shrink_scores, pred0_sft , valid, score_i, score_mean);
  __TOC__(get_cc_eval)

  __TIC__(points_ptr)
  std::vector<std::pair<int,int>> points_ptr;
  points_ptr.reserve(20000);

  for(int i=0; i<output_h; i++) {
    for(int j=0; j<output_w; j++) {
       if (   pred0_sft[i*output_w + j ] >score_thres_pixel 
           && score_i.ptr<int32_t>(i)[j]==0 ){
          points_ptr.emplace_back( std::pair(j, i) );
          // std::cout <<"points : " << j << " " << i << "\n";
       }
    }
  }
  __TOC__(points_ptr)

  if (points_ptr.size()==0 ) {
     return result;
  }

  // 3.3 groupSearch()
  __TIC__(groupsearch)
  groupSearch( points_ptr, mpool, score_i , pred0_sft );
  __TOC__(groupsearch)

  // 3.4 image_idx_Tobox
  __TIC__(image_idx_tobox)
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  cv::Mat score_dst;
  score_i.convertTo(score_dst, CV_8U);

  cv::Mat mat1, mat2;
  RotatedRect box;
  Point2f vertices[4];
  Point2d vert_dest[4];
  for(int i=0; i<(int)valid.size(); i++) {
     if( !valid[i] ) continue;
     cv::threshold(score_dst, mat1, i+1, 0, THRESH_TOZERO_INV );
     cv::threshold(mat1, mat2, i, 0, THRESH_TOZERO );

     cv::findContours(mat2, 
                   contours, 
                   hierarchy,
                   cv::RETR_TREE,
                   cv::CHAIN_APPROX_SIMPLE);
     // std::cout <<"contours.size : " << i << "  " << contours.size() <<"\n";
     box = cv::minAreaRect( contours[0]);
     box.points(vertices);
     // for(int j=0; j<4; j++) std::cout << vertices[j].x << "  " << vertices[j].y << "\n";
     fix_scale(vertices, vert_dest, idx);
     if (   XLNX_TEXTMOUNTAIN_FILTERIN 
         && (filtered(vert_dest[0], vert_dest[1]) || filtered(vert_dest[3], vert_dest[0]))) {
           continue;
     }
     result.res.push_back(TextMountainResult::tmitem(vert_dest, score_mean[i])); 
  }
  __TOC__(image_idx_tobox)
  return result;
}

}  // namespace ai
}  // namespace vitis

