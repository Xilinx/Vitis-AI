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
#include <cctype>
#include <numeric>
#include <sys/stat.h>
#include <boost/algorithm/string.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/math.hpp>
#include "ocr_postimp.hpp"


DEF_ENV_PARAM(XLNX_OCR_GETWORD_THREAD, "2"); 
DEF_ENV_PARAM(XLNX_OCR_MARKERPOSLEN, "100"); 
// max length of word
DEF_ENV_PARAM(XLNX_OCR_IMPORT_POST, "0");
// 0: no fast;  1: fast (not for accuracy)
DEF_ENV_PARAM(XLNX_OCR_FAST_LEXICON, "1");  
DEF_ENV_PARAM(XLNX_OCR_POST_ROUND, "0");

#define MIN_WORD_LEN 3
#define MAX_WORD_LEN 25

using namespace cv;
using namespace std;
namespace vitis {
namespace ai {

inline float sigmoid(float x) {
  return 1.0/(1.0+exp(-x) );
}

template<typename T>
void myreadfile(T* conf, int size1, std::string filename){
  ifstream Tin;  
  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {      cout<<"Can't open the file!";      return ;  }
  Tin.read( (char*)conf, size1*sizeof(T));
}

int getfilelen(const std::string& file)
{
  struct stat statbuf;
  if(stat(file.c_str(), &statbuf)!=0){
    std::cerr << " bad file stat " << file << std::endl;
    exit(-1);
  }
  return statbuf.st_size;
}


OCRPostImp::OCRPostImp(
    const vector<vitis::ai::library::InputTensor>& input_tensors,
    const vector<vitis::ai::library::OutputTensor>& output_tensors,
    const std::string& cfgpath, 
    int batch_size,
    int& real_batch_size,
    std::vector<int>& target_h8,
    std::vector<int>& target_w8,
    std::vector<float>& ratioh,
    std::vector<float>& ratiow,
    std::vector<cv::Mat>& oriimg_in
  ) :
      input_tensors_ (input_tensors),
      output_tensors_(output_tensors),
      cfgpath_(cfgpath),
      batch_size_(batch_size),
      real_batch_size_(real_batch_size),
      target_h8_(target_h8),
      target_w8_(target_w8),
      ratioh_(ratioh),
      ratiow_(ratiow),
      oriimg(oriimg_in)
{
  XLNX_OCR_GETWORD_THREAD   = ENV_PARAM(XLNX_OCR_GETWORD_THREAD);
  XLNX_OCR_IMPORT_POST      = ENV_PARAM(XLNX_OCR_IMPORT_POST);
  XLNX_OCR_FAST_LEXICON     = ENV_PARAM(XLNX_OCR_FAST_LEXICON);
  XLNX_OCR_POST_ROUND       = ENV_PARAM(XLNX_OCR_POST_ROUND);
  // 0: y2: 480x480x37   pred
  // 1: y:  240x240x2    textlink :: need sigmoid
  int Lsig = 0, Lsft = 1;
  if (output_tensors_[1].channel < output_tensors_[0].channel ) {
    Lsig = 1;
    Lsft = 0;
  }

  output_w = output_tensors_[Lsig].width;   // 240  240x240x2  480x480x37
  output_h = output_tensors_[Lsig].height;  // 240
  output_c = output_tensors_[Lsft].channel; // 38  -->37

  // std::cout <<" output whc : "  << output_w << " " << output_h << " " << output_c <<"\n";
  regions.resize(output_c);
  for(int i=0; i<output_c; i++) {
    regions[i].reserve( ENV_PARAM(XLNX_OCR_MARKERPOSLEN));
  }

  prepare_lexicon();

  scale_o_sig = tensor_scale( output_tensors_[Lsig] );  // sigmoid : 0.25
  scale_o_sft = tensor_scale( output_tensors_[Lsft] );  // softmax : 0.25

  low_text           = (-log(1.0/low_text          -1.0))/scale_o_sig;
  link_threshold     = (-log(1.0/link_threshold    -1.0))/scale_o_sig;
  text_threshold     = (-log(1.0/text_threshold    -1.0))/scale_o_sig;
  textmap_thresh_low = (-log(1.0/textmap_thresh_low-1.0))/scale_o_sig;

  // std::cout <<"low_text link_threshold text_threshold:" << low_text << " " << link_threshold <<" " << text_threshold <<"\n";

  p_textlink.resize(batch_size_);
  p_pred.resize(batch_size_);

  softmax_data.resize( output_w*output_h*output_c);
  softmax_data_src.resize( output_w*output_h*output_c);
  for(int i=0; i<batch_size_; i++) {
     p_pred[i] = (int8_t*)output_tensors_[Lsft].get_data(i);  //   480 480 37
     p_textlink[i] = (int8_t*)output_tensors_[Lsig].get_data(i);  //  240 240 2
  }
  box4.resize(4);
  vth_label.reserve(XLNX_OCR_GETWORD_THREAD);
}

OCRPostImp::~OCRPostImp() {}

std::string OCRPostImp::match_lexicon(const std::string& intext){

    int min_dist = 2, min_idx=0;
    int l = intext.size();
    int len[]={0, -1 , 1};
    int k_idx = 0;
    for(int k=0; k<3; k++) { 
      for(int i=0; i<(int)lexiconL[l+len[k]].size(); i++) {
        int ed = editdistance_eval( intext, lexiconL[l+len[k]][i].first);
        if (ed==0) {
           return intext;
        }

        if( ed == 1 && ( min_dist>1 || ( XLNX_OCR_FAST_LEXICON==0 &&  lexiconL[l+len[k]][i].second < lexiconL[k_idx][min_idx].second ))
          ) {
           min_dist = 1;
           min_idx = i;
           k_idx = l+len[k];
        }

      }
    }
    return  min_dist ==1 ? lexiconL[k_idx][min_idx].first : "";
}

void OCRPostImp::prepare_lexicon(){
  lexiconL.resize(MAX_WORD_LEN);
  ifstream Tin;
  std::string str;
  Tin.open(cfgpath_, ios_base::in);
  if(!Tin)  {
     std::cout<<"Can't open the file " << cfgpath_ << "\n";      exit(-1);
  }
  int pos=0;
  while( getline(Tin, str)) {
     lexiconL[str.size()].emplace_back( std::make_pair(str,pos));
     pos++;
  }
  Tin.close();
}

float OCRPostImp::linalgnorm(const cv::Mat& b0, const cv::Mat& b1) {
  float vx = b0.ptr<float>(0)[0] - b1.ptr<float>(0)[0];
  float vy = b0.ptr<float>(0)[1] - b1.ptr<float>(0)[1];
  return std::sqrt( std::pow(vx,2) + std::pow(vy,2));
}

vector<vector<cv::Point2f>> OCRPostImp::getDetBoxes(int idx, std::vector<std::string>& words) {
  vector<vector<cv::Point2f>> ret;  

  int img_h=target_h8_[idx], img_w=target_w8_[idx];
  cv::Mat mat_text(img_h, img_w, CV_8UC1, cvScalar(0));
  cv::Mat mat_link(img_h, img_w, CV_8UC1, cvScalar(0));
  cv::Mat mat_comb(img_h, img_w, CV_8UC1, cvScalar(0));

  __TIC__(getDetBox_init_textlink)

  for(int i=0; i<target_h8_[idx]; i++) {
    for(int j=0; j<target_w8_[idx]; j++)  {
       if ( p_textlink[idx][ i*output_w*2 + j*2+0 ]> low_text ) {
          mat_text.ptr<uchar>(i)[j] = 1;  
          mat_comb.ptr<uchar>(i)[j] = 1;  
       }
       if ( p_textlink[idx][ i*output_w*2 + j*2+1 ] > link_threshold ) {
          mat_link.ptr<uchar>(i)[j] = 1;  
          mat_comb.ptr<uchar>(i)[j] = 1;  
       }
    }
  }
  __TOC__(getDetBox_init_textlink)

  cv::Mat labels, stats, centroids_nouse;
  int nccomps = cv::connectedComponentsWithStats (
                  mat_comb, 
                  labels,    
                  stats,
                  centroids_nouse, 
                  4 );
  // std::cout <<" nccomps : " << nccomps << "\n";
  if (nccomps < 2) {
    return vector<vector<cv::Point2f>>{};
  }

  __TIC__(getDetBox_get_labels)  // 27ms  --> 9ms
  vector<vector<std::pair<int,int>>> vlabels = get_labels(idx, labels, nccomps);
  __TOC__(getDetBox_get_labels)

  __TIC__(getDetBox_inner_loop_outside)

  // std::cout <<"nccomps : " << nccomps <<"\n";
  for(int k=1; k<nccomps; k++) {
    __TIC__(getDetBox_inner_loop_each)

    auto size = stats.at<int>(k, cv::CC_STAT_AREA);
    // std::cout <<"k size : " << k << " " << size <<  "          nccompts :" << nccomps << "\n";
    if ( (int)size<10 || check_max_less_thresh(idx, vlabels, k)) {
       // std::cout <<" continue meet in loop \n";
       __TOC__(getDetBox_inner_loop_each)
       continue;
    }

    // make segmentation map
    int x = stats.at<int>(k,cv::CC_STAT_LEFT);
    int y = stats.at<int>(k,cv::CC_STAT_TOP);
    int w = stats.at<int>(k,cv::CC_STAT_WIDTH);
    int h = stats.at<int>(k,cv::CC_STAT_HEIGHT);

    int niter = int(std::sqrt(size * std::min(w, h) *1.0 / (w * h * 1.0)) * 2.0);
    int sx = std::max(x - niter, 0);
    int ex = std::min(x + w + niter + 1, img_w);
    int sy = std::max(y - niter, 0);
    int ey = std::min(y + h + niter + 1, img_h);

    if (0) {  
       std::cout <<"xywh: " << x << " " << y << " " << w << " " << h
                 << "     " << niter << "     " 
                 << sx << " " << ex << " " << sy << " " << ey << "\n";
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1+niter, 1+niter));
    
    // seg map
    cv::Mat segmap(img_h, img_w, CV_8UC1, cv::Scalar(0));
    // segmap[labels==k] = 255
    // segmap[np.logical_and(link_score==1, text_score==0)] = 0 
    for(unsigned int i=0; i<vlabels[k-1].size(); i++) {
       if (!(   mat_text.ptr<uchar>( vlabels[k-1][i].first)[vlabels[k-1][i].second] == 0  
             && mat_link.ptr<uchar>( vlabels[k-1][i].first)[vlabels[k-1][i].second] == 1 )) {
          segmap.ptr<uchar>( vlabels[k-1][i].first)[vlabels[k-1][i].second ] = 255;
       }
    }
    if(0){
       int loopx = 0;
       for(int i=0; i<img_h; i++) { for(int j=0; j<img_w; j++)  {
            if (segmap.ptr<uchar>( i)[j] != 0 ) { loopx++; }
       } }
    }
    // segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
    cv::Rect rect(sx, sy, ex-sx, ey-sy );
    cv::Mat mat_dilate(segmap(rect)); 
    cv::dilate(mat_dilate, mat_dilate, kernel);

    int max_contours = 4000;
    while(1) {
        bool bmax = false;
        np_contoursx =  cv::Mat::zeros(max_contours, 2, CV_32F);  // 5000 is enough !
        np_contoursx_size = 0;
        for(int i=0; i<img_h; i++)   {
          for(int j=0; j<img_w; j++)  {
             if (segmap.ptr<uchar>( i )[j] != 0 ) {
                if (np_contoursx_size == max_contours) { 
                  // std::cout <<" np_contoursx_size exceed " << max_contours << " ! \n";
                  bmax = true;
                  max_contours+=2000;
                  goto out; 
                }
                np_contoursx.ptr<float>(np_contoursx_size)[0] = j; // std::cout << j << " " ;
                np_contoursx.ptr<float>(np_contoursx_size)[1] = i; // std::cout << i << "\n";
                np_contoursx_size++;
             }
          }
        }
   out: 
        if (!bmax) break;
    }
    // std::cout << " np_contoursx_size :" << np_contoursx_size <<"\n";
    cv::Mat np_contours = np_contoursx(cv::Rect(0, 0, 2, np_contoursx_size ));
    // for(int i=0; i<np_contoursx_size; i++) { std::cout <<  np_contours.ptr<float>(i)[0] << "  " <<  np_contours.ptr<float>(i)[1]  <<"\n"; }
    cv::RotatedRect rectangle = cv::minAreaRect(np_contours);

    cv::Mat box;
    cv::boxPoints(rectangle, box); 
    // std::cout <<"box ;" <<  box.rows << " " << box.cols << "   type : " <<  "\n";
    // for (int i=0; i<4; i++) { std::cout <<"boxPoints: " << 1.0*box.ptr<float>(i)[0] <<  "  " <<  1.0*box.ptr<float>(i)[1]  <<"\n"; }
    float ww = linalgnorm(box.row(1), box.row(2));
    float hh = linalgnorm(box.row(2), box.row(3));
    float box_ratio = std::max(ww, hh) / (std::min(ww, hh) + 0.00005);

    if( std::abs(1 - box_ratio) <= 0.1) {
      // std::cout <<"abs(<=0.1) : ww hh : " << ww << "  " << hh << "   ration :" << box_ratio << "\n" ;  // 33 cases run here ! need further debug .
      float min0 = 10000.0, max0 = 0.0, min1=10000.0, max1=0.0;
      for(int i=0; i<np_contoursx_size; i++) {
         auto x0 = np_contours.ptr<float>(i)[0], x1 = np_contours.ptr<float>(i)[1];
         if ( x0 < min0 ) min0 = x0;
         if ( x0 > max0 ) max0 = x0;
         if ( x1 < min1 ) min1 = x1;
         if ( x1 > max1 ) max1 = x1;
      }
      float l=min0, r=max0, t=min1, b=max1;
      // std::cout <<" lrtb :" << l << " " << r << " " << t << " " << b << "\n";
      box4[0] = cv::Point2f(l, t); 
      box4[1] = cv::Point2f(r, t);
      box4[2] = cv::Point2f(r, b);
      box4[3] = cv::Point2f(l, b);
      //  {r, t}, {r, b}, {l, b}};
    } else {
      for(int i=0;i<4; i++) {
        box4[i] = cv::Point2f(box.ptr<float>(i)[0], box.ptr<float>(i)[1]); 
      }
    }
    std::vector<cv::Point2f> bxcw = clockwise(box4);

    // continue with new logic : pred_rec is softmax_data
    cv::Mat mask( target_h8_[idx], target_w8_[idx], CV_8UC1, cv::Scalar(0));

    vector<cv::Point2i> boxi;
    boxi = vector<cv::Point2i>{ 
                cv::Point2i((int)bxcw[0].x,  (int)bxcw[0].y),
                cv::Point2i((int)bxcw[1].x,  (int)bxcw[1].y),
                cv::Point2i((int)bxcw[2].x,  (int)bxcw[2].y),
                cv::Point2i((int)bxcw[3].x,  (int)bxcw[3].y) };
  
    cv::fillPoly(mask, vector<vector<cv::Point2i>>{boxi}, cv::Scalar(1) );
    // tmp_region_score = textmap*mask  # get max here !  TODO

    __TIC__(watershed)
    bool balpha=true;
    std::string rets = post_watershed( mask, balpha, idx);
    __TOC__(watershed)
    __TIC__(watershed_after)
    
    if (!rets.empty()) {
       // std::cout <<" before match: " << rets <<"\n";
       if (balpha) { 
          rets = match_lexicon(rets);  
       }
       // std::cout <<" after  : " << rets <<"\n";
       if (!rets.empty()) {
          words.emplace_back(rets);
          ret.emplace_back( std::move(bxcw));
       }
    }
    __TOC__(watershed_after)
    __TOC__(getDetBox_inner_loop_each)
  }
  __TOC__(getDetBox_inner_loop_outside)
  return ret;
}

std::string OCRPostImp::post_watershed( cv::Mat& mask,  bool& balpha, int idx  ){
  cv::Mat sure_bg( target_h8_[idx], target_w8_[idx], CV_8UC1, cv::Scalar(0));
  cv::Mat sure_fg( target_h8_[idx], target_w8_[idx], CV_8UC1, cv::Scalar(0));

  int max_fg=-100;
  for(int i=0; i< target_h8_[idx]; i++){
    for(int j=0; j< target_w8_[idx]; j++){
       if (  mask.ptr<uchar>(i)[j] ) {
          int8_t v = p_textlink[idx][ i*output_w*2 + j*2+0 ];
          if ( v > textmap_thresh_low) {
            sure_bg.ptr<uchar>(i)[j] = 255;
            if (v > max_fg ) {
                max_fg = v;
            }
          } 
       }// if(mask
    }// for j
  }// for i
  // std::cout << "\n max_fg " << max_fg << "  "  << textmap_thresh_low 
  //          << "  sig  " << sigmoid(max_fg*scale_o_sig)  << "  " << scale_o_sig << " \n";

  float tmp_high = sigmoid(max_fg*scale_o_sig)*textmap_thresh_high;
  float tmp_thresh_high = (-log(1.0/tmp_high-1.0))/scale_o_sig;

  // std::cout << "\n max_fg " << max_fg << "  "  << tmp_thresh_high << "\n";
  for(int i=0; i< target_h8_[idx]; i++){
    for(int j=0; j< target_w8_[idx]; j++){
       if (  mask.ptr<uchar>(i)[j] ) {
          int8_t v = p_textlink[idx][ i*output_w*2 + j*2+0 ];
          if (v > tmp_thresh_high ) {
            sure_fg.ptr<uchar>(i)[j] = 255;
          }
       }
    }
  }

  if(0) {
    std::cout << " in watershed sure_fg : " << "\n";
    for(int ii=0; ii< target_h8_[idx]; ii++) {
      for(int jj=0; jj< target_w8_[idx]; jj++) {
        if (sure_fg.at<uchar>(ii,jj) != 0) {
           std::cout << ii << " " << jj << " " << "\n";
        }
      }
    }
  }

  cv::Mat unknow( target_h8_[idx], target_w8_[idx], CV_8UC1, cv::Scalar(0));
  cv::subtract(sure_bg, sure_fg, unknow);

  if(0) { // test OK
    std::cout <<" unknown  : \n";
    for(int ii=0; ii< target_h8_[idx]; ii++) {
      for(int jj=0; jj< target_w8_[idx]; jj++) {
        if (unknow.at<uchar>(ii,jj) != 0) {
           std::cout << ii << " " << jj << " " << "\n";
        }
      }
    }
  }

  cv::Mat labels;
  int label_num = cv::connectedComponents(sure_fg, labels, 4, CV_32S);
  if (label_num <= MIN_WORD_LEN) { // should be <=
    return "";
  }

  if(0) {
    std::cout <<"labels: " << label_num << " \n";  
    for(int ii=0; ii< target_h8_[idx]; ii++) {
      for(int jj=0; jj< target_w8_[idx]; jj++) {
        if (labels.at<int32_t>(ii,jj) != 0) {
           std::cout << ii << " " << jj << " " << labels.at<int32_t>(ii,jj)  << "\n";
        }
      }
    }
  }

  cv::Mat markers( target_h8_[idx], target_w8_[idx], CV_32SC1, cv::Scalar(0));
  for(int i=0; i< target_h8_[idx]; i++){
    for(int j=0; j< target_w8_[idx]; j++){
       if ( unknow.ptr<uchar>(i)[j]==255 ) {
          markers.ptr<int32_t>(i)[j] = 0; 
       } else {
          markers.ptr<int32_t>(i)[j] = labels.ptr<int32_t>(i)[j]+1;
          // std::cout << i << " " << j << " " << markers.ptr<int32_t>(i)[j] <<"\n";  // test OK
       }
    }
  }

  cv::watershed(oriimg[idx], markers);

  if (0) {  // 
    std::cout <<"markers after  water: \n";
    for(int ii=0; ii< target_h8_[idx]; ii++) {
      for(int jj=0; jj< target_w8_[idx]; jj++) {
        if (markers.at<int32_t>(ii,jj) != 1) {
           std::cout << ii << " " << jj << " " << markers.at<int32_t>(ii,jj)  << "\n";
        }
      }
    }
  }

  std::vector<cv::Mat> np_contours(label_num+1);
  std::vector<std::pair<float,char>> vword;

  int max_contours = 4000;
  for(int k=2; k<label_num+1; k++) {//for each char..a
    int np_contoursx_size = 0;
    np_contours[k] =  cv::Mat::zeros(max_contours, 2, CV_32S);  // 5000 is enough !
    for(int i=0; i< target_h8_[idx]; i++){
      for(int j=0; j< target_w8_[idx]; j++){
        if(markers.ptr<int32_t>(i)[j] == k ) {
           np_contours[k].ptr<int32_t>(np_contoursx_size)[0] = j;  
           np_contours[k].ptr<int32_t>(np_contoursx_size)[1] = i;  
           np_contoursx_size++;
           // std::cout << k << " "  << i << " " << j << " \n";
        }
      }// end for(j
    }//end for(i
    np_contours[k] = np_contours[k](cv::Rect(0, 0, 2, np_contoursx_size ));
    // std::cout <<" height : " << np_contours[k].rows << "\n";
  }//end for(k

  for(int i=2; i<label_num+1; i++) {
    if (np_contours[i].rows == 0) {
       continue;
    }
    cv::RotatedRect rect = cv::minAreaRect(np_contours[i]);
    cv::Mat box;
    cv::boxPoints(rect, box);   
    // std::cout <<" type : " << type2str(box.type()) <<"\n";  // 32FC1
    // std::cout <<"boxP: " << 1.0*box.ptr<float>(0)[0] <<  "  " <<  1.0*box.ptr<float>(0)[1]  << "  " <<  1.0*box.ptr<float>(0)[2] <<  "  " << 1.0*box.ptr<float>(0)[3] << "\n"; 

    float sx = (box.ptr<float>(0)[0] + box.ptr<float>(2)[0] )/2.0;
    // std::cout <<" sx : " <<i << "  " << sx << "\n";
    
    for(int ii=0; ii<output_c; ii++) {
      regions[ii].clear();
    }
    for(int j=0; j<output_c; j++) {
      for(int k=0; k<(int)np_contours[i].rows; k++) {
        // regions = seg[:, markers==i]*(textmap[markers==i]np.newaxis,:])
        // seg is pred_rec, which is y2:softmax 
        int h = np_contours[i].ptr<int32_t>(k)[1];
        int w = np_contours[i].ptr<int32_t>(k)[0];
        float tmp = softmax_data[ h*target_w8_[idx]*output_c+w*output_c+j]*  sigmoid(scale_o_sig*p_textlink[idx][ h*output_w*2 + w*2+0 ]) ;
        regions[j].emplace_back(tmp);
        // std::cout << " jkt : " << j << " " <<k << "    " << h << " " << w << " " <<  softmax_data[ h*output_w*output_c+w*output_c+j] << " " <<  sigmoid(scale_o_sig*p_textlink[idx][ h*output_w*2 + w*2+0 ])*255 << "  " <<  tmp*255 << "\n";
      } 
    }
    for(int j=0;j<output_c;j++) {
       double sum = std::accumulate(std::begin(regions[j]), std::end(regions[j]), 0.0);
       regions[j][0] = sum/regions[j].size();
    }
    // choose = np.argmax(np.mean(regions, axis=1)
    auto res = std::max_element(regions.begin(), regions.end(), 
                                [](const std::vector<float>& i, const std::vector<float>& j){ 
                                   return i[0]<j[0]; } 
    );
    int choose = std::distance(regions.begin(), res);
    //  std::cout <<"choose : " << choose <<  "   " << int(num2char(choose)) << "  " << sx << "\n";
    if(choose) { 
      if (balpha && choose<11) { 
          balpha = false;
      }
      vword.emplace_back(std::make_pair(sx, num2char(choose)));
    }
  }// end for(i
  // sort the word;
  // if (vword.size() <= MIN_WORD_LEN) {
  if (vword.size() < MIN_WORD_LEN) {  // should be <
    return "";
  }
  std::sort(vword.begin(), vword.end(),
              [](const std::pair<float, char>& lhs,
                 const std::pair<float, char>& rhs) {
                return lhs.first < rhs.first;
            });
  std::string strout;
  for(int i=0; i<(int)vword.size(); i++) {
     strout.append(1, vword[i].second);
  }
  // std::cout <<"strout  " << strout << "\n";
  return strout;
}

vector<cv::Point2f> OCRPostImp::clockwise( vector<cv::Point2f>& box){
  vector<cv::Point2f> ret;
  auto p = std::min_element( box.begin(), box.end(), 
                            []( const cv::Point2f&i, const cv::Point2f& j)
                               { return i.x+i.y<j.x+j.y; } ); 

  for(int i=0; i<4-(p-box.begin()); i++) {
    ret.emplace_back(*(p+i)); 
  }
  for(int i=0; i < p-box.begin(); i++) {
    ret.emplace_back( *(  box.begin()+i)); 
  }
  // for(int i=0; i<4; i++) { std::cout << "clock: " << ret[i].x << "  " << ret[i].y << "\n"; }
  return ret;
}

int OCRPostImp::editdistance_eval(const std::string& str1, const std::string& str2) {
    int len1 = str1.length();
    int len2 = str2.length();
    int DP[2][len1 + 1];
 
    memset(DP, 0, sizeof DP);
 
    for (int i = 0; i <= len1; i++)
        DP[0][i] = i;
 
    for (int i = 1; i <= len2; i++) {
        for (int j = 0; j <= len1; j++) {
            if (j == 0) {
                DP[i % 2][j] = i;
            } 
            else if (str1[j - 1] == str2[i - 1]) {
                DP[i % 2][j] = DP[(i - 1) % 2][j - 1];
            }
            else {
                DP[i % 2][j] = 1 + min(DP[(i - 1) % 2][j],
                                       min(DP[i % 2][j - 1],
                                           DP[(i - 1) % 2][j - 1]));
            }
        }
    }
    return DP[len2 % 2][len1];
}

bool OCRPostImp::check_max_less_thresh(int idx, const  vector<vector<std::pair<int,int>>>& k_pos, int k ) {
  float tmpf, maxf = 0.0;
  for(unsigned int i=0; i<k_pos[k-1].size(); i++){
    if((tmpf=p_textlink[idx][ k_pos[k-1][i].first *output_w*2 + k_pos[k-1][i].second *2 +0 ]) > maxf) {
       maxf = tmpf;
    }
  }
  return maxf < text_threshold;
}

vector<vector<std::pair<int,int>>> OCRPostImp::get_labels(int idx, const cv::Mat& labels, int nlabels){
  vector<vector<std::pair<int,int>>> k_pos;

  float size = nlabels-1;
  int real_thread_num =  XLNX_OCR_GETWORD_THREAD> nlabels? nlabels : XLNX_OCR_GETWORD_THREAD;
  vector<vector<vector<std::pair<int,int>>>> k_posall(real_thread_num, vector<vector<std::pair<int,int>>>(nlabels-1));

  vector<int> start(real_thread_num);
  vector<int> len(real_thread_num);
  int len_x = round(size/real_thread_num);
  for(int i=0; i<real_thread_num; i++) {
     start[i] = i * len_x;
     len[i] = (i != real_thread_num -1) ? len_x : (size- (len_x*(real_thread_num-1))) ;
     // std::cout <<" start len : " << start[i] << " " << len[i] << " \n";
     vth_label.emplace_back( std::thread( &OCRPostImp::get_labels_thread, this, start[i], len[i], idx, std::cref(labels), nlabels, std::ref(k_posall[i]) ));
  }
  for(int i=0; i<real_thread_num; i++) {
     vth_label[i].join();
  }
  vth_label.clear();

  // combine result;
  for(int i=0; i<real_thread_num; i++) {
    k_pos.insert(k_pos.end(), k_posall[i].begin()+start[i], k_posall[i].begin()+start[i]+len[i]);
  }

  // for(int i=0;i<11;i++) std::cout <<"k_pos[" << i << "] :" << k_pos[i].size() <<" \n";
  return k_pos;
}

void OCRPostImp::get_labels_thread(
        int start, 
        int len, 
        int idx, 
        const cv::Mat& labels, 
        int nlabels, 
        vector<vector<std::pair<int,int>>>& k_pos ){
  for(int h=0; h<target_h8_[idx]; h++){
    for(int w=0; w<target_w8_[idx]; w++){
       for(int k=start+1; k<start+1+len; k++) {
          if (labels.at<int>(h, w) == k ) {
             k_pos[k-1].emplace_back(std::make_pair(h, w));
          }
       }
    }
  }
}

vector<OCRResult> OCRPostImp::process() {
  auto ret = vector<vitis::ai::OCRResult>{};
  ret.reserve(real_batch_size_);
  for (auto i = 0; i < real_batch_size_; ++i) {
    ret.emplace_back(process(i));
  }
  return ret;
}

OCRResult OCRPostImp::process(int idx) {
  target_h8_[idx]/=4;  
  target_w8_[idx]/=4;

  if (XLNX_OCR_IMPORT_POST) {
     std::string filen="/home/root/pp/ocr_output0.bin";  
     myreadfile( p_textlink[0] , getfilelen(filen),  filen);
     filen = "/home/root/pp/ocr_output1.bin";
     myreadfile( p_pred[0] , getfilelen(filen),  filen);
     filen="/home/root/pp/oriimage.bin";  
     myreadfile( oriimg[idx].ptr(0) , getfilelen(filen),  filen);
  }

  __TIC__(softmax_memcpy)
  // std::cout <<"softmax-src \n";
  for(int i=0; i<target_h8_[idx]; i++) {
     for(int j=0; j<target_w8_[idx]; j++) {
        memcpy( softmax_data_src.data() +i*target_w8_[idx]*output_c +j*output_c, p_pred[idx]+ (i*2)*(output_w*2)*output_c +j*2*output_c, output_c ); 
        // for(int k=0; k<output_c; k++) { std::cout <<  *(p_pred[idx]+ (i*2)*(output_w*2)*output_c + j*2*output_c +k)*0.125 << " "; } std::cout <<"\n";
     }
  } 
  softmax_data_src.resize(target_w8_[idx]*target_h8_[idx]*output_c);
  softmax_data.resize(target_w8_[idx]*target_h8_[idx]*output_c);
  __TOC__(softmax_memcpy)
  __TIC__(softmax)
  
  vitis::ai::softmax((int8_t*)softmax_data_src.data(), scale_o_sft, output_c, target_h8_[idx]*target_w8_[idx], softmax_data.data());
  __TOC__(softmax)
 
  if (0) {
    std::cout <<"softmax-result \n"; // test good
    for(int i=0; i<target_h8_[idx]; i++) {
       for(int j=0; j<target_w8_[idx]; j++) {
          for(int k=0; k<output_c; k++) { std::cout << softmax_data[i*target_w8_[idx]*output_c+j*output_c+k] << " "; } std::cout <<"\n";
       }
    }
  } 

  vector<std::string> words;
  __TIC__(getDetBoxes)
  vector<vector<cv::Point2f>> boxes = getDetBoxes(idx, words);
  __TOC__(getDetBoxes)

  OCRResult result{int(input_tensors_[0].width), int(input_tensors_[0].height) };

  result.words.swap(words);
  for(unsigned int i=0; i<boxes.size(); i++) {
    result.box.emplace_back( 
      XLNX_OCR_POST_ROUND ?
        vector<cv::Point>{
           cv::Point(round(boxes[i][0].x*ratiow_[idx]*4),  
                     round(boxes[i][0].y*ratioh_[idx]*4)),
           cv::Point(round(boxes[i][1].x*ratiow_[idx]*4),  
                     round(boxes[i][1].y*ratioh_[idx]*4)),
           cv::Point(round(boxes[i][2].x*ratiow_[idx]*4),  
                     round(boxes[i][2].y*ratioh_[idx]*4)),
           cv::Point(round(boxes[i][3].x*ratiow_[idx]*4),  
                     round(boxes[i][3].y*ratioh_[idx]*4)) }
      : 
        vector<cv::Point>{
           cv::Point(int(boxes[i][0].x*ratiow_[idx]*4),  
                     int(boxes[i][0].y*ratioh_[idx]*4)),
           cv::Point(int(boxes[i][1].x*ratiow_[idx]*4),  
                     int(boxes[i][1].y*ratioh_[idx]*4)),
           cv::Point(int(boxes[i][2].x*ratiow_[idx]*4),  
                     int(boxes[i][2].y*ratioh_[idx]*4)),
           cv::Point(int(boxes[i][3].x*ratiow_[idx]*4),  
                     int(boxes[i][3].y*ratioh_[idx]*4)) } 
    );

    if (0) {
      for(unsigned int j=0; j<boxes[0].size(); j++) {
         boxes[i][j].x *= ratiow_[idx]*4 ;
         boxes[i][j].y *= ratioh_[idx]*4 ;
         std::cout << boxes[i][j].x << "  " << boxes[i][j].y << "  ratio : " << ratiow_[idx]*4  << " " << ratioh_[idx]*4  << "\n";
      }
    }
  }
  return result;
}

}  // namespace ai
}  // namespace vitis

