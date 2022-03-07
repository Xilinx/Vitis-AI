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

#include <fstream>
#include <iostream>
#include <queue>
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

DEF_ENV_PARAM(XLNX_OCR_VISUAL, "0");
DEF_ENV_PARAM(XLNX_OCR_GETWORD_THREAD, "2"); 
DEF_ENV_PARAM(XLNX_MATCH_EXTLEN, "0"); 
DEF_ENV_PARAM(XLNX_OCR_FULLSOFTMAX, "0"); 

using namespace cv;
using namespace std;
namespace vitis {
namespace ai {

inline float sigmoid(float x) {
  return 1.0/(1.0+exp(-x) );
}

OCRPostImp::OCRPostImp(
    const vector<vitis::ai::library::InputTensor>& input_tensors,
    const vector<vitis::ai::library::OutputTensor>& output_tensors,
    const std::string& cfgpath, 
    int batch_size,
    int& real_batch_size,
    vector<int>& target_h8,
    vector<int>& target_w8,
    vector<float>& ratioh,
    vector<float>& ratiow
  ) :
      input_tensors_ (input_tensors),
      output_tensors_(output_tensors),
      cfgpath_(cfgpath),
      batch_size_(batch_size),
      real_batch_size_(real_batch_size),
      target_h8_(target_h8),
      target_w8_(target_w8),
      ratioh_(ratioh),
      ratiow_(ratiow)
{
  XLNX_OCR_VISUAL = ENV_PARAM(XLNX_OCR_VISUAL);
  XLNX_OCR_GETWORD_THREAD = ENV_PARAM(XLNX_OCR_GETWORD_THREAD);
  XLNX_MATCH_EXTLEN = ENV_PARAM( XLNX_MATCH_EXTLEN);
  XLNX_OCR_FULLSOFTMAX = ENV_PARAM(XLNX_OCR_FULLSOFTMAX);

  // 0: y2: 480x480x38   pred
  // 1: y:  240x240x2    textlink :: need sigmoid
  output_w = output_tensors_[1].width;   // 240
  output_h = output_tensors_[1].height;  // 240
  output_c = output_tensors_[0].channel; // 38

  prepare_pairs();
  g_color = creat_color_map(output_c, 255);

  scale_o_0 = tensor_scale( output_tensors_[0] );  // y2: 0.125
  scale_o_1 = tensor_scale( output_tensors_[1] );  // 0.5

  low_text       = (-log(1.0/low_text       -1.0))/scale_o_1;
  link_threshold = (-log(1.0/link_threshold -1.0))/scale_o_1;
  text_threshold = (-log(1.0/text_threshold -1.0))/scale_o_1;

  // std::cout <<"low_text link_threshold text_threshold:" << low_text << " " << link_threshold <<" " << text_threshold <<"\n";

  p_textlink.resize(batch_size_);
  p_pred.resize(batch_size_);

  for(int i=0; i<batch_size_; i++) {
     if (XLNX_OCR_FULLSOFTMAX) {
        softmax_data.resize( output_w*2*output_h*2*output_c);
     } else {
        softmax_data.resize( output_w*output_h*output_c);
        softmax_data_src.resize( output_w*output_h*output_c);
     }
     p_pred[i] = (int8_t*)output_tensors_[0].get_data(i);  //  output_c, 320, 560  --> 480 480 38
     p_textlink[i] = (int8_t*)output_tensors_[1].get_data(i);  //  320, 560, 2 -->240 240 2
  }
  box4.resize(4);
  small_lexicon_dict.resize(XLNX_OCR_GETWORD_THREAD);
  vth_word.reserve(XLNX_OCR_GETWORD_THREAD);
  vth_label.reserve(XLNX_OCR_GETWORD_THREAD);
}

OCRPostImp::~OCRPostImp() {}


V2I OCRPostImp::creat_color_map(int ncls , int width){
  V2I ret;
  int splits = std::ceil( std::pow(ncls*1.0, 1.0/3 ));
  for(int i=0; i<splits; i++) {
    int r = int(i * width * 1.0 / (splits-1));
    for(int j=0; j<splits; j++) {
      int g = int(j * width * 1.0 / (splits-1));
      for(int k=0; k<splits-1; k++){ 
        int b = int(k * width * 1.0 / (splits-1));
        ret.emplace_back(V1I{r,g,b});
        // std::cout <<"rgb:  " << i << " " << j << " " << k << "    " <<  r << " " << g << " " << b << "\n";
      }
    }
  }
  return ret;
}

void OCRPostImp::prepare_pairs(){
  ifstream Tin;
  std::string str;
  Tin.open(cfgpath_, ios_base::in);
  if(!Tin)  {
     std::cout<<"Can't open the file " << cfgpath_ << "\n";      exit(-1);
  }
  while( getline(Tin, str)) {
    std::size_t found = str.find_first_of(" ");
    if(found) {
       pairs[ str.substr(0, found)] = str.substr(found+1);
    }
  }
  Tin.close();
  // std::cout <<"pairs : " << pairs.size() <<  " \n";
}

float OCRPostImp::linalgnorm(const cv::Mat& b0, const cv::Mat& b1) {
  float vx = b0.ptr<float>(0)[0] - b1.ptr<float>(0)[0];
  float vy = b0.ptr<float>(0)[1] - b1.ptr<float>(0)[1];
  return std::sqrt( std::pow(vx,2) + std::pow(vy,2));
}

vector<vector<cv::Point2f>> OCRPostImp::getDetBoxes(int idx) {
  vector<vector<cv::Point2f>> ret;  //  list   array

  int img_h=target_h8_[idx], img_w=target_w8_[idx];
  cv::Mat mat_text(img_h, img_w, CV_8UC1, cvScalar(0));
  cv::Mat mat_link(img_h, img_w, CV_8UC1, cvScalar(0));
  cv::Mat mat_comb(img_h, img_w, CV_8UC1, cvScalar(0));

  __TIC__(getDetBox_init_textlink)

  //int loop1=0, loop2=0;
  for(int i=0; i<target_h8_[idx]; i++) {
    for(int j=0; j<target_w8_[idx]; j++)  {
       if(0){ 
            std::cout <<i << " " << j << " :  "
                << sigmoid(scale_o_1*p_textlink[idx][ i*output_w*2 + j*2+0 ]) << "   " << 1.0* p_textlink[idx][ i*output_w*2 + j*2+0 ]
                << "   " << 0.5* p_textlink[idx][ i*output_w*2 + j*2+0 ] << "                  "
                << sigmoid(scale_o_1*p_textlink[idx][ i*output_w*2 + j*2+1 ]) << "   " << 1.0* p_textlink[idx][ i*output_w*2 + j*2+1 ]
                << "   " << 0.5* p_textlink[idx][ i*output_w*2 + j*2+1 ]
                <<"\n";
       }
       if ( p_textlink[idx][ i*output_w*2 + j*2+0 ]> low_text ) {
          //loop1++;
          mat_text.ptr<uchar>(i)[j] = 1;  
          mat_comb.ptr<uchar>(i)[j] = 1;  
       }
       if ( p_textlink[idx][ i*output_w*2 + j*2+1 ] > link_threshold ) {
          //loop2++;
          mat_link.ptr<uchar>(i)[j] = 1;  
          mat_comb.ptr<uchar>(i)[j] = 1;  
       }
    }
  }
  __TOC__(getDetBox_init_textlink)

  cv::Mat labels, stats, centroids;
  int nccomps = cv::connectedComponentsWithStats (
                  mat_comb, 
                  labels,    
                  stats,
                  centroids, 
                  4 );

  __TIC__(getDetBox_get_labels)  // 27ms  --> 9ms
  vector<vector<std::pair<int,int>>> vlabels = get_labels(idx, labels, nccomps);
  __TOC__(getDetBox_get_labels)
  __TIC__(getDetBox_inner_loop_outside)
  for(int k=1; k<nccomps; k++) {
    auto size = stats.at<int>(k, cv::CC_STAT_AREA);
    // std::cout <<"k size : " << k << " " << size <<  "          nccompts :" << nccomps << "\n";
    if ( (int)size<10 || check_max_less_thresh(idx, vlabels, k)) {
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
       for(int i=0; i<img_h; i++)   {
         for(int j=0; j<img_w; j++)  {
            if (segmap.ptr<uchar>( i)[j] != 0 ) {
               loopx++;
            }
         }
       }
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
    // std::cout <<"box ;" <<  box.rows << " " << box.cols << "   type : " << type2str(box.type()) << "\n";
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
    ret.emplace_back(  clockwise(box4));
  }
  __TOC__(getDetBox_inner_loop_outside)
  return ret;
}

vector<cv::Point2f> OCRPostImp::clockwise( vector<cv::Point2f>& box){
  vector<cv::Point2f> ret;
  auto p = std::min_element( box.begin(), box.end(), []( const cv::Point2f&i, const cv::Point2f& j){ return i.x+i.y<j.x+j.y; } ); 

  for(int i=0; i<4-(p-box.begin()); i++) {
    ret.emplace_back(*(p+i)); 
  }
  for(int i=0; i < p-box.begin(); i++) {
    ret.emplace_back( *(  box.begin()+i)); 
  }
  // for(int i=0; i<4; i++) { std::cout << "clock: " << ret[i].x << "  " << ret[i].y << "\n"; }
  return ret;
}

std::string OCRPostImp::find_match_word(int ithread, const std::string& str,  V2F& scores ) {
   // scores: [x output_c ], only [2..37] need parse.   [0 1 ] is no use.
   if(str.empty())  {
      return str;
   }
   int dist_min_pre = 100; 
   int str_len = str.size();
   small_lexicon_dict[ithread].clear();

#if 1
   __TIC__(test_editdistance)  // 23 25 137 155 ms -->  15 29 65 89ms ..
   for(auto& it: pairs) {
      if ( abs(int(str_len - it.first.size())) > dist_min_pre+XLNX_MATCH_EXTLEN ) {
         continue;
      }
      int ed = editdistance_eval( str, it.first );
      // std::cout <<"ed : " << str << "  " << it.first << "   " <<  ed << "\n";
      if (ed <= dist_min_pre+XLNX_MATCH_EXTLEN ) {        
          dist_min_pre = ed; 
          small_lexicon_dict[ithread][it.first] = ed;      
      }
   }
   __TOC__(test_editdistance)
   // std::cout <<"ed---end small-dict-size: min_pre:   " << dist_min_pre <<  "  size : " << small_lexicon_dict.size() << "\n";
   int loop = 0;
   float dist_min = 100.0;
   std::string match_word;
   __TIC__(test_weighted_edit_distance)
   for(auto& it: small_lexicon_dict[ithread]) {
      if (it.second <= dist_min_pre+XLNX_MATCH_EXTLEN) {
        loop++;
        float ed = weighted_edit_distance(str, it.first, scores);
        // std::cout <<"ed : " << str << "  " << it.first << "   " <<  ed << "\n";
        if (ed < dist_min) {
           dist_min = ed;
           match_word = pairs[it.first];
           // std::cout <<"loop :" << loop << "   distmin matchword :" << dist_min << " " << match_word << "\n";
        }
      }
   } 
   __TOC__(test_weighted_edit_distance)
#else
   float dist_min = 100.0;
   __TIC__(test_editdistance)  // huge time
   for(auto& it: pairs) {
      if ( abs(int(str_len - it.first.size())) > dist_min+XLNX_MATCH_EXTLEN ) {
         continue;
      }
      float ed = weighted_edit_distance(str, it.first, scores);
      // std::cout <<"ed : " << str << "  " << it.first << "   " <<  ed << "\n";
      if (ed < dist_min) {
         dist_min = ed;
         match_word = pairs[it.first];
      }
   }
   __TOC__(test_editdistance)  // 23 25 137 155 ms
#endif

   // std::cout <<" loop in find_match_word : " << 0 << "   distmin : " << dist_min << "\n";
   return match_word;
}

int OCRPostImp::char2num(char c){
  int num = 0;
  if(std::isdigit(c)) {
     num = c - '0';
  } else if (std::isalpha(c)) {
     num = c - (c >= 'a' ? 'a' : 'A' ) + 10;
  } else {
     std::cerr <<" error occured for char " << int(c) << "\n";
  }
  return num;
}

float OCRPostImp::ed_delect_cost(int j, const std::string& word, V2F& scores ){
  int num = char2num( word[j]);
  // return scores[num][j]; // scores is dim<->exchanged: python: [36 3] c++: [3 output_c]
  return scores[j][num+2];
}

float OCRPostImp::ed_insert_cost(int i, const std::string& word, V2F& scores ){
  int c1 = char2num(word[i]);
  if (i< (int)(word.size() - 1) ){
     int c2 = char2num(word[i+1]);
     // return (scores[c1][i] + scores[c2][i+1])/2;
     return (scores[i][c1+2] + scores[i+1][c2+2])/2;
  }
  // return scores[c1][i];
  return scores[i][c1+2];
}

float OCRPostImp::ed_replace_cost(int i, int j, const std::string& word1, const std::string& word2, V2F& scores){
   int c1 = char2num(word1[i]);
   int c2 = char2num(word2[j]);
   // return std::max(1 - scores[c2][i]/scores[c1][i]*5, 0.0f);
   return std::max(1 - scores[i][c2+2]/scores[i][c1+2]*5, 0.0f);
}

float OCRPostImp::weighted_edit_distance(const std::string& s1, const std::string& s2, V2F& scores){
  // std::cout <<"weighted_edit_distance " << s1 << " " << s2 << " scores :" << scores.size() << " " << scores[0].size() << "\n";
  int m = s1.size();
  int n = s2.size();
  V2F dp(n+1, V1F(m+1, 0.0 ) );
  for(int i=0; i<m+1; i++) { 
    dp[0][i] = i; 
  }
  for(int i=0; i<n+1; i++) {
    dp[i][0] = i; 
  }
  for(int i=1; i<n+1; i++)  {
    for(int j=1; j<m+1; j++)  {
       float delect_cost = ed_delect_cost(j-1, s1, scores);
       float insert_cost = ed_insert_cost(j-1, s1, scores);
       float replace_cost = s1[j - 1] != s2[i - 1] ? ed_replace_cost(j-1, i-1, s1, s2, scores) : 0.0;
       dp[i][j] = std::min( {dp[i-1][j] + insert_cost, dp[i][j-1] + delect_cost, dp[i-1][j-1] + replace_cost} );
    }
  }
  return dp[n][m];
}

int OCRPostImp::editdistance_eval(const std::string& str1, const std::string& str2) {
    int len1 = str1.length();
    int len2 = str2.length();
 
    // Create a DP array to memoize result
    // of previous computations
    int DP[2][len1 + 1];
 
    // To fill the DP array with 0
    memset(DP, 0, sizeof DP);
 
    // Base condition when second string
    // is empty then we remove all characters
    for (int i = 0; i <= len1; i++)
        DP[0][i] = i;
 
    // Start filling the DP
    // This loop run for every
    // character in second string
    for (int i = 1; i <= len2; i++) {
        // This loop compares the char from
        // second string with first string
        // characters
        for (int j = 0; j <= len1; j++) {
            // if first string is empty then
            // we have to perform add character
            // operation to get second string
            if (j == 0) {
                DP[i % 2][j] = i;
            } 
            // if character from both string
            // is same then we do not perform any
            // operation . here i % 2 is for bound
            // the row number.
            else if (str1[j - 1] == str2[i - 1]) {
                DP[i % 2][j] = DP[(i - 1) % 2][j - 1];
            }
 
            // if character from both string is
            // not same then we take the minimum
            // from three specified operation
            else {
                DP[i % 2][j] = 1 + min(DP[(i - 1) % 2][j],
                                       min(DP[i % 2][j - 1],
                                           DP[(i - 1) % 2][j - 1]));
            }
        }
    }
 
    // after complete fill the DP array
    // if the len2 is even then we end
    // up in the 0th row else we end up
    // in the 1th row so we take len2 % 2
    // to get row
    return DP[len2 % 2][len1];
}

void OCRPostImp::getRecWords(int idx, 
                             vector<vector<cv::Point2f>>& det,
                             vector<float>& pred_rec, 
                             vector<std::string>& words, 
                             cv::Mat& color_map) {
  // std::cout <<"come in fillpoly " << det[0][0].x << " " << det[0][0].y << " " << det[0][1].x << " " << det[0][1].y << " " << det[0][2].x << " " << det[0][2].y << " " << det[0][3].x << " " << det[0][3].y << "\n" ; //  519 97   539 97   539 106   519 106

  if(  XLNX_OCR_VISUAL == 1) { // use env var to control: if parse color_map
    // TODO: full small logic here ....  ignore now
    __TIC__(getRecWords_mask_index)

    if (det.empty()) {
      return;
    }
    cv::Mat mask( target_h8_[idx], target_w8_[idx], CV_8UC1, cv::Scalar(0));
    vector<vector<cv::Point2i>> deti(det.size(),vector<cv::Point2i>( det[0].size()) );
    for(unsigned int i=0; i<det.size(); i++) {
      for(unsigned int j=0; j<det[0].size(); j++) {
         deti[i][j] = det[i][j];
         // std::cout <<i << " " << j << "   " << deti[i][j].x << " " << deti[i][j].y << "\n";
      }   
    }
    // std::cout <<" after deti\n";

    cv::fillPoly(mask, deti, cv::Scalar(1) );
    V1F pred_tmp(pred_rec); // 73ms
    // std::cout << "   " << pred_rec.size() << " " <<  pred_tmp.size() << "  ";
    // V1F pred_tmp = pred_rec;
    __TIC__(VISUAL_loop)
    for(int i=0; i<target_h8_[idx]; i++) {
      for(int j=0; j<target_w8_[idx]; j++) {
         // std::cout << pred_tmp[i*560*output_c+j*output_c+1] <<"\n";
         if (mask.ptr<uchar>(i)[j] == 0) {
            pred_tmp[i*target_w8_[idx]*output_c+j*output_c+0] = 1;
         }
         // else std::cout << i << "  " << j  << "\n";

         for(int k=1; k<output_c && mask.ptr<uchar>(i)[j]==0 ; k++) {
           // pred_tmp[i*560*output_c+j*output_c+k] *= mask.ptr<uchar>(i)[j];
           pred_tmp[i*target_w8_[idx]*output_c+j*output_c+k] = 0;
         }
         int v = std::max_element(pred_tmp.begin()+ i*target_w8_[idx]*output_c+j*output_c+0, 
                                     pred_tmp.begin()+ i*target_w8_[idx]*output_c+j*output_c+output_c)
                                      - (pred_tmp.begin()+i*target_w8_[idx]*output_c+j*output_c+0);
         color_map.ptr<cv::Vec3b>(i)[j]  = cv::Vec3b( g_color[v][0], g_color[v][1], g_color[v][2] );
         // if (v) std::cout <<"v: " << i << " " << j << "    " << v << "\n";
         //int tmp =  maxv - (pred_tmp.begin()+ i*560*output_c+j*output_c+0 );
         //if (tmp) mask_index.ptr<uchar>(i)[j] = tmp;
         // mask_index.ptr<uchar>(i)[j] = maxv - (pred_tmp.begin()+ i*560*output_c+j*output_c+0 );
         // if (mask.ptr<uchar>(i)[j] ) std::cout << i << " " << j << " " <<  int( mask.ptr<uchar>(i)[j]   ) << "\n";
      }
    }
    __TOC__(VISUAL_loop)
    __TOC__(getRecWords_mask_index)
  }

  vector<vector<std::string>> words_all(XLNX_OCR_GETWORD_THREAD);
  vector<vector<vector<cv::Point2f>>> det_all(XLNX_OCR_GETWORD_THREAD);
  int size = det.size(), start=0, len=0;
  int len_x = round(size/XLNX_OCR_GETWORD_THREAD);
  for(int i=0; i<XLNX_OCR_GETWORD_THREAD; i++) {
     start = i * len_x;
     len = (i != XLNX_OCR_GETWORD_THREAD-1) ?  len_x : (size- (len_x*(XLNX_OCR_GETWORD_THREAD-1))) ;
     vth_word.emplace_back( std::thread( 
                                  &OCRPostImp::getRecWords_thread, this, idx, i, start, len, std::ref(det), 
                                  std::ref(pred_rec), std::ref(words_all[i]),  std::ref(det_all[i]) ));
  }
  for(int i=0; i<XLNX_OCR_GETWORD_THREAD; i++) {
     vth_word[i].join();
  }
  vth_word.clear();

  det.clear();
  // combine result;
  for(int i=0; i<XLNX_OCR_GETWORD_THREAD; i++) {
     if (words_all[i].size()) {
        words.insert(words.end(), words_all[i].begin(), words_all[i].end()); 
        det.insert(det.end(), det_all[i].begin(), det_all[i].end()); 
     }
  }
}

void OCRPostImp::getRecWords_thread(int idx, int ithread, int start, int len, 
                                    vector<vector<cv::Point2f>>& det, 
                                    vector<float>& pred_rec, 
                                    vector<std::string>& words,
                                    vector<vector<cv::Point2f>>& det_out ) {
  V2F score2;
  vector<vector<cv::Point2i> > contours;
  __TIC__(getRecWords_box_loop)

  // for(auto &box: det) {
  for(int iloop=start; iloop<start+len; iloop++) {
    vector<cv::Point2i> box = vector<cv::Point2i>{ 
      // TODO: shili : use cv::Point2i( conv)
#if 1
                                cv::Point2i((int)det[iloop][0].x,  (int)det[iloop][0].y),
                                cv::Point2i((int)det[iloop][1].x,  (int)det[iloop][1].y),
                                cv::Point2i((int)det[iloop][2].x,  (int)det[iloop][2].y),
                                cv::Point2i((int)det[iloop][3].x,  (int)det[iloop][3].y) };
#else
                                // both bad
                                cv::Point2i(int(round(det[iloop][0].x)),  int(round(det[iloop][0].y))),
                                cv::Point2i(int(round(det[iloop][1].x)),  int(round(det[iloop][1].y))),
                                cv::Point2i(int(round(det[iloop][2].x)),  int(round(det[iloop][2].y))),
                                cv::Point2i(int(round(det[iloop][3].x)),  int(round(det[iloop][3].y))) };
                                // cv::Point2i(det[iloop][0]),
                                // cv::Point2i(det[iloop][1]),
                                // cv::Point2i(det[iloop][2]),
                                // cv::Point2i(det[iloop][3]) };
#endif

    __TIC__(getRecWords_box_loop_innerloop_all)  // 196 313 430 330 261 454 263 330 125ms -->21 output_c 76 10 8 101 8 10 5ms
    cv::Mat mask1(target_h8_[idx], target_w8_[idx], CV_8UC1, cv::Scalar(0));
    cv::fillPoly(mask1, vector<vector<cv::Point2i>>{box}, cv::Scalar(1) );

    __TIC__(getRecWords_core) // 122 224 327 292 224 327 224  292 89
    score2.clear();
    // std::string word = getRecWords_core(pred_tmp, mask1, score1, score2);
 
    __TIC__(getRecWords_core_initgray)  // 20ms
    cv::Mat gray(target_h8_[idx], target_w8_[idx], CV_8UC1, cv::Scalar(0));  
    for(int i=0; i<target_h8_[idx]; i++) {
      for(int j=0; j<target_w8_[idx]; j++) {
         if (mask1.ptr<uchar>(i)[j] != 0)  {
            if (XLNX_OCR_FULLSOFTMAX) { 
               gray.ptr<uchar>(i)[j] = 255.0* (1- pred_rec[i*2*output_w*2*output_c+j*2*output_c+0] );
               // std::cout <<i << " " << j << " " << int(gray.ptr<uchar>(i)[j]) << "  " << pred_rec[i*output_w*2*output_c+j*2*output_c] << "\n";
            } else {
               gray.ptr<uchar>(i)[j] = 255.0* (1- pred_rec[i*target_w8_[idx]*output_c+j*output_c+0] );
            }
         }
      }
    }
    __TOC__(getRecWords_core_initgray)
  
    cv::Mat thresh(target_h8_[idx], target_w8_[idx], CV_8UC1, cv::Scalar(0));
    cv::threshold(gray, thresh, threshold, 255.0, cv::THRESH_BINARY);
    vector<cv::Vec4i> hierarchy;
    contours.clear();
    cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  
    struct cmp1 {
      bool operator ()(const std::tuple<float,char, vector<float>>& a, const std::tuple<float,char, vector<float>>& b ) {
        return std::get<0>(a) >  std::get<0>(b);
      }
    };
    std::priority_queue< std::tuple<float,char, vector<float>>, 
                    vector<  std::tuple<float,char, vector<float>>  >, cmp1> chars;
  
    float scores=0.0;
    int n_scores=0;
    __TIC__(getRecWords_core_eachloop_outside)  // 4.9 7.3 .... ms

    // std::cout <<"contours.size :" << contours.size() << "\n";

    for(unsigned int ii=0; ii<contours.size(); ii++) {
      __TIC__(getRecWords_core_eachloop)  // 33ms
       cv::Mat temp(target_h8_[idx], target_w8_[idx], CV_8UC1, cv::Scalar(0));
       cv::drawContours(temp, contours, ii, cv::Scalar(255), -1); // if <0 is used, inner area is drawn
       cv::Rect rect = cv::boundingRect( contours[ii] );
       // std::cout <<"rect :" << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << "\n"; 
  
       vector<float> meanf(output_c, 0.0);
       int nmeanf=0;
       // 3612ms (EXTLEN:2)  482ms(for EXTLEN 0)
       // if k is outside, 5461ms  2323ms(for EXTLEN 0)
       for(int i = 0; i < target_h8_[idx]; i++) {
         for(int j = 0; j < target_w8_[idx]; j++) {
            if (temp.ptr<uchar>(i)[j] == 255) {
               nmeanf++;
               for(int k = 1; k< output_c; k++) { 
                  if (XLNX_OCR_FULLSOFTMAX) {
                     meanf[k] += pred_rec[i*2*output_w*2*output_c+j*2*output_c+k];
                  } else {
                     meanf[k] += pred_rec[i*target_w8_[idx]*output_c+j*output_c+k];
                  }   
               }
            }
         }
       }
       int num = std::max_element( meanf.begin()+1, meanf.end()) - meanf.begin();
       // std::cout <<"num :" << num << "\n";
       // char sym = num2char(num +1 );
       char sym = num2char(num );
       if (sym == '_' ) {
           continue;
       }

       std::transform(meanf.begin(), meanf.end(), meanf.begin(), [=](float f ){ return f/(nmeanf*1.0);} );
       // for(int k = 1; k<output_c; k++) { meanf[k]/=(1.0*nmeanf); }
       scores += meanf[num];
       n_scores++;
       chars.push(std::make_tuple(rect.x+rect.width/2.0, sym,  meanf ) );
       __TOC__(getRecWords_core_eachloop)
    } // end of inner loop
    __TOC__(getRecWords_core_eachloop_outside)  // 33ms
    if ( n_scores == 0 || scores/n_scores < 0.7 ) {
       __TOC__(getRecWords_core)
       __TOC__(getRecWords_box_loop_innerloop_all)  
       continue;
    }
    std::string word;
    while(!chars.empty()){
      word.append(1, toupper(std::get<1>(chars.top()))) ;
      score2.emplace_back( std::get<2>(chars.top() ) );
      chars.pop();
    }
    // std::cout <<"word score :" << word << "   " << scores << "\n"; 
    __TOC__(getRecWords_core)
    // printv2("score2 :", score2);
    // boost::algorithm::to_upper(word);
    __TIC__(getRecWords_find_match)  // 15/29/65/89ms  
    std::string match_word = find_match_word(ithread, word, score2);
    __TOC__(getRecWords_find_match)
    words.emplace_back(match_word);
    // det_out.emplace_back(box);
    det_out.emplace_back(  std::move(det[iloop]) );
    __TOC__(getRecWords_box_loop_innerloop_all)  
  }
  __TOC__(getRecWords_box_loop)
}

bool OCRPostImp::check_max_less_thresh(int idx, const  vector<vector<std::pair<int,int>>>& k_pos, int k ) {
  float tmpf, maxf = 0.0;
  for(unsigned int i=0; i<k_pos[k-1].size(); i++){
    if(  (tmpf =  p_textlink[idx][ k_pos[k-1][i].first *output_w*2 + k_pos[k-1][i].second *2 +0 ]) > maxf) {
       maxf = tmpf;
    }
  }
  return maxf < text_threshold;
}

vector<vector<std::pair<int,int>>> OCRPostImp::get_labels(int idx, const cv::Mat& labels, int nlabels){
  vector<vector<std::pair<int,int>>> k_pos;
  vector<vector<vector<std::pair<int,int>>>> k_posall(XLNX_OCR_GETWORD_THREAD, vector<vector<std::pair<int,int>>>(nlabels-1));

  float size = nlabels-1;
  vector<int> start(XLNX_OCR_GETWORD_THREAD);
  vector<int> len(XLNX_OCR_GETWORD_THREAD);
  int len_x = round(size/XLNX_OCR_GETWORD_THREAD);
  for(int i=0; i<XLNX_OCR_GETWORD_THREAD; i++) {
     start[i] = i * len_x;
     len[i] = (i != XLNX_OCR_GETWORD_THREAD-1) ? len_x : (size- (len_x*(XLNX_OCR_GETWORD_THREAD-1))) ;
     // std::cout <<" start len : " << start[i] << " " << len[i] << " \n";
     vth_label.emplace_back( std::thread( &OCRPostImp::get_labels_thread, this, start[i], len[i], idx, std::cref(labels), nlabels, std::ref(k_posall[i]) ));
  }
  for(int i=0; i<XLNX_OCR_GETWORD_THREAD; i++) {
     vth_label[i].join();
  }
  vth_label.clear();

  // combine result;
  for(int i=0; i<XLNX_OCR_GETWORD_THREAD; i++) {
    k_pos.insert(k_pos.end(), k_posall[i].begin()+start[i], k_posall[i].begin()+start[i]+len[i]);
  }

  // for(int i=0;i<11;i++) std::cout <<"k_pos[" << i << "] :" << k_pos[i].size() <<" \n";
  return k_pos;
}

void OCRPostImp::get_labels_thread(int start, int len, int idx, const cv::Mat& labels, int nlabels, 
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

  if (XLNX_OCR_FULLSOFTMAX) {
      __TIC__(softmax)
      vitis::ai::softmax( p_pred[idx], scale_o_0, output_c, output_w*2*output_h*2, softmax_data.data());
      // printv1(softmax_data, "softmaxVec", 38); 
      __TOC__(softmax)
  } else {
      __TIC__(softmax_memcpy)
      for(int i=0; i<target_h8_[idx]; i++) {
         for(int j=0; j<target_w8_[idx]; j++) {
            memcpy( softmax_data_src.data() +i*target_w8_[idx]*output_c +j*output_c, p_pred[idx]+ (i*2)*(output_w*2)*output_c +j*2*output_c, output_c ); 
            //for(int k=0; k<38; k++) { std::cout <<  *(p_pred[idx]+ (i*2)*(output_w*2)*38 + j*2*38 +k)*0.125 << " "; } 
            //std::cout <<"\n";
         }
      } 
      softmax_data_src.resize(target_w8_[idx]*target_h8_[idx]*output_c);
      softmax_data.resize(target_w8_[idx]*target_h8_[idx]*output_c);
      __TOC__(softmax_memcpy)
      __TIC__(softmax)
    
      vitis::ai::softmax((int8_t*)softmax_data_src.data(), scale_o_0, output_c, target_h8_[idx]*target_w8_[idx], softmax_data.data());
      __TOC__(softmax)
  }
 
  __TIC__(getDetBoxes)
  vector<vector<cv::Point2f>> boxes = getDetBoxes(idx);
  __TOC__(getDetBoxes)

  vector<std::string> words;
  OCRResult result{int(input_tensors_[0].width), int(input_tensors_[0].height) };
  if(  XLNX_OCR_VISUAL == 1) { // use env var to control: if parse it
    cv::Mat color_map(target_h8_[idx], target_w8_[idx], CV_8UC3, cv::Scalar(0,0,0)); 
    result.color_map = color_map;
  }

  __TIC__(getRecWords)
  getRecWords(idx, boxes, softmax_data, words, result.color_map);
  __TOC__(getRecWords)

  result.words.swap(words);
  if (result.words.empty()) {
      result.box.clear();
  } else {
      //adjust boxes;
      for(unsigned int i=0; i<boxes.size(); i++) {
        #if 0
        for(unsigned int j=0; j<boxes[0].size(); j++) {
           //std::cout <<"adjust : " << boxes[i][j].x << "  " << boxes[i][j].y << "  ratio : " << ratiow_[idx]*4  << " " << ratioh_[idx]*4  << "  |    ";
           boxes[i][j].x *= ratiow_[idx]*4 ;
           boxes[i][j].y *= ratioh_[idx]*4 ;
           //std::cout << boxes[i][j].x << "  " << boxes[i][j].y << "  ratio : " << ratiow_[idx]*4  << " " << ratioh_[idx]*4  << "\n";
        }
        #endif

        result.box.emplace_back( 
              vector<cv::Point>{
                      cv::Point(int(boxes[i][0].x*ratiow_[idx]*4),  int(boxes[i][0].y*ratioh_[idx]*4)),
                      cv::Point(int(boxes[i][1].x*ratiow_[idx]*4),  int(boxes[i][1].y*ratioh_[idx]*4)),
                      cv::Point(int(boxes[i][2].x*ratiow_[idx]*4),  int(boxes[i][2].y*ratioh_[idx]*4)),
                      cv::Point(int(boxes[i][3].x*ratiow_[idx]*4),  int(boxes[i][3].y*ratioh_[idx]*4))
                   } );
      }
  }
  return result;
}

}  // namespace ai
}  // namespace vitis

