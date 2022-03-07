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
#pragma once
#include <thread>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/nnpp/ocr.hpp>

namespace vitis {
namespace ai {

using namespace std;

class OCRPostImp : public OCRPost{
 public:

  OCRPostImp(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const std::string& cfgpath,
      int batch_size,
      int& real_batch_size,
      std::vector<int>& target_h8,
      std::vector<int>& target_w8,
      std::vector<float>& ratioh,
      std::vector<float>& ratiow
  );

  virtual ~OCRPostImp();

  virtual OCRResult process(int idx) override;
  virtual std::vector<OCRResult> process() override;

 private:
  // below 2 for initialize
  V2I creat_color_map(int ncls , int width);
  void prepare_pairs();

  float linalgnorm(const cv::Mat& b0, const cv::Mat& b1);
  std::vector<std::vector<cv::Point2f>> getDetBoxes(int idx);
  std::vector<cv::Point2f> clockwise( std::vector<cv::Point2f>& box);
  std::string find_match_word(int ithread, const std::string& str,  V2F& scores ) ;
  int char2num(char c);
  float ed_delect_cost(int j, const std::string& word, V2F& scores );
  float ed_insert_cost(int i, const std::string& word, V2F& scores );
  float ed_replace_cost(int i, int j, const std::string& word1, const std::string& word2, V2F& scores);
  float weighted_edit_distance(const std::string& s1, const std::string& s2, V2F& scores);
  void getRecWords(int idx, std::vector<std::vector<cv::Point2f>>& det, std::vector<float>& pred_rec, std::vector<std::string>& words, cv::Mat& color_map);
  void getRecWords_thread(int idx, int ithread, int start, int len, std::vector<std::vector<cv::Point2f>>& det, std::vector<float>& pred_rec, std::vector<std::string>& words, vector<vector<cv::Point2f>>& det_out );
  bool check_max_less_thresh(int idx, const std::vector<std::vector<std::pair<int,int>>>& k_pos, int k );
  std::vector<std::vector<std::pair<int,int>>> get_labels(int idx, const cv::Mat& labels, int nlabels);
  void get_labels_thread(int start, int len, int idx, const cv::Mat& labels, int nlabels, 
             std::vector<std::vector<std::pair<int,int>>>& k_pos );

  int editdistance_eval(const std::string& str1, const std::string& str2);
 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  std::string cfgpath_;
  int batch_size_;
  int& real_batch_size_;
  int output_w;
  int output_h;
  int output_c;
  std::vector<int>& target_h8_;
  std::vector<int>& target_w8_;
  std::vector<float>& ratioh_;
  std::vector<float>& ratiow_;

  std::unordered_map<std::string, std::string> pairs;
  inline char num2char(int i){ return g_chars[i]; }
  const char* g_chars="$_0123456789abcdefghijklmnopqrstuvwxyz";
  V2I g_color;
  // cv::Mat np_contoursx(5000, 2, CV_32F, cvScalar(0) );  // I think 5000 is enough?
  cv::Mat np_contoursx;
  int np_contoursx_size=5000;
  std::vector<int8_t*> p_textlink;
  std::vector<int8_t*> p_pred;
  std::vector<float> softmax_data;
  std::vector<int8_t> softmax_data_src;
  std::vector<std::unordered_map<std::string, int>> small_lexicon_dict;

  std::vector<cv::Point2f> box4;
  std::vector<std::thread> vth_label;
  std::vector<std::thread> vth_word;

  float scale_o_0 = 1.0, scale_o_1 = 1.0;
  float low_text = 0.5;
  float link_threshold = 0.2;
  float text_threshold = 0.85;
  int threshold = 192;

  int XLNX_OCR_VISUAL = 0;
  int XLNX_OCR_GETWORD_THREAD = 2;
  int XLNX_MATCH_EXTLEN = 0;
  int XLNX_OCR_FULLSOFTMAX = 0;
};

}  // namespace ai
}  // namespace vitis

