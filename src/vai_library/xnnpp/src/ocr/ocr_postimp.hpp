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
      std::vector<float>& ratiow,
      std::vector<cv::Mat>& oriimg
  );

  virtual ~OCRPostImp();
  virtual OCRResult process(int idx) override;
  virtual std::vector<OCRResult> process() override;

 private:
  void prepare_lexicon();

  std::string post_watershed(cv::Mat& mask, bool&, int idx);
  float linalgnorm(const cv::Mat& b0, const cv::Mat& b1);
  std::vector<std::vector<cv::Point2f>> getDetBoxes(int idx, std::vector<string>& words);
  std::vector<cv::Point2f> clockwise( std::vector<cv::Point2f>& box);
  std::string find_match_word(int ithread, const std::string& str,  V2F& scores ) ;
  bool check_max_less_thresh(int idx, const std::vector<std::vector<std::pair<int,int>>>& k_pos, int k );
  std::vector<std::vector<std::pair<int,int>>> get_labels(int idx, const cv::Mat& labels, int nlabels);
  void get_labels_thread(int start, int len, int idx, const cv::Mat& labels, int nlabels, 
             std::vector<std::vector<std::pair<int,int>>>& k_pos );

  int editdistance_eval(const std::string& str1, const std::string& str2);
  std::string match_lexicon(const std::string&);
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
  std::vector<cv::Mat>& oriimg;

  std::vector<std::vector<std::pair<std::string, int>>>  lexiconL;
  const char* g_chars="_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  inline char num2char(int i){ return g_chars[i]; }
  std::vector<std::vector<float>>regions;

  cv::Mat np_contoursx;
  int np_contoursx_size=5000;
  std::vector<int8_t*> p_textlink;
  std::vector<int8_t*> p_pred;
  std::vector<float> softmax_data;
  std::vector<int8_t> softmax_data_src;

  std::vector<cv::Point2f> box4;
  std::vector<std::thread> vth_label;

  float scale_o_sig = 1.0, scale_o_sft = 1.0;
  float low_text = 0.5;
  float link_threshold = 0.2;
  float text_threshold = 0.85;

  float textmap_thresh_low=0.4;
  float textmap_thresh_high=0.8;

  int XLNX_OCR_GETWORD_THREAD = 2;
  int XLNX_OCR_IMPORT_POST = 0;
  int XLNX_OCR_FAST_LEXICON = 1;
  int XLNX_OCR_POST_ROUND = 0;
};

}  // namespace ai
}  // namespace vitis

