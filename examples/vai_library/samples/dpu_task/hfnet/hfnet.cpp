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
#include "./hfnet.hpp"

#include <memory>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/math.hpp>

#define HW_SOFTMAX
//#define ENABLE_NEON

DEF_ENV_PARAM(DEBUG_HFNET, "0");
DEF_ENV_PARAM(DUMP_HFNET, "0");

using namespace std;
using namespace cv;

static vector<vitis::ai::library::OutputTensor> sort_tensors(
    const vector<vitis::ai::library::OutputTensor>& tensors,
    vector<size_t>& chas) {
  vector<vitis::ai::library::OutputTensor> ordered_tensors;
  for (auto i = 0u; i < chas.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].channel == chas[i]) {
        ordered_tensors.push_back(tensors[j]);
        LOG_IF(INFO, ENV_PARAM(DEBUG_HFNET))
          << "tensor name: " << tensors[j].name;
        break;
      }
  return ordered_tensors;
}

namespace vitis {
namespace ai {
class HFnetImp : public HFnet {
 public:
  HFnetImp(const std::string& model_name);

 public:
  virtual ~HFnetImp();
  virtual std::vector<HFnetResult> run(const std::vector<cv::Mat>& imgs) override;
 // virtual std::vector<cv::Mat> get_result() override;
  virtual size_t get_input_batch() override;
  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;

 private:
  vector<vector<float>> netvlad(const vitis::ai::library::OutputTensor& otensor1,
	     const vitis::ai::library::OutputTensor& otensor2, size_t count);
  void set_input(vitis::ai::library::InputTensor& tensor, float mean, float scale, vector<Mat>& img);
  void hfnet_run(const vector<cv::Mat>& input_images);
  bool verifyOutput(size_t count);

 public:
  std::unique_ptr<vitis::ai::DpuTask> task_;
  vector<vitis::ai::library::InputTensor> inputs_;
  vector<vitis::ai::library::OutputTensor> outputs_;

 private:
  int sWidth;
  int sHeight;
  size_t batch_;
  vector<size_t> chans_;

  size_t channel1;
  size_t channel2;
  size_t channel3;
  size_t channel4;
  size_t outputH;
  size_t outputW;
  size_t output2H;
  size_t output2W;
  float conf_thresh;
  size_t outputSize1;
  size_t outputSize2;
  vector<float> cluster;
  size_t clu_size;
  std::vector<HFnetResult> results_;
};

HFnet::HFnet(const std::string& model_name) {}

HFnet::~HFnet() {}

std::unique_ptr<HFnet> HFnet::create(const std::string& model_name) {
  return std::unique_ptr<HFnetImp>(new HFnetImp(model_name));
//  return std::make_unique<HFnetImp>(model_name);
}

HFnetImp::HFnetImp(const std::string& model_name): HFnet(model_name) {
  task_ = vitis::ai::DpuTask::create(model_name);
  inputs_ = task_->getInputTensor(0u);
  sWidth = inputs_[0].width;
  sHeight = inputs_[0].height;
  batch_ = inputs_[0].batch;
  chans_ = {65,256,32,1};
  outputs_ = sort_tensors(task_ -> getOutputTensor(0u), chans_);
  channel1 = outputs_[0].channel;
  channel2 = outputs_[1].channel;
  channel3 = outputs_[2].channel;
  channel4 = outputs_[3].size/batch_/outputs_[3].height/outputs_[3].width;
  outputH = outputs_[0].height;
  outputW = outputs_[0].width;
  output2H = outputs_[1].height;
  output2W = outputs_[1].width;
  conf_thresh = 0.015;

  LOG_IF(INFO, ENV_PARAM(DEBUG_HFNET)) 
    << "tensor1 info : " << outputs_[0].height << " " << outputs_[0].width  << " " << outputs_[0].channel << endl
    << "tensor2 info : " << outputs_[1].height << " " << outputs_[1].width  << " " << outputs_[1].channel << endl;

  outputSize1 = outputs_[0].channel * outputs_[0].height * outputs_[0].width;
  outputSize2 = outputs_[1].channel * outputs_[1].height * outputs_[1].width;

  clu_size = channel3*channel4;
  cluster.resize(clu_size);
  ifstream fin("cluster.bin", ios::binary);
  fin.read((char*)cluster.data(), clu_size*sizeof(float));
  fin.close();
}

HFnetImp::~HFnetImp() {}

//The int8_t input need a scale to convert float, but this operate will div scale as well.
template<typename Dtype>
inline vector<float> L2_normalization(Dtype* input, int channel, int group) {
  vector<float> output(channel*group);
  for (int i = 0; i < group; ++i) {
    float sum = 0.0;
    for (int j = 0; j < channel; ++j) {
      int pos = i*channel + j;
      float temp = static_cast<float>(input[pos]);
      sum += temp*temp;
    }
    float var = sqrt(sum);
    for (int j = 0; j < channel; ++j) {
      int pos = i*channel + j;
      output[pos] = static_cast<float>(input[pos]) / var;
    } 
  }
  return output;
}

void nms_mask(vector<vector<int>> &grid, int x, int y, int dist_thresh){
  int h=grid.size();
  int w=grid[0].size();
  for (int i=max(0,x-dist_thresh); i<min(h, x + dist_thresh + 1); ++i){
    for (int j=max(0, y-dist_thresh); j<min(w, y+dist_thresh+1); ++j){
      grid[i][j]=-1;
    }
  }
  grid[x][y]=1;
}

void nms_fast(const vector<int> &xs, const vector<int> &ys, const vector<float> &ptscore, 
                            vector<size_t> &keep_inds, const int inputW, const int inputH){
  //cout << xs.size() << " " << ys.size() << " " << ptscore.size() << " " << keep_inds.size() << endl;
  vector<vector <int>> grid(inputW, vector<int>(inputH, 0));
  vector<pair<float, size_t>> order;
  int dist_thresh = 4;
  for (size_t i=0; i< ptscore.size(); ++i){
    order.push_back({ptscore[i], i});
  }
  std::stable_sort(order.begin(), order.end(),
    [](const pair<float, size_t> &ls, const pair<float, size_t> &rs){
      return ls.first > rs.first;
    });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
    [](auto& km){return km.second; });

  for (size_t _i=0; _i<ordered.size(); ++_i){
    size_t i = ordered[_i];
    int x = xs[i];
    int y = ys[i];
    if(grid[x][y]==0 && x>=dist_thresh && x<inputW-dist_thresh && y >= dist_thresh && y<inputH-dist_thresh){  // remove points along border
      keep_inds.push_back(i);
      nms_mask(grid, x, y, dist_thresh);
    }
  }
}

float bilinear_interpolation(float v_xmin_ymin, float v_ymin_xmax, float v_ymax_xmin, float v_xmax_ymax, int xmin, int ymin, int xmax, int ymax, float x, float y, bool cout_value){
  float value = v_xmin_ymin * (xmax-x) * (ymax-y) + v_ymin_xmax * (ymax-y) * (x-xmin) +  v_ymax_xmin * (y-ymin) * (xmax-x) + v_xmax_ymax * (x-xmin) * (y-ymin); // (xmax-xmin)==ymax-ymin=1
  return value;
}

vector<vector<float>> grid_sample(const float* desc_map, const vector<pair<float, float>> &coarse_pts, const size_t channel, const size_t outputH, const size_t outputW){
  vector<vector<float>> desc(coarse_pts.size());
  for(size_t i = 0; i < coarse_pts.size(); ++i) {
    float x = (coarse_pts[i].first +1)/8 - 0.5;
    float y = (coarse_pts[i].second+1)/8 - 0.5;
    int xmin = floor(x);
    int ymin = floor(y);
    int xmax = xmin+1;
    int ymax = ymin+1;
    // bilinear interpolation
    {
      float divisor = 0.0;
      for(size_t j=0; j< channel; ++j){
        float value = bilinear_interpolation(
            desc_map[j + (ymin*outputW+xmin)*channel],
            desc_map[j + (ymin*outputW+xmax)*channel],
            desc_map[j + (ymax*outputW+xmin)*channel],
            desc_map[j + (ymax*outputW+xmax)*channel],
            xmin, ymin, xmax, ymax, x, y, i<2 && j<10);
        divisor += value*value;
        desc[i].push_back(value);
      }   
      for(size_t j=0; j < channel; ++j){
        desc[i][j] /= sqrt(divisor); //l2 normalize
      }
    }
  }
  return desc;
}

bool HFnetImp::verifyOutput(size_t count) {
  for (size_t n = 0; n < count; ++n) {
    HFnetResult result_;
    int8_t* out1 = (int8_t*)outputs_[0].get_data(n);
    int8_t* out2 = (int8_t*)outputs_[1].get_data(n);

    float scale1 = tensor_scale(outputs_[0]);
    float scale2 = tensor_scale(outputs_[1]);

    if(ENV_PARAM(DUMP_HFNET)) {
      ofstream ofs ("out1.bin", ios::binary);
      ofs.write((char*)out2, outputSize1);
      ofs.close();
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_HFNET)) 
      << "the scales: " << scale1 << " " << scale2 << endl;
    vector<float> output1(outputSize1);
    __TIC__(SOFTMAX)
#ifndef HW_SOFTMAX
    for (int i=0; i<outputH*outputW; ++i) {
      float sum{0.0f};
      int pos = i*channel1;
      for (int j=0; j<channel1; ++j){
        output1[pos + j] = std::exp(out1[j + pos]*scale1);
        sum += output1[pos + j];
      }
      for (int j=0; j<channel1; ++j){
        output1[pos+j] /= sum;
      }
    }
 #else
    vitis::ai::softmax(out1, scale1, channel1, outputH*outputW, output1.data());
 #endif
    __TOC__(SOFTMAX)

    __TIC__(HEATMAP)
    int reduced_size = (channel1-1)*outputH*outputW;
    vector<float> heatmap(reduced_size);
    // remove heatmap[-1,:,:]
    for (size_t i = 0; i < outputH*outputW; i++) {
      memcpy(heatmap.data()+i*(channel1-1), output1.data()+i*channel1, sizeof(float)*(channel1-1));
    }
    __TOC__(HEATMAP)
    
    vector<float> tmp;
    tmp.reserve(reduced_size);
    vector<int> xs, ys;
    vector<size_t> keep_inds;
    vector<float> ptscore;
    __TIC__(SORT)
    for (size_t m =0u; m<outputH; ++m){
      for (size_t i =0u; i<8; ++i){
        for (size_t n =0u; n < outputW; ++n){
          for (size_t j =0u; j<8; ++j){
            tmp.push_back(heatmap.at(i*8 + j + (m*outputW + n)*64)); //transpose heatmap
            if (tmp.back() >conf_thresh){
              ys.push_back(m*8+i);
              xs.push_back(n*8+j);
              ptscore.push_back(tmp.back());
            }
          }
        }
      }
    }
    __TOC__(SORT)

    __TIC__(NMS)
    nms_fast(xs, ys, ptscore, keep_inds, sWidth, sHeight);
    __TOC__(NMS)

    __TIC__(L2_NORMAL)
    LOG_IF(INFO, ENV_PARAM(DEBUG_HFNET)) 
      << "L2 normal: channel " << channel2 << " h: " << outputH << " w: " << outputW;
    auto output2 = L2_normalization(out2, channel2, output2H*output2W);
    __TOC__(L2_NORMAL)

    __TIC__(DESC)
    for (size_t i = 0; i < keep_inds.size(); ++i) {
        pair<float, float> pt(float(xs[keep_inds[i]]), float(ys[keep_inds[i]]));
        result_.keypoints.push_back(pt);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_HFNET)) 
      << "keypoints size: " << result_.keypoints.size();
    result_.descriptor = grid_sample(output2.data(), result_.keypoints, channel2, output2H, output2W);
    __TOC__(DESC)

    if(ENV_PARAM(DEBUG_HFNET)) {
      if (result_.descriptor.size() > 0) {
        cout<<"desc of pt0 :"<<endl;
        for (int i=0; i< 64; ++i){
            if(i%8==0){ cout<<endl;}
            cout<<result_.descriptor[0][i]<<"  ";
        }
        cout << endl;
        cout<<"desc of pt1 :"<<endl;
        for (int i=0; i< 64; ++i){
            if(i%8==0){ cout<<endl;}
            cout<<result_.descriptor[1][i]<<"  ";
        }
        cout << endl;
      }
    }
    results_.push_back(result_);
  }
  return true;
}

vector<vector<float>> HFnetImp::netvlad(const vitis::ai::library::OutputTensor& otensor1,
	     const vitis::ai::library::OutputTensor& otensor2, size_t count) {
  size_t vladH = otensor1.height;
  size_t vladW = otensor1.width;
  float scale3 = tensor_scale(otensor1);
  float scale4 = tensor_scale(otensor2);
  size_t size3 = vladH*vladW*channel3;
  //size_t size4 = vladH*vladW*channel4;
  LOG_IF(INFO, ENV_PARAM(DEBUG_HFNET)) 
    << "netvlad info: " << endl
    << "tensor1: " << vladH << " " << vladW << " " <<  channel3 << endl
    << "tensor2: " << otensor2.height << " " << otensor2.width << " " <<  channel4 << endl
    << "scales: " << scale3 << " " << scale4 << endl
    << "size: " << otensor1.size << " " << otensor2.size;
 
  vector<vector<float>> fc_input(count);
  for(size_t b = 0; b < count; ++b) {
    vector<float> output(size3);
    int8_t* out3 = (int8_t*)otensor1.get_data(b);
    int8_t* out4 = (int8_t*)otensor2.get_data(b);
    //softmax
    __TIC__(VLAD_SOFTMAX)
#ifndef HW_SOFTMAX
    for (size_t i=0; i<vladH*vladW; ++i) {
      float sum{0.0f};
      int pos = i*channel3;
      for (size_t j=0; j<channel3; ++j){
        output[pos + j] = std::exp(out3[j + pos]*scale3);
        sum += output[pos + j];
      }
      for (size_t j=0; j<channel3; ++j){
        output[pos+j] /= sum;
      }
    }
 #else
    vitis::ai::softmax(out3, scale3, channel3, vladH*vladW, output.data());
 #endif
    __TOC__(VLAD_SOFTMAX)

    __TIC__(MIX_COMPUTE)
    //mix_compute : matrix sub -> mul -> reduction_sum
    //auto mix_compute(cluster, out4, scale4, output,  channel3, channel4);
    vector<float> mix_res(cluster.size(), 0);
    float temp;
    for (size_t g = 0; g < vladH * vladW; ++g) {
      for (size_t c1 = 0; c1 < channel3; ++c1) {
        for (size_t c2 = 0; c2 < channel4; ++c2) {
          temp = cluster[c1*channel4+c2] - out4[g*channel4+c2] * scale4;
          mix_res[c1*channel4+c2] += output[g*channel3+c1] * temp;
        }
      }
    }
    __TOC__(MIX_COMPUTE)

    __TIC__(VLAD_L2NORMALS)
    vector<float> mix_l2 = L2_normalization(mix_res.data(), channel3, channel4);
    auto fc_in =  L2_normalization(mix_l2.data(), clu_size, 1);
    fc_input[b] =  L2_normalization(fc_in.data(), clu_size, 1);
    __TOC__(VLAD_L2NORMALS)
  }
  return fc_input;
}

void HFnetImp::set_input(vitis::ai::library::InputTensor& tensor, float mean, float scale, vector<Mat>& img) {
  float scale0 = vitis::ai::library::tensor_scale(tensor);
  size_t isize = tensor.size / tensor.batch;
  __TIC__(RESIZE)
  for (size_t i = 0; i < img.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_HFNET))
      << "batch " << i << endl
      << "img info(h,w): " << img[i].rows << " " << img[i].cols << endl
      << "dpu info(h,w): " << sHeight << " " <<  sWidth << endl
      << "scale: " << scale0 << " size: " << isize << endl;
    Mat mat;
    if (img[i].rows == sHeight && img[i].cols == sWidth) {
      mat = img[i];
    } else {
      resize(img[i], mat, cv::Size(sWidth, sHeight));
    }
  __TOC__(RESIZE)

  __TIC__(SET_IMG)
    int8_t* input_ptr = (int8_t*) tensor.get_data(i);
#ifndef NEON_GREY
    for (size_t j = 0; j < isize; ++j) {
      input_ptr[j] = static_cast<int8_t>((*(mat.data+j) - mean) * scale0 * scale);
      //input_ptr[j] = static_cast<int8_t>(inv[j]*scale0);
    }
#else
    // neon method
#endif
  __TOC__(SET_IMG)
    if (ENV_PARAM(DUMP_HFNET)) {
      ofstream fout("fin_"+to_string(i)+".bin", ios::binary);
      fout.write((char*)input_ptr, sWidth*sHeight);
      fout.close();
      LOG(INFO) << "The input scale is : " << scale0;
    }
  }
}

// run the fadnet
void HFnetImp::hfnet_run(const std::vector<cv::Mat>& input_image) {
  auto input_tensor = inputs_[0];
  auto group = input_image.size() / batch_;
  auto rest = input_image.size() % batch_;
  auto img_iter = input_image.begin();
  auto img_end = img_iter;
  if (rest>0) group += 1;
  size_t count = batch_;
  for(size_t g = 0; g < group; ++g) {
    __TIC__(PREPROCESS)
    size_t dist = std::distance(img_iter, input_image.end());
    if (dist > batch_)
      img_end += batch_;
    else {
      count = std::distance(img_iter, input_image.end());
      img_end = input_image.end();
      //cout << "count: " << count << endl;
    }
    vector<Mat> imgs(img_iter, img_end);
    img_iter = img_end;
    // set mean=128, scale=1/128.0
    set_input(input_tensor, 128, 0.0078125, imgs);
    __TOC__(PREPROCESS)
    //if(ifstream("./input.bin").read((char*)input_tensor.get_data(0), input_tensor.size/input_tensor.batch).good())
    //  cout << "succeed to read file";
    __TIC__(DPU_RUN)
    task_->run(0u);
    __TOC__(DPU_RUN)

    __TIC__(POSTPROCESS)
    verifyOutput(count);
    __TOC__(POSTPROCESS)

    __TIC__(VLAD_POST)
    auto fc_input = netvlad(outputs_[2], outputs_[3], count);
    __TOC__(VLAD_POST)

    auto fc_itensor = task_->getInputTensor(2u)[0];
    float fc_iscale = tensor_scale(fc_itensor);
    __TIC__(VLAD_FC)
    for (size_t j = 0; j < count; ++j) {
      results_[g*batch_+j].scale_w = imgs[j].cols/(float)sWidth;
      results_[g*batch_+j].scale_h = imgs[j].rows/(float)sHeight;
      int8_t* fc_inptr = (int8_t*)fc_itensor.get_data(j);
      for(size_t i = 0; i < fc_input[j].size(); ++i) {
        fc_inptr[i] = static_cast<int8_t>(fc_input[j][i]*fc_iscale);
      }
    }
    task_->run(2u);
    auto fc_otensor = task_->getOutputTensor(2u)[0];
    float fc_oscale = tensor_scale(fc_otensor);

    for (size_t j = 0; j < count; ++j) {
      int8_t* fc_res = (int8_t*)fc_otensor.get_data(j);
      results_[g*batch_+j].vlad = L2_normalization(fc_res, fc_otensor.size/fc_otensor.batch, 1);
    }
    __TOC__(VLAD_FC)
    LOG_IF(INFO, ENV_PARAM(DEBUG_HFNET))
      << "fc scale: " << fc_iscale << " " << fc_oscale << endl
      << "fc_out name: " << fc_otensor.name << endl;
  }
}

std::vector<HFnetResult> HFnetImp::run(const std::vector<cv::Mat>& imgs) {
  hfnet_run(imgs);
  return results_;
}

size_t HFnetImp::get_input_batch() { return task_->get_input_batch(0, 0); }
int HFnetImp::getInputWidth() const {
  return task_->getInputTensor(0u)[0].width;
}
int HFnetImp::getInputHeight() const {
  return task_->getInputTensor(0u)[0].height;
}

}  // namespace ai
}
