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
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <vitis/ai/library/tensor.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <math.h>
#include <utility>

#include <vitis/ai/profiling.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/image_util.hpp>

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

DEF_ENV_PARAM(DEBUG_FADNET, "0");
DEF_ENV_PARAM(DUMP_FADNET, "0");

using namespace std;
using namespace cv;

#ifdef ENABLE_NEON
static float vector_mul(const int8_t* input1, const int8_t* input2, float scale,
	       unsigned int group) {
  unsigned int batch = group / 32;
  int32x4_t sum = vdupq_n_s32(0);
  for (unsigned int i = 0; i < batch; ++i) {
    int8x8x4_t q01 = vld4_s8(input1);
    int8x8x4_t q02 = vld4_s8(input2);
    int16x8_t q2 = vmull_s8(q01.val[0], q02.val[0]);
    int16x8_t q3 = vmull_s8(q01.val[1], q02.val[1]);
    int16x8_t q4 = vmull_s8(q01.val[2], q02.val[2]);
    int16x8_t q5 = vmull_s8(q01.val[3], q02.val[3]);

    int16x4_t d10 = vget_low_s16(q2);
    int16x4_t d11 = vget_low_s16(q3);
    int16x4_t d12 = vget_low_s16(q4);
    int16x4_t d13 = vget_low_s16(q5);

    int16x4_t d14 = vget_high_s16(q2);
    int16x4_t d15 = vget_high_s16(q3);
    int16x4_t d16 = vget_high_s16(q4);
    int16x4_t d17 = vget_high_s16(q5);

    int32x4_t q10 = vaddl_s16(d10, d14);
    int32x4_t q11 = vaddl_s16(d11, d15);
    int32x4_t q12 = vaddl_s16(d12, d16);
    int32x4_t q13 = vaddl_s16(d13, d17);
    
    sum = vaddq_s32(sum, q10);
    sum = vaddq_s32(sum, q11);
    sum = vaddq_s32(sum, q12);
    sum = vaddq_s32(sum, q13);
    input1+=32;
    input2+=32;
  }
  int32x2_t s_low  = vget_low_s32(sum);
  int32x2_t s_high = vget_high_s32(sum);
  int32x2_t s = vpadd_s32(s_low, s_high);

  vector<int32_t> temp(2);
  vst1_s32(temp.data(), s);
  return (float)(temp[0]+temp[1])*scale;
}
#endif

void cost_volume(vitis::ai::library::OutputTensor input_l, vitis::ai::library::OutputTensor input_r, 
                        vitis::ai::library::InputTensor output) {
  __TIC__(FIRST_CHANNEL)
  CHECK_EQ(input_l.size, input_r.size) << "The two inputs' size not same, please check.";
  LOG_IF(INFO, ENV_PARAM(DEBUG_FADNET))
      << "cost_volume inputs shape: " << std::endl
      << "left: (" << input_l.height << ", " << input_l.width << ", " << input_l.channel << ") " 
      << "right: (" << input_r.height << ", " << input_r.width << ", " << input_r.channel << ") ";
  vector<float> temp(output.size);
      
  for (size_t a = 0; a < input_l.batch; ++a) {
    int8_t* left_input = (int8_t*)input_l.get_data(a);
    float left_scale = vitis::ai::library::tensor_scale(input_l);
    int8_t* right_input = (int8_t*)input_r.get_data(a);
    float right_scale = vitis::ai::library::tensor_scale(input_r);
  
    int8_t* volume = (int8_t*)output.get_data(a);
    float v_scale = vitis::ai::library::tensor_scale(output);
    size_t out_size = output.height * output.width * output.channel;
  
#ifdef ENABLE_NEON
    float dev_num = 1.0/input_l.channel;
    float real_scale = right_scale * left_scale * v_scale * dev_num;
    // compute the first channel of the cost volume
  
    __TOC__(FIRST_CHANNEL)
    
    __TIC__(REST_CHANNEL)
    // compute the rest channels of the cost volume
    for(size_t b = 0; b < input_l.height; ++b)
      for(size_t i = 0; i < output.channel; ++i)
        for(size_t c = i, e = 0; c < input_l.width; ++c, ++e) {
          size_t pos = b*input_l.width;
          size_t out_pos = (b*output.width + c)*output.channel + i;
          temp[out_pos] = vector_mul(left_input+(pos+c)*input_l.channel,
                          right_input+(pos+e)*input_l.channel,
                          real_scale,
                          input_l.channel);
        }
    __TOC__(REST_CHANNEL)
    __TIC__(LEAKY_RELU)
    // compute the average figure of volume
    for (size_t i = 0; i < out_size; ++i) {
      if (temp[i] >= 0)
        volume[i] = (int8_t)round(temp[i]);
      else {
        // leaky relu
        temp[i] *= 0.1015625;
        if (temp[i] - floor(temp[i]) == 0.5)
          volume[i] = (int8_t)ceil(temp[i]);
        else
          volume[i] = (int8_t)round(temp[i]);
      }
    }
    __TOC__(LEAKY_RELU)
#else
    // compute the first channel of the cost volume
    for (size_t b = 0; b < input_l.height; ++b)
      for (size_t c = 0; c < input_l.width; ++c) {
        float sum = 0.0;
        for (size_t d = 0; d < input_l.channel; ++d) {
          size_t pos = (b*input_l.width + c)*input_l.channel + d;
          // only write the first channel
          sum += (float)left_input[pos] * (float)right_input[pos];
        }
        size_t out_pos = (b*output.width + c)*output.channel;
        temp[out_pos] = sum;
      }
    __TOC__(FIRST_CHANNEL)
    
    __TIC__(REST_CHANNEL)
    // compute the rest channels of the cost volume
    for(size_t b = 0; b < input_l.height; ++b)
      for(size_t i = 1; i < output.channel; ++i)
        for(size_t c = i, e = 0; c < input_l.width; ++c, ++e) {
          float sum = 0.0;
          for(size_t d = 0; d < input_l.channel; ++d) {
            size_t pos = b*input_l.width*input_l.channel;
            sum += (float)left_input[pos + c*input_l.channel + d]
                   * (float)right_input[pos + e*input_l.channel + d];
          }
          size_t out_pos = a*out_size + (b*output.width + c)*output.channel + i;
  	      temp[out_pos] = sum;
      	}
    __TOC__(REST_CHANNEL)
    __TIC__(LEAKY_RELU)
    // compute the average figure of volume
    float real_scale = right_scale * left_scale * v_scale;
    float tmp = 0.0;
    for (size_t i = 0; i < out_size; ++i) {
      tmp = temp[i];
      if (tmp >= 0)
        volume[i] = (int8_t)round(tmp/(float)input_l.channel * real_scale);
      else {
        // leaky relu
        tmp = tmp/(float)input_l.channel * real_scale * 0.1015625;
        if (tmp - floor(tmp) == 0.5)
          volume[i] = (int8_t)ceil(tmp);
        else
          volume[i] = (int8_t)round(tmp);
      }
    }
    __TOC__(LEAKY_RELU)

#endif
    if(ENV_PARAM(DUMP_FADNET)) {
      for(size_t i = 0; i < output.batch; ++i) {
        std::ofstream ofs("cost_volume_" + to_string(i) + ".bin", ios::binary);
        ofs.write((char*)volume, out_size);
        ofs.close();
      }
    }
  }
}

int8_t dpu_round(float num) {
  if(num - floor(num) == 0.5) return ceil(num);
  else return round(num);
}

#ifdef ENABLE_NEON
void shift_and_copy(int8_t* output, int8_t* input, size_t size, int shift_num) {
  int group = size/64;
  int8x16_t all_1 = vdupq_n_s8(1);
  for(int i = 0; i < group; ++i) {
    int8x16x4_t q01 = vld4q_s8(input);
    if (shift_num > 0) {
      int8x16_t q2 = vshrq_n_s8(q01.val[0], shift_num);
      int8x16_t q3 = vandq_s8(vshrq_n_s8(q01.val[0], shift_num-1), all_1);
      q01.val[0] = vaddq_s8(q2, q3);

      q2 = vshrq_n_s8(q01.val[1], shift_num);
      q3 = vandq_s8(vshrq_n_s8(q01.val[1], shift_num-1), all_1);
      q01.val[1] = vaddq_s8(q2, q3);

      q2 = vshrq_n_s8(q01.val[2], shift_num);
      q3 = vandq_s8(vshrq_n_s8(q01.val[2], shift_num-1), all_1);
      q01.val[2] = vaddq_s8(q2, q3);

      q2 = vshrq_n_s8(q01.val[3], shift_num);
      q3 = vandq_s8(vshrq_n_s8(q01.val[3], shift_num-1), all_1);
      q01.val[3] = vaddq_s8(q2, q3);
    } else {
      int shift_new = abs(shift_num);
      q01.val[0] = vshlq_n_s8(q01.val[0], shift_new);
      q01.val[1] = vshlq_n_s8(q01.val[1], shift_new);
      q01.val[2] = vshlq_n_s8(q01.val[2], shift_new);
      q01.val[3] = vshlq_n_s8(q01.val[3], shift_new);
    }
    vst4q_s8(output, q01);
    input+=64;
    output+=64;
  }
  int rest = (size - group*64) /16;
  for (int i = 0; i < rest; ++i) {
    int8x16_t q4 = vld1q_s8(input);
    if (shift_num>0) {
      int8x16_t q2 = vshrq_n_s8(q4, shift_num);
      int8x16_t q3 = vandq_s8(vshrq_n_s8(q4, shift_num-1), all_1);
      q4 = vaddq_s8(q2, q3);
    } else {
      int shift_new = abs(shift_num);
      q4 = vshlq_n_s8(q4, shift_new);
    }
    vst1q_s8(output, q4);
    input+=16;
    output+=16;
  }
}
#endif

void copy_into_tensor (const vector<int8_t>& input,
	       	vitis::ai::library::InputTensor tensor, int o_fixpos) {
  __TIC__(COPY_INPUT)
  int i_fixpos = tensor.fixpos;
  auto size = tensor.height * tensor.width * tensor.channel;
  LOG_IF(INFO, ENV_PARAM(DEBUG_FADNET))
    << tensor.name << ": " << i_fixpos << " " << o_fixpos;
  for (size_t b = 0; b < tensor.batch; ++b) {
    auto data = (int8_t*)tensor.get_data(b);
    if(i_fixpos == o_fixpos) {
      memcpy(data, input.data() + b*size, size);
    } else {
  
      float o_scale = exp2f(-1.0 * o_fixpos);
      float i_scale = vitis::ai::library::tensor_scale(tensor);
      float scale = o_scale * i_scale;
#ifdef ENABLE_NEON
      int shift_num = o_fixpos - i_fixpos;
      shift_and_copy(data, (int8_t*)input.data() + b*size, size, shift_num);
      int rest = size % 16;
      if(rest > 0) {
        for(size_t i = size-rest; i < size; ++i)
          *(data+i) = dpu_round(input[i + b*size] * scale);
      }
    
#else
      for(size_t i = 0; i < size; ++i)
        data[i] = dpu_round(input[i+b*size] * scale);
#endif
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_FADNET))
      << "tensor shape: (" << tensor.height << ", " << tensor.width << ", " << tensor.channel << ").";
    if(ENV_PARAM(DUMP_FADNET)) {
      for(size_t i = 0; i < tensor.batch; ++i) {
        std::ofstream ofs("batch_"+tensor.name + to_string(i), ios::binary);
        ofs.write((char*)data, size);
        ofs.close();
      }
    }
  }
  __TOC__(COPY_INPUT)
}

vector<int8_t> copy_from_tensor (const vitis::ai::library::InputTensor tensor) {
  auto size = tensor.height * tensor.width * tensor.channel;
  vector<int8_t> output(tensor.size);
  for (size_t b = 0; b < tensor.batch; ++b) {
    auto data = tensor.get_data(b);
    memcpy(output.data()+b*size, data, size);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_FADNET))
    << "tensor shape: (" << tensor.height << ", " << tensor.width << ", " << tensor.channel;
  return output;
}

vector<int8_t> copy_from_tensor (const vitis::ai::library::OutputTensor tensor) {
  auto size = tensor.height * tensor.width * tensor.channel;
  vector<int8_t> output(tensor.size);
  for (size_t b = 0; b < tensor.batch; ++b) {
    auto data = tensor.get_data(b);
    memcpy(output.data()+b*size, data, size);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_FADNET))
    << "tensor shape: (" << tensor.height << ", " << tensor.width << ", " << tensor.channel;
  return output;
}

// reorder the tensors with name
vector<vitis::ai::library::InputTensor> sort_tensors (
            const vector<vitis::ai::library::InputTensor>& tensors, vector<string>& layer_names) {
  vector<vitis::ai::library::InputTensor> ordered_tensors;
  for (auto i = 0u; i < layer_names.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].name.find(layer_names[i]) != std::string::npos) {
        ordered_tensors.push_back(tensors[j]);
        break;
      }
  return ordered_tensors;
}

vector<vitis::ai::library::OutputTensor> sort_tensors (
            const vector<vitis::ai::library::OutputTensor>& tensors, vector<string>& layer_names) {
  vector<vitis::ai::library::OutputTensor> ordered_tensors;
  for (auto i = 0u; i < layer_names.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].name.find(layer_names[i]) != std::string::npos) {
        ordered_tensors.push_back(tensors[j]);
        break;
      }
  return ordered_tensors;
}

// run the fadnet
vector<Mat> FADnet_run(vector<unique_ptr<vitis::ai::DpuTask>>& task,
                       const vector<pair<cv::Mat, cv::Mat>>& input_images) {
  vector<cv::Mat> left_mats;
  vector<cv::Mat> right_mats;
  auto input_tensor_left = task[0]->getInputTensor(0u)[0];
  auto sWidth = input_tensor_left.width;
  auto sHeight = input_tensor_left.height;

  __TIC__(FADNET_RESIZE)
  for(size_t i = 0; i < input_tensor_left.batch; ++i) {
    cv::Mat left_mat;
    resize(input_images[i].first, left_mat, cv::Size(sWidth, sHeight));
    left_mats.push_back(left_mat);
    cv::Mat right_mat;
    resize(input_images[i].second, right_mat, cv::Size(sWidth, sHeight));
    right_mats.push_back(right_mat);
  }
  __TOC__(FADNET_RESIZE)

  // ### kernel 0 part ###
  __TIC__(FADNET_SET_IMG_LEFT)
  task[0]->setImageRGB(left_mats);
  // store the input 
  vector<int8_t> data_left = copy_from_tensor(input_tensor_left);
  __TOC__(FADNET_SET_IMG_LEFT)

  __TIC__(FADNET_DPU_LEFT)
  task[0]->run(0u);
  __TOC__(FADNET_DPU_LEFT)

  __TIC__(FADNET_STORE_KERNEL_0)
  // store the outputs of kernel_0
  auto outputs_l_unsort = task[0]->getOutputTensor(0u);
  vector<string> output_names_k0 = {"conv1", "conv2", "conv3"};
  auto outputs_l = sort_tensors(outputs_l_unsort, output_names_k0);

  vector<int8_t> data_conv1_l = copy_from_tensor(outputs_l[0]);
  vector<int8_t> data_conv2_l = copy_from_tensor(outputs_l[1]);
  vector<int8_t> data_conv3a_l = copy_from_tensor(outputs_l[2]);
  __TOC__(FADNET_STORE_KERNEL_0)

  // ### kernel 1 part ###
  __TIC__(FADNET_SET_IMG_RIGHT)
  auto input_tensor_right = task[1]->getInputTensor(0u)[0];
  task[1]->setImageRGB(right_mats);
  vector<int8_t> data_right = copy_from_tensor(input_tensor_right);
  __TOC__(FADNET_SET_IMG_RIGHT)

  __TIC__(FADNET_DPU_RIGHT)
  task[1]->run(0u);
  __TOC__(FADNET_DPU_RIGHT)

  //  cost volume
  __TIC__(FADNET_COST_VOLUME)
  auto output_tensor_l = outputs_l[2];
  auto output_tensor_r = task[1]->getOutputTensor(0u)[0];

  auto input_kernel_2_unsort = task[2]->getInputTensor(0u);
  vector<string> input_names_k2 = {"3585", "input_34", "3581",
                                   "3582", "3583", "4236_inserted_fix_30",
                                   "4236_inserted_fix_16", "4237"};
  auto input_kernel_2 = sort_tensors(input_kernel_2_unsort, input_names_k2); 
  if(ENV_PARAM(DEBUG_FADNET)) {
    for (size_t i = 0; i < input_kernel_2.size(); ++i)
      LOG(INFO) << input_kernel_2[i];
  }
  cost_volume(output_tensor_l, output_tensor_r, input_kernel_2[0]);
  __TOC__(FADNET_COST_VOLUME)

  // run the rest kernel
  __TIC__(FADNET_COPY_INPUT_K2)
  copy_into_tensor(data_conv3a_l, input_kernel_2[1], outputs_l[2].fixpos);
  copy_into_tensor(data_conv1_l,  input_kernel_2[2], outputs_l[0].fixpos);
  copy_into_tensor(data_conv2_l,  input_kernel_2[3], outputs_l[1].fixpos);
  copy_into_tensor(data_left,     input_kernel_2[4], input_tensor_left.fixpos);
  copy_into_tensor(data_left,     input_kernel_2[5], input_tensor_left.fixpos);
  copy_into_tensor(data_left,     input_kernel_2[6], input_tensor_left.fixpos);
  copy_into_tensor(data_right,    input_kernel_2[7], input_tensor_right.fixpos);
  __TOC__(FADNET_COPY_INPUT_K2)

  //exit(0);
  __TIC__(FADNET_DPU_LAST)
  task[2]->run(0u);
  __TOC__(FADNET_DPU_LAST)


  __TIC__(FADNET_POST_ARM)
  vector<Mat> rets;
  int ret_height = input_images[0].first.rows;
  int ret_width = input_images[0].first.cols;
  auto final_tensor = task[2]->getOutputTensor(0u)[0];

  if(ENV_PARAM(DUMP_FADNET)) {
    for (size_t b = 0; b < final_tensor.batch; ++b) {
      std::ofstream ofs("tensor_res_" + to_string(b) + ".bin", ios::binary);
      ofs.write((char*)final_tensor.get_data(b), final_tensor.width * final_tensor.height);
      ofs.close();
    }
  }

  float f_scale = vitis::ai::library::tensor_scale(final_tensor);
  for (size_t b = 0; b < final_tensor.batch; ++b) {
    Mat final_img(final_tensor.height, final_tensor.width, CV_8UC1);
    Mat ret;
    auto final_data = (int8_t*)final_tensor.get_data(b);
    if (f_scale == 1.f) {
      final_img = Mat(Size(final_tensor.width, final_tensor.height), CV_8UC1, (void*)final_data);
    } else {
      for(size_t i = 0; i < final_tensor.width * final_tensor.height; ++i)
        final_img.data[i] = (uint8_t)(final_data[i] * f_scale);
    }
    resize(final_img, ret, cv::Size(ret_width, ret_height));
    rets.push_back(ret);
  }
  __TOC__(FADNET_POST_ARM)
  return rets;
}



int main(int argc, char* argv[]) {
  // A kernel name, it should be samed as the dnnc result. e.g.
  // /usr/share/vitis_ai_library/models/FADNet_0_pt/FADNet_0_pt.xmodel
  auto kernel_name_0 = argv[1];
  auto kernel_name_1 = argv[2];
  auto kernel_name_2 = argv[3];

  Mat img_l = cv::imread(argv[4]);
  Mat img_r = cv::imread(argv[5]);

  // Create a dpu task object.
  vector<unique_ptr<vitis::ai::DpuTask>> task;
  task.emplace_back( vitis::ai::DpuTask::create(kernel_name_0));
  task.emplace_back( vitis::ai::DpuTask::create(kernel_name_1));
  task.emplace_back( vitis::ai::DpuTask::create(kernel_name_2));

  // Set the mean values and scale values.
  task[0]->setMeanScaleBGR({103.53, 116.28, 123.675},
                        {0.017429, 0.017507, 0.01712475});
  task[1]->setMeanScaleBGR({103.53, 116.28, 123.675},
                        {0.017429, 0.017507, 0.01712475});
  vector<pair<Mat, Mat>> imgs;
  imgs.push_back(make_pair(img_l, img_r));
  imgs.push_back(make_pair(img_l, img_r));
  imgs.push_back(make_pair(img_l, img_r));

  // Execute the FADnet post-processing.
  auto result = FADnet_run(task, imgs);
  //imshow("", result[0]);
  //waitKey(0);
  imwrite("result_fadnet.jpg", result[0]);

  return 0;
}
