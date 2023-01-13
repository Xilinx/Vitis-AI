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
#include "./psmnet.hpp"

#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/weak.hpp>

DEF_ENV_PARAM(DEBUG_PSMNET, "0");
DEF_ENV_PARAM(DEBUG_COST_VOLUME, "0");
DEF_ENV_PARAM(DUMP_PSMNET, "0");
DEF_ENV_PARAM(PSMNET_1, "0");
DEF_ENV_PARAM(PSMNET_SHOW, "0");
DEF_ENV_PARAM(IGNORE_DPU, "0");

using namespace std;

//DEF_ENV_PARAM_2(PSMNET_MODEL_DIR, "./PSMnet", std::string);

DEF_ENV_PARAM_2(PSMNET_MODEL_0, "PSMNet_pruned_0_pt/PSMNet_pruned_0_pt.xmodel", std::string);
DEF_ENV_PARAM_2(PSMNET_MODEL_1, "PSMNet_pruned_1_pt/PSMNet_pruned_1_pt.xmodel", std::string);
DEF_ENV_PARAM_2(PSMNET_MODEL_2, "PSMNet_pruned_2_pt/PSMNet_pruned_2_pt.xmodel", std::string);

namespace vitis {
namespace ai {
class PsmNetImp : public PsmNet {
 public:
  explicit PsmNetImp();
  PsmNetImp(const PsmNetImp&) = delete;
  PsmNetImp& operator=(const PsmNetImp&) = delete;

 public:
  virtual ~PsmNetImp();
  virtual void run(
      const std::vector<std::pair<cv::Mat, cv::Mat>>& imgs) override;
  virtual std::vector<cv::Mat> get_result() override;
  virtual size_t get_input_batch() override;
  virtual int get_input_width() const override;
  virtual int get_input_height() const override;

 private:
  void PSMnet_run(const vector<pair<cv::Mat, cv::Mat>>& input_images);
#ifdef ENABLE_CV_ACCEL
  CostVolumeAccel* cvAccel;
#endif

 public:
  std::vector<std::unique_ptr<vitis::ai::DpuTask>> tasks_;
  vector<vitis::ai::library::InputTensor> inputs_l_;
  vector<vitis::ai::library::OutputTensor> outputs_l_;
  vector<vitis::ai::library::InputTensor> inputs_r_;
  vector<vitis::ai::library::OutputTensor> outputs_r_;

//  std::vector<std::unique_ptr<vai_resize>> vai_res_;
//  std::unique_ptr<DpuSfm> dpu_sfm_;
  std::vector<std::unique_ptr<Resize>> cpu_res_;
  std::unique_ptr<CPUsfm> cpu_sfm_;
};

static vector<vitis::ai::library::InputTensor> sort_tensors(
    const vector<vitis::ai::library::InputTensor>& tensors,
    vector<string>& layer_names);
static vector<vitis::ai::library::OutputTensor> sort_tensors(
    const vector<vitis::ai::library::OutputTensor>& tensors,
    vector<string>& layer_names);

PsmNet::PsmNet() {}

PsmNet::~PsmNet() {}

std::unique_ptr<PsmNet> PsmNet::create() {
  return std::make_unique<PsmNetImp>();
}

//PsmNetImp::PsmNetImp() : tasks_{}, vai_res_{}, dpu_sfm_{}, cpu_res_{}, cpu_sfm_{} {
PsmNetImp::PsmNetImp() : tasks_{}, cpu_res_{}, cpu_sfm_{} {
  tasks_.emplace_back(vitis::ai::DpuTask::create(
      ENV_PARAM(PSMNET_MODEL_0)));
  tasks_.emplace_back(vitis::ai::DpuTask::create(
      ENV_PARAM(PSMNET_MODEL_1)));
  tasks_.emplace_back(vitis::ai::DpuTask::create(
      ENV_PARAM(PSMNET_MODEL_2)));
  tasks_[0]->setMeanScaleBGR({103.53, 116.28, 123.675},
                             {0.017429, 0.017507, 0.01712475});
  tasks_[1]->setMeanScaleBGR({103.53, 116.28, 123.675},
                             {0.017429, 0.017507, 0.01712475});
  // ### kernel 0 part ###
  vector<string> kernel0_task0_outs = {"input_106", "1478",     "input_152",
                                       "input_148", "input_144", "input_140"};
  vector<string> kernel0_task1_inps = {"input_106", "1478", "fix_0",
                                       "fix_1",     "fix_2", "fix_3"};
  // ### kernel 1 part ###
  vector<string> kernel1_task0_outs = {"input_262", "2591",     "input_308",
                                       "input_304", "input_300", "input_296"};
  vector<string> kernel1_task1_inps = {"input_262", "2591", "fix_0",
                                       "fix_1",     "fix_2", "fix_3"};

  auto outputs_l_unsort = tasks_[0]->getOutputTensor(0u);
  outputs_l_ = sort_tensors(outputs_l_unsort, kernel0_task0_outs);

  auto inputs_l_unsort = tasks_[0]->getInputTensor(1u);
  inputs_l_ = sort_tensors(inputs_l_unsort, kernel0_task1_inps);
  if (ENV_PARAM(DEBUG_PSMNET)) {
    for (size_t i = 0; i < outputs_l_.size(); ++i)
      LOG(INFO) << outputs_l_unsort[i].name;
    for (size_t i = 0; i < inputs_l_.size(); ++i)
      LOG(INFO) << inputs_l_unsort[i].name;
  }
  auto outputs_r_unsort = tasks_[1]->getOutputTensor(0u);
  outputs_r_ = sort_tensors(outputs_r_unsort, kernel1_task0_outs);

  auto inputs_r_unsort = tasks_[1]->getInputTensor(1u);
  inputs_r_ = sort_tensors(inputs_r_unsort, kernel1_task1_inps);

#ifdef ENABLE_CV_ACCEL
  cvAccel = new CostVolumeAccel("/run/media/mmcblk0p1/dpu.xclbin", 0);
#endif

  if (ENV_PARAM(DEBUG_PSMNET)) {
    for (size_t i = 0; i < outputs_r_.size(); ++i)
      LOG(INFO) << outputs_r_unsort[i].name;
    for (size_t i = 0; i < inputs_r_.size(); ++i)
      LOG(INFO) << inputs_r_unsort[i].name;
  }

//#ifndef ENABLE_AIE
  for (auto i : {2, 3, 4, 5}) {
    cpu_res_.emplace_back(std::make_unique<Resize>(outputs_l_[i], inputs_l_[i]));
  }
  for (auto i : {2, 3, 4, 5}) {
    cpu_res_.emplace_back(std::make_unique<Resize>(outputs_r_[i], inputs_r_[i]));
  }
  cpu_sfm_ = std::make_unique<CPUsfm>(tasks_[2]->getOutputTensor(0u)[0]);

//#else
//  for (auto i : {2, 3, 4, 5}) {
//    vai_res_.emplace_back(std::make_unique<vai_resize>(
//        "/media/sd-mmcblk0p1/dpu.xclbin", outputs_l_[i], inputs_l_[i]));
//  }
//  for (auto i : {2, 3, 4, 5}) {
//    vai_res_.emplace_back(std::make_unique<vai_resize>(
//        "/media/sd-mmcblk0p1/dpu.xclbin", outputs_r_[i], inputs_r_[i]));
//  }
//
//  dpu_sfm_ = std::make_unique<DpuSfm>("/media/sd-mmcblk0p1/dpu.xclbin",
//                                      tasks_[2]->getOutputTensor(0u)[0]);
//#endif
}
PsmNetImp::~PsmNetImp() {}

int8_t dpu_round(float num) {
  if (num - floor(num) == 0.5)
    return ceil(num);
  else
    return round(num);
}

#ifdef ENABLE_NEON
void shift_and_copy_neon(int8_t* input1, int8_t* input2, int8_t* output,
                         size_t size, int shift_num) {
  int8x16_t all_1 = vdupq_n_s8(1);
  int8x8x4_t d1 = vld4_s8(input1);
  int8x8x4_t d2 = vld4_s8(input2);
  int8x16x4_t q01;
  q01.val[0] = vcombine_s8(d1.val[0], d2.val[0]);
  q01.val[1] = vcombine_s8(d1.val[1], d2.val[1]);
  q01.val[2] = vcombine_s8(d1.val[2], d2.val[2]);
  q01.val[3] = vcombine_s8(d1.val[3], d2.val[3]);
  if (shift_num > 0) {
    int8x16_t q2 = vshrq_n_s8(q01.val[0], shift_num);
    int8x16_t q3 = vandq_s8(vshrq_n_s8(q01.val[0], shift_num - 1), all_1);
    q01.val[0] = vaddq_s8(q2, q3);

    q2 = vshrq_n_s8(q01.val[1], shift_num);
    q3 = vandq_s8(vshrq_n_s8(q01.val[1], shift_num - 1), all_1);
    q01.val[1] = vaddq_s8(q2, q3);

    q2 = vshrq_n_s8(q01.val[2], shift_num);
    q3 = vandq_s8(vshrq_n_s8(q01.val[2], shift_num - 1), all_1);
    q01.val[2] = vaddq_s8(q2, q3);

    q2 = vshrq_n_s8(q01.val[3], shift_num);
    q3 = vandq_s8(vshrq_n_s8(q01.val[3], shift_num - 1), all_1);
    q01.val[3] = vaddq_s8(q2, q3);
  } else {
    int shift_new = abs(shift_num);
    q01.val[0] = vshlq_n_s8(q01.val[0], shift_new);
    q01.val[1] = vshlq_n_s8(q01.val[1], shift_new);
    q01.val[2] = vshlq_n_s8(q01.val[2], shift_new);
    q01.val[3] = vshlq_n_s8(q01.val[3], shift_new);
  }
  vst4q_s8(output, q01);
}

void copy(int8_t* input1, int8_t* input2, int8_t* output, size_t size) {
  memcpy(output, input1, size);
  memcpy(output + size, input2, size);
}

void shift_and_copy(int8_t* output, int8_t* input, size_t size, int shift_num) {
  int group = size / 64;
  int8x16_t all_1 = vdupq_n_s8(1);
  for (int i = 0; i < group; ++i) {
    int8x16x4_t q01 = vld4q_s8(input);
    if (shift_num > 0) {
      int8x16_t q2 = vshrq_n_s8(q01.val[0], shift_num);
      int8x16_t q3 = vandq_s8(vshrq_n_s8(q01.val[0], shift_num - 1), all_1);
      q01.val[0] = vaddq_s8(q2, q3);

      q2 = vshrq_n_s8(q01.val[1], shift_num);
      q3 = vandq_s8(vshrq_n_s8(q01.val[1], shift_num - 1), all_1);
      q01.val[1] = vaddq_s8(q2, q3);

      q2 = vshrq_n_s8(q01.val[2], shift_num);
      q3 = vandq_s8(vshrq_n_s8(q01.val[2], shift_num - 1), all_1);
      q01.val[2] = vaddq_s8(q2, q3);

      q2 = vshrq_n_s8(q01.val[3], shift_num);
      q3 = vandq_s8(vshrq_n_s8(q01.val[3], shift_num - 1), all_1);
      q01.val[3] = vaddq_s8(q2, q3);
    } else {
      int shift_new = abs(shift_num);
      q01.val[0] = vshlq_n_s8(q01.val[0], shift_new);
      q01.val[1] = vshlq_n_s8(q01.val[1], shift_new);
      q01.val[2] = vshlq_n_s8(q01.val[2], shift_new);
      q01.val[3] = vshlq_n_s8(q01.val[3], shift_new);
    }
    vst4q_s8(output, q01);
    input += 64;
    output += 64;
  }
  int rest = (size - group * 64) / 16;
  for (int i = 0; i < rest; ++i) {
    int8x16_t q4 = vld1q_s8(input);
    if (shift_num > 0) {
      int8x16_t q2 = vshrq_n_s8(q4, shift_num);
      int8x16_t q3 = vandq_s8(vshrq_n_s8(q4, shift_num - 1), all_1);
      q4 = vaddq_s8(q2, q3);
    } else {
      int shift_new = abs(shift_num);
      q4 = vshlq_n_s8(q4, shift_new);
    }
    vst1q_s8(output, q4);
    input += 16;
    output += 16;
  }
}
#endif

// static int8_t get_output(int8_t* input_l, int8_t* input_r, int width,
//                          int channel, int h, int w, int m, int c, int
//                          act_fix) {
//   int8_t ret = 0;
//   if (w < m) {
//     // padding with zero
//   } else {
//     auto input = input_l;
//     auto offset = 0;
//     if (c < 32) {
//       input = input_l;
//       offset = 0;
//     } else {
//       input = input_r;
//       offset = 32;
//       w = w - m;
//     }
//     ret = input[h * width * channel + w * channel + c - offset];
//   }
//   auto cond = true;
//   cond = cond && h == 0;
//   cond = cond && w < 10;
//   cond = cond && (c == 0 || c == 1 || c == 31 || c == 32);
//   // cond = cond && m == 1;
//   cond = false;
//   LOG_IF(INFO, cond) << "h " << h << " "  //
//                      << "w " << w << " "  //
//                      << "m " << m << " "  //
//                      << "c " << c << " "  //
//                      << "ret " << std::hex << "0x" << (int)(ret & 0xff)
//                      << "act_fix " << act_fix << " "  //
//                      << std::dec << " "               //
//                      << endl;
//   ret = ret << act_fix;
//   return ret;
// }

#ifdef ENABLE_NEON
void cost_volume_neon(int8_t* input_l_ptr, int8_t* input_r_ptr,
                      int8_t* output_ptr, size_t height, size_t width,
                      size_t channel, int act_fix, size_t disp) {
  if (ENV_PARAM(DEBUG_COST_VOLUME)) {
    LOG(INFO) << "input_l_ptr " << std::hex << "0x" << (void*)input_l_ptr
              << std::dec << " "  //
              << "input_r_ptr " << std::hex << "0x" << (void*)input_r_ptr
              << std::dec << " "  //
              << "output_ptr " << std::hex << "0x" << (void*)output_ptr
              << std::dec << " "               //
              << "height " << height << " "    //
              << "width " << width << " "      //
              << "channel " << channel << " "  //
              << "act_fix " << act_fix << " "  //
              << "disp " << disp << " "        //
              << std::endl;
  }

  __TIC__(COST_VOLUME_NEON)
  if (act_fix == 0) {
    for (auto h = 0u; h < height; ++h) {
      for (auto w = 0u; w < width; ++w) {
        for (auto m = 0u; m < disp; ++m) {
          if (w < m) {
            int pos = ((h * width + w) * disp + m) * channel * 2;
            memset(output_ptr + pos, 0, channel * 2);
          } else {
            size_t i1_pos = (h * width + w) * channel;
            size_t i2_pos = (h * width + w - m) * channel;
            size_t o_pos = ((h * width + w) * disp + m) * channel * 2;
            memcpy(output_ptr + o_pos, input_l_ptr + i1_pos, channel);
            memcpy(output_ptr + o_pos + channel, input_r_ptr + i2_pos, channel);
          }
        }
      }
    }
  } else {
    for (auto h = 0u; h < height; ++h) {
      for (auto w = 0u; w < width; ++w) {
        for (auto m = 0u; m < disp; ++m) {
          if (w < m) {
            int pos = ((h * width + w) * disp + m) * channel * 2;
            memset(output_ptr + pos, 0, channel * 2);
          } else {
            size_t i1_pos = (h * width + w) * channel;
            size_t i2_pos = (h * width + w - m) * channel;
            size_t o_pos = ((h * width + w) * disp + m) * channel * 2;
            shift_and_copy_neon(input_l_ptr + i1_pos, input_r_ptr + i2_pos,
                                output_ptr + o_pos, channel, act_fix);
          }
        }
      }
    }
  }
  __TOC__(COST_VOLUME_NEON)
}
#endif

#ifdef ENABLE_CV_ACCEL
void cost_volume_accel(vitis::ai::library::OutputTensor input_l,
                      vitis::ai::library::OutputTensor input_r,
                      vitis::ai::library::InputTensor output,
                      CostVolumeAccel* cvAccel) {
  __TIC__(COST_VOLUME_ACCEL_TOTAL)
  CHECK_EQ(input_l.size, input_r.size)
      << "The two inputs' size not same, please check.";
  LOG_IF(INFO, ENV_PARAM(DEBUG_PSMNET))
      << "cost_volume inputs shape: " << std::endl
      << "left: (" << input_l.height << ", " << input_l.width << ", "
      << input_l.channel << ") "
      << "right: (" << input_r.height << ", " << input_r.width << ", "
      << input_r.channel << ") ";
  LOG_IF(INFO, ENV_PARAM(DEBUG_PSMNET))
      << "cost_volume output shape: " << std::endl
      << "output: (" << output.height << ", " << output.width << ", "
      << output.channel << ") " << " | Output.size: " << output.size;

  size_t input_l_size = input_l.size / input_l.batch;
  size_t input_r_size = input_r.size / input_r.batch;

  for (size_t a = 0; a < input_l.batch; ++a) {
    int8_t* input_l_ptr = (int8_t*)input_l.get_data(a);
    int8_t* input_r_ptr = (int8_t*)input_r.get_data(a);
    int8_t* output_ptr = (int8_t*)output.get_data(a);
    size_t out_size = output.size / output.batch;

    CHECK_EQ(input_l.height, input_r.height);
    CHECK_EQ(input_l.width, input_r.width);
    CHECK_EQ(input_l.channel, input_r.channel);
    

    __TIC__(COST_VOLUME_ACCEL)
    cvAccel->run(input_l_ptr, input_r_ptr, output_ptr);
    __TOC__(COST_VOLUME_ACCEL)


 
    if (ENV_PARAM(DUMP_PSMNET)) {
      static std::shared_ptr<std::mutex> mtx =
        vitis::ai::WeakStore<std::string, std::mutex>::create("dump-costvolume-accel");
      std::lock_guard<std::mutex> lock(*mtx);
      auto filename = std::string("cv_input_l_") + to_string(a) + ".bin";
      LOG(INFO) << "write to  " << filename;
      std::ofstream ofs_l(filename, ios::binary);
      ofs_l.write((char*)input_l_ptr, input_l_size);
      ofs_l.close();

      filename = std::string("cv_input_r_") + to_string(a) + ".bin";
      LOG(INFO) << "write to  " << filename;
      std::ofstream ofs_r(filename, ios::binary);
      ofs_r.write((char*)input_r_ptr, input_r_size);
      ofs_r.close();

      filename = std::string("cost_volume_accel_") + to_string(a) + ".bin";
      LOG(INFO) << "write to  " << filename;
      std::ofstream ofs(filename, ios::binary);
      ofs.write((char*)output_ptr, out_size);
      ofs.close();
    }
  }
  __TOC__(COST_VOLUME_ACCEL_TOTAL)
}
#endif

void cost_volume(vitis::ai::library::OutputTensor input_l,
                 vitis::ai::library::OutputTensor input_r,
                 vitis::ai::library::InputTensor output) {
  __TIC__(COST_VOLUME_TOTAL)
  CHECK_EQ(input_l.size, input_r.size)
      << "The two inputs' size not same, please check.";
  LOG_IF(INFO, ENV_PARAM(DEBUG_PSMNET))
      << "cost_volume inputs shape: " << std::endl
      << "left: (" << input_l.height << ", " << input_l.width << ", "
      << input_l.channel << ") "
      << "right: (" << input_r.height << ", " << input_r.width << ", "
      << input_r.channel << ") ";

  size_t input_l_size = input_l.size / input_l.batch;
  size_t input_r_size = input_r.size / input_r.batch;

  for (size_t a = 0; a < input_l.batch; ++a) {
    int8_t* input_l_ptr = (int8_t*)input_l.get_data(a);
    int8_t* input_r_ptr = (int8_t*)input_r.get_data(a);
    int8_t* output_ptr = (int8_t*)output.get_data(a);
    size_t out_size = output.size / output.batch;

    int left_fix = input_l.fixpos;
    // int right_fix = input_r.fixpos;
    int v_fix = output.fixpos;
    int act_fix = left_fix - v_fix;

    CHECK_EQ(input_l.height, input_r.height);
    CHECK_EQ(input_l.width, input_r.width);
    CHECK_EQ(input_l.channel, input_r.channel);
    auto height = input_l.height;
    auto width = input_l.width;
    auto channel = input_l.channel;
    // max_disp of psmnet, its value is 192/4
    size_t disp = 48;

#ifdef ENABLE_NEON
    // compute the first channel of the cost volume
    if (act_fix == 0 && channel == 32) {  // constant fold optimization
      cost_volume_neon(input_l_ptr, input_r_ptr, output_ptr, height, width, 32,
                       0, 48);
    } else {
      cost_volume_neon(input_l_ptr, input_r_ptr, output_ptr, height, width,
                       channel, act_fix, disp);
    }

#else

    __TIC__(COST_VOLUME_C)
    auto o_i = 0;
    for (auto h = 0u; h < height; ++h) {
      for (auto w = 0u; w < width; ++w) {
        for (auto m = 0u; m < disp; ++m) {
          for (auto c = 0u; c < channel * 2; ++c) {
            // output_ptr[o_i] = get_output(input_l_ptr, input_r_ptr, width,
            //                              channel, h, w, m, c, act_fix);
            o_i = o_i + 1;
          }
        }
      }
    }
    __TOC__(COST_VOLUME_C)
#endif

    if (ENV_PARAM(DUMP_PSMNET)) {
      static std::shared_ptr<std::mutex> mtx =
        vitis::ai::WeakStore<std::string, std::mutex>::create("dump-costvolume");
      std::lock_guard<std::mutex> lock(*mtx);
      auto filename = std::string("input_l_") + to_string(a) + ".bin";
      LOG(INFO) << "write to  " << filename;
      std::ofstream ofs_l(filename, ios::binary);
      ofs_l.write((char*)input_l_ptr, input_l_size);
      ofs_l.close();

      filename = std::string("input_r_") + to_string(a) + ".bin";
      LOG(INFO) << "write to  " << filename;
      std::ofstream ofs_r(filename, ios::binary);
      ofs_r.write((char*)input_r_ptr, input_r_size);
      ofs_r.close();

      filename = std::string("cost_volume_") + to_string(a) + ".bin";
      LOG(INFO) << "write to  " << filename;
      std::ofstream ofs(filename, ios::binary);
      ofs.write((char*)output_ptr, out_size);
      ofs.close();
    }
  }
  __TOC__(COST_VOLUME_TOTAL)
}

void copy_into_tensor(const vitis::ai::library::OutputTensor input,
                      vitis::ai::library::InputTensor tensor) {
  __TIC__(COPY_INPUT)
  int i_fixpos = tensor.fixpos;
  int o_fixpos = input.fixpos;
  auto size = tensor.height * tensor.width * tensor.channel;
  LOG_IF(INFO, ENV_PARAM(DEBUG_PSMNET))
      << tensor.name << ": " << i_fixpos << " " << o_fixpos;
  for (size_t b = 0; b < tensor.batch; ++b) {
    auto datai = (int8_t*)input.get_data(b);
    auto data = (int8_t*)tensor.get_data(b);
    if (i_fixpos == o_fixpos) {
      memcpy(data, datai, size);
    } else {
      float o_scale = exp2f(-1.0 * o_fixpos);
      float i_scale = vitis::ai::library::tensor_scale(tensor);
      float scale = o_scale * i_scale;
#ifdef ENABLE_NEON
      int shift_num = o_fixpos - i_fixpos;
      shift_and_copy(data, datai, size, shift_num);
      int rest = size % 16;
      if (rest > 0) {
        for (size_t i = size - rest; i < size; ++i)
          *(data + i) = dpu_round(datai[i] * scale);
      }

#else
      for (size_t i = 0; i < size; ++i) data[i] = dpu_round(datai[i] * scale);
#endif
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_PSMNET))
        << "tensor shape: (" << tensor.height << ", " << tensor.width << ", "
        << tensor.channel << ").";
    if (ENV_PARAM(DUMP_PSMNET)) {
      for (size_t i = 0; i < tensor.batch; ++i) {
        std::ofstream ofs("batch_" + tensor.name + to_string(i), ios::binary);
        ofs.write((char*)data, size);
        ofs.close();
      }
    }
  }
  __TOC__(COPY_INPUT)
}

// reorder the tensors with name
static vector<vitis::ai::library::InputTensor> sort_tensors(
    const vector<vitis::ai::library::InputTensor>& tensors,
    vector<string>& layer_names) {
  vector<vitis::ai::library::InputTensor> ordered_tensors;
  for (auto i = 0u; i < layer_names.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].name.find(layer_names[i]) != std::string::npos) {
        ordered_tensors.push_back(tensors[j]);
        break;
      }
  return ordered_tensors;
}

static vector<vitis::ai::library::OutputTensor> sort_tensors(
    const vector<vitis::ai::library::OutputTensor>& tensors,
    vector<string>& layer_names) {
  vector<vitis::ai::library::OutputTensor> ordered_tensors;
  for (auto i = 0u; i < layer_names.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].name.find(layer_names[i]) != std::string::npos) {
        ordered_tensors.push_back(tensors[j]);
        break;
      }
  return ordered_tensors;
}

//#ifdef ENABLE_AIE
//// aie op: resize
//static void dpu_resize(vai_resize* vai_res) {
//  static std::shared_ptr<std::mutex> mtx =
//      vitis::ai::WeakStore<std::string, std::mutex>::create("aie-sfm");
//  std::lock_guard<std::mutex> lock(*mtx);
//  vai_res->run();
//}
//#endif

template <typename TensorType>
static void read_tensor_from_file(TensorType& tensor) {
  auto file = tensor.name + ".bin";
  LOG(INFO) << "read " << file << " for " << tensor.name << " "
            << tensor.size / tensor.batch << " bytes";
  CHECK(std::ifstream(file)
            .read((char*)tensor.get_data(2), tensor.size / tensor.batch)
            .good())
      << "fail to read! filename=" << file;
}

template <typename TensorType>
static void write_tensor_to_file(TensorType& tensor) {
  for (auto a = 0u; a < tensor.batch; a++) {
    auto file = tensor.name + "." + std::to_string(a) + ".out";
    LOG(INFO) << "write " << file << " for " << tensor.name << " "
              << tensor.size / tensor.batch << " bytes";
    CHECK(std::ofstream(file)
              .write((char*)tensor.get_data(a), tensor.size / tensor.batch)
              .good())
        << "fail to write! filename=" << file;
  }
}

void PsmNetImp::PSMnet_run(const vector<pair<cv::Mat, cv::Mat>>& input_images) {
  vector<cv::Mat> left_mats;
  vector<cv::Mat> right_mats;
  auto input_tensor_left = tasks_[0]->getInputTensor(0u)[0];
  auto sWidth = input_tensor_left.width;
  auto sHeight = input_tensor_left.height;
  for (size_t i = 0; i < input_tensor_left.batch; ++i) {
    cv::Mat left_mat;
    cv::resize(input_images[i].first, left_mat, cv::Size(sWidth, sHeight));
    left_mats.push_back(left_mat);
    cv::Mat right_mat;
    cv::resize(input_images[i].second, right_mat, cv::Size(sWidth, sHeight));
    right_mats.push_back(right_mat);
  }

  __TIC__(PSMNET_SET_IMG_LEFT)
  tasks_[0]->setImageRGB(left_mats);
  __TOC__(PSMNET_SET_IMG_LEFT)
  if (ENV_PARAM(PSMNET_1) == 4) {
    for (auto& t : tasks_[0]->getInputTensor(0u)) {
      read_tensor_from_file(t);
    }
  }
  __TIC__(PSMNET_DPU_LEFT_0)
  tasks_[0]->run(0u);
  if (0)
    for (auto& t : tasks_[0]->getOutputTensor(0u)) {
      write_tensor_to_file(t);
    }
  __TOC__(PSMNET_DPU_LEFT_0)

  __TIC__(PSMNET_RESIZE_LEFT)
  // store the outputs of kernel_0
//#ifndef ENABLE_AIE
  for (auto i : {0, 1, 2, 3}) {
    cpu_res_[i]->run();
  }
//#else
//  for (auto i : {0, 1, 2, 3}) {
//    dpu_resize(vai_res_[i].get());
//  }
//#endif
  copy_into_tensor(outputs_l_[0], inputs_l_[0]);
  copy_into_tensor(outputs_l_[1], inputs_l_[1]);

  __TOC__(PSMNET_RESIZE_LEFT)

  __TIC__(PSMNET_DPU_LEFT_1)
  if (ENV_PARAM(PSMNET_1) == 3) {
    for (auto& t : tasks_[0]->getInputTensor(1u)) {
      read_tensor_from_file(t);
    }
  }
  if (0)
    for (auto& t : tasks_[0]->getInputTensor(1u)) {
      write_tensor_to_file(t);
    }
  tasks_[0]->run(1u);
  if (0)
    for (auto& t : tasks_[0]->getOutputTensor(1u)) {
      write_tensor_to_file(t);
    }
  __TOC__(PSMNET_DPU_LEFT_1)

  __TIC__(PSMNET_SET_IMG_RIGHT)
  tasks_[1]->setImageRGB(right_mats);
  __TOC__(PSMNET_SET_IMG_RIGHT)

  __TIC__(PSMNET_DPU_RIGHT_0)
  if (ENV_PARAM(PSMNET_1) == 4) {
    for (auto& t : tasks_[1]->getInputTensor(0u)) {
      read_tensor_from_file(t);
    }
  }
  tasks_[1]->run(0u);
  if (0)
    for (auto& t : tasks_[1]->getOutputTensor(0u)) {
      write_tensor_to_file(t);
    }
  __TOC__(PSMNET_DPU_RIGHT_0)

  __TIC__(PSMNET_RESIZE_RIGHT)
  // store the outputs of kernel_0
//#ifndef ENABLE_AIE
  for (auto i : {4, 5, 6, 7}) {
    cpu_res_[i]->run();
  }
//#else
//  for (auto i : {4, 5, 6, 7}) {
//    dpu_resize(vai_res_[i].get());
//  }
//#endif
  copy_into_tensor(outputs_r_[0], inputs_r_[0]);
  copy_into_tensor(outputs_r_[1], inputs_r_[1]);

  __TOC__(PSMNET_RESIZE_RIGHT)

  __TIC__(PSMNET_DPU_RIGHT_1)
  if (ENV_PARAM(PSMNET_1) == 3) {
    for (auto& t : tasks_[1]->getInputTensor(1u)) {
      read_tensor_from_file(t);
    }
  }
  tasks_[1]->run(1u);

  if (0)
    for (auto& t : tasks_[1]->getOutputTensor(1u)) {
      write_tensor_to_file(t);
    }

  __TOC__(PSMNET_DPU_RIGHT_1)

  //  cost volume
  __TIC__(PSMNET_COST_VOLUME)
  auto output_tensor_l = tasks_[0]->getOutputTensor(1u)[0];
  auto output_tensor_r = tasks_[1]->getOutputTensor(1u)[0];
  auto input_tensor_k2 = tasks_[2]->getInputTensor(0u)[0];

  if (ENV_PARAM(PSMNET_1) == 2) {
    read_tensor_from_file(output_tensor_l);
    read_tensor_from_file(output_tensor_r);
  }

#ifdef ENABLE_CV_ACCEL
  cost_volume_accel(output_tensor_l, output_tensor_r, input_tensor_k2, cvAccel);
#else
  cost_volume(output_tensor_l, output_tensor_r, input_tensor_k2);
#endif

  if (ENV_PARAM(PSMNET_1) == 1) {
    read_tensor_from_file(input_tensor_k2);
  }
  __TOC__(PSMNET_COST_VOLUME)

  __TIC__(PSMNET_DPU_LAST)
  tasks_[2]->run(0u);
  __TOC__(PSMNET_DPU_LAST)

  __TIC__(PSMNET_SFM)
//#ifndef ENABLE_AIE
  cpu_sfm_->run();
//#else
//  dpu_sfm_->run_with();
//#endif
  __TOC__(PSMNET_SFM)

  if (ENV_PARAM(DUMP_PSMNET)) {
    auto final_tensor = tasks_[2]->getOutputTensor(0u)[0];
    for (size_t b = 0; b < final_tensor.batch; ++b) {
      std::ofstream ofs("tensor_res_" + to_string(b) + ".bin", ios::binary);
      ofs.write((char*)final_tensor.get_data(b),
                final_tensor.width * final_tensor.height);
      ofs.close();
    }
  }
}  // namespace ai

void PsmNetImp::run(const std::vector<std::pair<cv::Mat, cv::Mat>>& imgs) {
  PSMnet_run(imgs);
  return;
}

std::vector<cv::Mat> PsmNetImp::get_result() {
  std::vector<cv::Mat> rets;
  float* sfm_output;
//#ifndef ENABLE_AIE
  sfm_output = cpu_sfm_->get_output();
//#else
//  sfm_output = dpu_sfm_->get_output();
//#endif
  auto final_tensor = tasks_[2]->getOutputTensor(0u)[0];
  for (size_t b = 0; b < final_tensor.batch; ++b) {
    float mmax = std::numeric_limits<float>::min();
    float mmin = std::numeric_limits<float>::max();
    cv::Mat ret = cv::Mat(cv::Size(final_tensor.width, final_tensor.height), CV_8UC3);
    int c = 0;
    for (auto h = 0u; h < final_tensor.height; ++h) {
      for (auto w = 0u; w < final_tensor.width; ++w) {
        auto value =
            sfm_output[final_tensor.width * final_tensor.height * b + c];
        int gray = (int)value * 1.7;
        uint8_t b = gray;
        uint8_t g = gray;
        uint8_t r = gray;
        ret.at<cv::Vec3b>(h, w) = cv::Vec3b(b, g, r);
        mmax = std::max(mmax, value);
        mmin = std::min(mmin, value);
        c = c + 1;
      }
    }
    rets.push_back(ret);
    cout << "mmax " << mmax << " "  //
         << "mmin " << mmin << " "  //
         << endl;
  }
  return rets;
}

size_t PsmNetImp::get_input_batch() { return tasks_[0]->get_input_batch(0, 0); }
int PsmNetImp::get_input_width() const {
  return tasks_[0]->getInputTensor(0u)[0].width;
}
int PsmNetImp::get_input_height() const {
  return tasks_[0]->getInputTensor(0u)[0].height;
}

}  // namespace ai
}  // namespace vitis
