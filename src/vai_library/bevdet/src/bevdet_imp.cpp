/*
 * Copyright 2019 xilinx Inc.
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
#include "./bevdet_imp.hpp"

#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vitis/ai/collection_helper.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/softmax.hpp>
#include <xir/graph/graph.hpp>
#include <xir/op/op.hpp>
#include <xir/tensor/tensor.hpp>

#include "utils.hpp"
using namespace std;
std::mutex mtx;
namespace vitis {
namespace ai {
DEF_ENV_PARAM(ENABLE_BEVdet_DEBUG, "0");
DEF_ENV_PARAM(USE_AIE, "0");
DEF_ENV_PARAM(USE_SOFTMAX_144, "0");

// BEVdetImp
BEVdetImp::BEVdetImp(const std::string& model_name0,
                     const std::string& model_name1,
                     const std::string& model_name2) {
  std::string aielib = "libbevdet_aie_runner.so";
  use_aie_ = false;
  aie_runner = nullptr;
  output0_80_ptr = output0_80;
  output0_64_ptr = output0_64;
  if (ENV_PARAM(USE_AIE)) {
    if (filesize("/usr/lib/" + aielib))
      use_aie_ = true;
    else
      LOG(INFO) << "didn't find /usr/lib/" << aielib << ", don't use aie";
  }
  if (use_aie_) {
    __TIC__(creat_aie_runner)
    auto attrs = xir::Attrs::create();
    attrs->set_attr("xclbin", "/run/media/mmcblk0p1/dpu.xclbin");
    attrs->set_attr("lib", std::map<std::string, std::string>{{"DPU", aielib}});
    auto ls1path = find_bevdet_1_pt_file(model_name1);
    LOG(INFO) << "use aie " << ls1path;
    auto graph = xir::Graph::deserialize(ls1path);
    auto root = graph->get_root_subgraph();
    for (auto c : root->get_children()) {
      if (c->get_attr<std::string>("device") == "DPU") {
        static std::unique_ptr<vart::Runner> aie_runner_static =
            vart::Runner::create_runner_with_attrs(c, attrs.get());
        aie_runner = aie_runner_static.get();
        aie_runner_inputs = vart::alloc_cpu_flat_tensor_buffers(
            aie_runner->get_input_tensors());
        aie_runner_outputs = vart::alloc_cpu_flat_tensor_buffers(
            aie_runner->get_output_tensors());
        output0_64_ptr = (char*)aie_runner_inputs[0]->data({0, 0, 0, 0}).first;
        output0_80_ptr = (char*)aie_runner_inputs[1]->data({0, 0, 0, 0}).first;
        break;
      }
    }
    if (aie_runner == nullptr) {
      use_aie_ = false;
      LOG(INFO) << "Failed to initialize aie, do not use aie";
    }
    __TOC__(creat_aie_runner)
  }

  dpu_attrs_ = xir::Attrs::create();
  model_0_ = ConfigurableDpuTask::create(model_name0, dpu_attrs_.get(), true);
  model_2_ = ConfigurableDpuTask::create(model_name2, dpu_attrs_.get(), false);
}

BEVdetImp::~BEVdetImp() {}

void BEVdetImp::run_aie(const std::vector<std::vector<char>>& input_bins,
                        int8_t* output) {
  uint64_t data, tensor_size;
  std::vector<float> fixed_pos{6, 6, 1, 7};
  std::vector<int> aie_input_idx{3, 1, 2, 0};
  __TIC__(create_aie_input);
  for (size_t i = 0; i < input_bins.size(); i++) {
    float input_fixed_scale = std::exp2f(fixed_pos[i]);
    std::tie(data, tensor_size) = aie_runner_inputs[i + 2]->data({0, 0, 0, 0});
    auto dst = (char*)data;
    float* src = (float*)input_bins[aie_input_idx[i]].data();
    if (i == 0 || i == 2) {
      CHECK_EQ(tensor_size, input_bins[aie_input_idx[i]].size() / 4);
      size_t dst_idx, src_idx;
      for (size_t b = 0; b < 6; b++) {
        for (size_t h = 0; h < 52; h++) {
          for (size_t w = 0; w < 704; w++) {
            dst_idx = ((b * 704 + w) * 52 + h) * 3;
            src_idx = ((b * 52 + h) * 704 + w) * 3;
            dst[dst_idx] = (int)std::round(src[src_idx] * input_fixed_scale);
            dst[dst_idx + 1] =
                (int)std::round(src[src_idx + 1] * input_fixed_scale);
            dst[dst_idx + 2] =
                (int)std::round(src[src_idx + 2] * input_fixed_scale);
          }
        }
      }
    } else if (i == 1) {
      std::vector<char> tmp(input_bins[aie_input_idx[i]].size() / 4);
      for (size_t j = 0; j < tmp.size(); j++) {
        tmp[j] = (int)std::round(src[j] * input_fixed_scale);
      }
      memset(dst, 0, tensor_size);
      for (size_t j = 0; j < 6; j++) {
        // 0 oc0,ic0  1 oc0,ic1  2 oc0,ic2
        for (size_t k = 0; k < 3; k++) {
          dst[384 * j + 16 * (0 + k) + 0] = dst[384 * j + 16 * (0 + k) + 12] =
              tmp[j * 9 + k];
          dst[384 * j + 16 * (3 + k) + 3] = dst[384 * j + 16 * (3 + k) + 15] =
              tmp[j * 9 + k];
          dst[384 * j + 16 * (4 + k) + 6] = tmp[j * 9 + k];
          dst[384 * j + 16 * (5 + k) + 9] = tmp[j * 9 + k];
          dst[384 * j + 16 * (8 + k) + 8] = tmp[j * 9 + k];
          dst[384 * j + 16 * (11 + k) + 5] = dst[384 * j + 16 * (11 + k) + 11] =
              tmp[j * 9 + k];
          dst[384 * j + 16 * (12 + k) + 2] = dst[384 * j + 16 * (12 + k) + 14] =
              tmp[j * 9 + k];
          dst[384 * j + 16 * (16 + k) + 4] = tmp[j * 9 + k];
          dst[384 * j + 16 * (19 + k) + 1] = dst[384 * j + 16 * (19 + k) + 7] =
              tmp[j * 9 + k];
          dst[384 * j + 16 * (20 + k) + 10] = tmp[j * 9 + k];
          dst[384 * j + 16 * (21 + k) + 13] = tmp[j * 9 + k];
        }
        // 3 oc1,ic0 4 oc1,ic1 5 oc1,ic2
        for (size_t k = 0; k < 3; k++) {
          dst[384 * j + 16 * (0 + k) + 1] = dst[384 * j + 16 * (0 + k) + 13] =
              tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (1 + k) + 4] = tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (4 + k) + 7] = tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (5 + k) + 10] = tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (8 + k) + 9] = tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (9 + k) + 0] = dst[384 * j + 16 * (9 + k) + 12] =
              tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (11 + k) + 6] = tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (12 + k) + 3] = dst[384 * j + 16 * (12 + k) + 15] =
              tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (16 + k) + 5] = tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (17 + k) + 8] = tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (19 + k) + 2] = tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (20 + k) + 11] = tmp[j * 9 + 3 + k];
          dst[384 * j + 16 * (21 + k) + 14] = tmp[j * 9 + 3 + k];
        }
        // 6 oc2,ic0  7 oc2,ic1 8 oc2,ic2
        for (size_t k = 0; k < 3; k++) {
          dst[384 * j + 16 * (0 + k) + 2] = dst[384 * j + 16 * (0 + k) + 14] =
              tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (1 + k) + 5] = tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (2 + k) + 8] = tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (5 + k) + 11] = tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (8 + k) + 4] = dst[384 * j + 16 * (8 + k) + 10] =
              tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (9 + k) + 1] = dst[384 * j + 16 * (9 + k) + 13] =
              tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (11 + k) + 7] = tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (16 + k) + 0] = dst[384 * j + 16 * (16 + k) + 6] =
              tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (17 + k) + 9] = tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (18 + k) + 12] = tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (19 + k) + 3] = tmp[j * 9 + 6 + k];
          dst[384 * j + 16 * (21 + k) + 15] = tmp[j * 9 + 6 + k];
        }
      }
    } else if (i == 3) {
      std::vector<char> tmp(input_bins[aie_input_idx[i]].size() / 4);
      for (size_t j = 0; j < tmp.size(); j++) {
        tmp[j] = (int)std::round(src[j] * input_fixed_scale);
      }
      for (size_t j = 0; j < 6; j++)
        for (size_t k = 0; k < 16; k++)
          for (size_t n = 0; n < 3; n++)
            dst[16 * 3 * j + k * 3 + n] = tmp[j * 3 + n];
    }
  }
  __TOC__(create_aie_input);
  __TIC__(run_aie);
  {
    std::lock_guard<std::mutex> lk(mtx);
    aie_runner->execute_async(
        vitis::ai::vector_unique_ptr_get(aie_runner_inputs),
        vitis::ai::vector_unique_ptr_get(aie_runner_outputs));
  }
  __TOC__(run_aie)
  __TIC__(copy_aie_output)
  auto data0 = (int8_t*)aie_runner_outputs[0]->data({0, 0, 0, 0}).first;
  auto data1 = (int8_t*)aie_runner_outputs[0]->data({1, 0, 0, 0}).first;
  LOG(INFO) << aie_runner_outputs[0]->data({1, 0, 0, 0}).second;
  size_t count = 128 * 128;
  for (size_t i = 0; i < count; i++) {
    memcpy(output + i * 80, data0 + i * 40, 40);
    memcpy(output + i * 80 + 40, data1 + i * 40, 40);
  }

  __TOC__(copy_aie_output)
}

void BEVdetImp::middle_process(const std::vector<std::vector<char>>& input1,
                               const vitis::ai::library::InputTensor& input2) {
  if (use_aie_) {
    auto data2 = (int8_t*)input2.get_data(0);
    run_aie(input1, data2);
  } else {
    // 1. ls 1 matmal
    // 2. ls 1 mal
    // 3. ls 1 add
    //(input1.3 x input1.1)*input.2+input.0
    float* input1_0 = (float*)input1[0].data();
    float* input1_1 = (float*)input1[1].data();
    float* input1_2 = (float*)input1[2].data();
    float* input1_3 = (float*)input1[3].data();
    float* output_1 = output1;
    for (int n = 0; n < 6;
         n++, input1_0 += 3, input1_1 += 9, output_1 += 109824) {
      for (int a = 0; a < 36608; a++, input1_2 += 3, input1_3 += 3) {
        float matmul0 = 0;
        float matmul1 = 0;
        float matmul2 = 0;
        for (auto b = 0; b < 3; b++) {
          matmul0 += input1_3[b] * input1_1[b];
          matmul1 += input1_3[b] * input1_1[3 + b];
          matmul2 += input1_3[b] * input1_1[6 + b];
        }

        output_1[(a % 704) * 156 + (a / 704) * 3] =
            matmul0 * input1_2[0] + input1_0[0];
        output_1[(a % 704) * 156 + (a / 704) * 3 + 1] =
            matmul1 * input1_2[1] + input1_0[1];
        output_1[(a % 704) * 156 + (a / 704) * 3 + 2] =
            matmul2 * input1_2[2] + input1_0[2];
      }
    }
    memset(voxel_output, 0, 1310720 * sizeof(float));
    // 4. ls 0 softmax
    // 5. ls 0 mal
    // 6. ls 2 voxel_pooling_scatter_sum
    constexpr float geom_sub_[3] = {51.2, 51.2, 10};
    constexpr float resolution_[3] = {1.25, 1.25, 0.05};
    float softmax_output[52];
    float* g_data = output1;
    auto x_data = (int8_t*)output0_80_ptr;
    auto output0_64_data = (int8_t*)output0_64_ptr;
    auto begin_idx = ENV_PARAM(USE_SOFTMAX_144) ? 12 : 0;
    for (size_t b = 0; b < 6; b++) {
      for (size_t w = 0; w < 704;
           w++, x_data += 80, output0_64_data += 52 + begin_idx) {
        my_softmax(output0_64_data, begin_idx, softmax_output);
        for (size_t h = 0; h < 52; h++, g_data += 3) {
          float softmaxh = softmax_output[h] * 0.5f;
          if (g_data[0] >= -51.2 && g_data[0] < 51.2 && g_data[1] >= -51.2 &&
              g_data[1] < 51.2 && g_data[2] >= -10 && g_data[2] < 10) {
            int ow = (g_data[0] + geom_sub_[0]) * resolution_[0];
            int oh = (g_data[1] + geom_sub_[1]) * resolution_[1];
            auto v_data = voxel_output + oh * 10240 + ow * 80;
            for (size_t c = 0; c < 80; c++) {
              v_data[c] = v_data[c] + softmaxh * float(x_data[c]);
            }
          }
        }
      }
    }
    auto data2 = (int8_t*)input2.get_data(0);
    for (int i = 0; i < 1310720; i++) {
      // LOG_IF(INFO, use_aie_ && 0 != data2[i])
      //     << "aie " << int(data2[i]) << " cpu "
      //     << (int)float2fix(voxel_output[i]);
      data2[i] = float2fix(voxel_output[i]);
    }
  }
}

void BEVdetImp::run_model_0(const std::vector<cv::Mat>& images, int idx) {
  auto batch = images.size();
  __TIC__(copy_input_from_image)
  model_0_->setInputImageRGB(images);
  __TOC__(copy_input_from_image)
  // model_0_ run
  model_0_->run(0);
  // do something
  int step = ENV_PARAM(USE_SOFTMAX_144) ? 45056 : 36608;
  auto tensors = model_0_->getOutputTensor();
  for (auto&& tensor : tensors[0])
    for (size_t j = 0; j < batch; j++) {
      if (tensor.channel == 80u) {
        memcpy(output0_80_ptr + (idx + j) * 56320, tensor.get_data(j), 56320);
      } else {
        memcpy(output0_64_ptr + (idx + j) * step, tensor.get_data(j), step);
      }
    }
}

std::vector<CenterPointResult> BEVdetImp::run(
    const std::vector<cv::Mat>& images,
    const std::vector<std::vector<char>>& input_bins) {
  __TIC__(BEVdet_run)
  CHECK_EQ(images.size(), 6);
  CHECK_EQ(input_bins.size(), 4);
  auto batch = model_0_->get_input_batch();
  auto input2 = model_2_->getInputTensor();
  // std::thread model_0_thread;
  for (size_t i = 0; i < images.size(); i += batch) {
    __TIC__(per_process)
    std::vector<cv::Mat> image_resize;
    for (size_t j = 0; j < batch && i + j < images.size(); j++) {
      image_resize.push_back(resize_and_crop_image(images[i + j]));
    }
    __TOC__(per_process)
    run_model_0(image_resize, i);
    // if (model_0_thread.joinable()) model_0_thread.join();
    // model_0_thread =
    //     std::thread(&BEVdetImp::run_model_0, this, image_resize, i);
  }
  // model_0_thread.join();
  __TIC__(middle_process)
  middle_process(input_bins, model_2_->getInputTensor()[0][0]);
  __TOC__(middle_process)
  __TIC__(run_2)
  model_2_->run(0);
  __TOC__(run_2)
  __TIC__(post_process)
  auto res = post_process(model_2_->getOutputTensor()[0], 0.05);
  __TOC__(post_process)
  __TOC__(BEVdet_run)
  return res;
}

// std::vector<CenterPointResult> BEVdetImp::run(
//     const std::vector<std::vector<char>>& input_bins) {
//   __TIC__(BEVdet_run)
//   CHECK_EQ(input_bins.size(), 5);
//   auto batch = model_0_->get_input_batch();
//   std::vector<cv::Mat> image_resize(batch);
//   auto input2 = model_2_->getInputTensor();
//   for (size_t i = 0; i < 6; i += batch) {
//     for (size_t j = 0; j < batch; j++) {
//       auto data2 = (int8_t*)model_0_->getInputTensor()[0][0].get_data(j);
//       for (int c = 0; c < 3; c++)
//         for (int hw = 0; hw < 180224; hw++)
//           data2[hw * 3 + c] =
//               input_bins[0][(i + j) * 180224 * 3 + c * 180224 + hw];
//     }
//     // model_0_ run
//     model_0_->run(0);
//     // do something
//     memcpy(output0 + i * 92928,
//     model_0_->getOutputTensor()[0][0].get_data(0),
//            model_0_->getOutputTensor()[0][0].size);
//   }
//   __TIC__(middle_process)
//   middle_process({input_bins[1], input_bins[2], input_bins[3],
//   input_bins[4]},
//                  model_2_->getInputTensor()[0][0]);
//   __TOC__(middle_process)
//   __TIC__(run_2)
//   model_2_->run(0);
//   __TOC__(run_2)
//   __TIC__(post_process)
//   auto res = post_process(model_2_->getOutputTensor()[0], 0.05);
//   __TOC__(post_process)
//   __TOC__(BEVdet_run)
//   return res;
// }
}  // namespace ai
}  // namespace vitis
