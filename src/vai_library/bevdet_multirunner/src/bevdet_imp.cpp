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
#include <vart/runner_helper.hpp>
#include <vitis/ai/collection_helper.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/multi_runner.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/graph/graph.hpp>
#include <xir/op/op.hpp>
#include <xir/tensor/tensor.hpp>

#include "utils.hpp"
using namespace std;
std::mutex mtx;
namespace vitis {
namespace ai {
DEF_ENV_PARAM(ENABLE_BEVdet_DEBUG, "0");

// BEVdetImp
BEVdetImp::BEVdetImp(const std::string& model_name, bool use_aie)
    : use_aie_(use_aie),
      model(model_name),
      runner{vitis::ai::MultiRunner::create(model)},
      mean(vitis::ai::getMean(runner.get())),
      scale(vitis::ai::getScale(runner.get())) {
  if (use_aie) {
    __TIC__(creat_aie_runner)
    auto attrs = xir::Attrs::create();
    attrs->set_attr("xclbin", "/run/media/mmcblk0p1/dpu.xclbin");
    attrs->set_attr("lib", std::map<std::string, std::string>{
                               {"DPU", "libbevdet_ls1_runner.so"}});
    auto graph = xir::Graph::deserialize(find_bevdet_1_pt_file(model));
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
        break;
      }
    }
    CHECK_NE(aie_runner, nullptr);
    __TOC__(creat_aie_runner)
  }
  auto inputs = runner->get_inputs();
  for (const auto& i : inputs) LOG(INFO) << i->get_tensor()->get_name();
}

BEVdetImp::~BEVdetImp() {}

std::vector<CenterPointResult> BEVdetImp::run(
    const std::vector<cv::Mat>& images,
    const std::vector<std::vector<char>>& input_bins) {
  auto inputs = runner->get_inputs();
  auto outputs = runner->get_outputs();
  __TIC__(per_process)
  __TIC__(copy_input_from_image)
  copy_input_from_image(images, inputs[0], mean, scale);
  __TOC__(copy_input_from_image)
  if (!use_aie_) {
    CHECK_GE(input_bins.size(), inputs.size() - 1);
    for (size_t i = 1; i < inputs.size(); i++) {
      float input_fixed_scale = vart::get_input_scale(inputs[i]->get_tensor());
      std::vector<char> tmp(input_bins[i - 1].size() / 4);
      float* src = (float*)input_bins[i - 1].data();
      for (size_t j = 0; j < tmp.size(); j++) {
        tmp[j] = (int)std::round(src[j] * input_fixed_scale);
      }
      copy_input_from_bin(tmp, inputs[i]);
    }
  } else {
    uint64_t data, tensor_size;
    std::vector<float> fixed_pos{6, 6, 1, 7};
    std::vector<int> aie_input_idx{3, 1, 2, 0};
    for (size_t i = 0; i < input_bins.size(); i++) {
      float input_fixed_scale = std::exp2f(fixed_pos[i]);  // 6617
      std::tie(data, tensor_size) = aie_runner_inputs[i]->data({0, 0, 0, 0});
      auto dst = (char*)data;
      float* src = (float*)input_bins[aie_input_idx[i]].data();
      if (i == 0 || i == 2) {
        CHECK_EQ(tensor_size, input_bins[aie_input_idx[i]].size() / 4);
        for (size_t j = 0; j < tensor_size; j++) {
          dst[j] = (int)std::round(src[j] * input_fixed_scale);
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
            dst[384 * j + 16 * (11 + k) + 5] =
                dst[384 * j + 16 * (11 + k) + 11] = tmp[j * 9 + k];
            dst[384 * j + 16 * (12 + k) + 2] =
                dst[384 * j + 16 * (12 + k) + 14] = tmp[j * 9 + k];
            dst[384 * j + 16 * (16 + k) + 4] = tmp[j * 9 + k];
            dst[384 * j + 16 * (19 + k) + 1] =
                dst[384 * j + 16 * (19 + k) + 7] = tmp[j * 9 + k];
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
            dst[384 * j + 16 * (12 + k) + 3] =
                dst[384 * j + 16 * (12 + k) + 15] = tmp[j * 9 + 3 + k];
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
            dst[384 * j + 16 * (16 + k) + 0] =
                dst[384 * j + 16 * (16 + k) + 6] = tmp[j * 9 + 6 + k];
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
  }
  __TOC__(per_process)
  if (use_aie_) {
    __TIC__(run_aie);
    {
      std::lock_guard<std::mutex> lk(mtx);
      aie_runner->execute_async(
          vitis::ai::vector_unique_ptr_get(aie_runner_inputs),
          vitis::ai::vector_unique_ptr_get(aie_runner_outputs));
    }
    __TOC__(run_aie)
    __TIC__(copy_aie_output)
    uint64_t data, tensor_size;
    std::tie(data, tensor_size) = aie_runner_outputs[0]->data({0, 0, 0, 0});
    auto size_per_batch = inputs[1]->get_tensor()->get_data_size();
    CHECK_EQ(size_per_batch, tensor_size / 4)
        << inputs[1]->get_tensor()->get_name();
    auto dst = (char*)inputs[1]->data({0, 0, 0, 0}).first;
    auto src = (int32_t*)data;
    for (int j = 0; j < size_per_batch; j++) dst[j] = src[j];
    __TOC__(copy_aie_output)
  }

  __TIC__(run_1)
  runner->execute_async(inputs, outputs);
  __TOC__(run_1)
  __TIC__(post_process)
  auto res = post_process(outputs, 0.05);
  __TOC__(post_process)
  return res;
}
// std::vector<CenterPointResult> run(
//     const std::vector<std::vector<char>>& input_bins) {
//   auto inputs = runner->get_inputs();
//   auto outputs = runner->get_outputs();
//   __TIC__(per_process)
//   CHECK_EQ(input_bins.size(), inputs.size());
//   for (size_t i = 0; i < inputs.size(); i++) {
//     if (i == 0) {
//       std::vector<char> tmp(input_bins[i].size());
//       for (int n = 0; n < 6; n++)
//         for (int c = 0; c < 3; c++)
//           for (int hw = 0; hw < 180224; hw++)
//             tmp[n * 180224 * 3 + hw * 3 + c] =
//                 input_bins[i][n * 180224 * 3 + c * 180224 + hw];
//       copy_input_from_bin(tmp, inputs[i]);
//       continue;
//     }
//     copy_input_from_bin(input_bins[i], inputs[i]);
//   }
//   __TOC__(per_process)
//   __TIC__(run_1)
//   runner->execute_async(inputs, outputs);
//   __TOC__(run_1)
//   __TIC__(post_process)
//   auto res = post_process(outputs, 0.05);
//   __TOC__(post_process)
//   return res;
// }

}  // namespace ai
}  // namespace vitis
