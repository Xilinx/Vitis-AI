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
#include <glog/logging.h>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/graph/graph.hpp>

#include "./anchor.hpp"
#include "./helper.hpp"
#include "./pointpillars.hpp"
#include "./pointpillars_post.hpp"
#include "./preprocess.hpp"
#include "vitis/ai/graph_runner.hpp"

namespace vitis {
namespace ai {
namespace pp {

std::vector<int> g_grid_size;
G_ANCHOR g_anchor;

}  // namespace pp
}  // namespace ai
}  // namespace vitis

using namespace vitis::ai::pp;
using namespace std;

std::vector<std::string> g_bin_files;
std::thread anchor_mask_t;
std::unique_ptr<PointPillarsPost> post_;
std::unique_ptr<PointPillarsPre> pre_;

struct tensors_attribute {
  int batchnum = 0;
  int realbatchnum = 0;

  int in_height0;
  int in_width0;
  float in_scale0;
  int in_channel0;

  int in_height1;
  int in_width1;
  float in_scale1;
  int in_channel1;

  int out_height0;
  int out_width0;
  float out_scale0;
} g_ta;

void get_grid_size() {
  for (int i = 0; i < 3; i++) {
    g_grid_size.emplace_back(
        int((cfg_point_cloud_range[i + 3] -
             cfg_point_cloud_range[i]) /
            cfg_voxel_size[i]));
  }
}

int get_fix_point(const xir::Tensor* tensor) {
  CHECK(tensor->has_attr("fix_point"))
      << "get tensor fix_point error! has no fix_point attr, tensor name is "
      << tensor->get_name();
  return tensor->template get_attr<int>("fix_point");
}

float get_scale(const xir::Tensor* input_tensor) {
  int fixpos = get_fix_point(input_tensor);
  float input_fixed_scale = std::exp2f(1.0f * (float)fixpos);
  return input_fixed_scale;
}

static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto ret = tensor->get_shape();
  std::fill(ret.begin(), ret.end(), 0);
  return ret;
}

int initialize(const std::vector<vart::TensorBuffer*>& input_tensor_buffers,
               const std::vector<vart::TensorBuffer*>& output_tensor_buffers) {
  get_grid_size();
  anchor_stride::create_all_anchors();

  auto input_tensor = input_tensor_buffers[1]->get_tensor();
  g_ta.batchnum = input_tensor->get_shape().at(0);
  g_ta.realbatchnum = std::min(g_ta.batchnum, (int)g_bin_files.size());  // 1
  g_ta.in_height0 = input_tensor->get_shape().at(1);   // 12000
  g_ta.in_width0 = input_tensor->get_shape().at(2);    // 100
  g_ta.in_channel0 = input_tensor->get_shape().at(3);  // 4
  g_ta.in_scale0 = get_scale(input_tensor);            // 64

  input_tensor = input_tensor_buffers[0]->get_tensor();
  g_ta.in_height1 = input_tensor->get_shape().at(1);  // 12000
  g_ta.in_width1 = input_tensor->get_shape().at(2);   // 4
  g_ta.in_channel1 = 1;                               // set to 1.
  g_ta.in_scale1 = 1;  // no this value. just set to 1

  std::vector<int8_t*> in_addr1;
  std::vector<float*> in_addr2;

  std::vector<std::int32_t> idx;
  uint64_t data_out = 0u;
  size_t size_out = 0u;

  // wait optimize ..
  for (int i = 0; i < g_ta.batchnum; i++) {
    idx = get_index_zeros(input_tensor_buffers[1]->get_tensor());
    idx[0] = i;
    std::tie(data_out, size_out) = input_tensor_buffers[1]->data(idx);
    in_addr1.emplace_back((int8_t*)data_out);

    idx = get_index_zeros(input_tensor_buffers[0]->get_tensor());
    idx[0] = i;
    std::tie(data_out, size_out) = input_tensor_buffers[0]->data(idx);
    in_addr2.emplace_back((float*)data_out);
  }

  pre_ = std::make_unique<PointPillarsPre>(
      in_addr1, (int)g_ta.in_scale0, g_ta.in_width0, g_ta.in_height0,
      g_ta.in_channel0, in_addr2, g_ta.in_width1, g_ta.in_height1,
      g_ta.batchnum, g_ta.realbatchnum);

  post_ = PointPillarsPost::create(output_tensor_buffers, g_ta.batchnum,
                                   g_ta.realbatchnum);
  return 0;
}

static void preprocess_run(const V2F& PointCloud) {
  for (int i = 0; i < g_ta.realbatchnum; i++) {
    pre_->process_net0(PointCloud[i].data(), PointCloud[i].size(), i);
  }

  // start the anchor_mask_t thread
  thread anchor_mask_t1(&PointPillarsPost::get_anchors_mask, post_.get(),
                        std::cref(pre_->pre_dict_));
  anchor_mask_t = std::move(anchor_mask_t1);

#if 0
  __TIC__(anchor_mask_t_wait_in_post)
  anchor_mask_t.join();
  __TOC__(anchor_mask_t_wait_in_post)
#endif
}

static std::vector<PointPillarsResult> postprocess_run(
    const std::vector<vart::TensorBuffer*>& output_tensor_buffers) {
  // wait for the anchor_mask_t thread
#if 1
  __TIC__(anchor_mask_t_wait_in_post)
  anchor_mask_t.join();
  __TOC__(anchor_mask_t_wait_in_post)
#endif
  __TIC__(postprocess_batch)
  std::vector<PointPillarsResult> results;
  for (int i = 0; i < g_ta.realbatchnum; i++) {
    results.emplace_back(post_->post_process(i));
  }
  __TOC__(postprocess_batch)
  return results;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <bin_url> " << std::endl;
    abort();
  }
  std::string g_xmodel_file = std::string(argv[1]);
  for (auto i = 2; i < argc; i++) {
    g_bin_files.push_back(std::string(argv[i]));
  }

  // create graph runner
  auto graph = xir::Graph::deserialize(g_xmodel_file);
  auto attrs = xir::Attrs::create();
  auto runner =
      vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
  CHECK(runner != nullptr);

  // get input/output tensor buffers
  auto input_tensor_buffers = runner->get_inputs();
  auto output_tensor_buffers = runner->get_outputs();

  int ret =
      initialize( input_tensor_buffers, output_tensor_buffers);
  CHECK_EQ(ret, 0) << "failed to initialize";

  // preprocess and fill input

  V2F PointCloud(g_ta.realbatchnum);
  for (int i = 0; i < g_ta.realbatchnum; i++) {
    int len = getfloatfilelen(g_bin_files[i]);
    PointCloud[i].resize(len);
    myreadfile(PointCloud[i].data(), len, g_bin_files[i]);
  }
  __TIC__(pp_total)
  __TIC__(pp_pre)
  preprocess_run(PointCloud);
  __TOC__(pp_pre)

  __TIC__(pp_model_run)
  // sync input tensor buffers
  for (auto& input : input_tensor_buffers) {
    input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                 input->get_tensor()->get_shape()[0]);
  }

  // run graph runner
  auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
  auto status = runner->wait((int)v.first, -1);
  CHECK_EQ(status, 0) << "failed to run the graph";

  // sync output tensor buffers
  for (auto output : output_tensor_buffers) {
    output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                 output->get_tensor()->get_shape()[0]);
  }

  __TOC__(pp_model_run)
  // postprocess and print resnet50 result
  __TIC__(pp_post)
  std::vector<PointPillarsResult> results =
      postprocess_run(output_tensor_buffers);
  __TOC__(pp_post)
  __TOC__(pp_total)

  for (int j = 0; j < g_ta.realbatchnum; j++) {
    std::cout << "result: " << j << "\n";
    for (unsigned int i = 0; i < results[j].ppresult.final_box_preds.size();
         i++) {
      std::cout << results[j].ppresult.label_preds[i] << "     " << std::fixed
                << std::setw(11) << std::setprecision(6) << std::setfill(' ')
                << results[j].ppresult.final_box_preds[i][0] << " "
                << results[j].ppresult.final_box_preds[i][1] << " "
                << results[j].ppresult.final_box_preds[i][2] << " "
                << results[j].ppresult.final_box_preds[i][3] << " "
                << results[j].ppresult.final_box_preds[i][4] << " "
                << results[j].ppresult.final_box_preds[i][5] << " "
                << results[j].ppresult.final_box_preds[i][6] << "     "
                << results[j].ppresult.final_scores[i] << "\n";
    }
  }
  return 0;
}
