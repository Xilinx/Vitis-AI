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

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/facelandmark.hpp>
#include <vitis/ai/nnpp/facelandmark.hpp>
#include <xir/graph/graph.hpp>
extern int g_last_frame_id;

// A struct that can storage model info
struct ModelInfo {
  int w;
  int h;
} model_info_landmark;

namespace vitis {
namespace ai {

static ModelInfo get_model_zise(const std::string& filename) {
  std::string model_name = "/usr/share/vitis_ai_library/models/" + filename +
                           "/" + filename + ".xmodel";
  auto graph = xir::Graph::deserialize(model_name);
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  if (children.empty()) {
    std::cout << "no subgraph" << std::endl;
  }
  for (auto c : children) {
    CHECK(c->has_attr("device"));
    auto device = c->get_attr<std::string>("device");
    if (device == "DPU") {
      auto inputs = c->get_sorted_input_tensors();
      // for (auto input : inputs) {
      if (inputs.size() > 0) {
        auto input = inputs[0];
        int height = input->get_shape().at(1);
        int width = input->get_shape().at(2);
        //        std::cout << "model width: " << width  << " model heigth: " <<
        //        height << std::endl;
        return ModelInfo{width, height};
      }
    }
  }
  return {0, 0};
}

static std::vector<std::string> split(const std::string& s,
                                      const std::string& delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}

struct FaceLandmarkAcc : public AccThread {
  FaceLandmarkAcc(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
  }

  virtual ~FaceLandmarkAcc() { of.close(); }

  static std::shared_ptr<FaceLandmarkAcc> instance(std::string output_file) {
    static std::weak_ptr<FaceLandmarkAcc> the_instance;
    std::shared_ptr<FaceLandmarkAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<FaceLandmarkAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (FaceLandmarkResult*)dpu_result.result_ptr.get();
    of << split(dpu_result.single_name, ".")[0] << " ";
    auto points = result->points;
    for (int i = 0; i < 5; ++i) {
      of << (int)(points[i].first * model_info_landmark.w) << " ";
    }
    for (int i = 0; i < 5; i++) {
      of << (int)(points[i].second * model_info_landmark.h) << " ";
    }
    of << std::endl;
  }

  virtual int run() override {
    if (g_last_frame_id == int(dpu_result.frame_id)) return -1;
    if (queue_->pop(dpu_result, std::chrono::milliseconds(50000)))
      process_result(dpu_result);
    return 0;
  }

  DpuResultInfo dpu_result;
  std::ofstream of;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << " usage: " << argv[0]
              << " <model_name> <image_list> <output_file>" << std::endl;  //
    abort();
  }

  model_info_landmark = vitis::ai::get_model_zise(argv[1]);
  return vitis::ai::main_for_accuracy_demo(
      argc, argv, [&] { return vitis::ai::FaceLandmark::create(argv[1]); },
      vitis::ai::FaceLandmarkAcc::instance(argv[3]), 2);
}
