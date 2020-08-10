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
#include <vart/dpu/dpu_runner_ext.hpp>  // from vitis runtime api
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <xir/graph/graph.hpp>
#include "graph_holder.hpp"

DEF_ENV_PARAM(ENABLE_DEBUG_DPBASE, "0");

namespace vitis {
namespace ai {

class DpuTaskImp : public DpuTask {
 public:
  DpuTaskImp(const std::string& kernel_name);

  DpuTaskImp(const DpuTaskImp&) = delete;
  DpuTaskImp& operator=(const DpuTaskImp&) = delete;
  virtual ~DpuTaskImp();

 public:
  virtual void run(size_t idx) override;
  virtual void setMeanScaleBGR(const std::vector<float>& mean,
                               const std::vector<float>& scale) override;
  virtual void setImageBGR(const cv::Mat& img) override;
  virtual void setImageRGB(const cv::Mat& img) override;

  virtual void setImageBGR(const std::vector<cv::Mat>& imgs) override;
  virtual void setImageRGB(const std::vector<cv::Mat>& imgs) override;
  //  virtual void setImageBGR(const uint8_t *input, int stride) override;
  // virtual void setImageRGB(const uint8_t *input, int stride) override;

  virtual std::vector<float> getMean() override;
  virtual std::vector<float> getScale() override;
  virtual std::vector<vitis::ai::library::InputTensor> getInputTensor(
      size_t idx) override;
  virtual std::vector<vitis::ai::library::OutputTensor> getOutputTensor(
      size_t idx) override;
  virtual size_t get_input_batch(size_t kernel_idx,
                                 size_t node_idx) const override;
  virtual size_t get_num_of_kernels() const override;
  virtual const xir::Graph* get_graph() const override;

 private:
  void setImageBGR(const uint8_t* input, int stride);
  void setImageRGB(const uint8_t* input, int stride);
  void set_num_of_inputs(size_t n);
  // size_t get_num_of_inputs(size_t n);
  // void clear_num_of_inputs(size_t n);

 private:
  const std::string model_name_;
  std::shared_ptr<GraphHolder> graph_holder_;
  //# Adding dirname_ back for DPUV1
  const std::string dirname_;							 
  std::vector<std::unique_ptr<vart::Runner>> runners_;
  std::vector<float> mean_;
  std::vector<float> scale_;
  bool do_mean_scale_;
  int num_of_inputs_ = -1;
};

}  // namespace ai
}  // namespace vitis

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: undecided-unix
// End:
