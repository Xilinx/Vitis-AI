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
#include <vart/runner_ext.hpp>  // from vitis runtime api
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
  DpuTaskImp(const std::string& kernel_name, xir::Attrs* attrs);

  DpuTaskImp(const DpuTaskImp&) = delete;
  DpuTaskImp& operator=(const DpuTaskImp&) = delete;
  virtual ~DpuTaskImp();

 public:
  virtual void run(size_t idx) override;
  virtual void run_with_xrt_bo(
      const std::vector<vart::xrt_bo_t>& input_bos) override;
  virtual void setMeanScaleBGR(const std::vector<float>& mean,
                               const std::vector<float>& scale) override;
  virtual void setImageBGR(const cv::Mat& img) override;
  virtual void setImageRGB(const cv::Mat& img, size_t ind) override;
  virtual void setInputDataArray(const std::vector<int8_t> input, size_t ind) override;
  virtual void setInputDataArray(
      const std::vector<std::vector<int8_t>> input, size_t ind) override;

  virtual void setImageBGR(const std::vector<cv::Mat>& imgs) override;
  virtual void setImageRGB(const std::vector<cv::Mat>&, size_t ind) override;
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
  virtual std::unique_ptr<xir::Attrs> get_attrs() const override;
  virtual int get_input_buffer_size() const override;
  virtual size_t get_input_offset() const override;
  virtual int get_input_fix_point() const override;

 private:
  void setImageBGR(const uint8_t* input, int stride);
  void setImageRGB(const uint8_t* input, int stride, size_t ind);
  void set_num_of_inputs(size_t n);
  // size_t get_num_of_inputs(size_t n);
  // void clear_num_of_inputs(size_t n);

 private:
  const std::string model_name_;
  std::shared_ptr<GraphHolder> graph_holder_;
  //# Adding dirname_ back for DPUV1
  const std::string dirname_;
  std::unique_ptr<xir::Attrs> default_attrs_;
  std::vector<std::unique_ptr<vart::Runner>> runners_;
  // # cache input_tensors & output_tensors
  // The function tensor.get_attr() is frequently called by deployed code,  there will be a certain performance loss.
  std::vector<std::vector<vitis::ai::library::InputTensor>> all_input_tensors_;
  std::vector<std::vector<vitis::ai::library::OutputTensor>> all_output_tensors_;
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
