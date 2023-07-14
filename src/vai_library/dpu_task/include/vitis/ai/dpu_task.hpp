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
#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <vart/assistant/xrt_bo_tensor_buffer.hpp>
#include <vart/experimental/runner_helper.hpp>
#include <vart/zero_copy_helper.hpp>
#include <vector>
#include <vitis/ai/library/tensor.hpp>
namespace xir {
class Graph;
class Attrs;
}  // namespace xir

namespace vitis {
namespace ai {

/**
 * @brief Base class for run a DPU task.
 */
class DpuTask {
  /**
   * @cond NOCOMMENTS
   */
 protected:
  DpuTask();

 public:
  virtual ~DpuTask();
  DpuTask(const DpuTask& other) = delete;
  DpuTask& operator=(const DpuTask& rhs) = delete;
  /**
   * @endcond
   */
 public:
  /**
   * @brief A static method to create a DPU task.
   *
   * @param kernel_name The dpu kernel name.
   * for example, if kernel_name is "resnet_50", the following dpu model files
   * are searched.
   *      ./libdpumodelrestnet_50.so
   *      /usr/lib/libdpumodelrestnet_50.so
   * @return A DpuTask instance.
   */
  static std::unique_ptr<DpuTask> create(const std::string& kernal_name);

  /**
   * @cond NOCOMMENTS
   */
  static std::unique_ptr<DpuTask> create(const std::string& kernal_name,
                                         xir::Attrs* attrs);
  /**
   * @endcond
   */

 public:
  /**
   * @brief Run the dpu task
   * @note Before invoking this function. An input data should be properly
   * copied to input tensors, via `setImageBGR` or `setImageRGB`.
   */
  virtual void run(size_t idx) = 0;
  /**
   * @cond NOCOMMENTS
   */
  virtual void run_with_xrt_bo(
      const std::vector<vart::xrt_bo_t>& input_bos) = 0;
  /**
   * @endcond
   */

  /**
   * @brief Set the mean/scale values.
   * @note By default, no mean-scale processing, after invoking this
   * function, mean-scale processing is enabled. You cannot turn it
   * off after enabling.
   * @param mean Mean, Normalization is used.
   * @param scale Scale, Normalization is used.
   */
  virtual void setMeanScaleBGR(const std::vector<float>& mean,
                               const std::vector<float>& scale) = 0;
  /**
   * @brief Copy a input image in BGR format to the input tensor.
   * @param img The input image (cv::Mat).
   */
  virtual void setImageBGR(const cv::Mat& img) = 0;
  /**
   * @cond NOCOMMENTS
   */
  virtual void setImageBGR(const std::vector<cv::Mat>& imgs) = 0;
  /**
   * @endcond
   */
  /**
   * @brief Copy a input image in RGB format to the input tensor.
   * @param img The input image(cv::Mat).
   */
  virtual void setImageRGB(const cv::Mat& img, size_t ind=0) = 0;
  /**
   * @cond NOCOMMENTS
   */
  virtual void setImageRGB(const std::vector<cv::Mat>& imgs, size_t ind=0) = 0;
  /**
   * @cond NOCOMMENTS
   */
  virtual void setInputDataArray(const std::vector<int8_t> input, size_t ind) = 0;
  /**
   * @cond NOCOMMENTS
   */
  virtual void setInputDataArray(
      const std::vector<std::vector<int8_t>> input, size_t ind) = 0;
  /**
   * @endcond
   */
  /**
   * @brief Get the mean values.
   * @return Mean values
   */
  virtual std::vector<float> getMean() = 0;
  /**
   * @brief Get the scale values.
   * @return Scale values
   */
  virtual std::vector<float> getScale() = 0;
  /**
   * @brief Get the input tensors.
   * @return The input tensors
   */
  virtual std::vector<vitis::ai::library::InputTensor> getInputTensor(
      size_t idx) = 0;
  /**
   * @brief Get the output tensors.
   * @return The output tensors.
   */
  virtual std::vector<vitis::ai::library::OutputTensor> getOutputTensor(
      size_t idx) = 0;
  /**
   * @brief get the number of tasks
   * @return The count of tasks
   * */
  virtual size_t get_num_of_kernels() const = 0;

  /**
   * @cond NOCOMMENTS
   */
  virtual size_t get_input_batch(size_t kernel_idx, size_t node_idx) const = 0;
  /**
   * @endcond
   */

  /**
   * @brief Get the graph of xmodel.
   * @return A xir graph.
   */
  virtual const xir::Graph* get_graph() const = 0;

  /**
   * @cond NOCOMMENTS
   */
  virtual std::unique_ptr<xir::Attrs> get_attrs() const = 0;

  /**
   * @cond NOCOMMENTS
   */
  virtual int get_input_buffer_size() const = 0;
  /**
   * @cond NOCOMMENTS
   */
  virtual size_t get_input_offset() const = 0;
  /**
   * @endcond
   */

  /**
   * @brief Get the fix point of the input tensor in this graph.
   * @return The input fix point.
   */
  virtual int get_input_fix_point() const = 0;
};

}  // namespace ai
}  // namespace vitis

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: undecided-unix
// End:
