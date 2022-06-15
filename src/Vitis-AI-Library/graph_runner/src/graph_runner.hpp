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
#include <unordered_map>
#include <xir/op/op.hpp>

#include "./tensor_buffer_linker.hpp"
#include "vart/assistant/tensor_buffer_allocator.hpp"
#include "vart/runner_ext.hpp"

namespace {

struct OutputInternal {
  std::unique_ptr<xir::Tensor> my_tensor;
  const xir::Tensor* subgraph_output_tensor;
  std::unique_ptr<vart::TensorBuffer> output_tensor_buffer;
  std::unique_ptr<TensorBufferLinker> linker;
};

struct GraphInternal {
  GraphInternal(const xir::Subgraph* sg);
  GraphInternal(GraphInternal&& other) = default;
  ~GraphInternal();
  void reset_buffers();
  const xir::Subgraph* subgraph;
  std::string device;
  std::unique_ptr<vart::Runner> runner;
  std::vector<std::unique_ptr<xir::Tensor>> input_tensors;  // add batch & ddr
  std::vector<std::unique_ptr<vart::TensorBuffer>> input_tensor_buffers;
  std::vector<const xir::Tensor*> subgraph_input_tensors;
  std::vector<OutputInternal> outputs;
};

class GraphTask : public vart::RunnerExt {
 public:
  explicit GraphTask(const xir::Subgraph* subgraph, xir::Attrs* attrs);
  GraphTask(const GraphTask& other) = delete;
  virtual ~GraphTask();

 private:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() override;
  virtual std::vector<vart::TensorBuffer*> get_inputs() override;
  virtual std::vector<vart::TensorBuffer*> get_outputs() override;

 private:
  void build_runners();
  void build_tensors();
  void build_tensors_for_dpu();
  void update_batch_size(const std::vector<xir::Tensor*>& tensors);
  void build_tensors_for_non_dpu();
  void build_tensors_for_device();
  void build_tensor_buffers();
  void link_tensor_buffers();
  void link_tensor_buffers(GraphInternal& cur,
                           std::vector<GraphInternal>::iterator to);
  std::vector<
      std::pair<std::unique_ptr<vart::TensorBuffer>*, const xir::Subgraph*>>
  get_slaves(std::vector<GraphInternal>::iterator down,
             const std::string& master);
  std::vector<const xir::Tensor*> build_input_tensors();
  std::vector<const xir::Tensor*> build_output_tensors();
  std::vector<vart::TensorBuffer*> build_input_tensor_buffers();
  std::vector<vart::TensorBuffer*> build_output_tensor_buffers();
  void build_subgraph_tensors();
  // bool is_same_tensor_buffer(const vart::TensorBuffer* up,
  //                         const vart::TensorBuffer* down);
  // const xir::Op* find_op(const std::string& tensor_name) const;
  const xir::Tensor* find_tensor(const std::string& tensor_name) const;
  vart::TensorBuffer* find_tensor_buffer(const std::string& tensor_name) const;
  std::unordered_map<std::string,
                     std::vector<std::unique_ptr<vart::TensorBuffer>>>
  allocate_all_tensor_buffers();
  void finalize_linkers();
  void after_invoke_runner(GraphInternal& i);
  std::vector<std::unique_ptr<vart::TensorBuffer>> map_to_single_batch(
      const std::vector<vart::TensorBuffer*>& tensor_buffers,
      size_t batch_index, size_t batch);
  void maybe_dump_tensor_buffers(
      const std::vector<vart::TensorBuffer*>& inputs,
      const std::vector<vart::TensorBuffer*>& outputs, size_t subgraph_index,
      size_t batch_index);

 private:
  const xir::Subgraph* subgraph_;
  std::unique_ptr<xir::Attrs> attrs_;
  std::vector<GraphInternal> internal_;
  size_t dpu_batch_size_;
  size_t nondpu_batch_size_;
  std::unique_ptr<vart::assistant::TensorBufferAllocator>
      tensor_buffer_allocator_;
  std::vector<const xir::Tensor*> input_tensors_;
  std::vector<const xir::Tensor*> output_tensors_;
  std::vector<vart::TensorBuffer*> input_tensor_buffers_;
  std::vector<vart::TensorBuffer*> output_tensor_buffers_;
};

}  // namespace
