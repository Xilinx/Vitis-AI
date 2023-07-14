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
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <memory>
#include <vector>

#include "vart/runner_helper.hpp"
#include "vart/simple_tensor_buffer.hpp"
#include "xir/graph/graph.hpp"
#include "xir/op/op_def.hpp"

namespace vitis {
namespace ai {

struct XmodelPostprocessorArgOpAndTensorBuffer {
  const xir::Op* op;
  vart::TensorBuffer* tensor_buffer;
};

struct XmodelPostprocessorArg {
  std::string name;
  std::vector<XmodelPostprocessorArgOpAndTensorBuffer> args;
};

struct XmodelPostprocessorInputs {
  std::vector<XmodelPostprocessorArg> inputs;
};

struct XmodelPostprocessorInitializationArgs {
  const xir::Graph* graph;
  const xir::Tensor* graph_input_tensor;
  const XmodelPostprocessorInputs& graph_output_tensor_buffers;
  const xir::Attrs* attrs;
};

class XmodelPostprocessorBase {
 public:
  explicit XmodelPostprocessorBase();
  virtual ~XmodelPostprocessorBase() = default;
  XmodelPostprocessorBase(const XmodelPostprocessorBase& other) = delete;
  XmodelPostprocessorBase& operator=(const XmodelPostprocessorBase& rhs) =
      delete;

 public:
  static std::unique_ptr<XmodelPostprocessorBase> create(
      const xir::Graph* graph);
  virtual void initialize(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) = 0;
  virtual std::vector<vitis::ai::proto::DpuModelResult> process(
      const XmodelPostprocessorInputs& tensor_buffers) = 0;
  virtual const xir::OpDef& get_def() const = 0;

 protected:
  /*  size_t batch_;
    size_t height_;
    size_t width_;
    size_t depth_;*/
};

class XmodelPostprocessorSingleBatch : public XmodelPostprocessorBase {
 public:
  explicit XmodelPostprocessorSingleBatch();
  virtual ~XmodelPostprocessorSingleBatch() = default;
  XmodelPostprocessorSingleBatch(const XmodelPostprocessorSingleBatch& other) =
      delete;
  XmodelPostprocessorSingleBatch& operator=(
      const XmodelPostprocessorSingleBatch& rhs) = delete;

 public:
  virtual vitis::ai::proto::DpuModelResult process_single_batch(
      const XmodelPostprocessorInputs& tensor_buffers) = 0;

 private:
  virtual std::vector<vitis::ai::proto::DpuModelResult> process(
      const XmodelPostprocessorInputs& tensor_buffers) override;
};

template <typename T>
struct arg_converter_t {
  static T convert(const XmodelPostprocessorInputs& graph_output_tensor_buffers,
                   const xir::OpDef* opdef, size_t arg_index);
};

template <typename T>
T arg_converter_t<T>::convert(
    const XmodelPostprocessorInputs& graph_output_tensor_buffers,
    const xir::OpDef* opdef, size_t arg_index) {
  LOG(FATAL) << "cannot convert argument: T=" << typeid(T).name();
  return T();
}

template <typename T>
struct arg_converter_t<vart::simple_tensor_buffer_t<T>> {
  static vart::simple_tensor_buffer_t<T> convert(
      const XmodelPostprocessorInputs& graph_output_tensor_buffers,
      const xir::OpDef* opdef, size_t arg_index) {
    auto input_arg_defs = opdef->input_args();
    CHECK_LT(arg_index, input_arg_defs.size()) << "wrong number of argument";
    CHECK_EQ(input_arg_defs.size(), graph_output_tensor_buffers.inputs.size());
    auto& arg_def = input_arg_defs[arg_index];
    CHECK(arg_def.occur_type == xir::OpArgDef::REQUIRED)
        << "it must be required single argument.";
    CHECK_EQ(graph_output_tensor_buffers.inputs[arg_index].args.size(), 1u)
        << "it must be single argument. name=" << arg_def.name;
    return vart::simple_tensor_buffer_t<T>::create(
        graph_output_tensor_buffers.inputs[arg_index].args[0].tensor_buffer);
  }
};

template <typename T>
struct arg_converter_t<
    std::unique_ptr<vart::simple_tensor_buffer_t<T>>> {
  static std::unique_ptr<vart::simple_tensor_buffer_t<T>> convert(
      const XmodelPostprocessorInputs& graph_output_tensor_buffers,
      const xir::OpDef* opdef, size_t arg_index) {
    auto input_arg_defs = opdef->input_args();
    CHECK_LT(arg_index, input_arg_defs.size()) << "wrong number of argument";
    CHECK_EQ(input_arg_defs.size(), graph_output_tensor_buffers.inputs.size());
    auto& arg_def = input_arg_defs[arg_index];
    CHECK(arg_def.occur_type == xir::OpArgDef::OPTIONAL)
        << "it must be required single argument.";
    if (graph_output_tensor_buffers.inputs[arg_index].args.size() == 1) {
      return std::make_unique<vart::simple_tensor_buffer_t<T>>(
          vart::simple_tensor_buffer_t<T>::create(
              graph_output_tensor_buffers.inputs[arg_index]
                  .args[0]
                  .tensor_buffer));
    }
    return nullptr;
  }
};

template <typename T>
struct arg_converter_t<
    std::vector<vart::simple_tensor_buffer_t<T>>> {
  static std::vector<vart::simple_tensor_buffer_t<T>> convert(
      const XmodelPostprocessorInputs& graph_output_tensor_buffers,
      const xir::OpDef* opdef, size_t arg_index) {
    auto input_arg_defs = opdef->input_args();
    CHECK_LT(arg_index, input_arg_defs.size()) << "wrong number of argument";
    CHECK_EQ(input_arg_defs.size(), graph_output_tensor_buffers.inputs.size());
    auto& arg_def = input_arg_defs[arg_index];
    CHECK(arg_def.occur_type == xir::OpArgDef::REQUIRED_AND_REPEATED)
        << "it must be required_and_repeated argument.";
    auto ret = std::vector<vart::simple_tensor_buffer_t<T>>();
    ret.reserve(graph_output_tensor_buffers.inputs[arg_index].args.size());
    for (auto& arg : graph_output_tensor_buffers.inputs[arg_index].args) {
      ret.emplace_back(vart::simple_tensor_buffer_t<T>::create(
          arg.tensor_buffer));
    };
    return ret;
  }
};

template <typename Imp>
class XmodelPostprocessor : public XmodelPostprocessorSingleBatch {
 public:
  explicit XmodelPostprocessor();
  virtual ~XmodelPostprocessor() = default;
  XmodelPostprocessor(const XmodelPostprocessor& other) = delete;
  XmodelPostprocessor& operator=(const XmodelPostprocessor& rhs) = delete;

 private:
  virtual vitis::ai::proto::DpuModelResult process_single_batch(
      const XmodelPostprocessorInputs& graph_output_tensor_buffers) override;
  virtual void initialize(
      XmodelPostprocessorInitializationArgs&& args) override;
  virtual const xir::OpDef& get_def() const override { return op_def_; }

 private:
  template <typename T>
  using remove_refcv_t =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;

  template <typename... Args, size_t... Index>
  vitis::ai::proto::DpuModelResult process_proxy0(
      vitis::ai::proto::DpuModelResult (Imp::*f)(Args...),
      const XmodelPostprocessorInputs& graph_output_tensor_buffers,
      std::integer_sequence<size_t, Index...> int_seq) {
    CHECK_EQ(op_def_.input_args().size(), sizeof...(Index))
        << "num of input arguments mismatch."
        << "op: " << to_string(&op_def_);
    return std::invoke(f, imp_.get(),
                       (arg_converter_t<remove_refcv_t<Args>>::convert(
                           graph_output_tensor_buffers, &op_def_, Index))...);
  }

  template <typename... Args>
  vitis::ai::proto::DpuModelResult process_proxy(
      vitis::ai::proto::DpuModelResult (Imp::*f)(Args...),
      const XmodelPostprocessorInputs& graph_output_tensor_buffers) {
    return process_proxy0(f, graph_output_tensor_buffers,
                          std::make_index_sequence<sizeof...(Args)>());
  }

 private:
  std::unique_ptr<Imp> imp_;
  xir::OpDef op_def_;
};

template <typename Imp>
XmodelPostprocessor<Imp>::XmodelPostprocessor()
    : XmodelPostprocessorSingleBatch(),
      imp_{nullptr},
      op_def_{Imp::get_op_def()} {}

template <typename Imp>
vitis::ai::proto::DpuModelResult XmodelPostprocessor<Imp>::process_single_batch(
    const XmodelPostprocessorInputs& tensor_buffers) {
  return process_proxy(&Imp::process, tensor_buffers);
}

template <typename Imp>
void XmodelPostprocessor<Imp>::initialize(
    XmodelPostprocessorInitializationArgs&& args) {
  imp_ = std::make_unique<Imp>(std::move(args));
}

}  // namespace ai
}  // namespace vitis
