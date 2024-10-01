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

#include <iterator>
#include <limits>
#include <thread>

#include <UniLog/UniLog.hpp>
#include "graph_runner.hpp"
#include "vart/batch_tensor_buffer_view.hpp"
#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/path_util.hpp"
#include "xir/graph/graph.hpp"

DEF_ENV_PARAM(DEBUG_GRAPH_RUNNER, "0")
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0")

namespace {

GraphTask::GraphTask(const xir::Subgraph* subgraph, xir::Attrs* attrs)
    : subgraph_{subgraph},
      attrs_{attrs == nullptr ? xir::Attrs::create()
                              : xir::Attrs::clone(attrs)},
      dpu_batch_size_{0u},
      nondpu_batch_size_{0u} {
  // dirty hack override original cpu-runner.so
  attrs_->set_attr("lib", std::map<std::string, std::string>{
                              {"CPU", "libvitis_ai_library-cpu_task.so.3"}});
  internal_ = vitis::ai::vec_map(
      subgraph->children_topological_sort(),
      [](const xir::Subgraph* subgraph_in) { return GraphInternal(subgraph_in); });
  build_runners();
  build_tensors();
  if (!attrs_->has_attr("__batch__")) {
    attrs_->set_attr<size_t>("__batch__", nondpu_batch_size_);
  }
  // CHECK(dpu_batch_size_ % nondpu_batch_size_ == 0u &&
  //      dpu_batch_size_ >= nondpu_batch_size_)
  //    << "dpu_batch_size_= " << dpu_batch_size_
  //    << ";nondpu_batch_size_= " << nondpu_batch_size_;
  UNI_LOG_CHECK((dpu_batch_size_ % nondpu_batch_size_ == 0u &&
                 dpu_batch_size_ >= nondpu_batch_size_),
                VAILIB_GRAPH_RUNNER_DPU_BATCH_ERROR)
      << "dpu_batch_size_= " << dpu_batch_size_
      << ";nondpu_batch_size_= " << nondpu_batch_size_;
  tensor_buffer_allocator_ =
      vart::assistant::TensorBufferAllocator::create(attrs_.get());
  build_tensor_buffers();
  link_tensor_buffers();
  finalize_linkers();
  input_tensors_ = build_input_tensors();
  output_tensors_ = build_output_tensors();
  input_tensor_buffers_ = build_input_tensor_buffers();
  output_tensor_buffers_ = build_output_tensor_buffers();
  build_subgraph_tensors();
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "GraphTask is created "
      << "@" << (void*)this << " "
      << "inputs:" << to_string(input_tensor_buffers_)
      << "outputs:" << to_string(output_tensor_buffers_);
}

static bool in_subgraph(const xir::Subgraph* subgraph, const xir::Op* op) {
  auto ops = subgraph->get_ops();
  return ops.find(op) != ops.end();
}

static std::vector<const xir::Op*> get_input_ops(const xir::Op* op) {
  auto ret = std::vector<const xir::Op*>();
  auto inputs = op->get_input_ops();
  for (auto& x : inputs) {
    for (auto& y : x.second) {
      ret.emplace_back(y);
    }
  }
  return ret;
}

std::set<const xir::Op*> get_head_ops(const xir::Subgraph* subgraph) {
  auto ops = subgraph->get_ops();
  auto ret = std::set<const xir::Op*>();
  for (auto& op : ops) {
    auto inputs = get_input_ops(op);
    auto is_head = std::all_of(
        inputs.begin(), inputs.end(),
        [subgraph](const xir::Op* op) { return !in_subgraph(subgraph, op); });
    if (is_head) {
      ret.insert(op);
    }
  }
  return ret;
}

static bool is_wired_orphan_subgraph(const xir::Subgraph* subgraph) {
  auto inputs = subgraph->get_sorted_input_tensors();
  if (!inputs.empty()) {
    return false;
  }
  auto ops = get_head_ops(subgraph);
  return std::all_of(ops.begin(), ops.end(), [](const xir::Op* op) {
    return op->get_type() == "const" || op->get_type() == "const-fix";
  });
}

void GraphTask::build_runners() {
  for (auto& i : internal_) {
    i.device = i.subgraph->has_attr("device")
                   ? i.subgraph->get_attr<std::string>("device")
                   : std::string("UNKNOWN");
  }
  // create DPU runner first, so that __batch__ is set properly
  for (auto& i : internal_) {
    if (i.device == "DPU") {
      i.runner =
          vart::Runner::create_runner_with_attrs(i.subgraph, attrs_.get());
    }
  }
  // create DPU runner first, so that __batch__ is set properly
  for (auto& i : internal_) {
    if (i.device != "DPU" && (i.subgraph->has_attr("runner"))) {
      // USER subgraph might have no runner, might have in case of testing.
      if (is_wired_orphan_subgraph(i.subgraph)) {
        // some orphan subgraph, which does not connect to any subgraph at all.
        continue;
      }
      i.runner =
          vart::Runner::create_runner_with_attrs(i.subgraph, attrs_.get());
    }
  }
}

static std::vector<const xir::Tensor*> output_internal_get_my_tensors(
    const std::vector<OutputInternal>& outputs) {
  return vitis::ai::vec_map(outputs, [](const OutputInternal& o) {
    return const_cast<const xir::Tensor*>(o.my_tensor.get());
  });
}

static std::vector<vart::TensorBuffer*> output_internal_get_tensor_buffers(
    const std::vector<OutputInternal>& outputs) {
  return vitis::ai::vec_map(outputs, [](const OutputInternal& o) {
    return const_cast<vart::TensorBuffer*>(o.output_tensor_buffer.get());
  });
}

void GraphTask::build_tensors() {
  build_tensors_for_dpu();
  build_tensors_for_non_dpu();
  for (auto& i : internal_) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
        << "create tensors for sg[" << i.subgraph->get_name() << "] on "
        << i.device << "\n"                                              //
        << "\tinput tensors: "                                           //
        << to_string(vitis::ai::vector_unique_ptr_get(i.input_tensors))  //
        << "\n"                                                          //
        << "\toutput tensors: "                                          //
        << to_string(output_internal_get_my_tensors(i.outputs));
    ;
  }
}

void GraphTask::build_tensors_for_dpu() {
  auto clone_tensor = [](const xir::Tensor* b) {
    // TODO: it is too heavy to clone all attributes, consider to use
    // runner_helper/src/tensor_mirror_attrs.hpp
    auto ret = xir::Tensor::clone(b);
    ret->set_attr<std::string>("device", "DPU");
    return ret;
  };
  for (auto& i : internal_) {
    if (i.device == "DPU") {
      i.input_tensors =
          vitis::ai::vec_map(i.runner->get_input_tensors(), clone_tensor);
      auto output_tensors =
          vitis::ai::vec_map(i.runner->get_output_tensors(), clone_tensor);
      update_batch_size(vitis::ai::vector_unique_ptr_get(i.input_tensors));
      update_batch_size(vitis::ai::vector_unique_ptr_get(output_tensors));
      i.outputs.resize(output_tensors.size());
      int c = 0;
      for (auto& output_tensor : output_tensors) {
        i.outputs[c].my_tensor = std::move(output_tensor);
        c = c + 1;
      }
    }
  }
}

void GraphTask::update_batch_size(const std::vector<xir::Tensor*>& tensors) {
  for (auto& b : tensors) {
    if (dpu_batch_size_ == 0u) {
      dpu_batch_size_ = (size_t)b->get_shape()[0];
    } else {
      // CHECK_EQ(dpu_batch_size_, (size_t)b->get_shape()[0])
      //    << "all tensor must have same batch size: " << b->to_string();
      UNI_LOG_CHECK(dpu_batch_size_ == (size_t)b->get_shape()[0],
                    VAILIB_GRAPH_RUNNER_DPU_BATCH_ERROR)
          << "all tensor must have same batch size: " << b->to_string();
    }
  }
}

void GraphTask::build_tensors_for_non_dpu() {
  std::string device = "";
  for (auto& i : internal_) {
    device = i.device;
    auto subgraph = i.subgraph;
    auto clone_tensor = [subgraph, this, &device](const xir::Tensor* b) {
      auto dims = b->get_shape();
      if (nondpu_batch_size_ == 0u) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "initialize nondpu subgraph batch=" << dims[0]
            << " subgraph=" << subgraph->get_name();
        nondpu_batch_size_ = (size_t)dims[0];
      } else {
        CHECK_EQ(nondpu_batch_size_, (size_t)dims[0])
            << "all tensor must have same batch size: " << b->to_string()
            << " subgraph=" << subgraph->get_name();
      }
      if (dpu_batch_size_ == 0u) {
        dpu_batch_size_ = nondpu_batch_size_;
      }
      dims[0] = (int)dpu_batch_size_;
      auto ret = xir::Tensor::create(b->get_name(), dims, b->get_data_type());
      ret->set_attrs(b->get_attrs());
      ret->set_attr<std::string>("device", device);
      return ret;
    };
    auto clone_tensor_unique = [subgraph, &clone_tensor](const xir::Tensor* b) {
      return clone_tensor(b);
    };
    if (i.device == "DPU") {
      continue;
    }
    auto output_tensors = std::vector<std::unique_ptr<xir::Tensor>>();
    if (i.runner == nullptr) {  // no runner for device USER
      if (is_wired_orphan_subgraph(i.subgraph)) {
        continue;
      }
      i.input_tensors = vitis::ai::vec_map(
          i.subgraph->get_sorted_input_tensors(), clone_tensor_unique);
      output_tensors = vitis::ai::vec_map(
          i.subgraph->get_sorted_output_tensors(), clone_tensor_unique);
    } else {
      i.input_tensors =
          vitis::ai::vec_map(i.runner->get_input_tensors(), clone_tensor);
      output_tensors =
          vitis::ai::vec_map(i.runner->get_output_tensors(), clone_tensor);
    }
    i.outputs.resize(output_tensors.size());
    int c = 0;
    for (auto& output_tensor : output_tensors) {
      i.outputs[c].my_tensor = std::move(output_tensor);
      c = c + 1;
    }
  }
}

template <typename T>
std::vector<std::unique_ptr<T>> create_view_vector_get(
    const std::vector<T*>& from) {
  return vitis::ai::vec_map(from, [](T* x) {
    return std::unique_ptr<T>(std::make_unique<vart::BatchTensorBufferView>(
        x, 0, x->get_tensor()->get_shape().at(0)));
  });
}

void GraphTask::build_tensor_buffers() {
  for (auto& i : internal_) {
    auto output_tensor_buffers =
        std::vector<std::unique_ptr<vart::TensorBuffer>>();
    auto runner_ext = dynamic_cast<vart::RunnerExt*>(i.runner.get());
    if (runner_ext != nullptr) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
          << i.subgraph->get_name() << " can create vart::RunnerExt!"
          << " use RunnerExt's input/output tensorbuffers to save CMA."
          << " Need release ownership to RunnerExt before GraphTask destroyed!";
      i.input_tensor_buffers = create_view_vector_get(runner_ext->get_inputs());
      output_tensor_buffers = create_view_vector_get(runner_ext->get_outputs());
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
          << i.subgraph->get_name() << " cannot create vart::RunnerExt!";
      std::tie(i.input_tensor_buffers, output_tensor_buffers) =
          tensor_buffer_allocator_->allocate(
              i.subgraph,
              vitis::ai::vector_unique_ptr_get_const(i.input_tensors),
              output_internal_get_my_tensors(i.outputs));
    }
    CHECK_EQ(output_tensor_buffers.size(), i.outputs.size());
    auto size = output_tensor_buffers.size();
    for (auto j = 0u; j < size; ++j) {
      i.outputs[j].output_tensor_buffer = std::move(output_tensor_buffers[j]);
      CHECK_EQ(i.outputs[j].output_tensor_buffer->get_tensor()->get_name(),
               i.outputs[j].my_tensor->get_name());
    }
  }
  for (auto& i : internal_) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
        << "create tensor buffers for sg[" << i.subgraph->get_name() << "] on "
        << i.device                      //
        << "\n\tinput tensor buffers: "  //
        << to_string(
               vitis::ai::vector_unique_ptr_get(i.input_tensor_buffers))  //
        << "\n\toutput tensor buffers: "                                  //
        << to_string(output_internal_get_tensor_buffers(i.outputs)) << "\n";
  }
}

void GraphTask::link_tensor_buffers() {
  for (auto i = internal_.begin(); i != internal_.end(); ++i) {
    link_tensor_buffers(*i, i + 1);
  }
  if (ENV_PARAM(DEBUG_GRAPH_RUNNER)) {
    for (auto i = internal_.begin(); i != internal_.end(); ++i) {
      for (auto& output : i->outputs) {
        auto linker = output.linker.get();
        if (linker) {
          LOG(INFO) << "linker: " << linker->to_string();
        }
      }
    }
  }
}

void GraphTask::link_tensor_buffers(GraphInternal& up,
                                    std::vector<GraphInternal>::iterator down) {
  for (auto& output : up.outputs) {
    auto master = &output.output_tensor_buffer;
    auto slaves = get_slaves(down, (*master)->get_tensor()->get_name());
    if (slaves.empty()) {
      continue;
    }
    output.linker = TensorBufferLinker::create(master);
    for (auto& t : slaves) {
      output.linker->add_slave(t.first, t.second);
    }
  }
}

std::vector<
    std::pair<std::unique_ptr<vart::TensorBuffer>*, const xir::Subgraph*>>
GraphTask::get_slaves(std::vector<GraphInternal>::iterator down,
                      const std::string& master) {
  std::vector<
      std::pair<std::unique_ptr<vart::TensorBuffer>*, const xir::Subgraph*>>
      ret;
  for (; down != internal_.end(); ++down) {
    for (auto& input : down->input_tensor_buffers) {
      auto is_same = input->get_tensor()->get_name() == master;
      if (is_same) {
        ret.emplace_back(std::make_pair(&input, down->subgraph));
      }
    }
  }
  return ret;
}

/*
bool GraphTask::is_same_tensor_buffer(const vart::TensorBuffer* up,
                                      const vart::TensorBuffer* down) {
  auto up_tensor_name = up->get_tensor()->get_name();
  auto down_tensor_name = down->get_tensor()->get_name();
  auto is_same_name = up_tensor_name == down_tensor_name;
  if (is_same_name) {
    return true;
  }
  auto down_op = find_op(down_tensor_name);
  if (down_op->get_type() == "upload") {
    auto upload_tensors = down_op->get_input_tensors();
    CHECK_EQ(upload_tensors.size(), 1u);
    return upload_tensors[0]->get_name() == up_tensor_name;
  }
  if (down_op->get_type() == "download") {
    auto download_tensors = down_op->get_input_tensors();
    CHECK_EQ(download_tensors.size(), 1u);
    auto download_tensor_name = download_tensors[0]->get_name();
    return download_tensor_name == up_tensor_name;
  }
  return false;
}


const xir::Op* GraphTask::find_op(const std::string& tensor_name) const {
  const xir::Op* ret = nullptr;
  for (auto op : subgraph_->get_ops()) {
    if (op->get_output_tensor()->get_name() == tensor_name) {
      ret = op;
      break;
    }
  }
  CHECK(ret != nullptr) << "cannot find op: tensor_name=" << tensor_name;
  return ret;
}
*/

const xir::Tensor* GraphTask::find_tensor(
    const std::string& tensor_name) const {
  const xir::Tensor* ret = nullptr;
  for (auto& i : internal_) {
    for (auto& output : i.outputs) {
      auto tensor = output.my_tensor.get();
      if (tensor->get_name() == tensor_name) {
        ret = tensor;
        break;
      }
    }
  }
  // CHECK(ret != nullptr) << "cannot find tensor: name=" << tensor_name;
  UNI_LOG_CHECK(ret != nullptr, VAILIB_GRAPH_RUNNER_NOT_FIND)
      << "cannot find tensor: name=" << tensor_name;
  return ret;
}

vart::TensorBuffer* GraphTask::find_tensor_buffer(
    const std::string& tensor_name) const {
  vart::TensorBuffer* ret = nullptr;
  for (auto& i : internal_) {
    for (auto& output : i.outputs) {
      auto tensor_buffer = output.output_tensor_buffer.get();
      if (tensor_buffer->get_tensor()->get_name() == tensor_name) {
        ret = tensor_buffer;
        break;
      }
    }
  }
  // CHECK(ret != nullptr) << "cannot find tensor buffer: name=" << tensor_name;
  UNI_LOG_CHECK(ret != nullptr, VAILIB_GRAPH_RUNNER_NOT_FIND)
      << "cannot find tensor buffer: name=" << tensor_name;
  return ret;
}

void GraphTask::finalize_linkers() {
  for (auto& i : internal_) {
    for (auto& output : i.outputs) {
      auto linker = output.linker.get();
      if (linker) {
        linker->finalize();
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "finalize linker: " << linker->to_string();
      }
    }
  }
}

std::vector<const xir::Tensor*> GraphTask::build_input_tensors() {
  auto ret = std::vector<const xir::Tensor*>();
  for (auto& i : internal_) {
    if (i.device == "USER") {
      for (auto& output : i.outputs) {
        ret.push_back(output.my_tensor.get());
      }
    }
  }
  return ret;
}

std::vector<const xir::Tensor*> GraphTask::build_output_tensors() {
  auto ret = std::vector<const xir::Tensor*>();
  for (auto& tensor : subgraph_->get_sorted_output_tensors()) {
    if (is_wired_orphan_subgraph(subgraph_->get_graph()->get_leaf_subgraph(
            tensor->get_producer()))) {
      continue;
    }
    ret.push_back(find_tensor(tensor->get_name()));
  }
  return ret;
}

std::vector<vart::TensorBuffer*> GraphTask::build_input_tensor_buffers() {
  return vitis::ai::vec_map(input_tensors_, [this](const xir::Tensor* tensor) {
    return find_tensor_buffer(tensor->get_name());
  });
}

std::vector<vart::TensorBuffer*> GraphTask::build_output_tensor_buffers() {
  return vitis::ai::vec_map(output_tensors_, [this](const xir::Tensor* tensor) {
    return find_tensor_buffer(tensor->get_name());
  });
}

void GraphTask::build_subgraph_tensors() {
  auto convert2 = [this](const std::unique_ptr<xir::Tensor>& tensor) {
    return subgraph_->get_graph()->get_tensor(tensor->get_name());
  };
  auto convert1 =
      [this, &convert2](std::vector<std::unique_ptr<xir::Tensor>>& tensors) {
        return vitis::ai::vec_map(tensors, convert2);
      };
  for (auto& i : internal_) {
    i.subgraph_input_tensors = convert1(i.input_tensors);
    for (auto& output : i.outputs) {
      output.subgraph_output_tensor = convert2(output.my_tensor);
    }
  }
}

GraphInternal::GraphInternal(const xir::Subgraph* sg)
    : subgraph{sg},
      device{},
      runner{},
      input_tensors{},
      input_tensor_buffers{},
      subgraph_input_tensors{},
      outputs{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "GraphInternal create "
      << "@" << (void*)this << " sugraph=" << this->subgraph->get_name();
}

void GraphInternal::reset_buffers() {
  outputs.clear();
  subgraph_input_tensors.clear();
  input_tensor_buffers.clear();
  input_tensors.clear();
}

GraphInternal::~GraphInternal() {
  runner.reset();
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "GraphInternal destroyed "
      << "@" << (void*)this << " sugraph=" << this->subgraph->get_name();
}

GraphTask::~GraphTask() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "@" << (void*)this << " graph task is destroyed";
  for (auto& i : internal_) {
    i.reset_buffers();
  }
}

static size_t get_batch_size(const std::vector<vart::TensorBuffer*>& input,
                             const std::vector<vart::TensorBuffer*>& output) {
  CHECK(!input.empty());
  auto ret = input[0]->get_tensor()->get_shape()[0];
  for (auto i = 1u; i < input.size(); ++i) {
    CHECK_EQ(ret, input[i]->get_tensor()->get_shape()[0]);
  }
  for (auto i = 0u; i < output.size(); ++i) {
    CHECK_EQ(ret, output[i]->get_tensor()->get_shape()[0]);
  }
  return (size_t)ret;
}

std::pair<uint32_t, int> GraphTask::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  auto user_batch = get_batch_size(input, output);
  auto effective_batch = std::min(user_batch, dpu_batch_size_);
  auto maybe_copy = [](const std::vector<vart::TensorBuffer*>& tbs1,
                       const std::vector<vart::TensorBuffer*>& tbs2) {
    auto size = tbs1.size();
    CHECK_EQ(size, tbs2.size());
    for (auto i = 0u; i < size; ++i) {
      if (tbs1[i] != tbs2[i]) {
        // CHECK(false) << "TODO: "
        // << " check implementation of copy_tensor_buffer, optimize "
        //    "the copy to itself";
        vart::TensorBuffer::copy_tensor_buffer(tbs1[i], tbs2[i]);
      }
    }
  };
  maybe_copy(input, get_inputs());
  size_t subgraph_index = 0u;
  for (auto& i : internal_) {
    auto output_tensor_buffers = output_internal_get_tensor_buffers(i.outputs);
    LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
        << "start the runner on " << i.device
        << " for subgraph: " << i.subgraph->get_name()  //
        << "\n\tinputs:"
        << to_string(vitis::ai::vector_unique_ptr_get(i.input_tensor_buffers))
        << "\n\toutputs:" << to_string(output_tensor_buffers);

    if (i.runner != nullptr) {  // "USER" device has no runner
      size_t runner_batch = 0u;
      if (i.device == "DPU") {
        runner_batch = effective_batch;
      } else {
        runner_batch = nondpu_batch_size_;
      }
      // effective_batch is either the user provided batch or dpu_batch_size_
      //
      // runner_batch is either the effective_batch or nondpu_batch_size_
      //
      // so that, for DPU runner, we run it only once. for other
      // runners, like CPU runners, we run each batch one by one.
      for (auto batch_index = 0u; batch_index < effective_batch;
           batch_index = batch_index + runner_batch) {
        attrs_->set_attr<int>("__batch_base__", (int)batch_index);
        auto single_batch_inputs = map_to_single_batch(
            vitis::ai::vector_unique_ptr_get(i.input_tensor_buffers),
            batch_index, runner_batch);
        auto single_batch_outputs = map_to_single_batch(
            output_tensor_buffers, batch_index, runner_batch);
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "start the runner on " << i.device
            << " for subgraph: " << i.subgraph->get_name()    //
            << " effective_batch: " << effective_batch        //
            << " runner_batch: " << runner_batch              //
            << " user_batch: " << user_batch                  //
            << " dpu_batch_size_: " << dpu_batch_size_        //
            << " nondpu_batch_size_: " << nondpu_batch_size_  //
            << "\n\tinputs:"
            << to_string(vitis::ai::vector_unique_ptr_get(single_batch_inputs))
            << "\n\toutputs:"
            << to_string(
                   vitis::ai::vector_unique_ptr_get(single_batch_outputs));
        auto status = i.runner->execute_async(
            vitis::ai::vector_unique_ptr_get(single_batch_inputs),
            vitis::ai::vector_unique_ptr_get(single_batch_outputs));
        auto ok = i.runner->wait((int)status.first, -1);
        CHECK(ok == 0);
        maybe_dump_tensor_buffers(
            vitis::ai::vector_unique_ptr_get(single_batch_inputs),
            vitis::ai::vector_unique_ptr_get(single_batch_outputs),
            subgraph_index, batch_index);
      }
    }
    after_invoke_runner(i);
    subgraph_index = subgraph_index + 1;
  }
  maybe_copy(get_outputs(), output);
  return std::make_pair(0u, 0);
}

void GraphTask::after_invoke_runner(GraphInternal& i) {
  for (auto& output : i.outputs) {
    auto linker = output.linker.get();
    if (linker) {
      linker->after_invoke_runner(i.subgraph);
    }
  }
}

int GraphTask::wait(int jobid, int timeout) { return 0; }

void GraphTask::maybe_dump_tensor_buffers(
    const std::vector<vart::TensorBuffer*>& inputs,
    const std::vector<vart::TensorBuffer*>& outputs, size_t subgraph_index,
    size_t batch_index) {
  if (!ENV_PARAM(XLNX_ENABLE_DUMP)) {
    return;
  }
  auto sname = vitis::ai::to_valid_file_name(subgraph_->get_name());
  auto dir = "dump/" + sname;
  for (auto& i : inputs) {
    auto s2_name = vitis::ai::to_valid_file_name(
        internal_[subgraph_index].subgraph->get_name());
    auto dirname = dir + "/" + s2_name + "/" + "i";
    vart::dump_tensor_buffer(dirname, i, batch_index);
  }
  for (auto& i : outputs) {
    auto s2_name = vitis::ai::to_valid_file_name(
        internal_[subgraph_index].subgraph->get_name());
    auto dirname = dir + "/" + s2_name + "/" + "o";
    vart::dump_tensor_buffer(dirname, i, batch_index);
  }
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphTask::map_to_single_batch(
    const std::vector<vart::TensorBuffer*>& tensor_buffers, size_t batch_index,
    size_t batch) {
  return vitis::ai::vec_map(
      tensor_buffers,
      [batch_index, batch](const vart::TensorBuffer* tensor_buffer) {
        return std::unique_ptr<vart::TensorBuffer>(
            std::make_unique<vart::BatchTensorBufferView>(
                const_cast<vart::TensorBuffer*>(tensor_buffer), batch_index,
                batch));
      });
}

// static std::vector<const xir::Tensor*> copy(
//     std::vector<std::unique_ptr<xir::Tensor>>& from) {
//   auto ret = std::vector<const xir::Tensor*>();
//   ret.reserve(from.size());
//   for (auto& b : from) {
//     ret.push_back(const_cast<const xir::Tensor*>(b.get()));
//   }
//   return ret;
// }

std::vector<const xir::Tensor*> GraphTask::get_input_tensors() {
  return input_tensors_;
}

std::vector<const xir::Tensor*> GraphTask::get_output_tensors() {
  return output_tensors_;
}

std::vector<vart::TensorBuffer*> GraphTask::get_inputs() {
  return input_tensor_buffers_;
}

std::vector<vart::TensorBuffer*> GraphTask::get_outputs() {
  return output_tensor_buffers_;
}

}  // namespace

extern "C" vart::Runner* create_runner_with_attrs(const xir::Subgraph* subgraph,
                                                  xir::Attrs* attrs) {
  return new GraphTask(subgraph, attrs);
}

extern "C" vart::Runner* create_runner(const xir::Subgraph* subgraph,
                                       const std::string& mode) {
  return new GraphTask(subgraph, nullptr);
}
