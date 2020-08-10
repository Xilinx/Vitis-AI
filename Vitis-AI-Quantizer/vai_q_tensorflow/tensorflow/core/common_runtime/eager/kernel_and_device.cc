/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"

#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {

std::function<void(std::function<void()>)>* KernelAndDevice::get_runner()
    const {
  if (runner_) {
    return runner_;
  } else {
    static auto* default_runner =
        new std::function<void(std::function<void()>)>(
            [](std::function<void()> f) { f(); });
    return default_runner;
  }
}

KernelAndDeviceFunc::~KernelAndDeviceFunc() {
  if (handle_ != kInvalidHandle) {
    Status status = pflr_->ReleaseHandle(handle_);
    if (!status.ok()) {
      LOG(INFO) << "Ignoring error status when releasing multi-device function "
                   "handle "
                << status.ToString();
    }
  }
}

Status KernelAndDeviceOp::Init(const NodeDef& ndef,
                               GraphCollector* graph_collector) {
  OpKernel* k = nullptr;
  if (flr_ == nullptr) {
    return errors::Internal(
        "A valid FunctionLibraryRuntime must be provided when running ops "
        "based on OpKernel.");
  }
  TF_RETURN_IF_ERROR(flr_->CreateKernel(ndef, &k));
  kernel_.reset(k);
  return Status::OK();
}

Status KernelAndDeviceFunc::Init(const NodeDef& ndef,
                                 GraphCollector* graph_collector) {
  const OpDef* op_def = nullptr;
  const FunctionDef* function_def;
  if (flr_ == nullptr) {
    // If function is being executed without an explicit device request,
    // lookup the FunctionDef in the CPU's FLR. All FLRs share the same
    // library.
    function_def = pflr_->GetFLR(host_cpu_device_->name())
                       ->GetFunctionLibraryDefinition()
                       ->Find(ndef.op());
  } else {
    function_def = flr_->GetFunctionLibraryDefinition()->Find(ndef.op());
  }

  if (function_def != nullptr) {
    op_def = &(function_def->signature());
  } else {
    TF_RETURN_IF_ERROR(OpDefForOp(ndef.op().c_str(), &op_def));
  }
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(ndef, *op_def, &input_dtypes_, &output_dtypes_));

  FunctionLibraryRuntime::InstantiateOptions options;
  options.target = device_ == nullptr ? "" : device_->name();
  options.is_multi_device_function = true;
  for (const Device* device : input_devices_) {
    options.input_devices.push_back(device->name());
  }
  options.input_tensor_shapes = input_tensor_shapes_;
  options.input_resource_dtypes_and_shapes = input_resource_dtypes_and_shapes_;

  const auto& it = ndef.attr().find("executor_type");
  if (it != ndef.attr().end()) {
    options.executor_type = it->second.s();
  }
#if !defined(IS_MOBILE_PLATFORM)
  // Android tf library does not include grappler.
  const auto& config_it = ndef.attr().find("config_proto");
  if (it != ndef.attr().end()) {
    if (!options.config_proto.ParseFromString(config_it->second.s())) {
      return errors::InvalidArgument(
          "Failed to parse config_proto attribute as tensorflow::ConfigProto "
          "proto.");
    }
    grappler::GrapplerItem::OptimizationOptions optimization_options;

    // Tensorflow 2.0 in eager mode with automatic control dependencies will
    // prune all nodes that are not in the transitive fanin of the fetch nodes.
    // However because the function will be executed via FunctionLibraryRuntime,
    // and current function implementation does not prune stateful and dataset
    // ops, we rely on Grappler to do the correct graph pruning.
    optimization_options.allow_pruning_stateful_and_dataset_ops = true;

    optimization_options.is_eager_mode = true;

    // All the nested function calls will be executed and optimized via
    // PartitionedCallOp, there is no need to optimize functions now.
    optimization_options.optimize_function_library = false;

    options.optimize_graph_fn = std::bind(
        grappler::OptimizeGraph, std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
        options.config_proto, function_def->signature().name(),
        optimization_options, std::placeholders::_6);
  }
#endif  // !IS_MOBILE_PLATFORM
  options.graph_collector = graph_collector;

  // In Eager mode we always inline all functions into the top-level
  // function body graph, to get a single executable graph, that could be
  // optimized across function boundaries (e.g. prune unused inputs and outputs
  // in a function call chain). This is required to mimic graph mode execution,
  // with aggressive pruning of nodes not in the transitive fanin of fetches.
  options.config_proto.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);

  TF_RETURN_IF_ERROR(
      pflr_->Instantiate(ndef.op(), AttrSlice(ndef), options, &handle_));
  return pflr_->GetOutputDevices(handle_, &output_devices_);
}

Status KernelAndDeviceOp::Run(const gtl::InlinedVector<TensorValue, 4>& inputs,
                              std::vector<Tensor>* outputs,
                              NodeExecStats* stats, StepStats* step_stats,
                              GraphCollector* graph_collector,
                              CancellationManager* cancellation_manager) {
  ScopedStepContainer step_container(0, [this](const string& name) {
    device_->resource_manager()->Cleanup(name).IgnoreError();
  });
  return this->Run(&step_container, inputs, outputs, stats, step_stats,
                   graph_collector, cancellation_manager);
}

Status KernelAndDeviceFunc::Run(
    const gtl::InlinedVector<TensorValue, 4>& inputs,
    std::vector<Tensor>* outputs, NodeExecStats* stats, StepStats* step_stats,
    GraphCollector* graph_collector,
    CancellationManager* cancellation_manager) {
  const std::vector<Device*> devices = pflr_->device_mgr()->ListDevices();
  ScopedStepContainer step_container(0, [&devices](const string& name) {
    for (Device* device : devices) {
      device->resource_manager()->Cleanup(name).IgnoreError();
    }
  });
  return this->Run(&step_container, inputs, outputs, stats, step_stats,
                   graph_collector, cancellation_manager);
}

namespace {
void UpdateStats(OpKernelContext* context,
                 StepStatsCollector* step_stats_collector,
                 NodeExecStats* stats) {
  for (const auto& allocator_pair : context->ConsumeWrappedAllocators()) {
    AllocatorMemoryUsed* memory = stats->add_memory();
    memory->set_allocator_name(allocator_pair.first->Name());
    auto sizes = allocator_pair.second->GetSizes();
    memory->set_total_bytes(std::get<0>(sizes));
    memory->set_peak_bytes(std::get<1>(sizes));
    memory->set_live_bytes(std::get<2>(sizes));

    absl::optional<AllocatorStats> allocator_stats =
        allocator_pair.first->GetStats();
    if (allocator_stats) {
      memory->set_allocator_bytes_in_use(allocator_stats->bytes_in_use);
    }
    allocator_pair.second->GetRecordsAndUnRef();
  }
  auto* ms = stats->mutable_memory_stats();
  ms->set_temp_memory_size(context->temp_memory_allocated());
  for (const auto& alloc_id : context->persistent_alloc_ids()) {
    ms->mutable_persistent_tensor_alloc_ids()->Add(alloc_id);
  }

  ms->set_persistent_memory_size(context->persistent_memory_allocated());
  step_stats_collector->Finalize();
}

// In certain contexts (e.g. TPU async executions), the CancellationManager is
// used to shut down the device in error scenarios (as opposed to using the
// AsyncCompute's DoneCallback). This is handled through the
// {inc,dec}_num_deferred_ops_function.
struct OpExecutionState : public core::RefCounted {
  // TODO(nareshmodi): consider refcounting the cancellation_manager.
  CancellationManager cancellation_manager;
};
}  // anonymous namespace

Status KernelAndDeviceOp::Run(ScopedStepContainer* step_container,
                              const gtl::InlinedVector<TensorValue, 4>& inputs,
                              std::vector<Tensor>* outputs,
                              NodeExecStats* stats, StepStats* step_stats,
                              GraphCollector* graph_collector,
                              CancellationManager* cancellation_manager) {
  gtl::InlinedVector<AllocatorAttributes, 4> in_attrs(kernel_->num_inputs());
  for (size_t i = 0; i < in_attrs.size(); ++i) {
    in_attrs[i].set_on_host(kernel_->input_memory_types()[i] ==
                            tensorflow::HOST_MEMORY);
  }
  std::vector<AllocatorAttributes> out_attrs(kernel_->num_outputs());
  for (size_t i = 0; i < out_attrs.size(); ++i) {
    out_attrs[i].set_on_host(kernel_->output_memory_types()[i] ==
                             tensorflow::HOST_MEMORY);
  }

  gtl::InlinedVector<DeviceContext*, 4> input_device_contexts;
  for (int i = 0; i < inputs.size(); i++) {
    DeviceContext* device_context = nullptr;
    if (device_->tensorflow_gpu_device_info() != nullptr) {
      device_context = device_->tensorflow_gpu_device_info()->default_context;
    }
    input_device_contexts.push_back(device_context);
  }

  OpKernelContext::Params params;
  params.is_eager = true;
  params.device = device_;
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = kernel_.get();
  params.resource_manager = device_->resource_manager();
  params.input_alloc_attrs = &in_attrs;
  params.output_attr_array = gtl::vector_as_array(&out_attrs);
  params.function_library = flr_;
  params.slice_reader_cache = &slice_reader_cache_;
  params.rendezvous = rendez_;
  OpExecutionState* op_execution_state = nullptr;
  if (cancellation_manager) {
    params.cancellation_manager = cancellation_manager;
  } else {
    op_execution_state = new OpExecutionState;
    params.cancellation_manager = &op_execution_state->cancellation_manager;
  }
  params.log_memory = log_memory_;
  params.inc_num_deferred_ops_function = [op_execution_state]() {
    if (op_execution_state != nullptr) {
      op_execution_state->Ref();
    }
  };
  params.dec_num_deferred_ops_function = [op_execution_state]() {
    if (op_execution_state != nullptr) {
      op_execution_state->Unref();
    }
  };
  std::unique_ptr<StepStatsCollector> step_stats_collector;
  if (stats != nullptr) {
    step_stats_collector.reset(new StepStatsCollector(step_stats));
    params.track_allocations = true;
    params.stats_collector = step_stats_collector.get();
    params.graph_collector = graph_collector;
  }
  params.runner = get_runner();

  params.step_container = step_container;
  params.collective_executor =
      collective_executor_ ? collective_executor_->get() : nullptr;
  params.input_device_contexts = &input_device_contexts;

  OpKernelContext context(&params);

  if (kernel_->def().op() == "_Recv") {
    // TODO(apassos) do not special-case _Recv. Currently the GPU device fails
    // if trying to run _Recv->Compute(), specifically checking for _Recv. To go
    // around this we call _Recv->ComputeAsync, to mimic graph mode behavior.
    AsyncOpKernel* async = kernel_->AsAsync();
    Notification done;
    device_->ComputeAsync(async, &context, [&done]() { done.Notify(); });
    done.WaitForNotification();
  } else {
    const string& op_name = kernel_->name();
    // 'ScopedActivity' will trace the OpKernel scheduling time on host.
    profiler::TraceMe activity(
        [&] {
          return absl::StrCat(op_name, ":", kernel_->type_string(), "#id=",
                              step_container ? step_container->step_id() : 0,
                              ",device=", device_->name(), ",async=false#");
        },
        profiler::TraceMeLevel::kInfo);
    // 'ScopedAnnotation' will trace the OpKernel execution time on device.
    tracing::ScopedAnnotation annotation(
        [&]() { return absl::StrCat(op_name, ":", kernel_->type_string()); });
    device_->Compute(kernel_.get(), &context);
  }

  // Clean up execution op_execution_state if deferred ops aren't running.
  if (op_execution_state != nullptr) {
    op_execution_state->Unref();
  }

  if (!context.status().ok()) return context.status();

  if (outputs != nullptr) {
    outputs->clear();
    for (int i = 0; i < context.num_outputs(); ++i) {
      outputs->push_back(Tensor(*context.mutable_output(i)));
    }
  }
  if (stats != nullptr) {
    UpdateStats(&context, step_stats_collector.get(), stats);
  }
  return Status::OK();
}

Status KernelAndDeviceFunc::Run(
    ScopedStepContainer* step_container,
    const gtl::InlinedVector<TensorValue, 4>& inputs,
    std::vector<Tensor>* outputs, NodeExecStats* stats, StepStats* step_stats,
    GraphCollector* graph_collector,
    CancellationManager* cancellation_manager) {
  FunctionLibraryRuntime::Options opts;

  // We don't pass rendezvous from eager context because we can get tensor
  // name collisions in send/recv ops when running multiple instances
  // of the same multi-device function concurrently.
  Rendezvous* rendezvous = rendezvous_creator_(opts.step_id);
  opts.rendezvous = rendezvous;
  opts.create_rendezvous = false;

  CancellationManager cm;
  if (cancellation_manager) {
    opts.cancellation_manager = cancellation_manager;
  } else {
    opts.cancellation_manager = &cm;
  }
  opts.allow_dead_tensors = true;
  opts.step_container = step_container;
  opts.collective_executor =
      collective_executor_ ? collective_executor_->get() : nullptr;

  std::unique_ptr<StepStatsCollector> step_stats_collector;
  if (stats != nullptr) {
    step_stats_collector.reset(new StepStatsCollector(step_stats));
  }
  opts.stats_collector = step_stats_collector.get();
  opts.runner = get_runner();

  Notification done;
  Status status;
  outputs->clear();
  std::vector<Tensor> input_vector;
  input_vector.reserve(inputs.size());
  for (const TensorValue& tensor_value : inputs) {
    input_vector.push_back(*tensor_value.tensor);
  }
  {
    profiler::TraceMe activity(
        [&] { return absl::StrCat("FunctionRun:", name()); },
        profiler::TraceMeLevel::kInfo);
    pflr_->Run(opts, handle_, input_vector, outputs,
               [&status, &done](const Status& s) {
                 status = s;
                 done.Notify();
               });
    done.WaitForNotification();
  }

  rendezvous->Unref();
  if (step_stats_collector != nullptr) {
    step_stats_collector->Finalize();
  }
  return status;
}

tensorflow::Device* KernelAndDeviceOp::OutputDevice(int idx) const {
  if (kernel_->output_memory_types()[idx] == HOST_MEMORY) {
    return nullptr;
  }
  return device_;
}

tensorflow::Device* KernelAndDeviceFunc::OutputDevice(int idx) const {
  if (output_dtypes_[idx] == DT_RESOURCE) {
    return nullptr;
  }
  return output_devices_[idx];
}

tensorflow::Device* KernelAndDeviceOp::OutputResourceDevice(int idx) const {
  if (kernel_->output_type(idx) == DT_RESOURCE) {
    return device_;
  }
  return nullptr;
}

tensorflow::Device* KernelAndDeviceFunc::OutputResourceDevice(int idx) const {
  if (output_dtypes_[idx] == DT_RESOURCE) {
    return output_devices_[idx];
  }
  return nullptr;
}

DataType KernelAndDeviceOp::input_type(int i) const {
  return kernel_->input_type(i);
}

DataType KernelAndDeviceFunc::input_type(int i) const {
  return input_dtypes_[i];
}

Device* KernelAndDeviceOp::InputDevice(int i) const {
  if (kernel_->input_memory_types()[i] == HOST_MEMORY) {
    return host_cpu_device_;
  }
  return device_;
}

Device* KernelAndDeviceFunc::InputDevice(int i) const {
  if (input_dtypes_[i] == DT_RESOURCE) {
    return host_cpu_device_;
  } else {
    return input_devices_[i];
  }
}

}  // namespace tensorflow
