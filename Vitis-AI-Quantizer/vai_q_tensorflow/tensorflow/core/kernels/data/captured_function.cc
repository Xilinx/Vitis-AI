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
#include "tensorflow/core/kernels/data/captured_function.h"

#include <utility>

#include "absl/time/clock.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/notification.h"

namespace tensorflow {
namespace data {
namespace {

// Simplistic implementation of the `StepStatsCollectorInterface` that only
// cares about collecting the CPU time needed to execute a captured function.
class SimpleStepStatsCollector : public StepStatsCollectorInterface {
 public:
  void IncrementProcessingTime(int64 delta) {
    mutex_lock l(mu_);
    processing_time_ += delta;
  }

  NodeExecStatsInterface* CreateNodeExecStats(const Node* node) override {
    return new SimpleNodeExecStats(this);
  }

  string ReportAllocsOnResourceExhausted(const string& err) override {
    return "";
  }

  int64 processing_time() {
    tf_shared_lock l(mu_);
    return processing_time_;
  }

 private:
  class SimpleNodeExecStats : public NodeExecStatsInterface {
   public:
    explicit SimpleNodeExecStats(SimpleStepStatsCollector* step_stats_collector)
        : step_stats_collector_(step_stats_collector) {}

    void Done(const string& device) override {
      step_stats_collector_->IncrementProcessingTime(end_time_ns_ -
                                                     start_time_ns_);
      delete this;
    }

    void RecordExecutorStarted() override {
      start_time_ns_ = absl::GetCurrentTimeNanos();
    }

    void RecordComputeStarted() override {}

    void RecordComputeEnded() override {}

    void RecordExecutorEnded() override {
      end_time_ns_ = absl::GetCurrentTimeNanos();
    }

    bool TrackAllocations() const override { return false; }

    void SetMemory(OpKernelContext* ctx) override {}

    void SetOutput(int slot, const Tensor* tensor) override {}

    void SetReferencedTensors(const TensorReferenceVector& tensors) override {}

    void SetScheduled(int64 nanos) override {}

   private:
    int64 start_time_ns_ = 0;
    int64 end_time_ns_ = 0;
    SimpleStepStatsCollector* step_stats_collector_;  // Not owned.
  };

  mutex mu_;
  int64 processing_time_ GUARDED_BY(mu_) = 0;
};

Status RunShortCircuit(const ShortCircuitInfo& info,
                       const std::vector<Tensor>& args,
                       const CapturedFunction* const func,
                       std::vector<Tensor>* rets) {
  VLOG(3) << "Running function " << func->func().name() << " short circuit";
  size_t num_args = args.size();
  for (size_t i = 0; i < info.indices.size(); ++i) {
    if (info.indices[i] < num_args) {
      rets->push_back(args[info.indices[i]]);
    } else {
      rets->push_back(func->captured_inputs()[info.indices[i] - num_args]);
    }
  }
  return Status::OK();
}

Status RunShortCircuit(const ShortCircuitInfo& info, std::vector<Tensor>&& args,
                       const CapturedFunction* const func,
                       std::vector<Tensor>* rets) {
  VLOG(3) << "Running function " << func->func().name() << " short circuit";
  size_t num_args = args.size();
  for (size_t i = 0; i < info.indices.size(); ++i) {
    if (info.indices[i] < num_args) {
      if (info.can_move[i]) {
        rets->push_back(std::move(args[info.indices[i]]));
      } else {
        rets->push_back(args[info.indices[i]]);
      }
    } else {
      rets->push_back(func->captured_inputs()[info.indices[i] - num_args]);
    }
  }
  return Status::OK();
}

Status CreateShortCircuitInfo(OpKernelConstruction* ctx,
                              const NameAttrList& func,
                              ShortCircuitInfo* info) {
  auto& indices = info->indices;

  FunctionLibraryRuntime::Handle fn_handle;
  TF_RETURN_IF_ERROR(ctx->function_library()->Instantiate(
      func.name(), AttrSlice(&func.attr()), &fn_handle));
  auto cleanup = gtl::MakeCleanup([ctx, fn_handle]() {
    Status s = ctx->function_library()->ReleaseHandle(fn_handle);
    if (!s.ok()) {
      LOG(WARNING) << "Failed to release handle: " << s.error_message();
    }
  });

  // If the function contains any stateful operations, we conservatively execute
  // the entire function.
  if (ctx->function_library()->IsStateful(func.name())) {
    return Status::OK();
  }

  const FunctionBody* fn_body =
      ctx->function_library()->GetFunctionBody(fn_handle);
  indices.resize(fn_body->ret_nodes.size());

  for (size_t i = 0; i < fn_body->ret_nodes.size(); ++i) {
    Node* ret_node = fn_body->ret_nodes[i];
    Node* ret_input_node;
    TF_RETURN_IF_ERROR(ret_node->input_node(0, &ret_input_node));

    while (ret_input_node->def().op() == "Identity") {
      TF_RETURN_IF_ERROR(ret_input_node->input_node(0, &ret_input_node));
    }

    if (ret_input_node->def().op() == FunctionLibraryDefinition::kArgOp) {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(ret_input_node->def(), "index", &(indices[i])));
    } else {
      indices.clear();
      break;
    }
  }

  // Compute the `can_move` vector.
  if (!indices.empty()) {
    auto& can_move = info->can_move;
    std::map<int, int> last_use;
    for (size_t i = 0; i < indices.size(); ++i) {
      last_use[indices[i]] = i;
    }
    can_move.resize(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      can_move[i] = last_use[indices[i]] == i;
    }
  }

  return Status::OK();
}

Status CreateFunctionLibraryDefinition(
    const FunctionLibraryDefinition* lib_def, const string& func_name,
    std::unique_ptr<FunctionLibraryDefinition>* result) {
  DCHECK(lib_def != nullptr);
  const FunctionDef* fdef = lib_def->Find(func_name);
  if (TF_PREDICT_FALSE(fdef == nullptr)) {
    return errors::FailedPrecondition(strings::StrCat(
        "Could not find required function definition ", func_name));
  }
  *result = absl::make_unique<FunctionLibraryDefinition>(
      lib_def->ReachableDefinitions(*fdef));
  return (*result)->CopyFunctionDefFrom(func_name, *lib_def);
}

Status IsNodeStateful(const FunctionLibraryDefinition& library,
                      const NodeDef& node);

Status IsFunctionStateful(const FunctionLibraryDefinition& library,
                          const FunctionDef& function_def) {
  if (!function_def.signature().is_stateful()) {
    return Status::OK();
  }

  for (const NodeDef& node_def : function_def.node_def()) {
    TF_RETURN_IF_ERROR(IsNodeStateful(library, node_def));
  }
  return Status::OK();
}

// Returns whether an op has been whitelisted as stateless. Uses a heuristic to
// whitelist source dataset ops which have been marked stateful due to
// b/65524810. Also looks up the `op_def->name` in the global
// `WhitelistedStatefulOpRegistry`.
bool IsOpWhitelisted(const OpDef* op_def) {
  return (op_def->output_arg_size() == 1 &&
          op_def->output_arg(0).type() == DT_VARIANT &&
          (absl::EndsWith(op_def->name(), "Dataset") ||
           absl::EndsWith(op_def->name(), "DatasetV2"))) ||
         WhitelistedStatefulOpRegistry::Global()->Contains(op_def->name());
}

Status IsNodeStateful(const FunctionLibraryDefinition& library,
                      const NodeDef& node) {
  const OpDef* op_def;

  // TODO(jsimsa): Fix C++ unit tests so that we do not have to ignore
  // `LookUpOpDef` errors here.
  if (!OpRegistry::Global()->LookUpOpDef(node.op(), &op_def).ok() ||
      IsOpWhitelisted(op_def) || !op_def->is_stateful() ||
      op_def->name() == "Assert") {
    return Status::OK();
  }

  if (op_def->name() == "If") {
    const FunctionDef* then_func =
        library.Find(node.attr().at("then_branch").func().name());
    const FunctionDef* else_func =
        library.Find(node.attr().at("else_branch").func().name());
    if (then_func != nullptr) {
      TF_RETURN_IF_ERROR(IsFunctionStateful(library, *then_func));
    }
    if (else_func != nullptr) {
      TF_RETURN_IF_ERROR(IsFunctionStateful(library, *else_func));
    }
    return Status::OK();
  }

  if (op_def->name() == "While") {
    const FunctionDef* cond_func =
        library.Find(node.attr().at("cond").func().name());
    const FunctionDef* body_func =
        library.Find(node.attr().at("body").func().name());
    if (cond_func != nullptr) {
      TF_RETURN_IF_ERROR(IsFunctionStateful(library, *cond_func));
    }
    if (body_func != nullptr) {
      TF_RETURN_IF_ERROR(IsFunctionStateful(library, *body_func));
    }
    return Status::OK();
  }

  return errors::FailedPrecondition(op_def->name(), " is stateful.");
}

}  // namespace

Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const std::vector<Tensor>& input_element,
    int64 thread_index, const InstantiatedCapturedFunction& inst_captured_func,
    StringPiece prefix, std::unique_ptr<IteratorBase>* out_iterator) {
  std::vector<Tensor> return_values;

  TF_RETURN_IF_ERROR(inst_captured_func.RunWithBorrowedArgs(ctx, input_element,
                                                            &return_values));

  if (!(return_values.size() == 1 && return_values[0].dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(return_values[0].shape()))) {
    return errors::InvalidArgument(
        "Function must return a single scalar of dtype DT_VARIANT.");
  }

  // Retrieve the dataset that was created in `f`.
  DatasetBase* returned_dataset;
  TF_RETURN_IF_ERROR(
      GetDatasetFromVariantTensor(return_values[0], &returned_dataset));

  // Create an iterator for the dataset that was returned by `f`.
  return returned_dataset->MakeIterator(
      ctx, strings::StrCat(prefix, "[", thread_index, "]"), out_iterator);
}

/* static */
Status FunctionMetadata::Create(
    OpKernelConstruction* ctx, const string& func_name, Params params,
    std::shared_ptr<FunctionMetadata>* out_metadata) {
  NameAttrList func;
  TF_RETURN_IF_ERROR(ctx->GetAttr(func_name, &func));
  return Create(ctx, std::move(func), params, out_metadata);
}

Status FunctionMetadata::Create(
    OpKernelConstruction* ctx, NameAttrList&& func, Params params,
    std::shared_ptr<FunctionMetadata>* out_metadata) {
  out_metadata->reset(new FunctionMetadata(std::move(func), params));
  TF_RETURN_IF_ERROR(CreateFunctionLibraryDefinition(
      ctx->function_library()->GetFunctionLibraryDefinition(),
      (*out_metadata)->func_.name(), &(*out_metadata)->lib_def_));
  TF_RETURN_IF_ERROR(CreateShortCircuitInfo(
      ctx, (*out_metadata)->func_, &(*out_metadata)->short_circuit_info_));
  (*out_metadata)->ValidateMultiDevice();
  return Status::OK();
}

void FunctionMetadata::ValidateMultiDevice() {
  const FunctionDef* fdef = lib_def_->Find(func_.name());
  if (is_multi_device_function_) {
    auto attr = fdef->attr().find(FunctionLibraryDefinition::kIntsOnDeviceAttr);
    if (attr != fdef->attr().end() && attr->second.b()) {
      LOG(WARNING)
          << "Disabling multi-device execution for a function that uses the "
          << FunctionLibraryDefinition::kIntsOnDeviceAttr << " attribute.";
      is_multi_device_function_ = false;
      return;
    }
    auto validate_arg = [this](const OpDef::ArgDef& arg) {
      if (!arg.number_attr().empty() || !arg.type_list_attr().empty()) {
        LOG(WARNING) << "Disabling multi-device execution for a function with "
                        "a vector argument "
                     << arg.name() << ".";
        is_multi_device_function_ = false;
      }
    };
    for (const auto& arg : fdef->signature().input_arg()) {
      validate_arg(arg);
    }
    for (const auto& arg : fdef->signature().output_arg()) {
      validate_arg(arg);
    }
  }
}

/* static */
Status CapturedFunction::Create(
    OpKernelContext* ctx,
    const std::shared_ptr<const FunctionMetadata> metadata,
    const string& argument_name,
    std::unique_ptr<CapturedFunction>* out_function) {
  OpInputList inputs;
  TF_RETURN_IF_ERROR(ctx->input_list(argument_name, &inputs));
  std::vector<Tensor> captured_inputs(inputs.begin(), inputs.end());
  return Create(ctx, metadata, std::move(captured_inputs), out_function);
}

/* static */
Status CapturedFunction::Create(
    OpKernelContext* ctx,
    const std::shared_ptr<const FunctionMetadata> metadata,
    std::vector<Tensor>&& captured_inputs,
    std::unique_ptr<CapturedFunction>* out_function) {
  *out_function = absl::WrapUnique(
      new CapturedFunction(metadata, std::move(captured_inputs)));
  return Status::OK();
}

Status CapturedFunction::AddToGraph(
    SerializationContext* ctx, DatasetBase::DatasetGraphDefBuilder* b,
    std::vector<Node*>* other_arguments,
    DataTypeVector* other_arguments_types) const {
  other_arguments->reserve(captured_inputs_.size());
  other_arguments_types->reserve(captured_inputs_.size());
  for (const Tensor& t : captured_inputs_) {
    Node* node;
    DatasetBase* input;
    Status s = GetDatasetFromVariantTensor(t, &input);
    if (s.ok()) {
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &node));
    } else {
      TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
    }
    other_arguments->emplace_back(node);
    other_arguments_types->emplace_back(t.dtype());
  }
  TF_RETURN_IF_ERROR(
      b->AddFunction(ctx, metadata_->func().name(), *metadata_->lib_def()));
  return Status::OK();
}

Status CapturedFunction::Instantiate(
    IteratorContext* ctx, std::unique_ptr<InstantiatedCapturedFunction>*
                              instantiated_captured_function) {
  // The context's runtime will be used for all subsequent calls.
  FunctionLibraryRuntime* lib = ctx->flr();
  FunctionLibraryRuntime::InstantiateOptions inst_opts;
  inst_opts.lib_def = metadata_->lib_def();
  inst_opts.create_kernels_eagerly = true;
  if (!metadata_->use_inter_op_parallelism()) {
    inst_opts.executor_type = "SINGLE_THREADED_EXECUTOR";
  }
  inst_opts.is_multi_device_function = metadata_->is_multi_device_function();

  // We infer the target device from the function library runtime.
  DCHECK(lib->device() != nullptr);
  inst_opts.target = lib->device()->name();

  if (metadata_->is_multi_device_function()) {
    // Compute devices of non-captured inputs.
    //
    // We infer the number of non-captured inputs by subtracting the number
    // of captured inputs from the number of input arguments and we infer the
    // input devices from the function library runtime.
    const FunctionDef* fdef =
        metadata_->lib_def()->Find(metadata_->func().name());
    if (fdef == nullptr) {
      return errors::InvalidArgument(
          "Failed to find function ", metadata_->func().name(),
          " in function library: ", lib->GetFunctionLibraryDefinition());
    }
    size_t num_non_captured_inputs =
        fdef->signature().input_arg_size() - captured_inputs_.size();
    for (size_t i = 0; i < num_non_captured_inputs; ++i) {
      inst_opts.input_devices.push_back(inst_opts.target);
    }
    // Compute devices of captured inputs.
    // TODO(jsimsa): Correctly handle tensors on devices other than CPU:0.
    Device* cpu_device;
    TF_RETURN_IF_ERROR(lib->device_mgr()->LookupDevice("CPU:0", &cpu_device));
    std::unordered_map<int, DtypeAndPartialTensorShape>&
        input_resource_variable_dtypes_and_shapes =
            inst_opts.input_resource_dtypes_and_shapes;
    for (size_t i = 0; i < captured_inputs_.size(); ++i) {
      const auto& input = captured_inputs_[i];
      DataType dtype = input.dtype();
      if (dtype == DT_RESOURCE) {
        const ResourceHandle& handle = input.flat<ResourceHandle>()(0);
        inst_opts.input_devices.push_back(handle.device());
        const auto& dtypes_and_shapes = handle.dtypes_and_shapes();
        // Set dtypes and shapes for resource variable inputs.
        if (!dtypes_and_shapes.empty()) {
          input_resource_variable_dtypes_and_shapes[num_non_captured_inputs +
                                                    i] =
              dtypes_and_shapes.at(0);
        }
      } else if (MTypeFromDType(dtype) == HOST_MEMORY) {
        inst_opts.input_devices.push_back(cpu_device->name());
      } else {
        // Fall back to using the function library runtime device.
        inst_opts.input_devices.push_back(inst_opts.target);
      }
    }

    for (size_t i = 0; i < fdef->signature().output_arg_size(); ++i) {
      inst_opts.output_devices.push_back(inst_opts.target);
    }
  }

  FunctionLibraryRuntime::Handle f_handle;
  TF_RETURN_IF_ERROR(ctx->function_handle_cache()->Instantiate(
      metadata_->func().name(), AttrSlice(&metadata_->func().attr()), inst_opts,
      &f_handle));

  DataTypeVector ret_types;
  TF_RETURN_IF_ERROR(lib->GetRetTypes(f_handle, &ret_types));

  *instantiated_captured_function =
      absl::WrapUnique<InstantiatedCapturedFunction>(
          new InstantiatedCapturedFunction(lib, f_handle, std::move(ret_types),
                                           *ctx->runner(),
                                           ctx->cancellation_manager(), this));
  return Status::OK();
}

bool CapturedFunction::IsStateful() const { return !CheckExternalState().ok(); }

Status CapturedFunction::CheckExternalState() const {
  for (const auto& name : lib_def()->ListFunctionNames()) {
    TF_RETURN_IF_ERROR(
        IsFunctionStateful(*lib_def(), *(lib_def()->Find(name))));
  }
  return Status::OK();
}

namespace {
class CallFrameBase : public CallFrameInterface {
 public:
  explicit CallFrameBase(DataTypeSlice ret_types)
      : ret_types_(ret_types), retvals_(ret_types.size()) {}

  // Caller methods.
  Status ConsumeRetvals(std::vector<Tensor>* retvals) {
    retvals->reserve(retvals_.size());
    int i = 0;
    for (auto&& val : retvals_) {
      if (!val) {
        return errors::Internal("No return value for index ", i, ".");
      }
      retvals->emplace_back(std::move(val.value()));
      ++i;
    }
    return Status::OK();
  }

  size_t num_retvals() const override { return retvals_.size(); }

  // Callee methods.
  Status SetRetval(int index, const Tensor& val) override {
    if (index < retvals_.size() && val.dtype() == ret_types_[index] &&
        !retvals_[index]) {
      retvals_[index] = val;
      return Status::OK();
    } else if (index >= retvals_.size()) {
      return errors::InvalidArgument("Return value ", index,
                                     " is out of range.");
    } else if (val.dtype() != ret_types_[index]) {
      return errors::InvalidArgument("Expected type ",
                                     DataTypeString(ret_types_[index]),
                                     " for return value ", index, " but got ",
                                     DataTypeString(val.dtype()), ".");
    } else {
      return errors::Internal("Attempted to set return value ", index,
                              " more than once.");
    }
  }

 private:
  DataTypeSlice ret_types_;
  std::vector<gtl::optional<Tensor>> retvals_;
  TF_DISALLOW_COPY_AND_ASSIGN(CallFrameBase);
};

class OwnedArgsCallFrame : public CallFrameBase {
 public:
  OwnedArgsCallFrame(std::vector<Tensor>&& args,
                     const std::vector<Tensor>* captured_inputs,
                     DataTypeSlice ret_types)
      : CallFrameBase(ret_types),
        args_(std::move(args)),
        captured_inputs_(captured_inputs) {}

  size_t num_args() const override {
    return args_.size() + captured_inputs_->size();
  }

  // Callee methods.
  Status GetArg(int index, Tensor* val) const override {
    if (index < args_.size()) {
      // TODO(mrry): Consider making `CallFrameInterface::GetArg` non-const in
      // order to be able to `std::move(args_[index])` into `*val`.
      *val = args_[index];
      return Status::OK();
    } else if (index < args_.size() + captured_inputs_->size()) {
      *val = (*captured_inputs_)[index - args_.size()];
      return Status::OK();
    } else {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
    }
  }

 private:
  std::vector<Tensor> args_;
  const std::vector<Tensor>* const captured_inputs_;  // Not owned.
};

class BorrowedArgsCallFrame : public CallFrameBase {
 public:
  BorrowedArgsCallFrame(const std::vector<Tensor>& args,
                        const std::vector<Tensor>* captured_inputs,
                        DataTypeSlice ret_types)
      : CallFrameBase(ret_types),
        args_(args),
        captured_inputs_(captured_inputs) {}

  size_t num_args() const override {
    return args_.size() + captured_inputs_->size();
  }

  // Callee methods.
  Status GetArg(int index, Tensor* val) const override {
    if (index < args_.size()) {
      *val = args_[index];
      return Status::OK();
    } else if (index < args_.size() + captured_inputs_->size()) {
      *val = (*captured_inputs_)[index - args_.size()];
      return Status::OK();
    } else {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
    }
  }

 private:
  const std::vector<Tensor>& args_;                   // Not owned.
  const std::vector<Tensor>* const captured_inputs_;  // Not owned.
};

}  // namespace

InstantiatedCapturedFunction::InstantiatedCapturedFunction(
    FunctionLibraryRuntime* lib, FunctionLibraryRuntime::Handle f_handle,
    DataTypeVector ret_types, std::function<void(std::function<void()>)> runner,
    CancellationManager* cancellation_manager, CapturedFunction* captured_func)
    : lib_(lib),
      f_handle_(f_handle),
      ret_types_(std::move(ret_types)),
      captured_runner_(std::move(runner)),
      cancellation_manager_(cancellation_manager),
      captured_func_(captured_func) {}

// NOTE: We don't release f_handle_ here and instead delegate the function
// handle releasing to the FunctionHandleCache. This is because in some cases
// (RepeatDatasetOp in particular), we want to keep the function state (e.g.
// random number generator) even after the Iterator is reset after going through
// one epoch.
InstantiatedCapturedFunction::~InstantiatedCapturedFunction() {}

Status InstantiatedCapturedFunction::Run(IteratorContext* ctx,
                                         std::vector<Tensor>&& args,
                                         std::vector<Tensor>* rets) const {
  auto& info = captured_func_->short_circuit_info();
  if (!info.indices.empty()) {
    return RunShortCircuit(info, std::move(args), captured_func_, rets);
  }

  FunctionLibraryRuntime::Options f_opts;
  ScopedStepContainer step_container(
      f_opts.step_id, [this](const string& name) {
        lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  f_opts.create_rendezvous = ShouldCreateRendezvous();
  CancellationManager cancellation_manager;
  f_opts.cancellation_manager = &cancellation_manager;
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(ConnectCancellationManagers(
      cancellation_manager_, &cancellation_manager, &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));

  OwnedArgsCallFrame frame(std::move(args), &captured_func_->captured_inputs(),
                           ret_types_);
  Notification n;
  Status s;
  lib_->Run(f_opts, f_handle_, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

Status InstantiatedCapturedFunction::RunWithBorrowedArgs(
    IteratorContext* ctx, const std::vector<Tensor>& args,
    std::vector<Tensor>* rets) const {
  auto& info = captured_func_->short_circuit_info();
  if (!info.indices.empty()) {
    return RunShortCircuit(info, args, captured_func_, rets);
  }

  FunctionLibraryRuntime::Options f_opts;
  ScopedStepContainer step_container(
      f_opts.step_id, [this](const string& name) {
        lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  f_opts.create_rendezvous = ShouldCreateRendezvous();
  CancellationManager cancellation_manager;
  f_opts.cancellation_manager = &cancellation_manager;
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(ConnectCancellationManagers(
      cancellation_manager_, &cancellation_manager, &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));

  BorrowedArgsCallFrame frame(args, &captured_func_->captured_inputs(),
                              ret_types_);
  Notification n;
  Status s;

  lib_->Run(f_opts, f_handle_, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

Status InstantiatedCapturedFunction::RunInstantiated(
    const std::vector<Tensor>& args, std::vector<Tensor>* rets) {
  auto& info = captured_func_->short_circuit_info();
  if (!info.indices.empty()) {
    return RunShortCircuit(info, args, captured_func_, rets);
  }

  FunctionLibraryRuntime::Options f_opts;
  ScopedStepContainer step_container(
      f_opts.step_id, [this](const string& name) {
        lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = &step_container;
  f_opts.runner = &captured_runner_;
  f_opts.create_rendezvous = ShouldCreateRendezvous();
  CancellationManager cancellation_manager;
  f_opts.cancellation_manager = &cancellation_manager;
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(ConnectCancellationManagers(
      cancellation_manager_, &cancellation_manager, &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));

  BorrowedArgsCallFrame frame(args, &captured_func_->captured_inputs(),
                              ret_types_);
  Notification n;
  Status s;

  lib_->Run(f_opts, f_handle_, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

void InstantiatedCapturedFunction::RunAsync(
    IteratorContext* ctx, std::vector<Tensor>&& args, std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done, const string& prefix) const {
  auto& info = captured_func_->short_circuit_info();
  if (!info.indices.empty()) {
    // Run the `done` callback on a threadpool thread, because it will
    // potentially do a non-trivial amount of (e.g. copying) work, and we may
    // want to run that concurrently with the next invocation.
    Status s = RunShortCircuit(info, std::move(args), captured_func_, rets);
    (*ctx->runner())(
        std::bind([s](FunctionLibraryRuntime::DoneCallback& done) { done(s); },
                  std::move(done)));
    return;
  }

  // NOTE(mrry): This method does not transfer ownership of `ctx`, and it may
  // be deleted before `done` is called. Take care not to capture `ctx` in any
  // code that may execute asynchronously in this function.
  OwnedArgsCallFrame* frame = new OwnedArgsCallFrame(
      std::move(args), &captured_func_->captured_inputs(), ret_types_);

  FunctionLibraryRuntime::Options f_opts;
  ResourceMgr* resource_mgr = lib_->device()->resource_manager();
  ScopedStepContainer* step_container = new ScopedStepContainer(
      f_opts.step_id, [resource_mgr](const string& name) {
        resource_mgr->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = step_container;
  f_opts.runner = ctx->runner();
  f_opts.create_rendezvous = ShouldCreateRendezvous();
  auto cancellation_manager = absl::make_unique<CancellationManager>();
  f_opts.cancellation_manager = cancellation_manager.get();
  std::function<void()> deregister_fn;
  Status s = ConnectCancellationManagers(
      ctx->cancellation_manager(), cancellation_manager.get(), &deregister_fn);
  if (!s.ok()) {
    done(s);
    return;
  }

  std::shared_ptr<SimpleStepStatsCollector> stats_collector;
  if (ctx->model() || ctx->stats_aggregator()) {
    stats_collector = absl::make_unique<SimpleStepStatsCollector>();
  }
  f_opts.stats_collector = stats_collector.get();

  // Transfer ownership of the cancellation manager to `callback`.
  CancellationManager* raw_cancellation_manager =
      cancellation_manager.release();
  auto callback = std::bind(
      [this, rets, step_container, raw_cancellation_manager, frame](
          const FunctionLibraryRuntime::DoneCallback& done,
          IteratorContext* ctx, const std::function<void()>& deregister_fn,
          const string& prefix,
          const std::shared_ptr<SimpleStepStatsCollector>& stats_collector,
          // Begin unbound arguments.
          Status s) {
        delete step_container;
        deregister_fn();
        delete raw_cancellation_manager;
        if (s.ok()) {
          s = frame->ConsumeRetvals(rets);
        }
        delete frame;
        if (ctx->model()) {
          // TODO(b/129085499) Utilize the `node_name` which would be unique
          // than the prefix for the function execution time statistics.
          // prefix_with_func_name would then be node_name + func_name.
          if (ctx->stats_aggregator()) {
            string prefix_end =
                str_util::Split(prefix, "::", str_util::SkipEmpty()).back();
            string prefix_with_func_name =
                strings::StrCat(prefix_end, stats_utils::kDelimiter,
                                captured_func_->func().name());
            ctx->stats_aggregator()->AddToHistogram(
                stats_utils::ExecutionTimeHistogramName(prefix_with_func_name),
                {static_cast<float>(stats_collector->processing_time())},
                ctx->model()->NumElements(prefix));
          }
          ctx->model()->AddProcessingTime(prefix,
                                          stats_collector->processing_time());
          ctx->model()->RecordStart(prefix, false /* stop_output */);
        }
        done(s);
        if (ctx->model()) {
          ctx->model()->RecordStop(prefix, false /* start_output */);
        }
      },
      std::move(done), ctx, std::move(deregister_fn), prefix,
      std::move(stats_collector), std::placeholders::_1);

  lib_->Run(f_opts, f_handle_, frame, std::move(callback));
}

bool InstantiatedCapturedFunction::ShouldCreateRendezvous() const {
  return lib_->device()->device_type() != DEVICE_CPU ||
         captured_func_->is_multi_device_function();
}

CapturedFunction::CapturedFunction(
    const std::shared_ptr<const FunctionMetadata> metadata,
    std::vector<Tensor> captured_inputs)
    : metadata_(metadata), captured_inputs_(std::move(captured_inputs)) {}

}  // namespace data
}  // namespace tensorflow
