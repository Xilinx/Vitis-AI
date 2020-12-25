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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_

// Support for eager execution of TensorFlow kernels.

#include <memory>
#include <unordered_map>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {

// Forward declaration for proto class NodeExecStats so we do not need to
// include the proto header
class NodeExecStats;
class StepStats;
class ProcessFunctionLibraryRuntime;
class FunctionLibraryRuntime;

// KernelAndDevice encapsulates the logic needed to run a computation eagerly.
// The computation can be a single instantiated kernel (implemented by
// KernelAndDeviceOp below) or a multi-device function (implemented by
// KernelAndDeviceFunc below).
//
// Also see:
// https://www.tensorflow.org/code/tensorflow/core/common_runtime/kernel_benchmark_testlib.h
// and
// https://www.tensorflow.org/code/tensorflow/core/kernels/ops_testutil.h
class KernelAndDevice : public core::RefCounted {
 public:
  // Populates this with a kernel appropriate for 'ndef'.
  //
  // The provided FunctionLibraryRuntime MUST outlive all calls to
  // Run() on the returned KernelAndDevice.
  virtual Status Init(const NodeDef& ndef, GraphCollector* graph_collector) = 0;

  // Non-multi-device functions are run using regular CallOp and look like
  // primitive operations from KernelAndDevice perspective.
  // `flr` can be nullptr if the operation is not run on any specific device
  // (currently can happen only for multi-device functions).
  KernelAndDevice(
      FunctionLibraryRuntime* flr,
      std::function<void(std::function<void()>)>* runner,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      Device* host_cpu_device)
      : device_(flr == nullptr ? nullptr : flr->device()),
        host_cpu_device_(host_cpu_device),
        flr_(flr),
        collective_executor_(std::move(collective_executor)),
        runner_(runner) {}

  // Not thread safe.
  virtual ~KernelAndDevice() {}

  // TODO(ashankar): Handle list-valued inputs.
  virtual Status Run(const gtl::InlinedVector<TensorValue, 4>& inputs,
                     std::vector<Tensor>* outputs, NodeExecStats* stats,
                     StepStats* step_stats, GraphCollector* graph_collector,
                     CancellationManager* cancellation_manager) = 0;

  virtual Status Run(ScopedStepContainer* step_container,
                     const gtl::InlinedVector<TensorValue, 4>& inputs,
                     std::vector<Tensor>* outputs, NodeExecStats* stats,
                     StepStats* step_stats, GraphCollector* graph_collector,
                     CancellationManager* cancellation_manager) = 0;

  virtual Device* InputDevice(int i) const = 0;
  virtual Device* OutputDevice(int idx) const = 0;
  // If idx'th output is a resource, returns the device backing the resource.
  // Else, returns nullptr.
  virtual Device* OutputResourceDevice(int idx) const = 0;

  // Returns the kernel that will be used to run this.
  // Returns nullptr if this will be run using function library runtime.
  virtual const OpKernel* kernel() const = 0;

  // Returns the device on which this kernel will run. In the case of
  // multi-device functions, this is the default device that is passed to the
  // placer but actual computation can happen on a different set of devices.
  // Also, outputs can be produced on devices different from what this method
  // returns.
  Device* device() const { return device_; }

  virtual const DataTypeVector& output_dtypes() const = 0;

  virtual DataType input_type(int i) const = 0;
  virtual int num_inputs() const = 0;
  virtual int num_outputs() const = 0;
  virtual const string& name() const = 0;

 protected:
  std::function<void(std::function<void()>)>* get_runner() const;

  Device* const device_;               // can be null
  Device* const host_cpu_device_;      // non-null
  FunctionLibraryRuntime* const flr_;  // can be null
  const std::unique_ptr<CollectiveExecutor::Handle> collective_executor_;

 private:
  std::function<void(std::function<void()>)>* const runner_;  // can be null
};

// Represents an op kernel and the device it will be run on.
class KernelAndDeviceOp final : public KernelAndDevice {
 public:
  KernelAndDeviceOp(
      tensorflow::Rendezvous* rendez, bool log_memory,
      FunctionLibraryRuntime* flr,
      std::function<void(std::function<void()>)>* runner,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      Device* host_cpu_device)
      : KernelAndDevice(flr, runner, std::move(collective_executor),
                        host_cpu_device),
        rendez_(rendez),
        log_memory_(log_memory) {}

  ~KernelAndDeviceOp() override {}

  Status Init(const NodeDef& ndef, GraphCollector* graph_collector) override;

  Status Run(const gtl::InlinedVector<TensorValue, 4>& inputs,
             std::vector<Tensor>* outputs, NodeExecStats* stats,
             StepStats* step_stats, GraphCollector* graph_collector,
             CancellationManager* cancellation_manager) override;

  Status Run(ScopedStepContainer* step_container,
             const gtl::InlinedVector<TensorValue, 4>& inputs,
             std::vector<Tensor>* outputs, NodeExecStats* stats,
             StepStats* step_stats, GraphCollector* graph_collector,
             CancellationManager* cancellation_manager) override;

  const OpKernel* kernel() const override { return kernel_.get(); }

  Device* InputDevice(int i) const override;
  Device* OutputDevice(int idx) const override;
  Device* OutputResourceDevice(int idx) const override;

  DataType input_type(int i) const override;
  const DataTypeVector& output_dtypes() const override {
    return kernel_->output_types();
  }
  int num_inputs() const override { return kernel_->num_inputs(); }
  int num_outputs() const override { return kernel_->num_outputs(); }
  const string& name() const override { return kernel_->name(); }

 private:
  std::unique_ptr<OpKernel> kernel_;
  Rendezvous* const rendez_;
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_;
  const bool log_memory_;
};

// Represents a multi-device function. Functions can also be run using
// various function-calling kernels including CallOp and PartitionedCallOp.
// In such cases, KernelAndDeviceOp is used.
class KernelAndDeviceFunc final : public KernelAndDevice {
 public:
  // `flr` can be nullptr.
  // `pflr` must not be nullptr.
  // `host_cpu_device` must not be nullptr.
  KernelAndDeviceFunc(
      FunctionLibraryRuntime* flr, ProcessFunctionLibraryRuntime* pflr,
      std::vector<Device*> input_devices,
      std::unordered_map<int, TensorShape> input_tensor_shapes,
      std::unordered_map<int, DtypeAndPartialTensorShape>
          input_resource_dtypes_and_shapes,
      std::function<void(std::function<void()>)>* runner,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      Device* host_cpu_device, const string& name,
      std::function<Rendezvous*(const int64)> rendezvous_creator)
      : KernelAndDevice(flr, runner, std::move(collective_executor),
                        host_cpu_device),
        pflr_(pflr),
        handle_(kInvalidHandle),
        input_devices_(std::move(input_devices)),
        input_tensor_shapes_(std::move(input_tensor_shapes)),
        input_resource_dtypes_and_shapes_(
            std::move(input_resource_dtypes_and_shapes)),
        name_(name),
        rendezvous_creator_(std::move(rendezvous_creator)) {}

  virtual ~KernelAndDeviceFunc();

  Status Init(const NodeDef& ndef, GraphCollector* graph_collector) override;

  Status Run(const gtl::InlinedVector<TensorValue, 4>& inputs,
             std::vector<Tensor>* outputs, NodeExecStats* stats,
             StepStats* step_stats, GraphCollector* graph_collector,
             CancellationManager* cancellation_manager) override;
  Status Run(ScopedStepContainer* step_container,
             const gtl::InlinedVector<TensorValue, 4>& inputs,
             std::vector<Tensor>* outputs, NodeExecStats* stats,
             StepStats* step_stats, GraphCollector* graph_collector,
             CancellationManager* cancellation_manager) override;

  const OpKernel* kernel() const override { return nullptr; }

  Device* InputDevice(int i) const override;
  Device* OutputDevice(int idx) const override;
  Device* OutputResourceDevice(int idx) const override;

  DataType input_type(int i) const override;
  const DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }
  int num_inputs() const override { return input_dtypes_.size(); }
  int num_outputs() const override { return output_dtypes_.size(); }
  const string& name() const override { return name_; };

 private:
  ProcessFunctionLibraryRuntime* const pflr_;  // non-null
  FunctionLibraryRuntime::Handle handle_;
  // CPU devices are null. Resource handles' devices are actual backing
  // devices.
  std::vector<Device*> output_devices_;
  // CPU devices are not null. Resource handles' devices are actual backing
  // devices.
  std::vector<Device*> input_devices_;
  std::unordered_map<int, TensorShape> input_tensor_shapes_;
  std::unordered_map<int, DtypeAndPartialTensorShape>
      input_resource_dtypes_and_shapes_;

  DataTypeVector input_dtypes_;
  DataTypeVector output_dtypes_;
  string name_;

  std::function<Rendezvous*(const int64)> rendezvous_creator_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_
