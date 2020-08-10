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

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

class TestEnv {
 public:
  TestEnv() : flib_def_(OpRegistry::Global(), {}) {
    std::vector<std::unique_ptr<Device>> devices;
    devices.push_back(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));
    cpu_device_ = devices.back().get();
    device_mgr_ = absl::make_unique<DeviceMgr>(std::move(devices));
    OptimizerOptions opts;
    pflr_ = tensorflow::MakeUnique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, &flib_def_,
        opts, /*default_thread_pool=*/nullptr, /*cluster_flr=*/nullptr);

    flr_ = pflr_->GetFLR("/job:a/replica:0/task:0/device:CPU:0");
    CHECK(flr_ != nullptr);
  }

  FunctionLibraryRuntime* function_library_runtime() const { return flr_; }
  ProcessFunctionLibraryRuntime* pflr() const { return pflr_.get(); }
  Device* cpu_device() { return cpu_device_; }

 private:
  FunctionLibraryDefinition flib_def_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  FunctionLibraryRuntime* flr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  Device* cpu_device_;
};

void BM_CreateGraph(int iters) {
  for (int i = 0; i < iters; ++i) {
    Scope root = Scope::NewRootScope();
    auto C = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}});
    auto M = ops::MatMul(root, C, C);
    TF_CHECK_OK(root.status());
  }
}
BENCHMARK(BM_CreateGraph);

void BM_RunGraph(int iters) {
  tensorflow::testing::StopTiming();
  Scope root = Scope::NewRootScope();
  auto C = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}});
  auto M = ops::MatMul(root, C, C);
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(1);
  opts.config.set_intra_op_parallelism_threads(1);
  ClientSession sess(root, opts);
  std::vector<Tensor> outputs;
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    outputs.clear();
    TF_CHECK_OK(sess.Run({M}, &outputs));
  }
}
BENCHMARK(BM_RunGraph);

void BM_CreateAndDestroySession(int iters) {
  tensorflow::testing::StopTiming();
  Scope root = Scope::NewRootScope();
  auto C = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}});
  auto M = ops::MatMul(root, C, C);
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    ClientSession sess(root);
  }
}
BENCHMARK(BM_CreateAndDestroySession);

void BM_KernelAndDeviceInit(int iters) {
  tensorflow::testing::StopTiming();
  NodeDef ndef(AttrBuilder("MatMul")
                   .Set("T", DT_FLOAT)
                   .Set("transpose_a", false)
                   .Set("transpose_b", false)
                   .NumInputs(2)
                   .BuildNodeDef());
  TestEnv env;
  KernelAndDeviceOp k(nullptr, false, env.function_library_runtime(), nullptr,
                      nullptr, env.cpu_device());
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(k.Init(ndef, nullptr));
  }
}
BENCHMARK(BM_KernelAndDeviceInit);

void BM_KernelAndDeviceRun(int iters) {
  tensorflow::testing::StopTiming();
  Tensor t(Input({{1.0f, 2.0f}, {3.0f, 4.0f}}).tensor());
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(TensorValue(&t));
  inputs.push_back(TensorValue(&t));
  std::vector<Tensor> outputs;
  NodeDef ndef(AttrBuilder("MatMul")
                   .Set("T", DT_FLOAT)
                   .Set("transpose_a", false)
                   .Set("transpose_b", false)
                   .NumInputs(inputs.size())
                   .BuildNodeDef());
  TestEnv env;
  KernelAndDeviceOp k(nullptr, false, env.function_library_runtime(), nullptr,
                      nullptr, env.cpu_device());
  TF_CHECK_OK(k.Init(ndef, nullptr));
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(k.Run(inputs, &outputs, nullptr, nullptr, nullptr, nullptr));
  }
}
BENCHMARK(BM_KernelAndDeviceRun);
}  // namespace
}  // namespace tensorflow
