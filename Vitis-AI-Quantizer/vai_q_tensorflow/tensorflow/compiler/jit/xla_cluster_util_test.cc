/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_cluster_util.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(CreateCycleDetectionGraph, ConnectivityThroughEnterExitRegion) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output enter =
      ops::internal::Enter(root.WithOpName("enter"), a, "only_frame");
  Output exit = ops::internal::Exit(root.WithOpName("exit"), enter);
  Output b = ops::Add(root.WithOpName("b"), a, exit);

  FixupSourceAndSinkEdges(root.graph());

  GraphCycles cycles;
  TF_ASSERT_OK(CreateCycleDetectionGraph(root.graph(), &cycles).status());
  EXPECT_FALSE(cycles.ContractEdge(a.node()->id(), b.node()->id()));
}

TEST(CreateCycleDetectionGraph, ConnectivityThroughMultipleEnterExitRegions) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output enter_0 =
      ops::internal::Enter(root.WithOpName("enter_0"), a, "frame_0");
  Output exit_0 = ops::internal::Exit(root.WithOpName("exit_0"), enter_0);
  Output enter_1 =
      ops::internal::Enter(root.WithOpName("enter_1"), a, "frame_1");
  Output exit_1 = ops::internal::Exit(root.WithOpName("exit_1"), enter_1);
  Output b = ops::Add(root.WithOpName("b"), a, exit_1);

  FixupSourceAndSinkEdges(root.graph());

  GraphCycles cycles;
  TF_ASSERT_OK(CreateCycleDetectionGraph(root.graph(), &cycles).status());
  EXPECT_FALSE(cycles.ContractEdge(a.node()->id(), b.node()->id()));
}

TEST(CreateCycleDetectionGraph, ReachingEnterExit) {
  // TODO(b/127521408): We can lift this limitation with some work.
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output enter_0 =
      ops::internal::Enter(root.WithOpName("enter_0"), a, "frame_0");
  Output exit_0 = ops::internal::Exit(root.WithOpName("exit_0"), enter_0);

  Output add = ops::Add(root.WithOpName("add"), exit_0, exit_0);

  Output enter_1 =
      ops::internal::Enter(root.WithOpName("enter_1"), add, "frame_0");
  Output exit_1 = ops::internal::Exit(root.WithOpName("exit_1"), enter_1);

  FixupSourceAndSinkEdges(root.graph());

  GraphCycles cycles;
  TF_ASSERT_OK_AND_ASSIGN(bool ok,
                          CreateCycleDetectionGraph(root.graph(), &cycles));
  EXPECT_FALSE(ok);
}

const char* kCPU0 = "/job:localhost/replica:0/task:0/device:CPU:0";
const char* kGPU0 = "/job:localhost/replica:0/task:0/device:GPU:0";
const char* kGPU1 = "/job:localhost/replica:0/task:0/device:GPU:1";

TEST(IsSingleGpuGraph, ReturnsTrue) {
  Scope root = Scope::NewRootScope().WithAssignedDevice(kGPU0).ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output b = ops::Add(root.WithOpName("b"), a, a);
  Output c = ops::Add(root.WithOpName("c"), b, b);

  FixupSourceAndSinkEdges(root.graph());

  EXPECT_TRUE(IsSingleGpuGraph(*root.graph()));
}

TEST(IsSingleGpuGraph, ReturnsFalseForCpuGraph) {
  Scope root = Scope::NewRootScope().WithAssignedDevice(kCPU0).ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output b = ops::Add(root.WithOpName("b"), a, a);
  Output c = ops::Add(root.WithOpName("c"), b, b);

  FixupSourceAndSinkEdges(root.graph());

  EXPECT_FALSE(IsSingleGpuGraph(*root.graph()));
}

TEST(IsSingleGpuGraph, ReturnsFalseForMultiGpuGraph) {
  Scope root = Scope::NewRootScope().WithAssignedDevice(kGPU0).ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output b = ops::Add(root.WithOpName("b").WithAssignedDevice(kGPU1), a, a);
  Output c = ops::Add(root.WithOpName("c"), b, b);

  FixupSourceAndSinkEdges(root.graph());

  EXPECT_FALSE(IsSingleGpuGraph(*root.graph()));
}
}  // namespace
}  // namespace tensorflow
