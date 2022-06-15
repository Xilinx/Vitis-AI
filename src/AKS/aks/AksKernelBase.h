/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __AKS_KERNEL_BASE_H
#define __AKS_KERNEL_BASE_H

#include <iostream>
#include <vector>

#include <xir/tensor/tensor.hpp>
#include <vart/tensor_buffer.hpp>

namespace AKS
{
  struct NodeParams;
  struct DynamicParamValues;

  /// Base interface for an AKS kernel
  /// Every kernel implementation should inherit from KernelBase
  class KernelBase
  {
    public:
      virtual ~KernelBase() {}

      /// Returns total number of CPU threads to be allocated for this kernel.
      /// If there are multiple nodes of same kernel, allocated threads are shared.
      /// Alternately, it can be specified in kernel JSON also.
      /// Default is -1. It means number provided in JSON is used.
      /// If it is not mentioned in JSON also, a single thread is used.
      virtual int getNumCUs() {return -1;}

      /// Returns whether kernel execution is an asynchronous operation.
      virtual bool isExecAsync() { return false; }

      /// Actual operation to be executed
      /// It could be either blocking or non-blocking(async) call.
      /// If async, it is mandatory to implement wait() method also.
      /// @param in Input data to the kernel
      /// @param out Output data of the kernel
      /// @param params Node parameters. Unique to a particular node.
      /// @param dynParams Input parameters. It changes with each input.
      /// @return execID An ID unique to this operation. Used for waiting for its result in async call.
      virtual int exec_async (
        std::vector<vart::TensorBuffer *> &in,
        std::vector<vart::TensorBuffer *> &out,
        NodeParams* params,
        DynamicParamValues* dynParams) = 0;

      /// Wait operation for asynchronous kernel execution
      /// Required only if isExecAsync() is true.
      /// @param execID An ID returned by exec_async call in async call.
      /// @params Node parameters. Unique to a particular node.
      virtual void wait (int, NodeParams*) {}

      /// Initialize a Node
      /// AKS performs this operation for each node in a graph as soon as graph is loaded.
      /// Any setup operations w.r.t a node could be implemented here
      /// @param params Node parameters. Unique to a particular node.
      virtual void nodeInit(NodeParams*) {}

      /// Report any info by each node
      /// If any kernel wants to print any info after all jobs, it could be added here.
      /// Eg: Accuracy kernel wants to report final accuracy over a full dataset.
      /// It is invoked for every node by SysManagerExt::report() method.
      virtual void report(AKS::NodeParams* nodeParams) {}
  };
}
#endif
