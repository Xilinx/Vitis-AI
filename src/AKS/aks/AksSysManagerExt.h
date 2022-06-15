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
#ifndef __AKS_SYS_MANAGER_EXT_H_
#define __AKS_SYS_MANAGER_EXT_H__

#include <iostream>
#include <string>
#include <future>
#include <vector>

#include <xir/tensor/tensor.hpp>
#include <vart/tensor_buffer.hpp>

namespace AKS
{
  class NodeParams;
  class AIGraph;
  template<typename T> using VecPtr = std::vector<std::unique_ptr<T>>;
  /// Top-level System Manager responsible for managing graphs and jobs.
  class SysManagerExt
  {
    public:
      static SysManagerExt * getGlobal();
      static void deleteGlobal();

      /// Load all the kernels from the given directory
      /// @param dir Directory containing all the kernel JSON files
      int loadKernels(std::string dir);

      ///. Load a graph
      /// @param graphJson graph (in JSON fromat) to be executed
      void loadGraphs(std::string graphJson);

      /// Get a handle to a loaded graph.
      /// @param graphName name of the graph as provided in graph JSON
      AIGraph* getGraph(std::string graphName);

      /// Enqueue a job to a particular graph
      /// This operation is non-blocking. As soon as an image is enqueued, it returns
      /// the control to the caller.
      /// To wait for the result, either use the returned future or waitForAllResults()
      /// @param graph Handle to the graph. Use getGraph() to get the handle.
      /// @param filePath Path to an image
      /// @param inputs Vector of input buffers to be passed (optional)
      /// @param userArgs Any other arguments (optional)
      /// @return future Output buffers of last node is returned by this future object.

      std::future<std::vector<std::unique_ptr<vart::TensorBuffer>>> enqueueJob(
          AIGraph* graph, const std::string& filePath,
          std::vector<std::unique_ptr<vart::TensorBuffer>> inputs = {},
          AKS::NodeParams* userArgs = nullptr);

      std::future<std::vector<std::unique_ptr<vart::TensorBuffer>>> enqueueJob(
          AIGraph* graph, const std::vector<std::string>& filePaths,
          std::vector<std::unique_ptr<vart::TensorBuffer>> inputs = {},
          AKS::NodeParams* userArgs = nullptr);

      /// Wait for all the jobs enqueued to SysManager to be finished
      /// This is a blocking call
      void waitForAllResults();

      /// Wait for all the jobs specific to a particular graph to be finished.
      /// This is a blocking call
      /// @param graph Handle to a loaded graph. Use getGraph() to get the handle.
      void waitForAllResults(AIGraph* graph);

      /// This function invokes report method of every node in the graph
      /// @param graph Handle to a loaded graph. Use getGraph() to get the handle.
      void report(AIGraph* graph);

      /// Reset the internal timers in the SysManagerExt.
      /// This is useful to remove any initialization overhead
      /// while calculating the performance statistics
      /// Since SysManagerExt is singleton, Use this once just before your enqueueJob
      void resetTimer();

      /// Print the average performance statistics of a graph execution
      /// Available only in TRACE mode
      void printPerfStats();

      /// wrapper for exposing python bindings
      void pyEnqueueJob(AIGraph* graph, const std::string& filePath);
      void pyEnqueueJob(AIGraph* graph, const std::vector<std::string>& filePaths);

    private:
      static SysManagerExt *_global;
      SysManagerExt();
      ~SysManagerExt();
  };
}

#endif // __AKS_SYS_MANAGER_EXT_H__
