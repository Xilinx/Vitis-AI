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
// Kernel Functions Implementation

#include <vector>
#include <fstream>
#include <map>
#include <atomic>
#include <tuple>
#include <cassert>
#include <filesystem>

#include <aks/AksTensorBuffer.h>
#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>
#include <aks/AksLogger.h>

#include "AksDPUHelper.h"

// DPUNodeObject specifies everything associated with a
// single DPU node in the graph
// So each node in the graph can have its own #cores etc. 
struct DPUNodeObject {
  DPUNodeObject() = default;
  std::vector<std::unique_ptr<vart::Runner>> runners;
  GraphInfo shapes;
  std::unique_ptr<xir::Graph, std::default_delete<xir::Graph>> graph;
  std::vector<xir::Subgraph const*> subgraphs;
  std::atomic<unsigned int> core_id {0};
  unsigned int core_count;
};

class DPURunner: public AKS::KernelBase {
  public:
    // For each node, there are multiple runners as defined in the graph
    bool isExecAsync() { return false; }
    void nodeInit(AKS::NodeParams*);
    int exec_async (
        std::vector<vart::TensorBuffer *> &in,
        std::vector<vart::TensorBuffer *> &out,
        AKS::NodeParams* params,
        AKS::DynamicParamValues* dynParams);

    std::map<AKS::NodeParams*, DPUNodeObject> nodes;
};

extern "C" {

  AKS::KernelBase* getKernel (AKS::NodeParams* params) {
    /// Create kernel object
    DPURunner * kbase = new DPURunner();
    return kbase;
  }

} // extern C

void DPURunner::nodeInit(AKS::NodeParams* params) {

  nodes.emplace(std::piecewise_construct,
      std::forward_as_tuple(params),
      std::forward_as_tuple());

  // Get xmodel path from graph
  auto modelFile = params->getValue<string>("model_file");
  // Get root dir of xmodel from env
  const char* axr = std::getenv("AKS_XMODEL_ROOT") ?
                    std::getenv("AKS_XMODEL_ROOT") :
                    "graph_zoo/";

  // Update model path
  std::string xmodelFile = std::string(axr) + "/" + modelFile;

  namespace fs = std::filesystem;
  fs::directory_entry xmodelFs (xmodelFile);
  if (!xmodelFs.exists()) {
    // Fall back to path provided in graph json
    xmodelFile = modelFile;
  }

  auto num_runners = params->hasKey<int>("num_runners") ? params->getValue<int>("num_runners") : 1;
  // Load graph
  nodes[params].graph = std::move(xir::Graph::deserialize(xmodelFile));
  // Get DPU subgraph and create runners
  nodes[params].subgraphs = std::move(get_dpu_subgraph(nodes[params].graph.get()));
  for(int i = 0; i < num_runners; ++i) {
    std::unique_ptr<vart::Runner> runner_ = vart::Runner::create_runner(nodes[params].subgraphs.back(), "run");
    nodes[params].runners.push_back(std::move(runner_));
  }
  nodes[params].core_count = num_runners;
}

int DPURunner::exec_async (
    vector<vart::TensorBuffer*>& in, vector<vart::TensorBuffer*>& out,
    AKS::NodeParams* params, AKS::DynamicParamValues* dynParams)
{
  auto& curNode = nodes[params];
  const auto& curRunners = curNode.runners;
  unsigned int tmpID = curNode.core_id++;
  unsigned int runnerID = tmpID % curNode.core_count;

  auto runner = curRunners[runnerID].get();

  auto inputTensors  = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

  /// Fill input
  int index = 0;
  for (auto & tensor: inputTensors) {
    auto in_shape = in[index]->get_tensor()->get_shape();
    assert ((in_shape == tensor->get_shape()) && "[ERROR] Input shape does not match DPU input shape!");
    if (tensor->get_name() != in[index]->get_tensor()->get_name()) {
      xir::Tensor* inputTensor = const_cast<xir::Tensor*>(in[index]->get_tensor());
      inputTensor->rename(tensor->get_name());
    }
    inputsPtr.push_back(in[index++]);
  }

  /// Create output
  for (auto & tensor: outputTensors) {
    // Create o/p buffers
    AKS::AksTensorBuffer * outDD = 
      new AKS::AksTensorBuffer(xir::Tensor::create(
                                 tensor->get_name(), tensor->get_shape(),
                                 xir::create_data_type<float>()
                              ));
    out.push_back(outDD);
    outputsPtr.push_back(out.back());
  }

  /// Run DPU
  auto job_id = runner->execute_async(inputsPtr, outputsPtr);
  runner->wait(job_id.first, -1);

  return 0;
}
