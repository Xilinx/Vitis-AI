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

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>
#include <aks/AksLogger.h>
#include "common.h"

// DPUNodeObject specifies everything associated with a
// single DPUv2 node in the graph
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

class DPUCZDX8GRunner: public AKS::KernelBase {
  public:
    // For each node, there are multiple runners as defined in the graph
    map<AKS::NodeParams*, DPUNodeObject> nodes;
    bool isExecAsync() { return false; }
    void nodeInit(AKS::NodeParams*);
    int exec_async (
        std::vector<AKS::DataDescriptor *> &in,
        std::vector<AKS::DataDescriptor *> &out,
        AKS::NodeParams* params,
        AKS::DynamicParamValues* dynParams);
  private:
    std::map<std::string, xir::Tensor*> _ioTensors;
};

extern "C" {

  AKS::KernelBase* getKernel (AKS::NodeParams* params) {
    /// Create kernel object
    DPUCZDX8GRunner * kbase = new DPUCZDX8GRunner();
    return kbase;
  }

} // extern C

void DPUCZDX8GRunner::nodeInit(AKS::NodeParams* params) {
  nodes.emplace(std::piecewise_construct,
      std::forward_as_tuple(params),
      std::forward_as_tuple());

  auto modelFile = params->getValue<string>("model_file");
  auto num_runners = params->hasKey<int>("num_runners") ? params->getValue<int>("num_runners") : 1;

  nodes[params].graph = std::move(xir::Graph::deserialize(modelFile));
  nodes[params].subgraphs = std::move(get_dpu_subgraph(nodes[params].graph.get()));
  for(int i=0; i<num_runners; ++i) {
    std::unique_ptr<vart::Runner> runner_ = vart::Runner::create_runner(nodes[params].subgraphs.back(), "run");
    nodes[params].runners.push_back(std::move(runner_));
  }
  nodes[params].core_count = num_runners;
}

int DPUCZDX8GRunner::exec_async (
    vector<AKS::DataDescriptor *>& in, vector<AKS::DataDescriptor *>& out,
    AKS::NodeParams* params, AKS::DynamicParamValues* dynParams)
{
  auto& curNode = nodes[params];
  const auto& curRunners = curNode.runners;
  unsigned int tmpID = curNode.core_id++;
  unsigned int runnerID = tmpID % curNode.core_count;

  auto runner = curRunners[runnerID].get();
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::unique_ptr<xir::Tensor>> xirTensors;

  int in_idx = 0;
  for(const auto& iTensor: inputTensors) {
    const auto& in_dims = iTensor->get_shape();
    xirTensors.push_back(xir::Tensor::create(iTensor->get_name(), in_dims, xir::DataType::FLOAT, sizeof(float) * 8u));
    inputsPtr.push_back(new CpuFlatTensorBuffer(in[in_idx]->data(), xirTensors.back().get()));
    in_idx++;
  }

  for(const auto& oTensor: outputTensors) {
    const auto& out_dims = oTensor->get_shape();
    out.push_back(new AKS::DataDescriptor(out_dims, AKS::DataType::FLOAT32));
    xirTensors.push_back(xir::Tensor::create(oTensor->get_name(), out_dims, xir::DataType::FLOAT, sizeof(float) * 8u));
    outputsPtr.push_back(new CpuFlatTensorBuffer(out.back()->data(), xirTensors.back().get()));
  }

  auto job_id = runner->execute_async(inputsPtr, outputsPtr);
  runner->wait(job_id.first, -1);

  for(int i=0; i<inputsPtr.size(); ++i) {
    delete inputsPtr[i];
  }

  for(int i=0; i<outputsPtr.size(); ++i) {
    delete outputsPtr[i];
  }
  return -1;
}
