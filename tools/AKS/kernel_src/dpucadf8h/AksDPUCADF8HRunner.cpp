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

class DPUCADF8HRunner: public AKS::KernelBase {
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
    DPUCADF8HRunner * kbase = new DPUCADF8HRunner();
    return kbase;
  }

} // extern C

void DPUCADF8HRunner::nodeInit(AKS::NodeParams* params) {
  nodes.emplace(std::piecewise_construct,
      std::forward_as_tuple(params),
      std::forward_as_tuple());

  auto modelFile = params->getValue<string>("model_file");
  auto num_runners = params->hasKey<int>("num_runners") ? params->getValue<int>("num_runners") : 1;

  nodes[params].graph = std::move(xir::Graph::deserialize(modelFile));
  nodes[params].subgraphs = std::move(get_dpu_subgraph(nodes[params].graph.get()));
  for(int i = 0; i < num_runners; ++i) {
    std::unique_ptr<vart::Runner> runner_ = vart::Runner::create_runner(nodes[params].subgraphs.back(), "run");
    nodes[params].runners.push_back(std::move(runner_));
  }
  nodes[params].core_count = num_runners;
}

int DPUCADF8HRunner::exec_async (
    vector<AKS::DataDescriptor *>& in, vector<AKS::DataDescriptor *>& out,
    AKS::NodeParams* params, AKS::DynamicParamValues* dynParams)
{
  auto& curNode = nodes[params];
  const auto& curRunners = curNode.runners;
  unsigned int tmpID = curNode.core_id++;
  unsigned int runnerID = tmpID % curNode.core_count;

  auto runner = curRunners[runnerID].get();

  auto inputs = dynamic_cast<vart::RunnerExt*>(runner)->get_inputs();
  auto outputs = dynamic_cast<vart::RunnerExt*>(runner)->get_outputs();

  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  auto & out_dims = outputTensors[0]->get_shape();
  auto & in_dims = inputTensors[0]->get_shape();

  // Fill input data to input buffers
  uint8_t* std_data = (uint8_t*)inputs[0]->data().first;
  int inSize = in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3];
  uint8_t* pData = (uint8_t*)in[0]->data();
  for(uint32_t i=0; i< inSize; i++)
  {
    std_data[i] = pData[i];
  }

  auto job_id = runner->execute_async(inputs, outputs);
  runner->wait(job_id.first, -1);

  AKS::DataDescriptor * outDD = new AKS::DataDescriptor(out_dims, AKS::DataType::FLOAT32);
  int outSize = out_dims[0] * out_dims[1] * out_dims[2] * out_dims[3];
  float* oData = (float*)outDD->data();
  int8_t* std_data_out = (int8_t*)outputs[0]->data().first;
  for(uint32_t i = 0; i < outSize; i++)
  {
    oData[i] = (float)(std_data_out[i]);
    std_data_out[i] = 0;
  }

  out.push_back(outDD);
  return 0;
}
