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

#include "AksAIGraphInAction.h"
#include "AksAIJob.h"

using namespace AKS;

AIGraphInAction::AIGraphInAction(AIGraph* graph, uint64_t jobID, DynamicParamValues* dynParams)
  :_graph(graph), _dynParams(dynParams),
  _nodeInRefCount(graph->getInDegree()),
  _nodeOutRefCount(graph->getOutDegree()), _jobID(jobID)
  {
    _graph->incNumJobs();
  }


AIGraphInAction::~AIGraphInAction() {
  // Delete dynParams
  if(_dynParams) {
    delete _dynParams;
    _dynParams = nullptr;
  }

  // Delete AliveNodes
  for(auto& item: _activeNodes) {
    if(item.second) {
      delete item.second;
      item.second = nullptr;
    }
  }

  // Decrement number of jobs in its graph
  _graph->decNumJobs();
}

void AIGraphInAction::setFinished(AKS::AIJob* lastJob) {
  auto lastNode  = lastJob->getCurrentNode();
  auto aliveNode = _activeNodes.at(lastNode);
  std::vector<vart::TensorBuffer*>& outputs = aliveNode->getOutputs();
  std::vector<std::unique_ptr<vart::TensorBuffer>> v;

  // @abidk : Creating uptr from raw ptr is risky. Needs a better way
  for(int i=0; i<outputs.size(); ++i) {
    v.emplace_back(outputs[i]);
    outputs[i] = nullptr;
  }
  _promise.set_value(std::move(v));
}

