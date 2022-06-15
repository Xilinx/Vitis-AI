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
#ifndef __AKS_AI_GRAPH_IN_ACTION_H_
#define __AKS_AI_GRAPH_IN_ACTION_H_

#include <vector>
#include <string>
#include <future>
#include <mutex>

#include "aks/AksNodeParams.h"
#include "AksAIGraph.h"
#include "AksAliveNode.h"

namespace AKS {

  class AliveNode;
  class AIJob;

  class AIGraphInAction {
    public:
      /// Mutex used for locking In/Out degrees and enqueue next job
      std::mutex degreeMtx;

      AIGraphInAction(AIGraph* graph, uint64_t jobID, DynamicParamValues* dynParams=nullptr);
      ~AIGraphInAction();

      std::future<std::vector<std::unique_ptr<vart::TensorBuffer>>> getFuture() { return _promise.get_future(); };
      void setFinished(AKS::AIJob* lastJob);

      AIGraph* getGraph() { return _graph; }
      DynamicParamValues* getDynamicParams() { return _dynParams; }

      void insertAliveNode(AIGraphNode* node, AliveNode* value) { _activeNodes[node] = value; }
      AliveNode* getAliveNode(AIGraphNode* node) { return _activeNodes.at(node); }

      /// Decrement the inDegree of a node
      /// param ID Node ID
      /// return updated value
      int decInRefCount(int ID) {return --_nodeInRefCount[ID]; }

      /// Decrement the outDegree of a node (provide the node ID)
      /// param ID Node ID
      /// return updated value
      int decOutRefCount(int ID){return --_nodeOutRefCount[ID];}

      uint64_t getJobID() const{ return _jobID; }

    private:
      AIGraph* _graph;
      std::map<AIGraphNode*, AliveNode*> _activeNodes;
      DynamicParamValues *_dynParams = nullptr;
      std::vector<int> _nodeInRefCount;
      std::vector<int> _nodeOutRefCount;
      std::promise<std::vector<std::unique_ptr<vart::TensorBuffer>>> _promise;
      const uint64_t _jobID;
  };


} // namespace AKS

#endif // __AKS_AI_GRAPH_IN_ACTION_H_
