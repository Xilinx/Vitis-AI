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
#ifndef __AKS_AI_JOB_H_
#define __AKS_AI_JOB_H_

#include <xir/tensor/tensor.hpp>
#include <vart/tensor_buffer.hpp>

namespace AKS {

  class AIGraphInAction;
  class AIGraphNode;

  class AIJob {
    public:
      AIJob(AIGraphInAction* curGraph, AIGraphNode* nextNode)
        : _curGraphInAction(curGraph), _curNode(nextNode) {}

      /// Getter for Current GraphInAction
      AIGraphInAction* getCurrentGraphInAction() { return _curGraphInAction; }

      /// Getter for Current Node
      AIGraphNode* getCurrentNode() { return _curNode; }

      int getJobID() { return _jid; }
      int getWorkerID() { return _workerID; }
      std::vector<vart::TensorBuffer*> getOutputs() { return _outputs; }

      void setJobID(int jid) { _jid = jid; }
      void setWorkerID(int workerID) { _workerID = workerID; }
      void setOutputs(std::vector<vart::TensorBuffer*> v) { _outputs = v; }


      ~AIJob() {}

    private:
      AIGraphInAction* _curGraphInAction;
      AIGraphNode* _curNode;
      int _jid=-1;
      int _workerID=-1;
      std::vector<vart::TensorBuffer*> _outputs = {};
  };


} // namespace AKS

#endif // __AKS_AI_JOB_H_

