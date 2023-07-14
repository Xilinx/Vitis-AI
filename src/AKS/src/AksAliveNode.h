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
#ifndef __AKS_ALIVE_NODE_H_
#define __AKS_ALIVE_NODE_H_

#include <vector>
#include <string>
#include <utility>

#include <xir/tensor/tensor.hpp>
#include <vart/tensor_buffer.hpp>

namespace AKS {

  class AIGraphInAction;
  class AIGraphNode;

  class AliveNode {
    public:
      AliveNode(std::vector<vart::TensorBuffer*>& outputs)
        :_outputs(outputs) {}

      AliveNode(std::vector<vart::TensorBuffer*> outputs)
        :_outputs(std::move(outputs)) {}

      ~AliveNode() {
        // @abidk : Each element might have been set NULL by AIGraphInAction::setFinished()
        for(auto i=0; i<_outputs.size(); ++i) {
          if(_outputs[i]) delete _outputs[i];
          _outputs[i] = nullptr;
        }
      }

      std::vector<vart::TensorBuffer*>& getOutputs() { return _outputs; }

    private:
      /// Pointers to outputs of the alive node, so that its child nodes can take input from here.
      std::vector<vart::TensorBuffer*> _outputs;

  };


} // namespace AKS

#endif // __AKS_ALIVE_NODE_H_
