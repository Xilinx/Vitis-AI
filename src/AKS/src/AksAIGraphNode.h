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
#ifndef __AKS_AI_GRAPH_NODE_H_
#define __AKS_AI_GRAPH_NODE_H_

#include <map>
#include <string>
#include <map>
#include <set>
#include <vector>
using namespace std;

namespace AKS
{

  class KernelDef;
  class AIGraph;
  struct NodeParams;
  class AIGraphNode
  {
    public:
      AIGraphNode(string name,AIGraph* parentGraph);
      ~AIGraphNode();
      
      string getName() const { return _name;}
      void   addExecKernel(KernelDef* kernel, NodeParams *pvalues);
      void   provideExecKernels(set<KernelDef*> &kernels);
      void   addNextNodeStr(string nname);
      void   addNextNode(AIGraphNode *nextNode);
      void   addPrevNode(AIGraphNode *prevNode);
      void   dump(string prefix);
    
      const  set<string>& getNextNodesStr() { return _nextNodesStr;}
      bool   isNodeANextNode(string nname);
      bool   isNodeOK(string &err);
      std::map<KernelDef*, NodeParams*>& getKernels(void) { return _execKernels; } 

      // TODO : It returns only the first kernel
      KernelDef* getKernel() { return _execKernels.begin()->first; }
      NodeParams* getOpParam(KernelDef* kernel) { return _execKernels.at(kernel); }

      // Getters for prev nodes and next nodes
      const vector<AIGraphNode*> & getPrevNodes() { return _prevNodes; }
      const vector<AIGraphNode*> & getNextNodes() { return _nextNodes; }
      
      void     clearNextPrev() { _nextNodes.clear(); _prevNodes.clear();}

      void     setIndex(int index) { _index = index;}
      int      getIndex() const { return _index;}
      bool     isInitialized() { return initialized; }
      void     initDone() { initialized = true; }
    private:
      string _name;
      map<KernelDef*, NodeParams*> _execKernels;
      set<string> _nextNodesStr;
      AIGraph *_parentGraph;
      vector<AIGraphNode*> _nextNodes;
      vector<AIGraphNode*> _prevNodes;
      int _index;
      bool initialized = false;
  };
}

#endif
