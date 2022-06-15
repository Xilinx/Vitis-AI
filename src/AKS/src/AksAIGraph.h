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
#ifndef __AKS_AI_GRAPH_H_
#define __AKS_AI_GRAPH_H_

#include <map>
#include <string>
#include <vector>
#include <set>
#include <atomic>
using namespace std;

namespace AKS
{
  class AIGraphNode;
  class KernelDef;

  class AIGraph
  {
    public:
      AIGraph(string name);
      ~AIGraph(); 
      
      string getName() const { return _name;}
      void   setVersion(string v) { _version  = v;}
      string getVersion() const {return _version;}
      void   addNode(AIGraphNode* node);
      AIGraphNode* findNode(string nodeName);
      set<KernelDef*> getUniqueKernels();
      void    dump(string prefix); 
      bool    isGraphOK(string &err);
      AIGraphNode* getFirstNode() {return _firstNode;}; 
  
      const vector<AIGraphNode*>& getNodesVector() {return _nodesVector;}
      const vector<int> getInDegree() { return _inDegree;}
      const vector<int> getOutDegree() { return _outDegree;}
      
      void incNumJobs() { ++_numJobs; }
      void decNumJobs() { --_numJobs; }
      int64_t getNumJobs() { return _numJobs.load(); }

    private:
      string _name;
      string _version;
      vector<AIGraphNode*> _nodesVector;
      vector<int> _inDegree;
      vector<int> _outDegree;
      AIGraphNode* _firstNode = nullptr;
      std::atomic<int64_t> _numJobs{0};
  };
}

#endif
