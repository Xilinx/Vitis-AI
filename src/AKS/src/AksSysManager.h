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
#ifndef __AKS_SYS_MANAGER_H_
#define __AKS_SYS_MANAGER_H_

#include <string>
#include <map>
#include <future>

#include <xir/tensor/tensor.hpp>
#include <vart/tensor_buffer.hpp>

#include "AksQueue.h"
#include "AksTracer.h"


namespace AKS
{
  class AIJob;
  class AIGraph;
  class NodeParams;
  template<typename T> using Queue = UnlimitedQueue<T>;
  template<typename T> using VecPtr = std::vector<std::unique_ptr<T>>;
  class SysManager
  {

    public:
      static SysManager * getGlobal();
      static void deleteGlobal();

      void loadGraphs(std::string graphJson);
      int readKernelJsons(std::string dir);
      AIGraph* getGraph(std::string graphName);

      std::future<VecPtr<vart::TensorBuffer>> enqueueJob(
          AIGraph* graph, const std::string& filePath,
          VecPtr<vart::TensorBuffer> inputs,
          AKS::NodeParams* userArgs = nullptr);
      std::future<VecPtr<vart::TensorBuffer>> enqueueJob(
          AIGraph* graph, const std::vector<std::string>& filePaths,
          VecPtr<vart::TensorBuffer> inputs,
          AKS::NodeParams* userArgs = nullptr);
      void waitForAllResults();
      void waitForAllResults(AIGraph* graph);
      void report(AIGraph* graph);
      void printPerfStats();
      void resetTimer();

    private:
      static SysManager *_global;

      SysManager();
      ~SysManager();

      void performJob(AKS::Queue<AIJob*>* qin, AKS::Queue<AIJob*>* qout, int workerID);
      void updateQueuesAndWorkers();
      int readGraphJsons(std::string dir);
      int loadGraphJson(std::string jsonFile);
      void saveTrace(const std::string& filename="graph.trace");
      bool isQueueActive(std::string queueName);
      void addNewQueue(Queue<AIJob*>* que);
      int  removeQueue(std::string name);
      int  clearQueues(void);
      void listActiveQueues(void);
      Queue<AIJob*> * getQueue(std::string name) { return _queues.at(name); }
      std::vector<std::thread> _workers;
      std::vector<WorkerLog> _logs;

      ///. Stuffs to be done after a Job execution
      ///. It is common for all the jobs
      ///. param job Pointer to job that is executed
      ///. param outputs output of the job
      ///. param workerID ID of worker thread executing this
      
      // void postExecRoutine(AIJob* job, std::vector<AKS::DataDescriptor*>& outputs, int workerID=-1);
      void postExecRoutine(AIJob* job, std::vector<vart::TensorBuffer*>& outputs, int workerID=-1);

      ///. If a job is async call, then a new thread is created which will wait for the
      ///. job to finish and then do the postExecRoutine.
      ///. param jobID id of the job to wait for finish
      ///. param job Pointer to job that is executed
      ///. param outputs output of the job which waitRoutine is waiting to finish
      ///. param workerID ID of worker thread executing this
      // void waitRoutine(int workerID);
      void waitRoutine(AIJob* job);


      AIGraph *_loadedGraphs;
      /// Queues for all the operations
      std::map<std::string, Queue<AIJob*>*> _queues;

      std::string _path;
      std::string _name;
      std::string _version;

      std::atomic<uint64_t> _jobID{0};
      std::atomic<int> _numJobs{0};
      // This is used for log entry, waitForAllJobs etc.
      std::condition_variable _cv;
      std::mutex _mtx;
      std::atomic<uint32_t> _numThreads{0};
      Queue<AIJob*> _waitQueue{"waitQ", 32};
      std::chrono::time_point<std::chrono::steady_clock> _t_start;
      std::chrono::duration<float> _t_enq_dur{0};
      // This is used for exec waiting on _numThreads
      std::condition_variable _waitCV;
      std::mutex _waitMtx;
      std::mutex _timeMtx;

      int _max_concurrent_jobs = 128;
  };
}

// extern "C" {
//   AKS::SysManager* createSysManager();
//   void loadKernels(AKS::SysManager* sysMan, const char* kernelDir);
//   void loadGraphs(AKS::SysManager* sysMan, const char* graphPath);
//   AKS::AIGraph* getGraph(AKS::SysManager* sysMan, const char* graphName);
//   void enqueueJob(AKS::SysManager* sysMan, AKS::AIGraph* graph,
//       const char* imagePath, AKS::NodeParams* params);
//   void wait(AKS::SysManager* sysMan);
//   void report(AKS::SysManager* sysMan, AKS::AIGraph* graph);
//   void deleteSysManager();
// }

#endif // __AKS_SYS_MANAGER_H_
