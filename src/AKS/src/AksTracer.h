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
#ifndef __AKS_TRACER_
#define __AKS_TRACER_

#include <string>
#include <iostream>
#include <map>
#include <list>
#include <vector>
#include <chrono>
#include <sstream>
#include <limits>
#include "AksCommonDefs.h"

namespace AKS
{
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  ///. ph = K: kernel exec, W: kernel_async exec, T: wait w.r.t W
  class TraceInfo {
    public:
      std::string name;
      TimePoint ts; 
      TimePoint te; 
      uint64_t jobID;
      char ph; 
      TraceInfo(const std::string& _name, TimePoint _ts, TimePoint _te,
          uint64_t _jobID, char _ph)
        :name(_name), ts(_ts), te(_te), jobID(_jobID), ph(_ph) { }

      TraceInfo(std::string&& _name, TimePoint _ts, TimePoint _te,
          uint64_t _jobID, char _ph)
        :name(std::move(_name)), ts(_ts), te(_te), jobID(_jobID), ph(_ph) { }

      ~TraceInfo() = default;

      std::string getTraceFormat(
          int workerID=0, 
          std::string thread_name = "",
          TimePoint start_t = TimePoint()) 
      {
        // return {.....}
        std::string sep = ", ";
        std::stringstream ss;
        ss << "{";

        ss << "\"name\":" << "\"" << name << "\"" << sep;
        if(ph != 'F') {
          char tph = (ph == 'K' || ph == 'W' || ph == 'T') ? 'X' : ph; 
          ss << "\"ph\":" << "\""<< tph << "\"" << sep;
        }
        ss << "\"pid\": " << 0 << sep;
        ss << "\"tid\": " << "\"" << thread_name << "\"" << sep;

        ss.precision(std::numeric_limits<double>::max_digits10);
        ss << "\"ts\":" << std::chrono::duration<double, std::micro>{ts-start_t}.count() << sep;
        ss << "\"dur\":" << std::chrono::duration<double, std::micro>{te-ts}.count() << sep;

        // "args": {"jobID":10, "ph": "W"}
        ss << "\"args\": {";
        ss << "\"jobID\":" << jobID;
        ss << sep << "\"ph\":" << "\"" << ph << "\"";
        ss << "}";

        ss << "}";

        // std::cout << ss.str() << std::endl;
        return ss.str();
      }

  }; 

  class WorkerLog {
    public:
      /// List of tracePoints
      std::vector<TraceInfo> tracePoints;
      /// 0-indexed worker ID
      int workerID;
      /// Actual thread ID
      std::thread::id thread_id;
      /// name of thread
      std::string name;

      WorkerLog(int _workerID, std::thread::id _threadID, std::string _name="", int _logSize=LOGSIZE)
        :workerID(_workerID), thread_id(_threadID), name(_name) { tracePoints.reserve(_logSize); }

      ~WorkerLog() = default;

      void addEntry(const std::string& name, TimePoint ts, 
          TimePoint te, uint64_t jobID, char ph) 
      {
        tracePoints.emplace_back(name, ts, te, jobID, ph);
      }

      void addEntry(std::string&& name, TimePoint ts, 
          TimePoint te, uint64_t jobID, char ph) 
      {
        tracePoints.emplace_back(std::move(name), ts, te, jobID, ph);
      }

      void dumpTrace(const std::string& filename = "graph_exec.trace") {
        for(auto& t: tracePoints) {
          std::cout << t.name << " " << std::endl;
        }
      }
  };

} 

#endif // __AKS_TRACER_
