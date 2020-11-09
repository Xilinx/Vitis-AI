/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <assert.h>
#include <mutex>
#include <iomanip>

#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/sysinfo.h>
#define gettid() syscall(SYS_gettid)

#define VAI_TRACE_TAG_LEN (10)
#define VAI_TRACE_INFO_LEN (128)

enum TraceEventType {
    VAI_EVENT_HOST_START,
    VAI_EVENT_HOST_END,
    VAI_EVENT_INFO,

    VAI_EVENT_DEVICE_START,
    VAI_EVENT_DEVICE_END,
    VAI_EVENT_MARKER,
    VAI_EVENT_COUNTER
};

inline std::string event_type_str(TraceEventType __t)
{
  switch (__t) {
  case VAI_EVENT_HOST_START:
    return std::string("EVENT_HOST_START");
  case VAI_EVENT_HOST_END:
    return std::string("EVENT_HOST_END");
  case VAI_EVENT_DEVICE_START:
    return std::string("EVENT_DEVICE_START");
  case VAI_EVENT_DEVICE_END:
    return std::string("EVENT_DEVICE_END");
  case VAI_EVENT_MARKER:
    return std::string("EVENT_MARKER");
  case VAI_EVENT_INFO:
    return std::string("EVENT_INFO");
  case VAI_EVENT_COUNTER:
    return std::string("EVENT_COUNTER");
  }

  return std::string("EVENT_UNKNOW");
}


namespace vitis
{
namespace ai
{

enum TimestampType {BOOT, x86_TSC};
struct vaiTraceEvent {
 public:
  char tag[VAI_TRACE_TAG_LEN];
  int pid;
  int cpuid;
  int dev_id;
  double ts;
  uint64_t tsc;
  TraceEventType type;
  const char *info;
 public:
  friend std::ostream& operator << (std::ostream& os, const vaiTraceEvent& e) {
    os << event_type_str(e.type) <<" " <<
       e.pid     <<" " <<
       e.cpuid   <<" " <<
       e.tag <<" " << e.dev_id << " ";

    if (e.ts != 0)
      os << std::setiosflags(std::ios::fixed) << std::setprecision(6) << e.ts      <<" " ;
    if (e.tsc != 0)
      os << e.tsc <<" ";

    os << e.info;
    os << std::endl;

    return os ;
  }
};

class TracePoint
{
 public:
  TracePoint();
  ~TracePoint();

 public:
  void trace(TraceEventType t, const char *tag, int core_id, const std::string& info);

 private:
  TimestampType m_ts_type;
  std::ofstream m_logger;
  int m_pid;
  bool m_enabled;
  std::mutex m_trace_mutex;
};

extern TracePoint tp_;

inline TracePoint& tp_instance()
{
  return vitis::ai::tp_;
}

// APIs
inline void tracepoint(TraceEventType t, const char* tag)
{
  vitis::ai::tp_instance().trace(t, tag, -1, std::string());
}

inline void tracepoint(TraceEventType t, const char* tag, const std::string& info)
{
  vitis::ai::tp_instance().trace(t, tag, -1, info);
}

inline void tracepoint(TraceEventType t, const char* tag, int dev_id, const std::string& info)
{
  vitis::ai::tp_instance().trace(t, tag, int(dev_id), info);
}

inline void tracepoint(TraceEventType t, const char* tag, int dev_id)
{
  vitis::ai::tp_instance().trace(t, tag, int(dev_id), std::string());
}
}  // namespace ai
}  // namespace vitis
