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

#include <vitis/ai/tracepoint.hpp>

inline uint64_t get_tsc(void)
{
#if defined(__i386__) or defined(__x86_64__)
  unsigned long long high, low;
  asm volatile("rdtsc" : "=a" (low), "=d" (high));
  return (low + (high << 32));
#else
  return 0;
#endif
}

inline double get_ts(void)
{

  auto tp = std::chrono::steady_clock::now().time_since_epoch();
  auto ts = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1,1>>>(tp).count();

  return ts;
}

inline double get_xrt_ts(void)
{
#ifdef ENABLE_XRT_TIMESTAMP
  auto xrt_time_ns = xrt_core::time_ns();
  double xrt_time_s = xrt_time_ns / 1000.0 / 1000.0 / 1000.0;

  return xrt_time_s;
#else
  return 0;
#endif
}

namespace vitis
{
namespace ai
{

TracePoint tp_;

TracePoint::TracePoint() : m_trace_mutex {} {
  m_pid = getpid();
  m_enabled = std::getenv("VAI_TRACE_ENABLE") ? true : false;

  if (!m_enabled)
	 return; 

  auto dir = std::getenv("VAI_TRACE_DIR") ? std::getenv("VAI_TRACE_DIR") : std::string("/tmp/");
  auto logger_file = dir + "vaitrace_" + std::to_string(m_pid);

  std::lock_guard<std::mutex> lock(m_trace_mutex);

  m_logger.open(logger_file, std::ios::out);

  if (!m_logger.is_open()) {
    m_enabled = false;
    std::cout << "Error: failed to open trace log file: " << logger_file << std::endl;
    return;
  }

  auto ts_type = std::getenv("VAI_TRACE_TS");
  if (!ts_type) {
    m_enabled = false;
    std::cout << "Error: no timestamp type defined " << std::endl;
    return;
  } else {
    if (std::strcmp(ts_type, "boot") == 0)
      m_ts_type = BOOT;
    if (std::strcmp(ts_type, "x86-tsc") == 0)
      m_ts_type = x86_TSC;
    if (std::strcmp(ts_type, "XRT") == 0)
      m_ts_type = XRT;
  }

  if (m_enabled) {
    std::cout << "Vaitrace tracepoint enabled log file: " << logger_file
    << " timestamp type: " << m_ts_type
    << std::endl;
    TracePoint::tracepoint_sync_time(m_ts_type, ts_type);
  }
}

void TracePoint::tracepoint_sync_time(enum TimestampType ts_type, const char* tag)
{
  
  vitis::ai::vaiTraceEvent trace_sync_event;

  trace_sync_event.type = VAI_EVENT_TIME_SYNC;

  trace_sync_event.pid = 0;
  trace_sync_event.cpuid = -1;
  trace_sync_event.dev_id = -1;

  trace_sync_event.ts = 0;
  trace_sync_event.tsc = 0;

  if (ts_type == BOOT) {
    trace_sync_event.ts = get_ts();
  } else if (ts_type == x86_TSC) {
    trace_sync_event.tsc = get_tsc();
  } else if (ts_type == XRT) {
    trace_sync_event.ts = get_xrt_ts();
  }

  std::strncpy(trace_sync_event.tag, tag, VAI_TRACE_TAG_LEN-1);
  trace_sync_event.tag[VAI_TRACE_TAG_LEN-1] = '\0';

  std::stringstream t_sync;
  auto tp = std::chrono::steady_clock::now().time_since_epoch();
  auto ts = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1,1>>>(tp).count();

  t_sync << std::setiosflags(std::ios::fixed) << std::setprecision(9) << ts;
  auto t_sync_str = std::string(t_sync.str());
  trace_sync_event.info = t_sync_str.c_str();

  m_logger << trace_sync_event;
}

void TracePoint::enable() {
  std::lock_guard<std::mutex> lock(m_trace_mutex);
  m_enabled = true;
}

void TracePoint::disable() {
  std::lock_guard<std::mutex> lock(m_trace_mutex);
  m_enabled = false;
}

TracePoint::~TracePoint()
{
  m_logger.close();
}

void TracePoint::trace(TraceEventType t, const char* tag, int dev_id, const std::string& info)
{
  if (!m_enabled)
    return ;

  auto cpuid = sched_getcpu();
  auto pid = gettid();

  vitis::ai::vaiTraceEvent trace_event;

  trace_event.type = t;

  trace_event.pid = pid;
  trace_event.cpuid = cpuid;
  trace_event.dev_id = dev_id;

  trace_event.ts = 0;
  trace_event.tsc = 0;

  if (m_ts_type == BOOT) {
    trace_event.ts = get_ts();
  } else if (m_ts_type == x86_TSC) {
    trace_event.tsc = get_tsc();
  } else if (m_ts_type == XRT) {
    trace_event.ts = get_xrt_ts();
  }

  std::strncpy(trace_event.tag, tag, VAI_TRACE_TAG_LEN-1);
  trace_event.tag[VAI_TRACE_TAG_LEN-1] = '\0';
  trace_event.info = info.c_str();

  std::lock_guard<std::mutex> lock(m_trace_mutex);

  m_logger << trace_event;

  if (m_logger.bad()) {
    m_enabled = false;
    std::cout << "Vaitrace tracepoint error, stoped" << std::endl;
  }

  return ;
}

extern "C" {
void tracepoint_py_func(bool start, const char* func_name)
{
  if (start)
    vitis::ai::tp_instance().trace(VAI_EVENT_PY_FUNC_START, "PY", -1, std::string(func_name));
  else
    vitis::ai::tp_instance().trace(VAI_EVENT_PY_FUNC_END, "PY", -1, std::string(func_name));
}

void vaitrace_enable() {
    vitis::ai::tp_instance().enable();
}

void vaitrace_disable() {
    vitis::ai::tp_instance().disable();
}
}
} // namespace ai
} // namespace vitis
