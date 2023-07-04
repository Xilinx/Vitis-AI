/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#include <signal.h>

#include <cstdlib>
#include <map>
#include <vitis/ai/trace.hpp>

#include "internal.hpp"
#if _WIN32
#  include <windows.h>
#else
#  include <sys/syscall.h>
#  include <sys/sysinfo.h>
#  include <sys/types.h>

#  define gettid() syscall(SYS_gettid)
#  define getpid() syscall(SYS_getpid)
#endif
namespace vitis::ai::trace {

// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;

using namespace std::chrono;

DEF_ENV_PARAM(DEBUG_VAITRACE, "0");

void handler(int param) {
  dump();
  exit(param);
}

void print_opt(vai_trace_options_t& options) {
  for (auto& o : options) {
    VAITRACE_DBG << "[options]" << o.first << ":" << o.second;
  }
};

trace_controller::trace_controller(vai_trace_options_t options)
    : enabled(false) {
  enabled = bool(stoi(options["enable"], nullptr));

  if (!enabled) return;

  print_opt(options);

  size_t buf_size_mb = stoi(options["buf_size_mb"], nullptr);
  p_rbuf = new vai_trace_q_t(buf_size_mb);

  CHECK(p_rbuf != nullptr);

  logger_dir_path = options["trace_log_dir"];
  logger_file_path = options["logger_file_path"];

  signal(SIGINT, handler);
  signal(SIGTERM, handler);
};

trace_controller::~trace_controller() {
  dump();
  free(p_rbuf);
};

string trace_controller::get_logger_file_path() { return logger_file_path; };
string trace_controller::get_logger_dir_path() { return logger_dir_path; };

void trace_controller::push_info(trace_entry_t i) {
  std::lock_guard<std::mutex> lock(infobase_lock);
  infobase.push_back(i);
}

mutex global_lock;
mutex core_lock[CORE_N_MAX];

void trace_time_sync(const char* tag) {
  auto tp = steady_clock::now().time_since_epoch();
  auto ts = duration_cast<duration<double>>(tp).count();

  auto xrt_ts = get_xrt_ts();
  add_info("trace_timesync", "xrt_ts", xrt_ts, "steady_clock", ts, "unit", "s");
}

/* Return 1 if env VAI_TRACE_ENABLE set */
bool check_env(vai_trace_options_t& options) {
  auto pid =
#if _WIN32
      GetCurrentProcessId();
#else
      gettid();
#endif
  options["pid"] = to_string(pid);

  auto trace_env = my_getenv_s("VAI_TRACE_ENABLE", "false");
  auto enable = trace_env == "true";
  options["enable"] = to_string(enable);
  if (!enable) return false;

  auto buf_size_mb = my_getenv_s("VAI_TRACE_RBUF_MB", "2");
  options["buf_size_mb"] = buf_size_mb;

  auto trace_log_dir = my_getenv_s("VAI_TRACE_DIR", "/temp/");
  options["trace_log_dir"] = trace_log_dir;

  string logger_file_path = trace_log_dir + "vaitrace_" + to_string(pid);
  options["logger_file_path"] = logger_file_path;

  return enable;
}

vai_trace_options_t initialize() {
  vai_trace_options_t options;
  check_env(options);

  VAITRACE_DBG << "initialize...";

  return options;
};

bool is_enabled() { return get_trace_controller_inst().is_enabled(); };

void lock(void) {
  if (!is_enabled()) return;
  global_lock.lock();
};

void lock(size_t& core_idx) {
  if (!is_enabled()) return;
  core_lock[core_idx].lock();
};

void lock(std::mutex& mutex) {
  if (!is_enabled()) return;
  mutex.lock();
};

void unlock(void) {
  if (!is_enabled()) return;
  global_lock.unlock();
};

void unlock(size_t& core_idx) {
  if (!is_enabled()) return;
  core_lock[core_idx].unlock();
};

void unlock(std::mutex& mutex) {
  if (!is_enabled()) return;
  mutex.unlock();
};

void disable_trace() {
  VAITRACE_DBG << "Disabling...";
  get_trace_controller_inst().disable();
};

void push_info(trace_entry_t i) { get_trace_controller_inst().push_info(i); };

void dump() {
  if (!is_enabled()) return;

  disable_trace();

  VAITRACE_DBG << "Dumping...";

  vector<trace_entry_t> o_data;

  trace_entry_t section_flag;

  section_flag.insert(std::make_pair("#SECTION", "INFO"));

  // Add timesync
  trace_time_sync("vart_tracer");

  // Get Info
  o_data.push_back(section_flag);
  auto infobase = get_infobase();
  for_each(infobase->begin(), infobase->end(),
           [&o_data](auto n) { o_data.push_back(n); });

  // Get Trace
  section_flag.erase("#SECTION");
  section_flag.insert(std::make_pair("#SECTION", "TRACE"));
  o_data.push_back(section_flag);
  get_rbuf()->lock();

  for_each(get_rbuf()->begin(), get_rbuf()->end(),
           [&o_data](auto n) { o_data.push_back(n->get()); });

  get_rbuf()->unlock();

  auto o_file = get_trace_controller_inst().get_logger_file_path();
  VAITRACE_DBG << "Dumping to:" << o_file;
  dump_to(o_data, o_file);
};

extern "C" {
void stop() {
  if (!is_enabled()) return;

  dump();

  VAITRACE_DBG << "Stoping...";
};
void start() {
  if (!is_enabled()) return;
};
}

auto options = vitis::ai::trace::initialize();
trace_controller tc(options);

}  // namespace vitis::ai::trace
