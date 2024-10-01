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

#ifndef __COMMON_H_
#define __COMMON_H_

#include "event.hpp"
#include "payload.hpp"
#include "ringbuf.hpp"
#include "vaitrace_dbg.hpp"

namespace vitis::ai::trace {
// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;
using std::map;
using vai_trace_header_t = traceEventBase;
using vai_trace_q_t = RingBuf<vai_trace_header_t>;
using vai_trace_opt_t = std::pair<string, string>;
using vai_trace_options_t = map<string, string>;

class trace_controller {
 public:
  trace_controller(map<string,string> options);
  ~trace_controller();
  bool is_enabled() {return enabled;}
  void disable() {enabled = false;}
  string get_logger_file_path(void);
  string get_logger_dir_path(void);
  void push_info(trace_entry_t i);
  vai_trace_q_t* p_rbuf;
  vector<trace_entry_t> infobase;

 private:
  std::mutex infobase_lock;
  string logger_file_path;
  string logger_dir_path;
  bool enabled;
};

extern trace_controller tc;

inline trace_controller &get_trace_controller_inst() {
    return vitis::ai::trace::tc;
}

inline vector<trace_entry_t>* get_infobase() {
  return &get_trace_controller_inst().infobase;
}

inline vai_trace_q_t* get_rbuf() {
    return get_trace_controller_inst().p_rbuf;
}
}  // namespace vitis::ai::trace

#endif
