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

#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include <utility>

#include "event.hpp"
#include "vaitrace_dbg.hpp"

namespace vitis::ai::trace {
// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;
using std::list;
using std::lock_guard;
using std::mutex;
using std::cout;
using std::endl;
using std::vector;
using std::string;
template <typename T>
class RingBuf : public list<T*> {
 public:
  RingBuf(size_t size_mb = 1, size_t substitution_rate = 20)
      : cur_size_(0), substitution_rate(substitution_rate), m_lock{} {
    CHECK(size_mb > 0 && size_mb <= 32);
    max_size_ = size_mb * 1024 * 1024;
  }

  ~RingBuf() {}

  void push(T* data) {
    lock_guard<mutex> lock(m_lock);
    // m_lock.lock();
    this->push_back(data);
    cur_size_ += data->get_size();
    // m_lock.unlock();

    // This step can be asynchronous
    maintain();
  }

  void lock(void) {
      m_lock.lock();
  }

  void unlock(void) {
      m_lock.unlock();
  }

 private:
  size_t max_size_;
  size_t cur_size_;
  size_t substitution_rate;
  std::mutex m_lock;
  bool overflowed(void) { return (cur_size_ > max_size_); }

  void put_slots(void) {
    size_t put_bytes = int(max_size_ * substitution_rate / 100);

    CHECK(put_bytes <= cur_size_);

    for (size_t i = 0; i < put_bytes;) {
      auto entry = this->front();
      auto entry_size = entry->get_size();
      cur_size_ -= entry_size;
      i += entry_size;
      delete entry;
      this->pop_front();
    }
  }

  void maintain(void) {
    if (overflowed()) {
      put_slots();
    }
  }
};
}  // namespace vitis::ai::trace
