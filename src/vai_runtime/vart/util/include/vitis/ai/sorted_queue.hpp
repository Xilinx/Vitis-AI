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
#ifndef DEEPHI_SORTED_QUEUE_HPP_
#define DEEPHI_SORTED_QUEUE_HPP_

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>

namespace vitis {
namespace ai {
/**
 * A thread safe, bounded, priority queue
 */
template <typename T>
class SortedQueue {
 public:
  explicit SortedQueue(std::size_t capacity) : capacity_(capacity) {}

  /**
   * Return the maxium size of the queue.
   */
  std::size_t capacity() const { return this->capacity_; }

  /**
   * Copy the value to the end of this queue.
   * This is blocking.
   */
  void push(const T& new_value) {
    std::unique_lock<std::mutex> lock(this->mtx_);
    this->cond_not_full_.wait(
        lock, [this]() { return this->internal_size() < this->capacity_; });
    this->internal_push(new_value);
    this->cond_not_empty_.notify_one();
  }

  /**
   * Copy the value to the end of this queue.
   * This will fail and return false if blocked for more than rel_time.
   */
  bool push(const T& new_value, const std::chrono::milliseconds& rel_time) {
    std::unique_lock<std::mutex> lock(this->mtx_);
    if (this->cond_not_full_.wait_for(lock, rel_time, [this]() {
          return this->internal_size() < this->capacity_;
        }) == false) {
      return false;
    }
    this->internal_push(new_value);
    this->cond_not_empty_.notify_one();
    return true;
  }

  /**
   * Return the first element in the queue and remove it from the queue.
   * This is blocking.
   */
  void pop(T& value) {
    std::unique_lock<std::mutex> lock(this->mtx_);
    this->cond_not_empty_.wait(lock,
                               [this]() { return !this->internal_empty(); });
    this->internal_pop(value);
    this->cond_not_full_.notify_one();
  }

  /**
   * Return the first element in the queue and remove it from the queue.
   * This will fail and return false if blocked for more than rel_time.
   */
  bool pop(T& value, const std::chrono::milliseconds& rel_time) {
    std::unique_lock<std::mutex> lock(this->mtx_);
    if (this->cond_not_empty_.wait_for(lock, rel_time, [this]() {
          return !this->internal_empty();
        }) == false) {
      return false;
    }
    this->internal_pop(value);
    this->cond_not_full_.notify_one();
    return true;
  }

 protected:
  inline virtual std::size_t internal_size() const {
    return this->internal_.size();
  }
  inline virtual bool internal_empty() const { return this->internal_.empty(); }
  inline virtual void internal_push(const T& new_value) {
    this->internal_.emplace(new_value);
  }
  inline virtual void internal_pop(T& value) {
    value = std::move(this->internal_.top());
    this->internal_.pop();
  }

  std::mutex mtx_;
  std::size_t capacity_;
  std::priority_queue<T, std::deque<T>, std::greater<T>> internal_;
  std::condition_variable cond_not_empty_;
  std::condition_variable cond_not_full_;
};
}  // namespace ai
}  // namespace vitis
#endif
