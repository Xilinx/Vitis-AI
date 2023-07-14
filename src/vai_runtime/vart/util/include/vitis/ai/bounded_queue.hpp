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
#ifndef DEEPHI_BOUNDED_QUEUE_HPP_
#define DEEPHI_BOUNDED_QUEUE_HPP_

#include "shared_queue.hpp"

namespace vitis {
namespace ai {
/**
 * A thread safe queue with a size limit.
 * It will block on push if full, and on pop if empty.
 */
template <typename T>
class BoundedQueue : public SharedQueue<T> {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  /**
   * Return the maxium size of the queue.
   */
  std::size_t capacity() const { return this->capacity_; }

  /**
   * Copy the value to the end of this queue.
   * This is blocking.
   */
  void push(const T& new_value) override {
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
   * Look at the top of the queue. i.e. the element that would be poped.
   * This will fail and return false if blocked for more than rel_time.
   */
  bool top(T& value, const std::chrono::milliseconds& rel_time) {
    std::unique_lock<std::mutex> lock(this->mtx_);
    if (this->cond_not_empty_.wait_for(lock, rel_time, [this]() {
          return !this->internal_empty();
        }) == false) {
      return false;
    }
    this->internal_top(value);
    return true;
  }

  /**
   * Return the first element in the queue and remove it from the queue.
   * This is blocking.
   */
  void pop(T& value) override {
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
  bool pop(T& value, const std::chrono::milliseconds& rel_time) override {
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

  /**
   * Return the first element in the queue that satisfies cond, and remove it
   * from the queue.
   * This is blocking.
   */
  bool pop(T& value, std::function<bool(const T&)>& cond) override {
    std::lock_guard<std::mutex> lock(this->mtx_);
    if (this->internal_empty()) return false;
    auto it =
        std::find_if(this->internal_.begin(), this->internal_.end(), cond);
    if (it != this->internal_.end()) {
      value = std::move(*it);
      this->internal_.erase(it);
      this->cond_not_full_.notify_one();
      return true;
    }
    return false;
  }

  /**
   * Return the first element in the queue that satisfies cond, and remove it
   * from the queue.
   * This will fail and return false if blocked for more than rel_time.
   */
  bool pop(T& value, std::function<bool(const T&)>& cond,
           const std::chrono::milliseconds& rel_time) override {
    auto now = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(this->mtx_);
    // Wait until not empty
    if (!this->cond_not_empty_.wait_for(
            lock, rel_time, [this]() { return !this->internal_empty(); })) {
      return false;
    }
    auto it =
        std::find_if(this->internal_.begin(), this->internal_.end(), cond);
    while (it == this->internal_.end()) {
      if (this->cond_not_empty_.wait_until(lock, now + rel_time) ==
          std::cv_status::timeout) {
        break;
      }
      it = std::find_if(this->internal_.begin(), this->internal_.end(), cond);
    }
    it = std::find_if(this->internal_.begin(), this->internal_.end(), cond);
    if (it != this->internal_.end()) {
      value = std::move(*it);
      this->internal_.erase(it);
      this->cond_not_full_.notify_one();
      return true;
    }
    return false;
  }

 protected:
  std::size_t capacity_;

  std::condition_variable cond_not_full_;
};
}  // namespace ai
}  // namespace vitis
#endif
