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
#ifndef DEEPHI_SHARED_QUEUE_HPP_
#define DEEPHI_SHARED_QUEUE_HPP_

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

namespace vitis {
namespace ai {
/**
 * A thread safe queue.
 */
template <typename T>
class SharedQueue {
 public:
  /**
   * Return the size of the queue.
   */
  virtual std::size_t size() const {
    std::lock_guard<std::mutex> lock(this->mtx_);
    return this->internal_size();
  }

  /**
   * Return true if the queue is empty, and false otherwise.
   */
  virtual bool empty() const {
    std::lock_guard<std::mutex> lock(this->mtx_);
    return this->internal_empty();
  }

  /**
   * Copy the value to the end of this queue.
   */
  virtual void push(const T& new_value) {
    std::lock_guard<std::mutex> lock(this->mtx_);
    this->internal_push(new_value);
    this->cond_not_empty_.notify_one();
  }

  /**
   * Get the first element in the queue and remove it from the queue.
   * This is blocking
   */
  virtual void pop(T& value) {
    std::unique_lock<std::mutex> lock(this->mtx_);
    this->cond_not_empty_.wait(lock,
                               [this]() { return !(this->internal_empty()); });
    this->internal_pop(value);
  }

  /**
   * Get the first element in the queue and remove it from the queue.
   * This will fail and return false if blocked for more than rel_time.
   */
  virtual bool pop(T& value, const std::chrono::milliseconds& rel_time) {
    std::unique_lock<std::mutex> lock(this->mtx_);
    if (this->cond_not_empty_.wait_for(lock, rel_time, [this]() {
          return !(this->internal_empty());
        }) == false) {
      return false;
    }
    this->internal_pop(value);
    return true;
  }

  /**
   * Get the first element in the queue that satisfies cond, and remove it
   * from the queue.
   * This will fail and return false if no such element.
   */
  virtual bool pop(T& value, std::function<bool(const T&)>& cond) {
    std::lock_guard<std::mutex> lock(this->mtx_);
    if (this->internal_empty()) return false;
    auto it = std::find_if(internal_.begin(), internal_.end(), cond);
    if (it != internal_.end()) {
      value = std::move(*it);
      internal_.erase(it);
      return true;
    }
    return false;
  }

  /**
   * Get the first element in the queue that satisfies cond, and remove it
   * from the queue.
   * This will fail and return false if no such element, or blocked for more
   * than rel_time.
   */
  virtual bool pop(T& value, std::function<bool(const T&)>& cond,
                   const std::chrono::milliseconds& rel_time) {
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
      return true;
    }
    return false;
  }

 protected:
  inline virtual std::size_t internal_size() const {
    return this->internal_.size();
  }
  inline virtual bool internal_empty() const { return this->internal_.empty(); }
  inline virtual void internal_push(const T& new_value) {
    this->internal_.emplace_back(new_value);
  }
  inline virtual void internal_pop(T& value) {
    value = std::move(this->internal_.front());
    this->internal_.pop_front();
  }
  inline virtual void internal_top(T& value) {
    value = this->internal_.front();
  }

  mutable std::mutex mtx_;
  std::condition_variable cond_not_empty_;

  std::deque<T> internal_;
};
}  // namespace ai
}  // namespace vitis
#endif
