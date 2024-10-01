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
#ifndef DEEPHI_RING_QUEUE_HPP_
#define DEEPHI_RING_QUEUE_HPP_

#include <condition_variable>
#include <mutex>
#include <vector>

namespace vitis {
namespace ai {
/**
 * A thread safe queue with a size limit.
 * When it is full, it will overwrite the earliest element.
 */
template <typename T>
class RingQueue {
 public:
  explicit RingQueue(std::size_t capacity)
      : capacity_(capacity),
        buffer_(capacity),
        size_(0U),
        front_(0U),
        rear_(0U) {}
  /**
   * Return the maxium size of the queue.
   */
  std::size_t capacity() const { return capacity_; }

  /**
   * Return the size of the queue.
   */
  std::size_t size() const { return size_; }

  /**
   * Copy the value to the end of this queue.
   */
  void push(const T& new_value) {
    std::lock_guard<std::mutex> lock(mtx_);
    buffer_[rear_] = new_value;
    rear_ = (rear_ + 1) % capacity_;
    if (size_ >= capacity_) {
      front_ = rear_;
    } else {
      ++size_;
    }
  }

  /**
   * Get the first element in the queue and remove it from the queue.
   * Return false if the queue is empty
   */
  bool pop(T& value) {
    std::lock_guard<std::mutex> lock(mtx_);
    bool res = false;
    if (size_ > 0) {
      value = buffer_[front_];
      front_ = (front_ + 1) % capacity_;
      --size_;
      res = true;
    }
    return res;
  }

  /**
   * Look at the top of the queue. i.e. the element that would be poped.
   * Warning: the returned pointer can become dangled
   */
  T* pop() {
    std::lock_guard<std::mutex> lock(mtx_);
    T* res = nullptr;
    if (size_ > 0) {
      res = &buffer_[front_];
      front_ = (front_ + 1) % capacity_;
      --size_;
    }
    return res;
  }

  /**
   * Return the first element in the queue and remove it from the queue.
   * Warning: the returned pointer can become dangled
   */
  T* top() {
    std::lock_guard<std::mutex> lock(mtx_);
    T* res = nullptr;
    if (size_ > 0) {
      res = &buffer_[front_];
    }
    return res;
  }

 private:
  std::size_t capacity_;
  std::vector<T> buffer_;
  std::size_t size_;
  std::size_t front_;
  std::size_t rear_;

  mutable std::mutex mtx_;
};
}  // namespace ai
}  // namespace vitis
#endif
