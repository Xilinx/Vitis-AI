/*
 * Copyright 2019 xilinx Inc.
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
#include <glog/logging.h>

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>

namespace vitis {
namespace ai {
template <typename MessageType>
class ErlMsgBox {
  using mutex_type = std::mutex;

 private:
  class Cons {
   public:
    Cons(std::unique_ptr<MessageType>&& car, std::unique_ptr<Cons>&& cdr)
        : car_(std::move(car)), cdr_(std::move(cdr)) {}
    ~Cons() {}
    Cons() = delete;
    Cons(const Cons&) = delete;
    Cons& operator()(const Cons&) = delete;

    std::unique_ptr<MessageType>& Car() { return car_; }

   private:
    std::unique_ptr<MessageType> car_;
    std::unique_ptr<Cons> cdr_;
    friend class ErlMsgBox;
  };

 public:
  /// @brief create a ErlMsgBox when `capacity` is -1, it means no
  /// limitation, otherwise, it will become blocking.
  ErlMsgBox(size_t capacity = std::numeric_limits<size_t>::max(),
            bool use_lock = true)
      : capacity_{capacity},
        use_lock_{use_lock},
        size_{0},
        lock_for_recv_{},
        lock_for_queue_{},
        not_full{},
        not_empty{},
        header_{nullptr},
        tail_{&header_} {}
  ~ErlMsgBox() {
    auto cur = std::move(header_);
    // properly destruct everything to avoid stack overflow
    while (cur) {
      cur = std::move(cur->cdr_);
    }
  }
  size_t size() const {
    auto lock = std::unique_ptr<std::lock_guard<std::mutex>>();
    if (use_lock_) {
      lock = std::make_unique<std::lock_guard<std::mutex>>(lock_for_queue_);
    }
    return size_;
  }
  size_t capacity() const {
    auto lock = std::unique_ptr<std::lock_guard<std::mutex>>();
    if (use_lock_) {
      lock = std::make_unique<std::lock_guard<std::mutex>>(lock_for_queue_);
    }
    return capacity_;
  }
  bool empty() const {
    auto lock = std::unique_ptr<std::lock_guard<std::mutex>>();
    if (use_lock_) {
      lock = std::make_unique<std::lock_guard<std::mutex>>(lock_for_queue_);
    }
    return size_ == 0u;
  }
  bool full() const {
    auto lock = std::unique_ptr<std::lock_guard<std::mutex>>();
    if (use_lock_) {
      lock = std::make_unique<std::lock_guard<std::mutex>>(lock_for_queue_);
    }
    return size_ >= capacity_;
  }

  template <typename... Args>
  void emplace_send(Args&&... args) {
    auto cur = std::make_unique<MessageType>(std::forward<Args>(args)...);
    while ((cur = send_ptr(std::move(cur), std::chrono::milliseconds(1000))) !=
           nullptr) {
      // empty body
    }
    return;
  }

  template <typename... Args>
  bool emplace_push(Args&&... args) {
    auto lock = std::unique_ptr<std::lock_guard<std::mutex>>();
    if (use_lock_) {
      lock = std::make_unique<std::lock_guard<std::mutex>>(lock_for_queue_);
    }
    if (size_ >= capacity_) {
      return false;
    }
    std::unique_ptr<typename ErlMsgBox<MessageType>::Cons> x =
        std::unique_ptr<Cons>(
            new Cons(std::make_unique<MessageType>(std::forward<Args>(args)...),
                     nullptr));
    if (tail_ == &header_) {
      tail_ = &x->cdr_;
    }
    x->cdr_ = std::move(header_);
    header_ = std::move(x);
    size_++;
    return true;
  }

  std::unique_ptr<MessageType> push(std::unique_ptr<MessageType> obj) {
    auto lock = std::unique_ptr<std::lock_guard<std::mutex>>();
    if (use_lock_) {
      lock = std::make_unique<std::lock_guard<std::mutex>>(lock_for_queue_);
    }
    if (size_ >= capacity_) {
      return obj;
    }
    std::unique_ptr<typename ErlMsgBox<MessageType>::Cons> x =
        std::unique_ptr<Cons>(new Cons(std::move(obj), nullptr));

    if (tail_ == &header_) {
      tail_ = &x->cdr_;
    }
    x->cdr_ = std::move(header_);
    header_ = std::move(x);
    size_++;
    // do not not_empty.notify
    // because this method won't wake up a `recv` thread;
    return nullptr;
  }

  std::unique_ptr<MessageType> pop() {
    auto lock = std::unique_ptr<std::lock_guard<std::mutex>>();
    if (use_lock_) {
      lock = std::make_unique<std::lock_guard<std::mutex>>(lock_for_queue_);
    }
    auto ret = std::move(header_);
    if (ret != nullptr) {
      header_ = std::move(ret->cdr_);
    }
    return ret == nullptr ? nullptr : std::move(ret->car_);
  }
  void send_ptr(std::unique_ptr<MessageType> obj) {
    std::unique_ptr<MessageType> cur;
    while ((cur = send_ptr(std::move(obj), std::chrono::milliseconds(1000))) !=
           nullptr) {
      // empty body
    }
  }

  /// @brief ms max wait time for in milliseconds
  ///
  /// if timeout, the
  /// original object will be returned, i.e. release the ownship,
  ///
  /// if no timeout, the object is consumed so that nullptr is returned.
  template <class Rep, class Period>
  std::unique_ptr<MessageType> send_ptr(
      std::unique_ptr<MessageType> obj,
      const std::chrono::duration<Rep, Period>& rel_time =
          std::chrono::duration<Rep, Period>::zero()) {
    if (wait_for_not_full(rel_time) == std::cv_status::timeout) {
      return obj;
    }
    // 注意要先构造对象，再拿锁。构造过程中，也许（不太可能）会调
    // 用 send/receieve 函数，但是不会递归拿锁。
    // 这里先不实现 bounded 的功能。因为没想好。
    // 设想是现实：先等待 not_full.wait ，释放锁，再构造对象，再等待
    // not_full.wait 。
    // 如果拿锁构造对象的话，也许会出现循环加锁？
    std::unique_ptr<typename ErlMsgBox<MessageType>::Cons> x =
        std::unique_ptr<Cons>(
            new Cons(std::unique_ptr<MessageType>(std::move(obj)), nullptr));
    send_elt(std::move(x));
    return nullptr;
  }

  // __attribute__ ((noinline)) is necessary, otherwise, '-Os' will result in
  // the following error terminate called after throwing an instance of
  // 'std::system_error'
  //   what():  Operation not permitted
  //  Aborted
  // but 'O2' and 'O0' has no such problem.
  // if add someting like std::cerr << "hello", no such error again, I guess
  // it might due the function inline.
 private:
  void
#if _WIN32
      __declspec(noinline)
#else
   __attribute__((noinline))
#endif
  send_elt(std::unique_ptr<typename ErlMsgBox<MessageType>::Cons> p_new_elt) {
    //
    // 这里可以优化，变成无锁算法？
    // 不行，因为也许多个线程同时添加元素。compare and set
    // 也许可以，但是没有继续研究了。
    auto lock = std::unique_ptr<std::lock_guard<std::mutex>>();
    if (use_lock_) {
      lock = std::make_unique<std::lock_guard<std::mutex>>(lock_for_queue_);
    }
    // 在结尾添加这个新构造的对象，tail_ 是一个 Cons_ 的指针。
    *tail_ = std::move(p_new_elt);
    tail_ = &((*tail_)->cdr_);
    size_++;
    not_empty.notify_one();
    return;
  }

 public:
  /// @brief recieve
  /// rel_time == 0 means do not wait.
  template <class Rep = uint64_t, class Period = std::milli>
  std::unique_ptr<MessageType> recv(
      const std::chrono::duration<Rep, Period>& rel_time =
          std::chrono::duration<Rep, Period>::zero()) {
    return recv(nullptr, rel_time);
  }
  /// @brief, cond is for testing
  /// 这个函数不会导致递归拿锁。
  /// rel_time == 0 means do not wait.
  ///
  template <class Rep, class Period>
  std::unique_ptr<MessageType> recv(
      const std::function<bool(const MessageType& p)>& cond,
      const std::chrono::duration<Rep, Period>& rel_time =
          std::chrono::duration<Rep, Period>::zero()) {
    /*
      THIS IS THE PREMATURE OPTIMIZATION, COMMENT IT OUT FOR YOUR REFERENCE.

      // first search without locking
      // tail_ might not be up to date, but it is always consistent,
      // i.e. tail_ == &last_elt.cdr; see send();

      for (p = &header_; p != tail_; p = &(*p)->cdr_) {
      if (!cond || cond(*((*p)->Car()))) {
        // is it necessary to lock?
        std::lock_guard<mutex_type> guard(lock_for_queue_);
        std::unique_ptr<MessageType> ret = std::move((*p)->Car());
        // deconstruct the element;
        RemoveElementFromList(p);
        return ret;
      }
    }
    */
    // search with lock
    auto timeout_time = std::chrono::system_clock::now() + rel_time;
    auto do_not_wait = rel_time == std::chrono::duration<Rep, Period>::zero();
    auto wait_for_ever = false;  // TODO not implmented.
    auto only_one_thread_can_enter_recv =
        std::unique_ptr<std::lock_guard<mutex_type>>();
    if (use_lock_) {
      only_one_thread_can_enter_recv =
          std::make_unique<std::lock_guard<mutex_type>>(lock_for_recv_);
    }
    // 注意这个锁放在了 while 循环外面。原来放在里面的，因为有
    // condition_variable not_empty。这个 wait_for_not_empty() 会在循
    // 环结尾，释放这个锁。
    auto guard = std::unique_ptr<std::unique_lock<mutex_type>>();
    if (use_lock_) {
      guard = std::make_unique<std::unique_lock<mutex_type>>(lock_for_queue_);
    }
    std::unique_ptr<Cons>* p = &header_;
    while (true) {
      // *p is likely to be null, i.e. that last element, waiting for
      // *some one invoking send() in the context of another thread.
      while (*p) {
        // *p is unique_ptr<Cons>, it means: `while it is not the
        // last element`.
        // if cond is empty function, assuming it is true for any
        // message.
        if (!cond || cond(*((*p)->Car()))) {
          std::unique_ptr<MessageType> ret = std::move((*p)->Car());
          RemoveElementFromList(p);
          size_--;
          if (use_lock_) {
            not_full.notify_one();
          }
          return ret;
        } else {
          p = &(*p)->cdr_;
        }
      }
      // 释放锁， 再次拿锁
      if (do_not_wait) {
        break;
      }
      auto cv_status =
          wait_for_not_empty(timeout_time, guard.get(), wait_for_ever);
      if (cv_status == std::cv_status::timeout) {
        break;  // 超时返回
      } else {
        /// p is restarted from where it is left;
        // p = tail_;
        continue;  // 继续，有可能得到满足条件的值
      }
    }
    return nullptr;
  }

 private:
  template <class Rep, class Period>
  std::cv_status wait_for_not_full(
      const std::chrono::duration<Rep, Period>& rel_time) {
    if (use_lock_) {
      auto pred = [this]() { return size_ < capacity_; };
      std::unique_lock<mutex_type> guard(lock_for_queue_);
      if (rel_time == std::chrono::duration<Rep, Period>::zero()) {
        not_full.wait(guard, pred);
      }
      not_full.wait_for(guard, rel_time, pred);
      return pred() ? std::cv_status::no_timeout : std::cv_status::timeout;
    }
    return full() ? std::cv_status::timeout : std::cv_status::no_timeout;
  }
  template <class Clock, class Duration>
  std::cv_status wait_for_not_empty(
      const std::chrono::time_point<Clock, Duration>& timeout_time,
      std::unique_lock<mutex_type>* guard, bool forever) {
    if (guard) {  // use lock
      auto pred = [this]() { return size_ > 0; };
      if (forever) {
        not_empty.wait(*guard, pred);
      } else {
        not_empty.wait_until(*guard, timeout_time, pred);
      }
      return pred() ? std::cv_status::no_timeout : std::cv_status::timeout;
    }
    return empty() ? std::cv_status::timeout : std::cv_status::no_timeout;
  }
  const size_t capacity_;
  const bool use_lock_;
  size_t size_;
  mutex_type lock_for_recv_;  // make sure only one c
  mutable mutex_type lock_for_queue_;
  std::condition_variable not_full;
  std::condition_variable not_empty;

 private:
  std::unique_ptr<Cons> header_;
  std::unique_ptr<Cons>* tail_;
  void RemoveElementFromList(std::unique_ptr<Cons>* p) {
    if (!((*p)->cdr_)) {
      // if it is the last element, update the tail_ pointer.
      assert(tail_ == &((*p)->cdr_));
      tail_ = p;
    }
    *p = std::move((*p)->cdr_);
  }
};
}  // namespace ai
}  // namespace vitis
