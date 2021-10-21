// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_DATA_PARALLEL_H_
#define PIK_DATA_PARALLEL_H_

// Portable, low-overhead C++11 ThreadPool alternative to OpenMP for
// data-parallel computations.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>  // max
#include <atomic>
#include <condition_variable>  //NOLINT
#include <cstdlib>
#include <mutex>   //NOLINT
#include <thread>  //NOLINT
#include <vector>

#include "pik/bits.h"
#include "pik/status.h"

namespace pik {

// Scalable, lower-overhead thread pool, especially suitable for data-parallel
// computations in the fork-join model, where clients need to know when all
// tasks have completed.
//
// Thread pools usually store small numbers of heterogeneous tasks in a queue.
// When tasks are identical or differ only by an integer input parameter, it is
// much faster to store just one function of an integer parameter and call it
// for each value. Conventional vector-of-tasks can be run in parallel using a
// lambda function adapter that simply calls task_funcs[task].
//
// This thread pool can efficiently load-balance millions of tasks using an
// atomic counter, thus avoiding per-task virtual or system calls. With 48
// hyperthreads and 1M tasks that add to an atomic counter, overall runtime is
// 10-20x higher when using std::async, and ~200x for a queue-based ThreadPool.
//
// Usage:
//   ThreadPool pool;
//   pool.Run(0, 1000000, [](int task, int thread) { Func1(task, thread); });
//
// When Run returns, all of its tasks have finished. The destructor waits until
// all worker threads have exited cleanly. "thread" is useful for accessing
// thread-local data, typically a pre-allocated array of kMaxThreads
// cache-aligned elements.
class ThreadPool {
 public:
  // For per-thread arrays. Can increase if needed.
  static constexpr int kMaxThreads = 256;

  // Starts the given number of worker threads and blocks until they are ready.
  // "num_worker_threads" defaults to one per hyperthread. If zero, all tasks
  // run on the main thread.
  explicit ThreadPool(
      const int num_worker_threads = std::thread::hardware_concurrency());

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator&(const ThreadPool&) = delete;

  // Waits for all threads to exit.
  ~ThreadPool();

  // Returns number of worker threads created (some may be sleeping and never
  // wake up in time to participate in Run). Useful for characterizing
  // performance; 0 means "run on main thread".
  size_t NumWorkerThreads() const { return num_worker_threads_; }

  // Returns maximum number of main/worker threads that may call Func. Useful
  // for allocating per-thread storage.
  size_t NumThreads() const { return num_threads_; }

  // Runs func(task, thread) on worker thread(s) for every task in [begin, end).
  // "thread" is 0 if NumThreads() == 0, otherwise [0, NumThreads()).
  // Not thread-safe - no two calls to Run may overlap.
  // Subsequent calls will reuse the same threads.
  //
  // Precondition: 0 <= begin <= end.
  template <class Func>
  void Run(const int begin, const int end, const Func& func,
           const char* caller = "") {
    //    printf("ThreadPool::Run: %s\n", caller);
    PIK_ASSERT(0 <= begin && begin <= end);
    if (begin == end) {
      return;
    }

    if (num_worker_threads_ == 0) {
      const int thread = 0;
      for (int task = begin; task < end; ++task) {
        func(task, thread);
      }
      return;
    }

    if (depth_.fetch_add(1, std::memory_order_acq_rel) != 0) {
      PIK_ASSERT(false);  // Must not re-enter.
    }

    const WorkerCommand worker_command = (WorkerCommand(end) << 32) + begin;
    // Ensure the inputs do not result in a reserved command.
    PIK_ASSERT(worker_command != kWorkerWait);
    PIK_ASSERT(worker_command != kWorkerOnce);
    PIK_ASSERT(worker_command != kWorkerExit);

    func_ = &CallClosure<Func>;
    arg_ = &func;
    num_reserved_.store(0, std::memory_order_relaxed);

    StartWorkers(worker_command);
    WorkersReadyBarrier();

    if (depth_.fetch_add(-1, std::memory_order_acq_rel) != 1) {
      PIK_ASSERT(false);
    }
  }

  // Runs func(thread, thread) on all thread(s) that may participate in Run.
  // If NumThreads() == 0, runs on the main thread with thread == 0, otherwise
  // concurrently called by each worker thread in [0, NumThreads()).
  template <class Func>
  void RunOnEachThread(const Func& func) {
    if (num_worker_threads_ == 0) {
      const int thread = 0;
      func(thread, thread);
      return;
    }

    func_ = reinterpret_cast<TypeErasedFunc>(&CallClosure<Func>);
    arg_ = &func;
    StartWorkers(kWorkerOnce);
    WorkersReadyBarrier();
  }

 private:
  // After construction and between calls to Run, workers are "ready", i.e.
  // waiting on worker_start_cv_. They are "started" by sending a "command"
  // and notifying all worker_start_cv_ waiters. (That is why all workers
  // must be ready/waiting - otherwise, the notification will not reach all of
  // them and the main thread waits in vain for them to report readiness.)
  using WorkerCommand = uint64_t;

  // Special values; all others encode the begin/end parameters.
  static constexpr WorkerCommand kWorkerWait = ~0ULL;
  static constexpr WorkerCommand kWorkerOnce = ~1ULL;
  static constexpr WorkerCommand kWorkerExit = ~2ULL;

  // Calls f(task, thread). Used for type erasure of Func arguments. The
  // signature must match TypeErasedFunc, hence a const void* argument.
  template <class Closure>
  static void CallClosure(const void* f, const int task, const int thread) {
    (*reinterpret_cast<const Closure*>(f))(task, thread);
  }

  void WorkersReadyBarrier() {
    std::unique_lock<std::mutex> lock(mutex_);
    // Typically only a single iteration.
    while (workers_ready_ != threads_.size()) {
      workers_ready_cv_.wait(lock);
    }
    workers_ready_ = 0;

    // Safely handle spurious worker wakeups.
    worker_start_command_ = kWorkerWait;
  }

  // Precondition: all workers are ready.
  void StartWorkers(const WorkerCommand worker_command) {
    mutex_.lock();
    worker_start_command_ = worker_command;
    // Workers will need this lock, so release it before they wake up.
    mutex_.unlock();
    worker_start_cv_.notify_all();
  }

  // Attempts to reserve and perform some work from the global range of tasks,
  // which is encoded within "command". Returns after all tasks are reserved.
  static void RunRange(ThreadPool* self, const WorkerCommand command,
                       const int thread) {
    const int begin = command & 0xFFFFFFFF;
    const int end = command >> 32;
    const int num_tasks = end - begin;
    const int num_worker_threads = static_cast<int>(self->num_worker_threads_);

    // OpenMP introduced several "schedule" strategies:
    // "single" (static assignment of exactly one chunk per thread): slower.
    // "dynamic" (allocates k tasks at a time): competitive for well-chosen k.
    // "guided" (allocates k tasks, decreases k): computing k = remaining/n
    //   is faster than halving k each iteration. We prefer this strategy
    //   because it avoids user-specified parameters.

    for (;;) {
      // guided
      const int num_reserved =
          self->num_reserved_.load(std::memory_order_relaxed);
      const int num_remaining = num_tasks - num_reserved;
      const int my_size = std::max(num_remaining / (num_worker_threads * 4), 1);
      const int my_begin = begin + self->num_reserved_.fetch_add(
                                       my_size, std::memory_order_relaxed);
      const int my_end = std::min(my_begin + my_size, begin + num_tasks);
      // Another thread already reserved the last task.
      if (my_begin >= my_end) {
        break;
      }
      for (int task = my_begin; task < my_end; ++task) {
        self->func_(self->arg_, task, thread);
      }
    }
  }

  // What task to run on a worker thread. Points to code generated via
  // CallClosure. Arguments are arg_ (points to the lambda), task, thread.
  using TypeErasedFunc = void (*)(const void*, int, int);

  static void ThreadFunc(ThreadPool* self, const int thread);

  // Unmodified after ctor, but cannot be const because we call thread::join().
  std::vector<std::thread> threads_;

  const size_t num_worker_threads_;  // == threads_.size()
  const size_t num_threads_;

  std::atomic<int> depth_{0};  // detects if Run is re-entered (not supported).

  std::mutex mutex_;  // guards both cv and their variables.
  std::condition_variable workers_ready_cv_;
  size_t workers_ready_ = 0;
  std::condition_variable worker_start_cv_;
  WorkerCommand worker_start_command_;

  // Written by main thread, read by workers (after mutex lock/unlock).
  TypeErasedFunc func_;
  const void* arg_;

  // Updated by workers; alignment/padding avoids false sharing.
  alignas(64) std::atomic<int> num_reserved_{0};
  int padding[15];
};

// Wrappers to enable ThreadPool* == nullptr (cheaper than constructing a
// ThreadPool(0)). Do not call pool->* directly.

static inline size_t NumWorkerThreads(ThreadPool* pool) {
  return pool == nullptr ? 0 : pool->NumWorkerThreads();
}

static inline size_t NumThreads(ThreadPool* pool) {
  return pool == nullptr ? 1 : pool->NumThreads();
}

template <class Func>
void RunOnPool(ThreadPool* pool, const int begin, const int end,
               const Func& func, const char* caller = "") {
  if (pool == nullptr) {
    const int thread = 0;
    for (int task = begin; task < end; ++task) {
      func(task, thread);
    }
    return;
  }
  pool->Run(begin, end, func, caller);
}

template <class Func>
void RunOnEachThread(ThreadPool* pool, const Func& func) {
  if (pool == nullptr) {
    const int thread = 0;
    func(thread, thread);
    return;
  }
  pool->RunOnEachThread(func);
}

// Adapters for zero-cost switching between ThreadPool and non-threaded loop.

struct ExecutorLoop {
  // Lambda must accept int task = [begin, end) and int thread = 0 arguments.
  template <class Lambda>
  void Run(const int begin, const int end, const Lambda& lambda,
           const char* caller = "") const {
    for (int i = begin; i < end; ++i) {
      lambda(i, 0);
    }
  }
};

struct ExecutorPool {
  explicit ExecutorPool(ThreadPool* pool) : pool(pool) {}

  // Lambda must accept int task = [begin, end) and int thread arguments.
  template <class Lambda>
  void Run(const int begin, const int end, const Lambda& lambda,
           const char* caller) const {
    RunOnPool(pool, begin, end, lambda, caller);
  }

  ThreadPool* pool;  // not owned
};

// Accelerates multiple unsigned 32-bit divisions with the same divisor by
// precomputing a multiplier. This is useful for splitting a contiguous range of
// indices (the task index) into 2D indices. Exhaustively tested on dividends
// up to 4M with non-power of two divisors up to 2K.
class Divider {
 public:
  // "d" is the divisor (what to divide by).
  Divider(const uint32_t d) : shift_(FloorLog2Nonzero(d)) {
    // Power of two divisors (including 1) are not supported because it is more
    // efficient to special-case them at a higher level.
    PIK_ASSERT((d & (d - 1)) != 0);

    // ceil_log2 = floor_log2 + 1 because we ruled out powers of two above.
    const uint64_t next_pow2 = 1ULL << (shift_ + 1);

    mul_ = ((next_pow2 - d) << 32) / d + 1;
  }

  // "n" is the numerator (what is being divided).
  inline uint32_t operator()(const uint32_t n) const {
    // Algorithm from "Division by Invariant Integers using Multiplication".
    // Its "sh1" is hardcoded to 1 because we don't need to handle d=1.
    const uint32_t hi = (uint64_t(mul_) * n) >> 32;
    return (hi + ((n - hi) >> 1)) >> shift_;
  }

 private:
  uint32_t mul_;
  const int shift_;
};

}  // namespace pik

#endif  // PIK_DATA_PARALLEL_H_
