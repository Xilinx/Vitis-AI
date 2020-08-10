/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/lib/core/threadpool.h"

#include <atomic>

#include "absl/synchronization/barrier.h"
#include "absl/synchronization/blocking_counter.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace thread {

static const int kNumThreads = 30;

TEST(ThreadPool, Empty) {
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    ThreadPool pool(Env::Default(), "test", num_threads);
  }
}

TEST(ThreadPool, DoWork) {
  Context outer_context(ContextKind::kThread);
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    const int kWorkItems = 15;
    std::atomic<bool> work[kWorkItems];
    for (int i = 0; i < kWorkItems; i++) {
      work[i] = false;
    }
    {
      ThreadPool pool(Env::Default(), "test", num_threads);
      for (int i = 0; i < kWorkItems; i++) {
        pool.Schedule([&outer_context, &work, i]() {
          Context inner_context(ContextKind::kThread);
          ASSERT_EQ(outer_context, inner_context);
          ASSERT_FALSE(work[i].exchange(true));
        });
      }
    }
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_TRUE(work[i]);
    }
  }
}

void RunSharding(int64 block_size, int64 total, ThreadPool* threads) {
  mutex mu;
  int64 num_shards = 0;
  int64 num_done_work = 0;
  std::vector<std::atomic<bool>> work(total);
  for (int i = 0; i < total; i++) {
    work[i] = false;
  }
  threads->TransformRangeConcurrently(
      block_size, total,
      [=, &mu, &num_shards, &num_done_work, &work](int64 start, int64 end) {
        VLOG(1) << "Shard [" << start << "," << end << ")";
        EXPECT_GE(start, 0);
        EXPECT_LE(end, total);
        mutex_lock l(mu);
        ++num_shards;
        for (; start < end; ++start) {
          EXPECT_FALSE(work[start].exchange(true));  // No duplicate
          ++num_done_work;
        }
      });
  LOG(INFO) << block_size << " " << total;
  EXPECT_EQ(num_done_work, total);
  for (int i = 0; i < total; i++) {
    ASSERT_TRUE(work[i]);
  }
  const int64 num_workers = (total + block_size - 1) / block_size;
  if (num_workers < threads->NumThreads()) {
    // If the intention is to limit the parallelism explicitly, we'd
    // better honor it. Ideally, even if per_thread_max_parallelism >
    // num_workers, we should expect that Shard() implementation do
    // not over-shard. Unfortunately, ThreadPoolDevice::parallelFor
    // tends to over-shard.
    EXPECT_LE(num_shards, 1 + num_workers);
  }
}

// Adapted from work_sharder_test.cc
TEST(SparseUtilsTest, TransformRangeConcurrently) {
  ThreadPool threads(Env::Default(), "test", 16);
  for (auto block_size : {1, 7, 10, 64, 100, 256, 1000, 9999}) {
    for (auto diff : {0, 1, 11, 102, 1003, 10005, 1000007}) {
      const int64 total = block_size + diff;
      RunSharding(block_size, total, &threads);
    }
  }
}

TEST(SparseUtilsTest, NumShardsUsedByTransformRangeConcurrently) {
  ThreadPool threads(Env::Default(), "test", 16);
  EXPECT_EQ(1, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 3 /* total */));
  EXPECT_EQ(2, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 4 /* total */));
  EXPECT_EQ(2, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 5 /* total */));
  EXPECT_EQ(2, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 6 /* total */));
  EXPECT_EQ(3, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 7 /* total */));
  EXPECT_EQ(7, threads.NumShardsUsedByTransformRangeConcurrently(
                   1 /* block_size */, 7 /* total */));
  EXPECT_EQ(1, threads.NumShardsUsedByTransformRangeConcurrently(
                   0 /* block_size */, 7 /* total */));
}

TEST(ThreadPool, ParallelFor) {
  Context outer_context(ContextKind::kThread);
  // Make ParallelFor use as many threads as possible.
  int64 kHugeCost = 1 << 30;
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    const int kWorkItems = 15;
    std::atomic<bool> work[kWorkItems];
    ThreadPool pool(Env::Default(), "test", num_threads);
    for (int i = 0; i < kWorkItems; i++) {
      work[i] = false;
    }
    pool.ParallelFor(kWorkItems, kHugeCost,
                     [&outer_context, &work](int64 begin, int64 end) {
                       Context inner_context(ContextKind::kThread);
                       ASSERT_EQ(outer_context, inner_context);
                       for (int64 i = begin; i < end; ++i) {
                         ASSERT_FALSE(work[i].exchange(true));
                       }
                     });
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_TRUE(work[i]);
    }
  }
}

TEST(ThreadPool, ParallelForWithWorkerId) {
  // Make ParallelForWithWorkerId use as many threads as possible.
  int64 kHugeCost = 1 << 30;
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    const int kWorkItems = 15;
    std::atomic<bool> work[kWorkItems];
    ThreadPool pool(Env::Default(), "test", num_threads);
    for (int i = 0; i < kWorkItems; i++) {
      work[i] = false;
    }
    std::atomic<bool> threads_running[kNumThreads + 1];
    for (int i = 0; i < num_threads + 1; i++) {
      threads_running[i] = false;
    }
    pool.ParallelForWithWorkerId(
        kWorkItems, kHugeCost,
        [&threads_running, &work](int64 begin, int64 end, int64 id) {
          // Store true for the current thread, and assert that another thread
          // is not running with the same id.
          ASSERT_LE(0, id);
          ASSERT_LE(id, kNumThreads);
          ASSERT_FALSE(threads_running[id].exchange(true));
          for (int64 i = begin; i < end; ++i) {
            ASSERT_FALSE(work[i].exchange(true));
          }
          ASSERT_TRUE(threads_running[id].exchange(false));
          threads_running[id] = false;
        });
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_TRUE(work[i]);
    }
    for (int i = 0; i < num_threads + 1; i++) {
      ASSERT_FALSE(threads_running[i]);
    }
  }
}

TEST(ThreadPool, Parallelism) {
  // Test that if we have N threads and schedule N tasks,
  // all tasks will be scheduled at the same time.
  // Failure mode for this test will be episodic timeouts (does not terminate).
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  for (int iter = 0; iter < 2000; iter++) {
    absl::Barrier barrier(kNumThreads);
    absl::BlockingCounter counter(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
      pool.Schedule([&]() {
        barrier.Block();
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }
}

static void BM_Sequential(int iters) {
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  // Decrement count sequentially until 0.
  int count = iters;
  mutex done_lock;
  bool done_flag = false;
  std::function<void()> work = [&pool, &count, &done_lock, &done_flag,
                                &work]() {
    if (count--) {
      pool.Schedule(work);
    } else {
      mutex_lock l(done_lock);
      done_flag = true;
    }
  };
  work();
  mutex_lock l(done_lock);
  done_lock.Await(Condition(&done_flag));
}
BENCHMARK(BM_Sequential);

static void BM_Parallel(int iters) {
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  // Decrement count concurrently until 0.
  std::atomic_int_fast32_t count(iters);
  mutex done_lock;
  bool done_flag = false;
  for (int i = 0; i < iters; ++i) {
    pool.Schedule([&count, &done_lock, &done_flag]() {
      if (count.fetch_sub(1) == 1) {
        mutex_lock l(done_lock);
        done_flag = true;
      }
    });
  }
  mutex_lock l(done_lock);
  done_lock.Await(Condition(&done_flag));
}
BENCHMARK(BM_Parallel);

static void BM_ParallelFor(int iters, int total, int cost_per_unit) {
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  // Decrement count concurrently until 0.
  std::atomic_int_fast32_t count(iters);
  mutex done_lock;
  bool done_flag = false;
  for (int i = 0; i < iters; ++i) {
    pool.ParallelFor(total, cost_per_unit,
                     [&count, &done_lock, &done_flag](int64 begin, int64 end) {
                       for (int64 i = begin; i < end; ++i) {
                         if (count.fetch_sub(1) == 1) {
                           mutex_lock l(done_lock);
                           done_flag = true;
                         }
                       }
                     });
  }
  mutex_lock l(done_lock);
  done_lock.Await(Condition(&done_flag));
}
BENCHMARK(BM_ParallelFor)
    ->ArgPair(1 << 10, 1)
    ->ArgPair(1 << 20, 1)
    ->ArgPair(1 << 10, 1 << 10)
    ->ArgPair(1 << 20, 1 << 10)
    ->ArgPair(1 << 10, 1 << 20)
    ->ArgPair(1 << 20, 1 << 20)
    ->ArgPair(1 << 10, 1 << 30)
    ->ArgPair(1 << 20, 1 << 30);

}  // namespace thread
}  // namespace tensorflow
