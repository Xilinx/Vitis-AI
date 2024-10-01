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
#include <gtest/gtest.h>
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>

#include <vitis/ai/c++14.hpp>
#include <vitis/ai/nocopy_bounded_queue.hpp>
#include <vitis/ai/profiling.hpp>

using namespace vitis::ai;

static constexpr int QUEUE_SIZE = 1000;

template <typename T>
class TestQueue : public ::testing::Test {
 public:
  TestQueue() {}
  virtual ~TestQueue() {}

  void SetUp() { queue = std::make_shared<NoCopyBoundedQueue<T>>(QUEUE_SIZE); }

  void TearDown() {}

  void enqueue(int num, std::function<T(int)> generator) {
    for (int i = 0; i < num; i++) {
      std::unique_ptr<T> in = std::unique_ptr<T>(new T);
      *in = generator(i);
      while (!queue->push(std::move(in), std::chrono::milliseconds(1000)))
        ;
    }
  }

  void dequeue(int num, std::function<void(T)> validator) {
    std::unique_ptr<T> out;
    for (int i = 0; i < num; i++) {
      do {
        out = queue->pop(std::chrono::milliseconds(1000));
      } while (!out);
      validator(*out);
    }
  }

  std::shared_ptr<NoCopyBoundedQueue<T>> queue;
};

class TestIntQueue : public TestQueue<int> {};

TEST_F(TestIntQueue, TestPushPop) {
  int expected_out = 5;
  std::unique_ptr<int> in = std::make_unique<int>(1);
  *in = expected_out;
  __TIC__(PUSH)
  bool ret = queue->push(std::move(in), std::chrono::milliseconds(1000));
  __TOC__(PUSH)
  EXPECT_TRUE(ret) << "should be able to push";
  __TIC__(TOP)
  int* outp = queue->top(std::chrono::milliseconds(1000));
  __TOC__(TOP)
  EXPECT_EQ(expected_out, *outp);
  __TIC__(POP)
  std::unique_ptr<int> out = queue->pop(std::chrono::milliseconds(1000));
  __TOC__(POP)
  EXPECT_EQ(expected_out, *out);
}

TEST_F(TestIntQueue, TestPopWithCond) {
  for (int i = 0; i < QUEUE_SIZE; i++) {
    std::unique_ptr<int> in = std::make_unique<int>(1);
    *in = i;
    queue->push(std::move(in));
  }
  std::function<bool(const std::unique_ptr<int>&)> cond =
      [](const std::unique_ptr<int>& i) { return *i == 55; };
  std::unique_ptr<int> out = queue->pop(cond, std::chrono::milliseconds(1000));
  EXPECT_EQ(*out, 55);
  out = queue->pop(cond, std::chrono::milliseconds(1000));
  EXPECT_FALSE(out) << "should fail to pop";
}

TEST_F(TestIntQueue, TestPopEmptyQueue) {
  std::unique_ptr<int> out = queue->pop(std::chrono::milliseconds(1000));
  EXPECT_FALSE(out) << "should fail to pop";
}

TEST_F(TestIntQueue, TestPushFullQueue) {
  for (int i = 0; i < QUEUE_SIZE; i++) {
    std::unique_ptr<int> in = std::make_unique<int>(1);
    *in = i;
    queue->push(std::move(in));
  }
  std::unique_ptr<int> in = std::make_unique<int>(1);
  *in = 5;
  bool ret = queue->push(std::move(in), std::chrono::milliseconds(1000));
  EXPECT_FALSE(ret) << "should fail to push";
}

TEST_F(TestIntQueue, TestMultiThread) {
  int expected_out = 0;
  std::function<int(int)> generator = [](int in) { return in; };
  std::function<void(int)> validator = [&expected_out](int out) {
    EXPECT_EQ(expected_out, out);
    expected_out++;
  };
  std::thread t1(&TestIntQueue_TestMultiThread_Test::enqueue, this, QUEUE_SIZE,
                 generator);
  std::thread t2(&TestIntQueue_TestMultiThread_Test::dequeue, this, QUEUE_SIZE,
                 validator);
  t1.join();
  t2.join();
}

TEST_F(TestIntQueue, TestMultiThread_10) {
  std::atomic<int> sum(0);
  std::function<int(int)> generator = [](int in) { return in; };
  std::function<void(int)> validator = [&sum](int out) { sum += out; };
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back(&TestIntQueue_TestMultiThread_10_Test::enqueue, this,
                         QUEUE_SIZE, generator);
  }
  for (int i = 0; i < 10; i++) {
    threads.emplace_back(&TestIntQueue_TestMultiThread_10_Test::dequeue, this,
                         QUEUE_SIZE, validator);
  }
  for (int i = 0; i < threads.size(); i++) {
    if (threads[i].joinable()) {
      threads[i].join();
    }
  }
  int expected_sum = 0;
  for (int i = 0; i < QUEUE_SIZE; i++) {
    expected_sum += 10 * i;
  }
  EXPECT_EQ(sum, expected_sum);
}

struct A {
  int x;
  std::string y;
};
class TestStructQueue : public TestQueue<A> {};

TEST_F(TestStructQueue, TestPushPop) {
  std::unique_ptr<A> in = std::make_unique<A>();
  A a{5, "test"};
  *in = a;
  bool ret = queue->push(std::move(in), std::chrono::milliseconds(1000));
  EXPECT_TRUE(ret) << "should be able to push";
  A* out = queue->top(std::chrono::milliseconds(1000));
  EXPECT_EQ(a.x, out->x);
  EXPECT_EQ(a.y, out->y);
  std::unique_ptr<A> outp = queue->pop(std::chrono::milliseconds(1000));
  EXPECT_EQ(a.x, outp->x);
  EXPECT_EQ(a.y, outp->y);
}
