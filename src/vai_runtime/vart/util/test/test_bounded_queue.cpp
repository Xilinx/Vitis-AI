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

#include <vitis/ai/bounded_queue.hpp>
#include <vitis/ai/profiling.hpp>

using namespace vitis::ai;

static constexpr int QUEUE_SIZE = 1000;

template <typename T>
class TestQueue : public ::testing::Test {
 public:
  TestQueue() {}
  virtual ~TestQueue() {}

  void SetUp() { queue = std::make_shared<BoundedQueue<T>>(QUEUE_SIZE); }

  void TearDown() {}

  void enqueue(int num, std::function<T(int)> generator) {
    for (int i = 0; i < num; i++) {
      while (!queue->push(generator(i), std::chrono::milliseconds(1000)))
        ;
    }
  }

  void dequeue(int num, std::function<void(T)> validator) {
    T out;
    for (int i = 0; i < num; i++) {
      while (!queue->pop(out, std::chrono::milliseconds(1000)))
        ;
      validator(out);
    }
  }

  std::shared_ptr<BoundedQueue<T>> queue;
};

class TestIntQueue : public TestQueue<int> {};

TEST_F(TestIntQueue, TestPushPop) {
  int in = 5;
  __TIC__(PUSH)
  bool ret = queue->push(in, std::chrono::milliseconds(1000));
  __TOC__(PUSH)
  EXPECT_TRUE(ret) << "should be able to push";
  int out;
  __TIC__(TOP)
  ret = queue->top(out, std::chrono::milliseconds(1000));
  __TOC__(TOP)
  EXPECT_TRUE(ret) << "should be able to top";
  EXPECT_EQ(in, out);
  out = -1;
  __TIC__(POP)
  ret = queue->pop(out, std::chrono::milliseconds(1000));
  __TOC__(POP)
  EXPECT_TRUE(ret) << "should be able to pop";
  EXPECT_EQ(in, out);
}

TEST_F(TestIntQueue, TestPopWithCond) {
  for (int i = 0; i < QUEUE_SIZE; i++) {
    queue->push(i);
  }
  int out;
  std::function<bool(const int&)> cond = [](const int& i) { return i == 55; };
  bool ret = queue->pop(out, cond, std::chrono::milliseconds(1000));
  EXPECT_TRUE(ret) << "should be able to pop";
  EXPECT_EQ(out, 55);
  ret = queue->pop(out, cond, std::chrono::milliseconds(1000));
  EXPECT_FALSE(ret) << "should failto pop";
}

TEST_F(TestIntQueue, TestPopEmptyQueue) {
  int out;
  bool ret = queue->pop(out, std::chrono::milliseconds(1000));
  EXPECT_FALSE(ret) << "should fail to pop";
}

TEST_F(TestIntQueue, TestPushFullQueue) {
  for (int i = 0; i < QUEUE_SIZE; i++) {
    queue->push(i);
  }
  int in = 5;
  bool ret = queue->push(in, std::chrono::milliseconds(1000));
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
  // check that the sum of all popped values is correct.
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
  for (int i = 0; i < (int)threads.size(); i++) {
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

class TestSharedPointerQueue : public TestQueue<std::shared_ptr<int>> {};

TEST_F(TestSharedPointerQueue, TestPushPop) {
  std::shared_ptr<int> p = std::make_shared<int>(10);
  bool ret = queue->push(p, std::chrono::milliseconds(1000));
  EXPECT_TRUE(ret) << "should be able to push";
  std::shared_ptr<int> out;
  ret = queue->top(out, std::chrono::milliseconds(1000));
  EXPECT_TRUE(ret) << "should be able to top";
  EXPECT_EQ(p, out);
  ret = queue->pop(out, std::chrono::milliseconds(1000));
  EXPECT_TRUE(ret) << "should be able to pop";
  EXPECT_EQ(p, out);
}

struct A {
  int x;
  std::string y;
};
class TestStructQueue : public TestQueue<A> {};

TEST_F(TestStructQueue, TestPushPop) {
  A a{5, "test"};
  bool ret = queue->push(a, std::chrono::milliseconds(1000));
  EXPECT_TRUE(ret) << "should be able to push";
  A out;
  ret = queue->top(out, std::chrono::milliseconds(1000));
  EXPECT_TRUE(ret) << "should be able to top";
  EXPECT_EQ(a.x, out.x);
  EXPECT_EQ(a.y, out.y);
  ret = queue->pop(out, std::chrono::milliseconds(1000));
  EXPECT_TRUE(ret) << "should be able to pop";
  EXPECT_EQ(a.x, out.x);
  EXPECT_EQ(a.y, out.y);
}
