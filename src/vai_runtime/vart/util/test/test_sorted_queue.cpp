/*
 * Copyright 2019 Xilinx Inc.
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
#include <iostream>
#include <memory>

#include <vitis/ai/profiling.hpp>
#include <vitis/ai/sorted_queue.hpp>

using namespace vitis::ai;

static constexpr int QUEUE_SIZE = 1000;

template <typename T>
class TestQueue : public ::testing::Test {
 public:
  TestQueue() {}
  virtual ~TestQueue() {}

  void SetUp() { queue = std::make_shared<SortedQueue<T>>(QUEUE_SIZE); }

  void TearDown() {}

  std::shared_ptr<SortedQueue<T>> queue;
};

class TestIntQueue : public TestQueue<int> {};

TEST_F(TestIntQueue, TestPushPop) {
  int in = 5;
  __TIC__(PUSH)
  bool ret = queue->push(in, std::chrono::milliseconds(1000));
  __TOC__(PUSH)
  EXPECT_TRUE(ret) << "should be able to push";
  int out;
  __TIC__(POP)
  ret = queue->pop(out, std::chrono::milliseconds(1000));
  __TOC__(POP)
  EXPECT_TRUE(ret) << "should be able to pop";
  EXPECT_EQ(in, out);
}

TEST_F(TestIntQueue, TestSort) {
  for (int i = 0; i < QUEUE_SIZE; i++) {
    queue->push(QUEUE_SIZE - 1 - i);
  }
  int out;
  for (int i = 0; i < QUEUE_SIZE; i++) {
    queue->pop(out);
    EXPECT_EQ(out, i);
  }
}
