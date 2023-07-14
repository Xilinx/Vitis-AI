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
#include <iostream>
#include <memory>

#include <vitis/ai/profiling.hpp>
#include <vitis/ai/shared_queue.hpp>

using namespace vitis::ai;

static constexpr int QUEUE_SIZE = 1000;

template <typename T>
class TestQueue : public ::testing::Test {
 public:
  TestQueue() {}
  virtual ~TestQueue() {}

  void SetUp() { queue = std::make_shared<SharedQueue<T>>(); }

  void TearDown() {}

  std::shared_ptr<SharedQueue<T>> queue;
};

class TestIntQueue : public TestQueue<int> {};

TEST_F(TestIntQueue, TestPushPop) {
  int in = 5;
  __TIC__(PUSH)
  queue->push(in);
  __TOC__(PUSH)
  int out;
  __TIC__(POP)
  queue->pop(out);
  __TOC__(POP)
  EXPECT_EQ(in, out);
}

TEST_F(TestIntQueue, TestPopWithCond) {
  for (int i = 0; i < QUEUE_SIZE; i++) {
    queue->push(i);
  }
  int out;
  std::function<bool(const int&)> cond = [](const int& i) { return i == 55; };
  queue->pop(out, cond);
  EXPECT_EQ(out, 55);
}
