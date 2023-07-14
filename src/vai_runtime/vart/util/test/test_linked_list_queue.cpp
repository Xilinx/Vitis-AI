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

#include <vitis/ai/linked_list_queue.hpp>
#include <vitis/ai/profiling.hpp>

using namespace vitis::ai;

static constexpr int QUEUE_SIZE = 1000;

template <typename T>
class TestQueue : public ::testing::Test {
 public:
  TestQueue() {}
  virtual ~TestQueue() {}

  void SetUp() { queue = std::make_shared<LinkedListQueue<T>>(); }

  void TearDown() {}

  std::shared_ptr<LinkedListQueue<T>> queue;
};

class TestIntQueue : public TestQueue<int> {};

TEST_F(TestIntQueue, TestPushPop) {
  int in = 5;
  __TIC__(PUSH)
  queue->send(in);
  __TOC__(PUSH)
  __TIC__(POP)
  std::unique_ptr<int> out = queue->receive();
  __TOC__(POP)
  EXPECT_EQ(in, *out);
}

TEST_F(TestIntQueue, TestPopWithCond) {
  for (int i = 0; i < QUEUE_SIZE; i++) {
    queue->send(i);
  }
  std::function<bool(const int&)> cond = [](const int& i) { return i == 55; };
  std::unique_ptr<int> out = queue->receive(cond);
  EXPECT_EQ(*out, 55);
}
