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
#include <glog/logging.h>

#include <cstring>
#include <iostream>
#include <thread>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/thread_pool.hpp"
DEF_ENV_PARAM(NUM_OF_THREADS, "1")
DEF_ENV_PARAM(NUM_OF_REQUESTS, "625000")
using namespace std;

int foo(int a, int b) {
  if (0)
    LOG_IF(INFO, true) << "a " << a << " "  //
                       << "b " << b << " "  //
        ;
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  return a + b;
}
int main1(int argc, char* argv[]) {
  auto p = vitis::ai::ThreadPool::create(ENV_PARAM(NUM_OF_THREADS));
  int a = 1;
  int b = 1;
  auto all = vector<std::future<int>>();
  for (auto i = 0; i < ENV_PARAM(NUM_OF_REQUESTS); ++i) {
    p->async(foo, a + i, b);
    // all.emplace_back();
  }
  for (auto& f : all) {
    LOG_IF(INFO, true) << "hello " << f.get() << endl;
  }
  return foo(a, b);
}
void foo2(std::promise<int>* p, int a, int b) {
  LOG_IF(INFO, true) << "a " << a << " "  //
                     << "b " << b << " ";
  p->set_value(a + b);
  delete p;
}
int main2(int argc, char* argv[]) {
  auto p = vitis::ai::ThreadPool::create(ENV_PARAM(NUM_OF_THREADS));
  auto all = vector<std::future<int>>();
  int a = 1;
  int b = 1;
  for (int i = 0; i < 10; ++i) {
    std::promise<int> promise;
    all.emplace_back(promise.get_future());
    p->async(foo2, new std::promise<int>(std::move(promise)), a + i, b);
  }
  for (auto& f : all) {
    LOG_IF(INFO, true) << "hello " << f.get() << endl;
  }
  return foo(a, b);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "usage " << argv[0] << " <test-case>: main1 main2";
    return 0;
  }
  if (strcmp(argv[1], "main1") == 0) {
    main1(argc, argv);
  } else if (strcmp(argv[1], "main2") == 0) {
    main2(argc, argv);
  }
  return 0;
}
