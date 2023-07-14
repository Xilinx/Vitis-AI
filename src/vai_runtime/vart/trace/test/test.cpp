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

#include <thread>
#include <vector>
#include <vitis/ai/trace.hpp>

#define N_LOOP (100000)
#define N_THREAD (5)

using namespace std;

void trace_thread(int id, size_t nloop) {
  cout << "id: " << id << " Thread start!" << endl;
  for (size_t i = 0; i < nloop; i++) {
    if (i % 20000 == 0) cout << "Thread " << id << ": Loop: " << (i/1000) << 'K' << endl;
    if (((id % 2) == 0) == true) {
      vitis::ai::trace::add_trace("Thread_1", i, 3.3, 'd', "hello");
    } else {
      vitis::ai::trace::add_trace("Thread_2", i * 8, 3.3, 'd', "world");
    }
  }
}

int main(int argc, char* argv[]) {
  size_t nth = N_THREAD;
  size_t nloop = N_LOOP;
  using Clock = std::chrono::steady_clock;

  if (argc == 3) {
    nth = atoi(argv[1]);
    nloop = atoi(argv[2]);
  };

  // vitis::ai::trace::initialize(4);
  vector<thread> ths;

  vitis::ai::trace::new_traceclass("Thread_1", {"morning", "noon", "night"});
  vitis::ai::trace::new_traceclass("Thread_2",
                                   {"spring", "summer", "autumn", "winter"});

  for (size_t i = 0; i < 20; i++) {
    vitis::ai::trace::add_info("Thread_1", "time", i++, "school", 987);
    vitis::ai::trace::add_info("Thread_2", "time", i++, "school", 876);
  }

  auto start = Clock::now();
  for (size_t i = 0; i < nth; i++) {
    auto t = thread(trace_thread, i, nloop);
    ths.push_back(move(t));
  }

  for (size_t i = 0; i < nth; i++) {
    ths[i].join();
  }

  auto end = Clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  auto entry_num = nth * nloop;
  cout << "Total trace entries: " << entry_num << endl;
  cout << "Elapsed time: " << elapsed_seconds.count() << " s\n";
  auto tpt = elapsed_seconds.count() * 1000 * 1000 / entry_num;
  cout << "Time per trace: " << tpt << " us" << endl;

  return 0;
}
