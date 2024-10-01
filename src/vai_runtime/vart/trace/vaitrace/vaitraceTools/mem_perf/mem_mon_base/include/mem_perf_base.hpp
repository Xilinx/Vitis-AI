/**
 * Copyright 2022-2023 Advanced Micro Devices Inc..
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

#ifndef MEM_PERF_H
#define MEM_PERF_H

#include <cstdint>
#include <cstddef>
#define DATA_BUFFER_SIZE (4 * 1024 * 1024)

struct record {
  double time;
  uint64_t data[10];
};

class MEM_PERF {
 public:
  uint32_t record_data_len = 0;
  int record_counter = 0;
  double act_period;
  struct record* data = NULL;

  MEM_PERF(){};
  virtual ~MEM_PERF(){};

  virtual void start_collect(double, void* data = NULL) = 0;
  virtual void stop_collect(void) = 0;
  virtual int pop_data(struct record* d) = 0;
  virtual double get_act_period(void) = 0;
  virtual int get_record_data_len(void) = 0;
};

#endif
