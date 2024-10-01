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

#ifndef NOC_H
#define NOC_H

#include <cstdint>
#include <thread>
#include <array>
#include <vector>
#include "mem_perf_base.hpp"

struct NOC_NSU_PERF {
  int op_type;  // read = 0; write = 1
  std::array<uint32_t, 4> noc_nsu_perf_addr;
};

class NOC : public MEM_PERF {
 protected:
  uint64_t mon_freq;
  int ddrmc_number;
  std::thread noc_thread;

  int devmem_read(int addr);
  int devmem_write(int addr, int value);
  int mon_lock(int addr);
  int mon_unlock(int addr);
  void mon_set_tbs(int addr, double period);
  double mon_get_tbs(int addr);
  void mon_setup(int addr, double period);

  void nsu_enable(int addr);
  void nsu_disable(int addr);
  void nsu_setup(int addr, int op_type);
  void create_nsu_base_addr(int addr_base);

 public:
  NOC(int freq, int addr[], int len);
  ~NOC();
  int* ddrmc_physics_addr = NULL;
  struct NOC_NSU_PERF* noc_nsu_perf = NULL;
  uint32_t nsu_indexs = 0;
  bool collecting = false;
  int nsu_sample(int addr);
  int get_record_data_len(void) override;
  double get_act_period(void) override;
  int pop_data(struct record* d) override;
  void start_collect(double interval_sec, void* _data = NULL) override;
  void stop_collect(void) override;
};

#endif
