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

#ifndef NMU_H
#define NMU_H

#include <array>
#include <cstdint>
#include <thread>
#include <vector>
#include <cstddef>
#define NMU_DATA_BUFFER_SIZE (4 * 1024 * 1024)
#define NOC_NMU_MAX_NUMBER 10

struct NOC_NMU_PERF {
  std::array<uint32_t, 2> noc_nmu_perf_addr;
};

struct record_nmu {
  double time;
  uint64_t data[16];
  //uint64_t data[NOC_NMU_MAX_NUMBER];
};

class NMU {
protected:
  uint64_t mon_freq;
  std::thread nmu_thread;

  int devmem_read(int addr);
  int devmem_write(int addr, int value);
  int lock(int addr);
  int unlock(int addr);
  void npi_set_tbs(int addr, double period);
  double npi_get_tbs(int addr);
  void npi_setup(int addr, double period);

  void noc_nmu_enable(int addr);
  void noc_nmu_disable(int addr);
  void noc_nmu_setup(int addr);
  void create_nmu_mon_addr(int *addr_base);

public:
  NMU(uint64_t freq, int nmu_addr[], int npi_addr, int len);
  ~NMU();
  int nmu_number;
  int *nmu_physics_addr = NULL;
  int npi_physics_addr = 0;
  struct NOC_NMU_PERF *noc_nmu_perf = NULL;
  uint32_t nsu_indexs = 0;
  bool collecting = false;
  uint32_t record_data_len = 0;
  int record_counter = 0;
  double act_period;
  struct record_nmu* data = NULL;

  uint64_t noc_nmu_sample(int addr);
  int get_record_data_len(void) ;
  double get_act_period(void) ;
  int pop_data(struct record_nmu *d) ;
  void start_collect(double interval_sec, void *_data = NULL) ;
  void stop_collect(void);
};

#endif
