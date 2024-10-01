
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

#ifndef APM_H
#define APM_H

#include "apm_definitions.hpp"

#include <cstdint>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include "mem_perf_base.hpp"

struct xapm_param {
  uint32_t mode;
  uint32_t max_slots;
  uint32_t event_cnt;
  uint32_t event_log;
  uint32_t sampled_cnt;
  uint32_t num_counters;
  uint32_t metric_width;
  uint32_t sampled_width;
  uint32_t global_cnt_width;
  uint32_t scale_factor;
  uint32_t isr;
  bool is_32bit_filter;
};

class APM : public MEM_PERF {
 protected:
  const int APM_MAP_SIZE = 4096;
  int apm_fd;
  std::string uio_device_name;
  void* apm_baseaddr;
  void* apm_physaddr;
  xapm_param* params;

  void load_phys_addr(int device_id);

  inline uint32_t readreg(unsigned int offset) volatile {
    return *(uint32_t*)((uint8_t*)apm_baseaddr + offset);
  }
  inline void writereg(unsigned int offset, uint32_t data) volatile {
    *(uint32_t*)((uint8_t*)apm_baseaddr + offset) = data;
  }

  std::thread apm_thread;

 public:
  APM(int uio_device_id = 0);
  ~APM();

  bool collecting = false;

  int get_mode(void);
  void set_metrics_counter(uint8_t slot, uint8_t metrics, uint8_t counter);
  void reset_metrics_counters(void);
  uint32_t get_metrics_counter(uint8_t counter);
  void collect(uint32_t sample_interval_clks, bool reset = true);

  void start_collect(double sample_interval_clks, void* data = NULL) override;
  void stop_collect(void) override;
  int get_record_data_len(void) override;
  double get_act_period(void) override;
  int pop_data(struct record* data) override;
};

#endif
