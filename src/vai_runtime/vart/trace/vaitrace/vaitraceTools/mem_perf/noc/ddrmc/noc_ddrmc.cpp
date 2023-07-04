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

#include <stdio.h>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include "noc_ddrmc.hpp"
const int NOC_NSU_PERF_SIZE = 1024 * 1024;

NOC::NOC(int freq, int addr[], int len) {
  mon_freq = freq * 1000 * 1000;
  ddrmc_physics_addr = addr;
  ddrmc_number = len;
  record_counter = 0;
}

NOC::~NOC() {
  if (data != NULL) free(data);
  if (noc_nsu_perf != NULL) free(noc_nsu_perf);

  ddrmc_physics_addr = NULL;
}

int NOC::devmem_read(int addr) {
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    printf("Err: can not open /dev/mem \n");
    return -1;
  }
  unsigned page_size = getpagesize();
  unsigned mapped_size = page_size;
  unsigned offset_in_page = (unsigned)addr & (page_size - 1);
  if (offset_in_page + 32 > page_size) {
    mapped_size *= 2;
  }
  void* map_ret = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                       fd, addr & ~(page_size - 1));
  if (map_ret == MAP_FAILED) {
    printf("Err: mmap get addr failed !\n");
    close(fd);
    return -1;
  } else if (map_ret == NULL) {
    printf("Err: mmap get a NULL !\n");
    close(fd);
    return -1;
  }

  void *map_virtual = (char*)map_ret + offset_in_page;
  int read_value = *(volatile uint32_t*)map_virtual;

  if (munmap(map_ret, mapped_size) == -1) {
    printf("Err: munmap failed !\n");
  }
  close(fd);

  return read_value;
}

int NOC::devmem_write(int addr, int value) {
  int read_value = -1;
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    printf("Err: can not open /dev/mem \n");
    return -1;
  }

  unsigned page_size = getpagesize();
  unsigned mapped_size = page_size;
  unsigned offset_in_page = (unsigned)addr & (page_size - 1);
  if (offset_in_page + 32 > page_size) {
    mapped_size *= 2;
  }
  void* map_ret = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                       fd, addr & ~(page_size - 1));

  if (map_ret == MAP_FAILED) {
    printf("Err: mmap get addr failed !\n");
    close(fd);
    return -1;
  } else if (map_ret == NULL) {
    printf("Err: mmap get a NULL !\n");
    close(fd);
    return -1;
  }

  void *map_virtual = (char*)map_ret + offset_in_page;
  *(volatile uint32_t*)map_virtual = value;

  read_value = *(volatile uint32_t*)map_virtual;

  if (munmap(map_ret, mapped_size) == -1) {
    printf("Err: munmap failed !\n");
  }
  close(fd);

  return read_value;
}

// MON function

int NOC::mon_lock(int addr) {
  int state;

  addr = addr + 0xC;
  state = devmem_write(addr, 1);

  return state;
}

int NOC::mon_unlock(int addr) {
  int state;

  addr = addr + 0xC;
  state = devmem_write(addr, 0xF9E8D7C6);

  return state;
}

double NOC::mon_get_tbs(int addr) {
  int tbs_0;
  double tbs;

  tbs_0 = devmem_read(addr + 0x4D4) & (0x1f);
  // 1 / mon_freq * pow(2,tbs_0)
  tbs = pow(2, tbs_0) / mon_freq;

  return tbs;
}

void NOC::mon_set_tbs(int addr, double period) {
  int tb_scale;
  double tb_value;

  tb_value = log(mon_freq * period) / log(2);
  tb_scale = int(tb_value + 0.5);
  devmem_write(addr + 0x4D4, tb_scale);
  act_period = mon_get_tbs(addr);

}

void NOC::mon_setup(int addr, double period) {
  if ((0.001 <= period) && (period <= 2)) {
    mon_set_tbs(addr, period);
    create_nsu_base_addr(addr);

  } else {
    std::cout << "mon period error" << std::endl;
  }
}

// NSU FUNCTION

void NOC::nsu_enable(int addr) {
  if (addr != 0) {
    devmem_write(addr, 1);
  }
}

void NOC::nsu_disable(int addr) {
  if (addr != 0) {
    devmem_write(addr, 0);
  }
}

void NOC::nsu_setup(int addr, int op_type) {
  if (addr != 0) {
    if (op_type == 0) {  // read
      devmem_write(addr + 0x4, 0x30);
    } else if (op_type == 1) {  // write
      devmem_write(addr + 0x4, 0xc0);
    } else {
      std::cout << "input op_type = " << op_type << std::endl;
      std::cout << "op_type error, op_type = 0/1 r/w " << std::endl;
      exit(1);
    }
  }
}

int NOC::nsu_sample(int addr) {
  int mon_1;
  int burst_acc;
  int byte_count;

  if (addr == 0) {
    return 0;
  }

  mon_1 = devmem_read(addr + 0x18);

  if ((mon_1 & (1 << 31)) != 0) {
    return -1;
  }

  burst_acc = mon_1 & ~(1 << 31);
  byte_count = burst_acc * 128 / 8;
  return byte_count;
}

void NOC::create_nsu_base_addr(int addr_base) {
  uint32_t i;

  if (addr_base == ddrmc_physics_addr[0]) {
    noc_nsu_perf[0].op_type = 0;
    noc_nsu_perf[1].op_type = 1;
    for (i = 0; i < 4; i++) {
      noc_nsu_perf[0].noc_nsu_perf_addr[i] = i * 64 + addr_base + 0x4D8;
      noc_nsu_perf[1].noc_nsu_perf_addr[i] = i * 64 + 32 + addr_base + 0x4D8;
    }
  } else if (addr_base == ddrmc_physics_addr[1]) {
    noc_nsu_perf[2].op_type = 0;
    noc_nsu_perf[3].op_type = 1;
    for (i = 0; i < 4; i++) {
      noc_nsu_perf[2].noc_nsu_perf_addr[i] = i * 64 + addr_base + 0x4D8;
      noc_nsu_perf[3].noc_nsu_perf_addr[i] = i * 64 + 32 + addr_base + 0x4D8;
    }
  } else if (addr_base == ddrmc_physics_addr[2]) {
    noc_nsu_perf[4].op_type = 0;
    noc_nsu_perf[5].op_type = 1;
    for (i = 0; i < 4; i++) {
      noc_nsu_perf[4].noc_nsu_perf_addr[i] = i * 64 + addr_base + 0x4D8;
      noc_nsu_perf[5].noc_nsu_perf_addr[i] = i * 64 + 32 + addr_base + 0x4D8;
    }
  } else if (addr_base == ddrmc_physics_addr[3]) {
    noc_nsu_perf[6].op_type = 0;
    noc_nsu_perf[7].op_type = 1;
    for (i = 0; i < 4; i++) {
      noc_nsu_perf[6].noc_nsu_perf_addr[i] = i * 64 + addr_base + 0x4D8;
      noc_nsu_perf[7].noc_nsu_perf_addr[i] = i * 64 + 32 + addr_base + 0x4D8;
    }
  } else {
    printf("Err: nsu sample physics addr error\n");
    exit(1);
  }
}

// COLLECT FUNCTION

void collecting_thread(NOC* noc, double sample_interval_clks) {
  uint32_t i, j;
  unsigned int byte_count = 0;

  while (noc->collecting) {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    noc->data[noc->record_counter].time =
        std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1, 1>>>(t)
            .count();

    for (i = 0; i < 10; i++) {
      if (i < 8) {
        for (j = 0; j < 4; j++) {
          byte_count =
              noc->nsu_sample(noc->noc_nsu_perf[i].noc_nsu_perf_addr[j]);
          if (byte_count >= 0) {
            noc->data[noc->record_counter].data[i] += byte_count;
          }
        }
      } else {
        noc->data[noc->record_counter].data[i] = 0;
      }
    }

    if (noc->record_counter * sizeof(struct record) == DATA_BUFFER_SIZE) {
      noc->collecting = false;
    }
    usleep(1000 * 1000 * sample_interval_clks);  // clks = s tranfer to ms
    noc->record_counter++;
  }
  std::cout << "NOC Stop Collecting" << std::endl;
}

void NOC::start_collect(double interval_sec, void* _data) {
  uint32_t i, j;

  if (_data != NULL)
    data = (struct record*)_data;
  else
    data = (struct record*)malloc(DATA_BUFFER_SIZE);

  memset(data, 0, DATA_BUFFER_SIZE);

  if (data == NULL) {
    std::cout << "Unable to alloc memory for data" << std::endl;
    exit(1);
  }

  if (noc_nsu_perf == NULL)
    noc_nsu_perf = (struct NOC_NSU_PERF*)malloc(NOC_NSU_PERF_SIZE);
  memset(noc_nsu_perf, 0, NOC_NSU_PERF_SIZE);

  if (noc_nsu_perf == NULL) {
    std::cout << "Unable to alloc memory for noc_nsu_perf" << std::endl;
    exit(1);
  }

  for (int l_n = 0; l_n < 8; l_n++) {
    noc_nsu_perf[l_n].op_type = -1;
    for (int l_m = 0; l_m < 4; l_m++) {
      noc_nsu_perf[l_n].noc_nsu_perf_addr[l_m] = 0;
    }
  }

  for (int n = 0; n < ddrmc_number; n++) {
    mon_unlock(ddrmc_physics_addr[n]);
    mon_setup(ddrmc_physics_addr[n], interval_sec);
  }

  for (i = 0; i < 8; i++) {
    for (j = 0; j < 4; j++) {
      nsu_setup(noc_nsu_perf[i].noc_nsu_perf_addr[j], noc_nsu_perf[i].op_type);
      nsu_enable(noc_nsu_perf[i].noc_nsu_perf_addr[j]);
    }
  }

  collecting = true;
  noc_thread = std::thread(collecting_thread, this, interval_sec);
}

void NOC::stop_collect(void) {
  collecting = false;
  noc_thread.join();
}

int NOC::pop_data(struct record* d) {
  if (record_counter == 0) return -1;
  d->time = data[record_counter - 1].time;
  for (int i = 0; i < 10; i++) {
    d->data[i] = data[record_counter - 1].data[i];
  }

  record_counter--;
  return 0;
}

double NOC::get_act_period(void) { return act_period; }

int NOC::get_record_data_len(void) {
  record_data_len = 10;
  return record_data_len;
}
