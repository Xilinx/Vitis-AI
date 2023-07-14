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

#include "noc_nmu.hpp"
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

const int NMU_NSU_PERF_SIZE = 1024 * 1024;

NMU::NMU(uint64_t freq, int nmu_addr[], int npi_addr, int len) {
  mon_freq = freq;
  nmu_physics_addr = nmu_addr;
  npi_physics_addr = npi_addr;
  nmu_number = len;
  record_counter = 0;
}

NMU::~NMU() {
  if (data != NULL)
    free(data);
  if (noc_nmu_perf != NULL)
    free(noc_nmu_perf);

  nmu_physics_addr = NULL;
}

int NMU::devmem_read(int addr) {
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
  void *map_ret = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED,
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

  void *map_virtual = (char *)map_ret + offset_in_page;
  int read_value = *(volatile uint32_t *)map_virtual;

  if (munmap(map_ret, mapped_size) == -1) {
    printf("Err: munmap failed !\n");
  }
  close(fd);

  return read_value;
}

int NMU::devmem_write(int addr, int value) {
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
  void *map_ret = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED,
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

  void *map_virtual = (char *)map_ret + offset_in_page;
  *(volatile uint32_t *)map_virtual = value;

  read_value = *(volatile uint32_t *)map_virtual;

  if (munmap(map_ret, mapped_size) == -1) {
    printf("Err: munmap failed !\n");
  }
  close(fd);

  return read_value;
}

// NMU_NPI & NMU_NMU LOCK/UNLOCK
int NMU::lock(int addr) {
  int state;
  addr += 0xC;
  state = devmem_write(addr, 1);
  return state;
}

int NMU::unlock(int addr) {
  int state;
  addr += 0xC;
  state = devmem_write(addr, 0xF9E8D7C6);
  return state;
}

// NMU_NPI function
double NMU::npi_get_tbs(int addr) {
  int tbs_0;
  double tbs;

  tbs_0 = (devmem_read(addr) >> 15) & (0x1f);
  // 1 / mon_freq * pow(2,tbs_0)
  tbs = pow(2, tbs_0) / mon_freq;

  return tbs;
}

void NMU::npi_set_tbs(int addr, double period) {
  int tb_scale;
  int value;
  double tb_value;

  /*scal value need write to reg*/
  tb_value = log(mon_freq * period) / log(2);
  tb_scale = int(tb_value + 0.5);

  /* read reg value and set bit15-bit19*/
  value = devmem_read(addr);
  value &= ~(0x1f << 15);
  value |= (tb_scale & 0x1f) << 15;

  devmem_write(addr, value);
  act_period = npi_get_tbs(addr);

}

void NMU::npi_setup(int addr, double period) {
  if ((0.001 <= period) && (period <= 2)) {
    npi_set_tbs(addr, period);
  } else {
    std::cout << "mon period error" << std::endl;
  }
}

// NMU_NMU FUNCTION

void NMU::noc_nmu_enable(int addr) {
  int value = 0;
  if (addr != 0) {
    value = devmem_read(addr);
    value |= 0x1;
    devmem_write(addr, value);
  }
}

void NMU::noc_nmu_disable(int addr) {
  int value = 0;
  if (addr != 0) {
    value = devmem_read(addr);
    value &= (~0x1);
    devmem_write(addr, value);
  }
}

uint64_t NMU::noc_nmu_sample(int addr) {
  uint64_t mon_low;
  uint64_t mon_upper;
  uint64_t byte_count;

  if (addr == 0) {
    return 0;
  }

  mon_low = devmem_read(addr - 0x4);
  mon_upper = devmem_read(addr - 0x8);
  byte_count = ((mon_upper & 0xff) << 32) | mon_low;
  return byte_count;
}

void NMU::create_nmu_mon_addr(int *addr_base) {
  for (int i = 0; i <= nmu_number; i++) {
    noc_nmu_perf[i].noc_nmu_perf_addr[0] =
        addr_base[i] + 0x88c; // REG_PERF_MON0_CTRL
    noc_nmu_perf[i].noc_nmu_perf_addr[1] =
        addr_base[i] + 0x8ac; // REG_PERF_MON1_CTRL
  }
}

// COLLECT FUNCTION

void collecting_thread(NMU *nmu, double sample_interval_clks) {
  int i, j, cnt;
  uint64_t byte_count = 0;

  while (nmu->collecting) {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    nmu->data[nmu->record_counter].time =
        std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1, 1>>>(t)
            .count();
    cnt = 0;
    for (i = 0; i < NOC_NMU_MAX_NUMBER ; i++) {
      if (i < nmu->nmu_number) {
        for (j = 0; j < 2; j++) {
          byte_count =
              nmu->noc_nmu_sample(nmu->noc_nmu_perf[i].noc_nmu_perf_addr[j]);
          if (byte_count >= 0) {
            nmu->data[nmu->record_counter].data[cnt++] = byte_count;
          }
        }
      } else { /* nmu->nmu_nmur < value < 44(total number of vek280 noc_nmu)*/
	nmu->data[nmu->record_counter].data[cnt++] = 0;	      
      }
    }

    int nmuber_record = NMU_DATA_BUFFER_SIZE / sizeof(struct record_nmu);
    if (nmu->record_counter == nmuber_record -1) {
      nmu->collecting = false;
    }
    usleep(1000 * 1000 * sample_interval_clks); // clks = s tranfer to ms
    nmu->record_counter++;
  }
  std::cout << "NMU Stop Collecting" << std::endl;
}

void NMU::start_collect(double interval_sec, void *_data) {

  if (_data != NULL)
    data = (struct record_nmu *)_data;
  else
    data = (struct record_nmu *)malloc(NMU_DATA_BUFFER_SIZE);

  memset(data, 0, NMU_DATA_BUFFER_SIZE);

  if (data == NULL) {
    std::cout << "Unable to alloc memory for data" << std::endl;
    exit(1);
  }

  if (noc_nmu_perf == NULL)
    noc_nmu_perf = (struct NOC_NMU_PERF *)malloc(NMU_NSU_PERF_SIZE);
  memset(noc_nmu_perf, 0, NMU_NSU_PERF_SIZE);

  if (noc_nmu_perf == NULL) {
    std::cout << "Unable to alloc memory for noc_nmu_perf" << std::endl;
    exit(1);
  }

  for (int l_n = 0; l_n < nmu_number; l_n++) {
    for (int l_m = 0; l_m < 2; l_m++) {
      noc_nmu_perf[l_n].noc_nmu_perf_addr[l_m] = 0;
    }
  }

  // NMU_NPI SETUP ** set sample period

  unlock(npi_physics_addr);
  npi_setup(npi_physics_addr + 0x100, interval_sec);

  /*unlock nmu mon*/
  for (int cnt = 0; cnt < nmu_number; cnt++) {
    unlock(nmu_physics_addr[cnt]);
  }
  create_nmu_mon_addr(nmu_physics_addr);
  for (int i = 0; i < nmu_number; i++) {
    for (int j = 0; j < 2; j++) {
      noc_nmu_enable(noc_nmu_perf[i].noc_nmu_perf_addr[j]);
    }
  }

  collecting = true;
  nmu_thread = std::thread(collecting_thread, this, interval_sec);
}

void NMU::stop_collect(void) {
  collecting = false;
  lock(npi_physics_addr);
  /*lock nmu mon*/
  for (int cnt = 0; cnt < nmu_number; cnt++) {
    lock(nmu_physics_addr[cnt]);
  }
  nmu_thread.join();
}

int NMU::pop_data(struct record_nmu *d) {
  if (record_counter == 0)
    return -1;
  d->time = data[record_counter - 1].time;
  for (int i = 0; i < (nmu_number * 2); i++) {
    d->data[i] = data[record_counter - 1].data[i];
  }

  record_counter--;
  return 0;
}

double NMU::get_act_period(void) { return act_period; }

int NMU::get_record_data_len(void) {
  record_data_len = nmu_number * 2;
  return record_data_len;
}
