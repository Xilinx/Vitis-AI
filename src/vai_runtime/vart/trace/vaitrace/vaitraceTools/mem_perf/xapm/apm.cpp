
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

#include "apm.hpp"
#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include <iostream>
#include <memory.h>

#ifdef BOARD_ULTRA96
#  define APM_CLK_FREQ 266666656
#else
#  define APM_CLK_FREQ 533333333
#endif

/**
 * @brief Initialize an APM instance
 *
 * @param uio_device_id - UIO device ID of the counter
 *
 * @return void
 */
APM::APM(int uio_device_id) {
  uio_device_name = "/dev/uio" + std::to_string(uio_device_id);

  load_phys_addr(uio_device_id);

  apm_fd = open(uio_device_name.c_str(), O_RDWR);
  if (apm_fd < 1) {
    std::cout << "Unable to open " << uio_device_name << std::endl;
  }

  apm_baseaddr =
      mmap(0, APM_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, apm_fd, 0);
  if (apm_baseaddr == MAP_FAILED) {
    close(apm_fd);
    printf("Unable to map APM memory\n");
  }

  params = (xapm_param*)mmap(0, APM_MAP_SIZE, PROT_READ | PROT_WRITE,
                             MAP_SHARED, apm_fd, getpagesize());
  if (params == MAP_FAILED) {
    munmap(apm_baseaddr, APM_MAP_SIZE);
    close(apm_fd);
    printf("Unable to map APM params\n");
  }
}

APM::~APM() {
  munmap(apm_baseaddr, APM_MAP_SIZE);
  munmap(params, APM_MAP_SIZE);
  if (data != NULL) free(data);
  close(apm_fd);
}

/**
 * @brief Return the APM hardware mode
 *
 * @return APM hardware mode
 */
int APM::get_mode(void) { return params->mode; }

/**
 * @brief Load the APM physical address from UIO
 *
 * @param device_id - UIO device ID for the APM instance
 *
 * @return void
 */
void APM::load_phys_addr(int device_id) {
  void* physaddr;
  std::string sysfs_reg_file =
      "/sys/class/uio/uio" + std::to_string(device_id) + "/device/of_node/reg";
  int sysfs_fd = open(sysfs_reg_file.c_str(), O_RDONLY);
  if (sysfs_fd < 1) {
    printf("Unable to open sysfs reg map for APM");
  }
  ssize_t bytes_read = read(sysfs_fd, &physaddr, sizeof(void*));
  close(sysfs_fd);
  if (bytes_read != sizeof(void*)) {
    printf("Unable to read physical APM address");
  }

  // In the register file byte order is swapped, we need to unswap
  apm_physaddr = (void*)__builtin_bswap64((uint64_t)physaddr);
}

/**
 * @brief Set a metrics counter slot
 *
 * @param slot - slot to set
 * @param metrics - which metric to record
 * @param counter - which counter to use
 *
 * @return void
 */
void APM::set_metrics_counter(uint8_t slot, uint8_t metrics, uint8_t counter) {
  uint32_t reg;
  uint32_t mask;
  uint32_t offset;

  // Set the mask value to force zero in counternum byte range
  switch (counter % 4) {
    case 0:
      mask = 0xffffff00;
      break;
    case 1:
      mask = 0xffff00ff;
      break;
    case 2:
      mask = 0xff00ffff;
      break;
    default:
      mask = 0x00ffffff;
  }

  if (counter < 4) {
    offset = XAPM_MSR0_OFFSET;
  } else if (counter < 8) {
    offset = XAPM_MSR1_OFFSET;
    counter -= 4;
  } else {
    offset = XAPM_MSR2_OFFSET;
    counter -= 8;
  }

  reg = readreg(offset);
  reg &= mask;
  reg |= metrics << (counter * 8);
  reg |= slot << (counter * 8 + 5);
  writereg(offset, reg);
}

/**
 * @brief Read the current value of a metrics counter
 *
 * @param counter - which counter to read
 *
 * @return uint32_t - value stored in the counter
 */
uint32_t APM::get_metrics_counter(uint8_t counter) {
  return (readreg(XAPM_SMC0_OFFSET + (counter * 16)) * params->scale_factor);
}

/**
 * @brief Reset all metrics counters of the instance
 *
 * @return void
 */
void APM::reset_metrics_counters(void) {
  // Set reset bit
  writereg(XAPM_CTL_OFFSET,
           readreg(XAPM_CTL_OFFSET) | XAPM_CR_MCNTR_RESET_MASK);
  // Unset reset bit
  writereg(XAPM_CTL_OFFSET,
           readreg(XAPM_CTL_OFFSET) & ~(XAPM_CR_MCNTR_RESET_MASK));
}

/**
 * @brief Run the metrics counters for a set time period to collect data
 *
 * @param sample_interval_clocks - Number of clocks to sample for
 * @param reset - Reset the counters before measuring?
 *
 * @return void
 */
void APM::collect(uint32_t sample_interval_clks, bool reset) {
  int tmp;

  if (reset) {
    reset_metrics_counters();
  }

  // Set the sample interval
  writereg(XAPM_SI_LOW_OFFSET, sample_interval_clks);

  // Load the sample interval to the sample interval counter
  writereg(XAPM_SICR_OFFSET, XAPM_SICR_LOAD_MASK);

  // Enable the sample interrupt
  writereg(XAPM_IE_OFFSET,
           readreg(XAPM_IE_OFFSET) | XAPM_IXR_SIC_OVERFLOW_MASK);

  // Enable global interrupts
  writereg(XAPM_GIE_OFFSET, 1);

  // Enable the metrics counters
  writereg(XAPM_CTL_OFFSET,
           readreg(XAPM_CTL_OFFSET) | XAPM_CR_MCNTR_ENABLE_MASK);

  // Enable the SIC
  writereg(XAPM_SICR_OFFSET, XAPM_SICR_ENABLE_MASK);

  if (read(apm_fd, &tmp, sizeof(int)) < 0) {
    printf("Unable to read from UIO");
  }
  if (params->isr & XAPM_IXR_SIC_OVERFLOW_MASK) {
    // Disable the SIC
    writereg(XAPM_SICR_OFFSET,
             readreg(XAPM_SICR_OFFSET) & ~(XAPM_SICR_ENABLE_MASK));
  }

  // Disable the metrics counter
  writereg(XAPM_CTL_OFFSET,
           readreg(XAPM_CTL_OFFSET) & ~(XAPM_CR_MCNTR_ENABLE_MASK));

  // Disable the sample interrupt
  writereg(XAPM_IE_OFFSET,
           readreg(XAPM_IE_OFFSET) ^ XAPM_IXR_SIC_OVERFLOW_MASK);

  // Disable global interrupts
  writereg(XAPM_GIE_OFFSET, 0);
}

void collecting_thread(APM* apm, uint32_t sample_interval_clks) {
  apm->record_counter = 0;

  while (apm->collecting) {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    apm->data[apm->record_counter].time =
        std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1, 1>>>(t)
            .count();

    apm->collect(sample_interval_clks);

    for (int j = 0; j < 10; j++) {
      apm->data[apm->record_counter].data[j] = apm->get_metrics_counter(j);
      // std::cout << "Metrics Counter"<< j << ": " <<
      // (apm->data[apm->record_counter].data[j]) << std::endl;
    }

    if (apm->record_counter * sizeof(struct record) == DATA_BUFFER_SIZE) {
      // Data overflow
      apm->collecting = false;
    }

    apm->record_counter++;
  }

  std::cout << "APM Stop Collecting" << std::endl;
}

void APM::start_collect(double sample_interval, void* _data) {
  if (_data != NULL)
    data = (struct record*)_data;
  else
    data = (struct record*)malloc(DATA_BUFFER_SIZE);

  memset(data, 0, DATA_BUFFER_SIZE);

  if (data == NULL) {
    printf("Unable to alloc memory for data\n");
    exit(1);
  }
  act_period = sample_interval;

  set_metrics_counter(1, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_0);
  set_metrics_counter(1, XAPM_METRIC_WRITE_BEAT_COUNT, XAPM_METRIC_COUNTER_1);
  set_metrics_counter(2, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_2);
  set_metrics_counter(2, XAPM_METRIC_WRITE_BYTE_COUNT, XAPM_METRIC_COUNTER_3);
  set_metrics_counter(3, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_4);
  set_metrics_counter(3, XAPM_METRIC_WRITE_BYTE_COUNT, XAPM_METRIC_COUNTER_5);
  set_metrics_counter(4, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_6);
  set_metrics_counter(4, XAPM_METRIC_WRITE_BYTE_COUNT, XAPM_METRIC_COUNTER_7);
  set_metrics_counter(5, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_8);
  set_metrics_counter(5, XAPM_METRIC_WRITE_BYTE_COUNT, XAPM_METRIC_COUNTER_9);

  unsigned int interval_clk = (unsigned int)(APM_CLK_FREQ * sample_interval);
  collecting = true;
  apm_thread = std::thread(collecting_thread, this, interval_clk);
}

void APM::stop_collect(void) {
  collecting = false;
  apm_thread.join();
}

int APM::pop_data(struct record* d) {
  if (record_counter == 0) return -1;

  d->time = data[record_counter - 1].time;
  for (int i = 0; i < 10; i++) {
    d->data[i] = data[record_counter - 1].data[i];
  }

  record_counter--;
  return 0;
}

double APM::get_act_period(void) { return act_period; }

int APM::get_record_data_len(void) {
  record_data_len = 10;
  return record_data_len;
}
