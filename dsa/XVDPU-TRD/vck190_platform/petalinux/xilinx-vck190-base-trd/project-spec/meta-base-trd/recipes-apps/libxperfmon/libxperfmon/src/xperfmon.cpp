/*
 * Copyright (C) 2020 – 2021 Xilinx, Inc.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * XILINX BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of the Xilinx shall not be used
 * in advertising or otherwise to promote the sale, use or other dealings in
 * this Software without prior written authorization from Xilinx.
 */  
 
#include "xperfmon.hpp"
#include "bind.hpp"

#define PAGE_SIZE 4096
#define READ_BYTE_COUNT 3
#define MET_CNT_EN 1
#define SICR_LOAD_DISABLE 0x02
#define SICR_RST_ENABLE 0x101
#define MS_TO_INTERVAL 400000
#define START true
#define STOP false
     
size_t inline HW_READ(uintptr_t addr){
  return *((volatile uint32_t*)(addr));
}

void inline HW_WRITE(uintptr_t addr, size_t val){
  *((volatile uint32_t*)(addr)) = val;
}

void inline HW_OR(uintptr_t addr, size_t val){
  *((volatile uint32_t*)(addr)) |= val;
}

void inline HW_XOR(uintptr_t addr, size_t val){
  *((volatile uint32_t*)(addr)) ^= val;
}     

APM::APM(boost::python::list l) {
  auto a = toVector(l); 
  for (int32_t i = 0; i < a.size(); i++) {
    port.append(APM(a[i])); 
  }
  int j = 0;   
}          
APM::APM(apm_t x) {
  apmnum = x.apm_num; 
  slotnum = x.slot;
  base_addr = x.addr;
  sampint = x.clock_freq;
  scale = 1.0/1000000.0;
  smcr = x.smcr;
  msr_read = x.msr_read;
  msr_write = x.msr_write;
  metsel = (msr_write >> ((slotnum%4)*8))&0x1F;
  lock = PTHREAD_MUTEX_INITIALIZER;
  state = STOP;
  fd = open("/dev/mem", O_RDWR | O_SYNC);
  assert(!(fd < 0));
  apm_ptr = (uintptr_t)mmap(0, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                            fd, base_addr);
  // write 1 to bit 0 of MET_CNT_EN of CR
  HW_WRITE(apm_ptr + APM_CR, MET_CNT_EN);
  // Set sample interval time; write 0 to SIR
  HW_WRITE(apm_ptr + APM_SIR, 0);
  munmap((void *)apm_ptr, PAGE_SIZE);
  close(fd);
}   
APM::~APM() {
  if (state) {
    stop();
  }
}
void APM::start() {
  if (!state) {
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    assert(!(fd < 0));
    apm_ptr = (uintptr_t)mmap(0, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                              fd, base_addr);
    // set the slot and apm info into the metric select register
    //msr_write = ((apmnum << 5) + metsel) << ((slotnum%4)*8);
    HW_XOR(apm_ptr + msr_read, HW_READ(apm_ptr+msr_read)&(0xFF << ((slotnum%4)*8)));
    //msr_write = ((apmnum << 5) + (metsel)) << ((slotnum%4)*8);
    HW_OR(apm_ptr + msr_read, msr_write);
    // Put sample interval time value into SIR: DDR_REF_CLK is 400 MHz: sample
    // Interrupt Configuration
    HW_WRITE(apm_ptr + 0x30, 0x01);
    HW_WRITE(apm_ptr + 0x34, 0x02);
    // interval of 1 = 2.5 nanoseconds
    HW_WRITE(apm_ptr + APM_SIR, sampint);
    // Load sample interval time value into APM counter; write 1 to bit 1 (LOAD)
    // of SICR, write 0 to bit 0 (ENABLE) of SICR
    HW_OR(apm_ptr + APM_SICR, SICR_LOAD_DISABLE);
    // read SISR to initiate sampling into the sampled metric counters (SMCs)
    int32_t temp = HW_READ(apm_ptr + APM_SISR);
    // Reset and enable down counter. Write 1 to bit 8 (MET_CNT_RST), 0 to bit 1
    // (LOAD) and 1 to bit 0(ENABLE) of SICR
    HW_WRITE(apm_ptr + APM_SICR, SICR_RST_ENABLE);
    munmap((void *)apm_ptr, PAGE_SIZE);
    close(fd);
    state = START;
  } 
}                 
void APM::stop() {
  if (state) {
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    assert(!(fd < 0));
    apm_ptr = (uintptr_t)mmap(0, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                              fd, base_addr);
    HW_XOR(apm_ptr + msr_read, msr_write);
    munmap((void *)apm_ptr, PAGE_SIZE);
    close(fd);
    state = STOP;
  }   
}
void APM::setInterval(double ms) {
  if (state) {
    stop();
  }
  sampint = ms * MS_TO_INTERVAL;
  scale *= 1000.0 / ms;
  start();
}
double APM::getInterval() { return (double)sampint; }
void APM::setMetric(size_t met_sel) {
  if (state) {
    stop();
  }
  metsel = met_sel;
  start();
}
size_t APM::getMetric() { return metsel; }
double APM::getThroughput(size_t metric) {
  if (!state) {
    start();
  }
  fd = open("/dev/mem", O_RDWR | O_SYNC);
  assert(!(fd < 0));
  apm_ptr = (uintptr_t)mmap(0, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                            fd, base_addr);
  int32_t temp;
  while (!(HW_READ(apm_ptr+0x38)&(0x02)));
  temp = HW_READ(apm_ptr + smcr);
  munmap((void *)apm_ptr, PAGE_SIZE);
  close(fd);
  if (metric == Gbps){
    return (double)(temp * scale * 8.0 / 1000.0);
  }
  else{
    return (double)(temp * scale);
  }
}



/*non-class member functions*/
bool operator ==(APM const &lhs, APM const &rhs){
  return lhs == rhs;
} 
