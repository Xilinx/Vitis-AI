/*
 * Copyright (C) 2019 – 2018 Xilinx, Inc.  All rights reserved.
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

#ifndef __XAPM_HPP__
#define __XAPM_HPP__

#include <assert.h>
#include <fcntl.h>
#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <boost/python.hpp>

#define APM_CR 0x300
#define APM_SIR 0x24
#define APM_SICR 0x28
#define APM_SISR 0x2C
#define APM_MAX_PORTS 6
#define MBs 0
#define Gbps 1

struct apm {
  uintptr_t addr;
  size_t slot;
  uint32_t msr_read;
  uint32_t msr_write;
  uint32_t smcr;
  uint32_t apm_num;
  uint32_t clock_freq;
} typedef apm_t;

class APM {
private:
  uintptr_t apm_ptr;
  uintptr_t base_addr;
  double scale;
  bool state;
  pthread_mutex_t lock;
  size_t sampint;
  size_t metsel;
  size_t slotnum;
  size_t smcr;
  size_t msr_write;
  size_t msr_read;
  size_t fd;
  size_t apmnum;
  APM(apm_t x);

public:
  APM(boost::python::list l);
  ~APM();
  boost::python::list port;
  void start();
  void stop();
  void setInterval(double ms);
  double getInterval();
  void setMetric(size_t met_sel);
  size_t getMetric();
  double getThroughput(size_t metric);
};

#endif
