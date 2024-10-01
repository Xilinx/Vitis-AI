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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char noc_nmu[] = "noc_nmu";

NMU  *noc_nmu_perf;

extern "C" {

void create_noc_nmu_instance(char *apm_type, uint64_t freq, int addr[],
                             int npi_addr, int len) {
  if (strcmp(apm_type, noc_nmu) == 0) {
    noc_nmu_perf = new NMU(freq, addr, npi_addr, len);

  } else {
    printf("unsupported  %s\n", apm_type);
  }
}

int start(double interval_in_sec) {
  noc_nmu_perf->start_collect(interval_in_sec);

  return EXIT_SUCCESS;
}

int stop(void) {
  noc_nmu_perf->stop_collect();
  return EXIT_SUCCESS;
}

int pop_data(struct record_nmu *d) { return noc_nmu_perf->pop_data(d); }

int get_record_data_len(void) { return noc_nmu_perf->get_record_data_len(); }

double get_act_period(void) { return noc_nmu_perf->get_act_period(); }
}
