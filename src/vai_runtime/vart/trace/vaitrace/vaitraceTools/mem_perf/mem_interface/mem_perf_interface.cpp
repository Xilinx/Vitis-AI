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
#include "noc_ddrmc.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char noc[] = "noc";
char apm[] = "apm";

MEM_PERF *noc_apm;

extern "C" {
void create_noc_instance(char *apm_type, int addr[], int len) {
  if (strcmp(apm_type, noc) == 0) {
    noc_apm = new NOC(1000, addr, len);

  } else {
    printf("unsupported  %s\n", apm_type);
  }
}

void create_apm_instance(char *apm_type) {
  if (strcmp(apm_type, apm) == 0) {
    noc_apm = new APM(1);

  } else {
    printf("unsupported  %s\n", apm_type);
  }
}

int start(double interval_in_sec) {
  noc_apm->start_collect(interval_in_sec);

  return EXIT_SUCCESS;
}

int stop(void) {
  noc_apm->stop_collect();
  return EXIT_SUCCESS;
}

int pop_data(struct record *d) { return noc_apm->pop_data(d); }

int get_record_data_len(void) { return noc_apm->get_record_data_len(); }

double get_act_period(void) { return noc_apm->get_act_period(); }
}
