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
#include "power_mon.hpp"

extern "C" {
int main(int argc, char* argv[]) {
  int volt = 0;
  int curr = 0;
  int power = 0;
  char volt_path[] = "/sys/class/hwmon/hwmon0/in1_input";
  char curr_path[] = "/sys/class/hwmon/hwmon0/curr1_input";
  char power_path[] = "/sys/class/hwmon/hwmon0/power1_input";
  volt = get_device_volt(volt_path);
  curr = get_device_curr(curr_path);
  power = get_device_power(power_path);

  printf("device curr=%d, volt=%d, power=%d\n", curr, volt, power);
  return 0;
}
}
