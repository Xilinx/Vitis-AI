/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/reboot.h>
#include <time.h>
#include <unistd.h>

static void print_version(const char* name) { printf("%s \n", name); }

static void usage(FILE* out) {
  fprintf(out, "Usage: upgrade-reboot\n");
  fprintf(out, "\n");
}
enum {
  OPT_VERSION = 128,
  OPT_HELP,
};

static char *short_options = "";
static struct option long_options[] = {
    {"version", 0, NULL, OPT_VERSION},
    {"help", 0, NULL, OPT_HELP},
    {0, 0, 0, 0},
};

int main(int argc, char *argv[]) {
  int i;
  int option_index;

  while ((i = getopt_long(argc, argv, short_options, long_options,
                          &option_index)) >= 0) {
    switch (i) {
      case 0:
        break;

      case OPT_VERSION:
        print_version("upgrade-reboot");
        exit(0);
        break;

      case OPT_HELP:
        usage(stdout);
        exit(0);
        break;

      default:
        usage(stderr);
        exit(1);
    }
  }

  sync();
  sleep(2);

  reboot(RB_AUTOBOOT);

  return 0;
}
