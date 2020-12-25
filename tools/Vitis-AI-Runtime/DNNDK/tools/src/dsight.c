/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "defs.h"

#define TOOL_PROFILER       "/usr/lib/libdsight.pyc"

/* for N2Cube Core library version */
char verDSight[] = "DSight version " DSIGHT_VER_STRING;

void run_profiler(char *traceFile);

void run_profiler(char *traceFile)
{
    int ret;
    char cmd[256];
    struct stat fstat;

    /* check if trace file exist */
    ret = stat(traceFile, &fstat);

    if (ret) {
        printf("Error: fail to locate DPU trace file: %s\n", traceFile);
        exit(-1);
    }

    sprintf(cmd, "python %s %s", TOOL_PROFILER, traceFile);

    ret = system(cmd);
    if (ret) {
        printf("Error: fail to run dsight\n");
        exit(-1);
    }
}

void print_usage()
{
    printf("Usage: dsight <option>\n");

    printf(" Options are:\n");
    printf(
        "  -p  --profile    Specify DPU trace file for profiling\n"
        "  -v  --version    Display DSight version info\n"
        "  -h  --help       Display this information\n");

}

int print_version(void)
{
    char buildLabel[256];

    /* print dsight version */
    printf("%s\n", verDSight);
    sprintf(buildLabel, "Build Label: %s %s", __DATE__,  __TIME__);
    printf("%s\n", buildLabel);

    return 0;
}



int main(int argc, char *argv[])
{
    int next_option;
    char *in_option;

    /* string listing valid short options letters. */
    const char* const short_options = "p:vh";

    /* array for describing valid long options. */
    const struct option long_options[] = {
        {"profile",   required_argument,    NULL, 'p'},
        {"version",   no_argument,          NULL, 'v'},
        {"help",      no_argument,          NULL, 'h'},
        {NULL,        0,                    NULL, 0}
    };

    do {
        next_option = getopt_long(argc, argv, short_options, long_options, NULL);
        switch (next_option) {
         case 'p':
            /* run profiler dsight */
            in_option = optarg;
            if (!in_option) {
                printf("Error: no DPU trace file specified for option [profile]");
                print_usage();
            }

            run_profiler(in_option);
            break;
        case 'v':
            /* display version info */
            print_version();
            break;
        case 'h':
            /* -h or --help */
            print_usage();
            break;
        case '?':
            /* -h or --help */
            print_usage();
        case -1:
            /* if no option is specified */
            if (1 == argc) {
                print_usage();
            }
            /* Done with options */
            break;
        default:
            /* Something else: unexpected */
            printf("Error: unexpected option is specified.");
            exit(-1);
        }
    } while (next_option != -1);

    return 0;
}
