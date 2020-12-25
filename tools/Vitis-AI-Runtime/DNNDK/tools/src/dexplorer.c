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
#include <stdint.h>
#include <string.h>
#include <limits.h>

#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dlfcn.h>

#include "defs.h"

extern int dpuOpen();
extern int dpuClose();
extern void n2cube_version(char buf[], int size);
extern void n2cube_builddate(char buf[], int size);
extern void dpuDumpSignatureInfo(char disVer);
extern void dnndk_version(char buf[], int size);
extern void dnndk_copyright(char buf[], int size);
extern int display_dpu_debug_info();
extern const char *dpu_get_n2cube_mode(void);
extern void dpu_set_n2cube_mode(char *mode);
extern uint32_t dpu_cache_status(void);
extern void dpu_set_n2cube_timeout(uint32_t second);
extern uint32_t dpu_get_n2cube_timeout(void);
extern void dpu_clear_counter(void);

static char *pMyPath = NULL;

void print_usage();
void run_profiler(char *traceFile);


int check_dpu_driver_exist()
{
    int ret;

    ret = dpuOpen();

    if (ret) {
        printf("Error: fail to open DPU device: please check if DPU Driver works as expected.\n");
        return -1;
    }

    return 0;
}

int display_dpu_status()
{
    if (display_dpu_debug_info()) {
        printf("Fail to display DPU Core status.\n");
        return -1;
    }

    return 0;
}

int display_dpu_info(char disVer)
{
    /* display DPU HW Caps info */
    dpuOpen();
    dpuDumpSignatureInfo(disVer);
    dpuClose();

    return 0;
}

int proc_extend(char *message)
{
    if (!strcasecmp(message, "w")) {
        return display_dpu_info(1);
    } else if (!strcasecmp(message, "h")) {
        printf("Usage: dexplorer -x <option>\n");
        printf("  dexplorer  -x w       Display the info of DPU cores\n");
        printf("  dexplorer  -x h       Display this information\n");
    } else {
        printf("Try 'dexplorer -h' for more information.\n");
        return -1;
    }

    return 0;
}

int set_n2cube_timeout(char *time)
{
    int i, value;

    for ( i=0; i<strlen(time); i++) {
        if ((time[i]<'0') || (time[i]>'9')) {
            printf("Error: invalid timeout value specified: %s\n", time);
            printf("Try 'dexplorer -h' for more information.\n");
            return -1;
        }
    }

    value = strtol(time, NULL, 10);

    if ((value < 1) || (value > 100) ) {
        printf("Error: invalid timeout value specified: %s\n", time);
        printf("Try 'dexplorer -h' for more information.\n");
        return -1;
    }

    dpu_set_n2cube_timeout(value);

    return 0;
}

int set_n2cube_mode(char *mode)
{
    int profEnable;
 //   char cmd[256];

    /* Param check */
    if (strcmp(mode, "normal") &&
        strcmp(mode, "profile") &&
        strcmp(mode, "debug") &&
        strcmp(mode, "dump")) {
            printf("Error: invalid mode value specified:%s\n", mode);
            printf("Try 'dexplorer -h' for more information.\n");
            return -1;
    }

    /* Update DPU mode */
    //sprintf(cmd, "echo %s > %s", mode, DPU_MODE);
    dpu_set_n2cube_mode(mode);
//    if (system(cmd)) {
//        printf("Error: fail to change DPU mode.\n");
//        return -1;
//    }

    /* Update DPU profEnable*/
    if (!strcmp(mode, "profile")) {
        profEnable = 1;
    } else {
        profEnable = 0;
    }
//    sprintf(cmd, "echo %d > %s", profEnable, DPU_PROFILER);
//    if (system(cmd)) {
//        printf("Error: fail to change DPU mode.\n");
//        return -1;
//    }

    return 0;
}

void print_usage()
{
    printf("Usage: dexplorer <option>\n");

    printf(" Options are:\n");
    printf(
        "  -v  --version    Display version info for each DNNDK component\n"
        "  -s  --status     Display the status of DPU cores\n"
        "  -w  --whoami     Display the info of DPU cores\n"
        "  -m  --mode       Specify DNNDK N2Cube running mode: normal, profile, or debug\n"
        "  -t  --timeout    Specify DPU timeout limitation in seconds under integer range of [1, 100]\n"
        "  -c  --clear      Clear the DPU schedule counter\n"
        "  -h  --help       Display this information\n");

}

int print_version(void)
{
    int ret;
    char buildLabel[256];
    char verDExplorer[] = "DExplorer version " DEXPLORER_VER_STRING;
    FILE *stream;
    char *p;

    /* print DNNDK release version */
    dnndk_version(buildLabel, sizeof(buildLabel));
    printf("Vitis AI for Edge DPU version %s\n", buildLabel);
    dnndk_copyright(buildLabel, sizeof(buildLabel));
    printf("%s\n\n", buildLabel);

    /* print dexplorer version */
    printf("%s\n", verDExplorer);
    sprintf(buildLabel, "Build Label: %s %s", __DATE__,  __TIME__);
    printf("%s\n", buildLabel);

    /* print dsight version */
    memset(buildLabel, '\0', sizeof(buildLabel));
    strcpy(buildLabel, pMyPath);
    p = buildLabel + strlen(buildLabel);
    while ((*p != '/') && (p != buildLabel)) p--;
    if (*p == '/') p++;
    strcpy(p, "dsight -v"); // Get dsight path
    stream = popen(buildLabel, "r");
    fread(buildLabel, sizeof(char), sizeof(buildLabel), stream); // Get $(dsight -v)
    pclose(stream);
    printf("\n%s", buildLabel);

    /* print ddump version */
    memset(buildLabel, '\0', sizeof(buildLabel));
    strcpy(buildLabel, pMyPath);
    p = buildLabel + strlen(buildLabel);
    while ((*p != '/') && (p != buildLabel)) p--;
    if (*p == '/') p++;
    strcpy(p, "ddump -v"); // Get dsight path
    stream = popen(buildLabel, "r");
    fread(buildLabel, sizeof(char), sizeof(buildLabel), stream); // Get $(dsight -v)
    pclose(stream);
    printf("\n%s", buildLabel);

    /* print DNNDK N2Cube Core library version */
    n2cube_version(buildLabel, sizeof(buildLabel));
    printf("\nN2Cube Core library version %s\n", buildLabel);
    n2cube_builddate(buildLabel, sizeof(buildLabel));
    printf("Build Label: %s\n", buildLabel);

    /* print DNNDK N2Cube Driver version */
//    ret = system("cat " DPU_DRV_VERSION);
//    if (ret) {
//        printf("Error: fail to display DPU Driver version info.\n");
//        return -1;
//    }

    return 0;
}

void strCaseless(char *str)
{
    int i;

    if (!str) return;

    for(i=0; str[i]; i++) {
        if((str[i]>='A') && (str[i]<='Z')) {
            str[i]=(str[i]-'A')+'a';
        }
    }
}

int main(int argc, char *argv[])
{
    int next_option;
    char *in_option;

    /* string listing valid short options letters. */
    const char* const short_options = "hm:wx:st:vc";

    /* array for describing valid long options. */
    const struct option long_options[] = {
        {"help",      no_argument,          NULL, 'h'},
        {"mode",      required_argument,    NULL, 'm'},
        {"whoami",    no_argument,          NULL, 'w'},
        {"extend",    required_argument,    NULL, 'x'},
        {"status",    no_argument,          NULL, 's'},
        {"timeout",   required_argument,    NULL, 't'},
        {"clear",     no_argument,          NULL, 'c'},
        {"version",   no_argument,          NULL, 'v'},
        {NULL,        0,                    NULL, 0}
    };

    /* Save path for find dsight path */
    pMyPath = argv[0];

    // Check DPU exist
    if (check_dpu_driver_exist() < 0) {
        return -1;
    }

    do {
        next_option = getopt_long(argc, argv, short_options, long_options, NULL);
        switch (next_option) {
        case 's':
            /* display DPU Core status */
            if (0 != display_dpu_status()) {
                goto dexplorer_error;
            }
            break;
        case 'm':
            /* sepcifiy DPU mode: dump or profile */
            in_option = optarg;
            strCaseless(in_option);
            if (0 != set_n2cube_mode(in_option)) {
                goto dexplorer_error;
            }
            break;
         case 'w':
            /* display DPU Core info */
            if (0 != display_dpu_info(0)) {
                goto dexplorer_error;
            }
            break;
         case 'x':
            /* display DPU Core info */
            in_option = optarg;
            if (0 != proc_extend(in_option)) {
                goto dexplorer_error;
            }
            break;
         case 't':
            /* sepcifiy DPU timeout value */
            in_option = optarg;
            if (0 != set_n2cube_timeout(in_option)) {
                goto dexplorer_error;
            }
            break;
         case 'c':
            dpu_clear_counter();
            break;
        case 'v':
            /* display version info */
            if (0 != print_version()) {
                goto dexplorer_error;
            }
            break;
        case 'h':
            /* -h or --help */
            print_usage();
            break;
        case '?':
            /* -h or --help */
            printf("Try 'dexplorer -h' for more information.\n");
            goto dexplorer_error;
        case -1:
            /* if no option is specified */
            if (1 == argc) {
                printf("Error: missing dexplorer option.\n");
                printf("Try 'dexplorer -h' for more information.\n");
                goto dexplorer_error;
            }
            if (argv[optind] != NULL) {
                printf("Error: invalid parameter \'%s\'.\n", argv[optind]);
                printf("Try 'dexplorer -h' for more information.\n");
                goto dexplorer_error;
            }
            /* Done with options */
            break;
        default:
            /* Something else: unexpected */
            printf("Error: unexpected option is specified.");
            goto dexplorer_error;
        }
    } while (next_option != -1);

    return 0;

dexplorer_error:
    return -1;
}
