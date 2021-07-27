/*
 * Copyright 2019 Xilinx Inc.
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
#ifndef XRNN_H
#define XRNN_H

#include <string>
#include <uuid/uuid.h>

#include "xclhal2.h" //XRT header 
#include "xclbin.h"

//#define XRNN_DEBUG_PRINT  // enable debug print

#if defined(XRNN_DEBUG_PRINT)
#define DEBUG_PRINT(...) \
do{ \
    fprintf(stdout, "%s(%d):%s ",__FILE__,__LINE__,__FUNCTION__); \
    fprintf(stdout, __VA_ARGS__); \
    fprintf(stdout,"\r\n"); \
}while(0)

#else
#define DEBUG_PRINT(...)

#endif

enum model_t
{
    SENTIMENT,
    SATISFACTION,
    OPENIE,
    RNNT
};


typedef struct 
{
    enum model_t type;
    xclDeviceHandle handle;
    std::string bitstreamFile;
    std::string halLogFile;
    unsigned index = 1;
    unsigned cu_index = 0;
    uint64_t cu_base_addr = 0;
    int first_mem = -1;
    uuid_t xclbinId;
    size_t alignment=128;
    bool ert;
    bool verbose = false;
    unsigned boHandle;
    char * boVirtual;
    uint64_t boDevAddr;
    size_t boSize;
}xrnn_t;

int xrnn_open(xrnn_t *xrnn, enum model_t type);
int xrnn_close(xrnn_t *xrnn);
int xrnn_run(xrnn_t *xrnn, char *in, uint64_t isize, int frame,  char *out, uint64_t osize);
int xrnn_init(xrnn_t *xrnn, char *model, uint64_t size);
//int xrnn_update_vector(xrnn_t *xrnn, char* vector, uint64_t size);
void upload_data(xclDeviceHandle handle, char* buf, unsigned long addr, unsigned long size);
void download_data(xclDeviceHandle handle, char* buf, unsigned long addr, unsigned long size);
int xrnn_reset(xrnn_t *xrnn);
int xrnn_ddr_test(xrnn_t *xrnn, uint64_t addr,  char *in, uint64_t isize, char *out, uint64_t osize);

long timespec_sub(struct timespec *t1, struct timespec *t2);


#endif

