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
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <iomanip>
#include <fstream>
#include <vector>
#include <mutex>
#include <xrt.h>

#include <getopt.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <sys/mman.h>
#include <time.h>
#include <chrono>
#include <thread>

// driver includes
#include "ert.h"

// host_src includes
#include "xclhal2.h" //XRT header 
#include "xclbin.h"

#include "xxrnn.h"

using namespace std;
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define DOMAIN xclBOKind(0)
#define VCK5000_DDR_OFFSET    0xc000000000
#define XVRNN_INPUT_ADDR      0xc005000000
#define XVRNN_OUTPUT_ADDR     0xc007100000

static uint32_t get_reg(xclDeviceHandle xcl_handle, uint64_t cu_addr) {
  uint32_t value = 0;
  size_t size = sizeof(value);
  auto read_result =
      xclRead(xcl_handle, XCL_ADDR_KERNEL_CTRL, cu_addr, &value, size);
  return value;
}

int xdpu_get_counter(xclDeviceHandle xcl_handle,
                             uint64_t cu_base_addr) {
  struct {
    char name[64];
    uint32_t addr;
  } regs[] = {
      {"APSTART", 0x00},    {"VERSION", 0x10},  //
      {"FRAME_LEN", 0x18},  {"INSTR_AL", 0x1c},  //
      {"INSTR_AH", 0x20},   {"MODEL_AL", 0x24},  //
      {"INPUT_AL", 0x28},   {"INPUT_AH", 0x2c},  //
      {"OUTPUT_AL", 0x30},  {"IN_STRIDE", 0x64}, //
      {"OUT_STRIDE", 0x68}, {"S0_TOP", 0x40},  //
      {"S1_LOAD", 0x44},    {"S2_SAVE", 0x48},  //
      {"S3_MMUL", 0x4c},    {"S4_ACTV", 0x50},  //
      {"S5_EMUL", 0x54},    {"S6_ADD", 0x58},  //
      {"S7_CYCLE", 0x5c},
  };
  int cnt = 0;
  for (const auto& reg : regs) {
    auto value = get_reg(xcl_handle, cu_base_addr + reg.addr);
    cout << " " << reg.name << " " << std::hex << value;
  }
  cout << endl;
  return 0;
}
static size_t get_size(int fd) {
  struct stat statbuf;
  fstat(fd, &statbuf);
  return statbuf.st_size;
}

struct timestamps {
  uint64_t total;
  uint64_t to_driver;
  uint64_t to_cu;
  uint64_t cu_complete;
  uint64_t done;
};
static inline uint64_t tp2ns(struct timespec* tp) {
  return (uint64_t)tp->tv_sec * 1000000000UL + tp->tv_nsec;
}
static void print_one_timestamp(const timestamps& ts) {
  std::cout << ts.total/1000 << "," << ts.to_driver/1000 << ","
    << ts.to_cu/1000 << "," << ts.cu_complete/1000 << ","
    << ts.done/1000 << std::endl;

}
static void print_timestamp(const uint64_t start, const uint64_t end,
                            cu_cmd_state_timestamps* c) {
  struct timestamps ts;
  ts.total = end - start;
  ts.to_driver = c->skc_timestamps[ERT_CMD_STATE_NEW] - start;
  ts.to_cu = c->skc_timestamps[ERT_CMD_STATE_RUNNING] -
             c->skc_timestamps[ERT_CMD_STATE_NEW];
  ts.cu_complete = c->skc_timestamps[ERT_CMD_STATE_COMPLETED] -
                   c->skc_timestamps[ERT_CMD_STATE_RUNNING];
  ts.done = end - c->skc_timestamps[ERT_CMD_STATE_COMPLETED];
  print_one_timestamp(ts);
}

void download_data_tofile(xclDeviceHandle handle, string filename) {
  unsigned long addr = XVRNN_OUTPUT_ADDR;
  unsigned long size = 0x800000;
  auto buf = std::vector<char>(size);
  auto flags = 0;
  auto ok = xclUnmgdPread(handle, flags, &buf[0], size, addr);
  cout<<"some buf: "<<endl;
  for(int j = 0; j < 8; ++j){
    cout<<buf[j]-'0'<<" ";
  }
  cout<<endl;
  assert(ok == 0);
  ofstream outfile(filename, ios::out | ios::binary);
  assert(outfile);
  int i = 0;
  while(i < size){
    outfile.write((char*)&buf[i], sizeof(char));
    ++i;
  }
}
void download_data(xclDeviceHandle handle, char* buf, unsigned long addr, unsigned long size) {
  auto flags = 0;
  auto ok = xclUnmgdPread(handle, flags, buf, size, addr);
  assert(ok == 0);
}


void upload_data(xclDeviceHandle handle, char* buf, unsigned long addr, unsigned long size) {
  auto flags = 0;
  auto ok = xclUnmgdPwrite(handle, flags, buf, size, addr);
  assert(ok == 0);
}


void load_xclbin(const string& filename, xrnn_t *xrnn){
  xclDeviceHandle handle = xrnn->handle; 
  auto fd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
  auto data = mmap(NULL, get_size(fd), PROT_READ, MAP_PRIVATE, fd, 0);
  const xclBin* blob = (const xclBin*)data;
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  xclLockDevice(handle);
  xclLoadXclBin(handle, blob);
  auto top = (const axlf*)data;
  string dsa = (const char*)top->m_header.m_platformVBNV;
  xuid_t uuid;
  memcpy(&uuid, top->m_header.uuid, sizeof(xuid_t));
  auto r = xclOpenContext(handle, &uuid[0], 0, true);
  auto ip = xclbin::get_axlf_section(top, IP_LAYOUT);
  ip_layout*  layout = (ip_layout*)(((char*)data) + ip->m_sectionOffset);
  uint64_t base_addr = layout->m_ip_data[0].m_base_address;
  xrnn-> cu_base_addr = base_addr;
  DEBUG_PRINT("dsa: %x, r: %x, cu_addr: 0x%x", dsa, r, base_addr);
}

int xrnn_open(xrnn_t *xrnn, enum model_t net)
{
    auto filename = xrnn->bitstreamFile;
    vector<xclDeviceHandle> handles;
    auto handle = xclOpen(0, NULL, XCL_INFO);
    xrnn->handle = handle;
    load_xclbin(filename, xrnn);
    handles.emplace_back(handle);
    DEBUG_PRINT("handle %x", handles[0]);

    return 0;
}

int xrnn_close(xrnn_t *xrnn)
{
    return 0;
}

int xrnn_init(xrnn_t *xrnn, char* ddr, uint64_t size)
{
    upload_data(xrnn->handle, ddr, VCK5000_DDR_OFFSET, size); 
    DEBUG_PRINT("xrnn_init done");

    return 0;
}

int xrnn_run(xrnn_t* xrnn, char *in, uint64_t isize, int frame, \
    char *out, uint64_t osize)
{
    xclDeviceHandle xcl_handle = xrnn->handle;
    unsigned int cu_mask =1u;
    
    for(int batch_i=0; batch_i<32; batch_i++){
      upload_data(xrnn->handle, (in+(batch_i*0x1f400)), (XVRNN_INPUT_ADDR+(batch_i*0x7d000)), 0x1f400);
    }

    auto bo_handle = xclAllocBO(xcl_handle, 4096, DOMAIN, XCL_BO_FLAGS_EXECBUF);
    auto bo_addr = xclMapBO(xcl_handle, bo_handle, true);
    auto ecmd = reinterpret_cast<ert_start_kernel_cmd*>(bo_addr);
    //for(int ii = 0; ii < 1; ii++){
      #if defined(XRNN_DEBUG_PRINT)
      struct timespec tp;
      clock_gettime(CLOCK_MONOTONIC, &tp);
      uint64_t start = tp2ns(&tp);
      #endif
      ecmd->cu_mask = cu_mask;
      ecmd->stat_enabled = 1;

      ecmd->state = ERT_CMD_STATE_NEW;
      ecmd->opcode = ERT_START_CU;
      ecmd->type = ERT_CTRL;
      ecmd->data[0x00000000 / 4] = 0x00000000;        // [0] APCTL=0,
      ecmd->data[0x00000018 / 4] = frame;
      ecmd->data[0x0000001c / 4] = 0x000000c0;
      ecmd->data[0x00000020 / 4] = 0x0b000000;
      ecmd->data[0x00000024 / 4] = 0x00000000;
      ecmd->data[0x00000028 / 4] = 0x000000c0;
      ecmd->data[0x0000002c / 4] = 0x05000000;
      ecmd->data[0x00000030 / 4] = 0x06000000;
      ecmd->data[0x00000034 / 4] = 0x000000c0;
      ecmd->data[0x00000038 / 4] = 0x0a000000;
      ecmd->data[0x0000003c / 4] = 0x00000000;
      ecmd->data[0x00000064 / 4] = 0x0007d000; 
      ecmd->data[0x00000068 / 4] = 0x0007d000; 

      ecmd->count = 30;
      auto state = 0;
      state = ecmd->state;
      DEBUG_PRINT("sizeof(ecmd) %x, ecmd->state %x, ecmd->cu_mask %x, ecmd->count %x, ecmd->opcode %x, ecmd->type %x", sizeof(*ecmd), ecmd->state, ecmd->cu_mask, ecmd->count, ecmd->opcode, ecmd->type);      

      //if(ENABLE_LOCK) g_mtx_ecmd.lock();
      xclExecBuf(xcl_handle, bo_handle);
      auto wait_value = 0;
      int wait_count = 0;
      while (ecmd->state != 4 && wait_count < 5) {
        DEBUG_PRINT("Waiting for DONE");
        wait_value = xclExecWait(xcl_handle, 1000);
        wait_count++;
      }
      //if(ENABLE_LOCK) g_mtx_ecmd.unlock();
      #if defined(XRNN_DEBUG_PRINT)
      clock_gettime(CLOCK_MONOTONIC, &tp);
      uint64_t end = tp2ns(&tp);

      state = ecmd->state;
      print_timestamp(start, end, ert_start_kernel_timestamps(ecmd));
      DEBUG_PRINT("wait_value %x, state %x, counter %x", wait_value, state, state);

      xdpu_get_counter(xcl_handle, xrnn->cu_base_addr);
      #endif

      int out_p = 0;
      int batches_osize = 0x7d00;
      for(int batch_i=0; batch_i<32; batch_i++){
        //0x40000: batch output stride in ddr, 0x7d00: the max ouput size of mlperf data
        download_data(xrnn->handle, (out+out_p), (XVRNN_OUTPUT_ADDR+(batch_i*0x40000)), batches_osize);
        out_p += batches_osize;
      } 

      //char rslt_file[100];
      //auto rc=sprintf (rslt_file, "./out/outf%d",0);
      //download_data_tofile(xcl_handle, rslt_file);
    //} //end for()

}

int xrnn_ddr_test(xrnn_t *xrnn, uint64_t addr, \
    char *in, uint64_t isize, char * out, uint64_t osize)
{

    unsigned boHandle = xclAllocBO(xrnn->handle, isize, \
        0, xrnn->first_mem);

    int *vbo = (int*) xclMapBO(xrnn->handle, boHandle, true);

    //std::cout << "in size" <<  isize << std::endl;
    memcpy(vbo, in, isize);

    if(xclSyncBO(xrnn->handle, boHandle, XCL_BO_SYNC_BO_TO_DEVICE, \
        isize, 0)) {
        return -EINVAL;
    }

    if(xclSyncBO(xrnn->handle, boHandle, XCL_BO_SYNC_BO_FROM_DEVICE, \
        osize, 0)) { 
        return -EINVAL;
    }

    //std::cout << "out size" <<  osize << std::endl;
    memcpy(out, vbo, osize);

    munmap(vbo, isize);
    xclFreeBO(xrnn->handle, boHandle);

    return 0;
}

static int timespec_check(struct timespec *t)
{
    if ((t->tv_nsec < 0) || (t->tv_nsec >= 1000000000))
        return -1;
    return 0;

}

long timespec_sub(struct timespec *t1, struct timespec *t2)
{
    if (timespec_check(t1) < 0) {
    	fprintf(stderr, "invalid time #1: %lld.%.9ld.\n",
    		(long long)t1->tv_sec, t1->tv_nsec);
    	return 0;
    }
    if (timespec_check(t2) < 0) {
    	fprintf(stderr, "invalid time #2: %lld.%.9ld.\n",
    		(long long)t2->tv_sec, t2->tv_nsec);
    	return 0;
    }
    t1->tv_sec -= t2->tv_sec;
    t1->tv_nsec -= t2->tv_nsec;
    if (t1->tv_nsec >= 1000000000) {
    	t1->tv_sec++;
    	t1->tv_nsec -= 1000000000;
    } else if (t1->tv_nsec < 0) {
    	t1->tv_sec--;
    	t1->tv_nsec += 1000000000;
    }

    return t1->tv_sec*1000000000 + t1->tv_nsec;
}

