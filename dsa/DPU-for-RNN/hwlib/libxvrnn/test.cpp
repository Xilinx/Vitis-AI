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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include "./include/lstm_xrnn.h"

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif
#define DEFAULT_ENC_FRM 20
#define PROFILER_OFFSET 0xa000000
#define DEC_FRM_OFFSET  0x7c40000
#define DEC_FRM_DSIZE   0X80
#define IN_STRIDE       0x7d000
#define OUT_STRIDE      0x40000
#define BATCH_SIZE      32
#define INPUT_SIZE      (IN_STRIDE * BATCH_SIZE)
#define OUTPUT_SIZE     (OUT_STRIDE * BATCH_SIZE)
#define DEC_FRM_F       "./32frm"
#define VECTORS_F       "./32input"
#define RESULT_F        "rslt/rslt_cpp_bin"

int main(int argc, char* argv[]) {
  int flag_p = 0;
  int prof_size = 0;
  char * junk;
  int c = 0;
  while((c = getopt(argc,argv,"hp:")) != -1) {
    switch(c){
      case 'h':
        printf("    -h : help\n");
        printf("    -p : download profiler, the profiler file is prof/prof_txt, -p <size>\n");
	return 1;
        break;
      case 'p':
	flag_p = 1;
        prof_size = strtol(optarg+2,&junk,16);
        break;
      case '?':
        printf("unknow option:%c\n",optopt);
        break;
      default:
        break;
    }
  }

  auto t0 = std::chrono::system_clock::now();
  xrnn xxx("rnnt");
  auto t1 = std::chrono::system_clock::now();
  printf("*** TEST START ***\r\n");
  int frame_num = DEFAULT_ENC_FRM;

  char* decoder_32frm = (char*)malloc(sizeof(char)*DEC_FRM_DSIZE);
  memset(decoder_32frm, 0, DEC_FRM_DSIZE);
  ifstream frm_in(DEC_FRM_F, ios::binary);
  frm_in.read(decoder_32frm, DEC_FRM_DSIZE);
  frm_in.close();
  xxx.rnnt_update_ddr(decoder_32frm, DEC_FRM_DSIZE, DEC_FRM_OFFSET);
  free(decoder_32frm);

  char* input = (char*)malloc(sizeof(char)*INPUT_SIZE);
  memset(input, 0, INPUT_SIZE);
  char* output = (char*)malloc(sizeof(char)*OUTPUT_SIZE);
  ifstream fin(VECTORS_F, ios::binary);
  fin.read(input, INPUT_SIZE);
  fin.close();

  auto t2 = std::chrono::system_clock::now();
  printf("*** hw run start\r\n");
  xxx.lstm_run(input, INPUT_SIZE, output, OUTPUT_SIZE, frame_num);
  printf("*** hw run done\r\n");
  auto t3 = std::chrono::system_clock::now();
  std::cout << "dpu init: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << std::endl;
  std::cout << "dpu run : " << std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << std::endl;
  ofstream fout(RESULT_F, ios::binary);
  fout.write(output, OUTPUT_SIZE);
  fout.close();

  if (flag_p == 1) { 
    char* prof_out = (char*)malloc(sizeof(char)*prof_size);
    memset(prof_out, 0, prof_size);
    xxx.rnnt_download_ddr(prof_out, prof_size, PROFILER_OFFSET);

    FILE *f;
    if((f = fopen("prof/prof_txt", "w")) == NULL) {
      printf("fail to write!");
      exit(1);
    }
    int reorder_size = prof_size - (prof_size%4);
    for (int i = 0; i<reorder_size; i+=4) {
      int32_t int32_buf = int32_buf & 0x00000000;
      int32_buf = int32_buf | (((int32_t)prof_out[i+3]<<24) & 0xff000000);
      int32_buf = int32_buf | (((int32_t)prof_out[i+2]<<16) & 0x00ff0000);
      int32_buf = int32_buf | (((int32_t)prof_out[i+1]<<8)  & 0x0000ff00);
      int32_buf = int32_buf | ((int32_t)prof_out[i]         & 0x000000ff);  
      fprintf(f, "%08x\n", int32_buf);
    }
    fclose(f);
    free(prof_out);
  }
  free(input);
  free(output);

  printf("*** TEST DONE ***\r\n");
  printf("*** result:  rslt/rslt_cpp_bin\r\n");

  return 0;
}

#ifdef __cplusplus
}
#endif
