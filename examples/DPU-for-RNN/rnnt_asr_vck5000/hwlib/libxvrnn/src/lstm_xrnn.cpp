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
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "lstm_xrnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BITSTEAM_FILE   "./xclbin/xvrnn.hw.xclbin"

xrnn::xrnn(char* model_name) {
  xrnn_dev.bitstreamFile = BITSTEAM_FILE;
  int slen = strlen(model_name);
  model_name_ = std::string(model_name);
  if (model_name_.compare("sentiment")==0) {
    rc = xrnn_open(&xrnn_dev, SENTIMENT);
  } else if (model_name_.compare("customer")==0) {
    rc = xrnn_open(&xrnn_dev, SATISFACTION);
  } else if (model_name_.compare("openIE")==0) {
    rc = xrnn_open(&xrnn_dev, OPENIE);
  } else if (model_name_.compare("rnnt")==0) {
    rc = xrnn_open(&xrnn_dev, RNNT);
  } else{
    printf("This model %s doesn't exist!",  model_name_);
    exit(0);
  }

  if(rc<0){
      printf("xrnn open error\r\n");
      assert(rc);
  }

  DEBUG_PRINT("[XRNN] GET XRNN WHOLE DDR");
  std::string model_path = "model/";
  model_path += model_name;
  model_path += ".bin";
  FILE *model = fopen(model_path.c_str(), "rb");
  if(!model){
      printf("model file open error\r\n");
      printf("The model path is %s\r\n", model_path);
      xrnn_close(&xrnn_dev);
      assert(rc);
  }
  fseek(model, 0, SEEK_END);
  long size=ftell(model);
  mbuf.resize(size);

  fseek(model, 0, SEEK_SET);
  fread(mbuf.data(), sizeof(char), size, model);
  rc = xrnn_init(&xrnn_dev, mbuf.data(), size);
  fclose(model);

}

// output data should be malloc before this function, Please Check!
void xrnn::lstm_run(char* input, int in_size, char* output, int out_size, int frame_num) {
  clock_t t1, t2;
  t1 = clock();
  rc = xrnn_run(&xrnn_dev, input, in_size, frame_num, output, out_size); 
  t2 = clock();
  //assert(rc);
  DEBUG_PRINT("dpu run time    : %f\t\n", ((float)(t2-t1))/CLOCKS_PER_SEC);
}

xrnn::~xrnn() {
    rc = xrnn_close(&xrnn_dev);
    if(rc<0){
        printf("xrnn close error\r\n");
    }
}

void xrnn::rnnt_update_ddr(char * in, int in_s, unsigned int offset) {

  DEBUG_PRINT("[XRNN] RNNT UPDATE DDR: addr---0x%x", (0xc000000000+offset));
  upload_data(xrnn_dev.handle, in, (0xc000000000+offset), in_s);
}

void xrnn::rnnt_download_ddr(char * out, int out_s, unsigned int offset) {

  DEBUG_PRINT("[XRNN] RNNT DOWNLOAD DDR: addr---0x%x", (0xc000000000+offset));
  download_data(xrnn_dev.handle, out, (0xc000000000+offset), out_s);
}

void xrnn::rnnt_reflash_ddr() {

  DEBUG_PRINT("[XRNN] RNNT REFLASH DDR");
  std::string file_path = "model/h_7a_bin";
  FILE *in_f = fopen(file_path.c_str(), "rb");
  if(!in_f){
      printf("the file model/h_1_bin open error\r\n");
      xrnn_close(&xrnn_dev);
  }
  long size=0x1600000;
  mbuf.resize(size);

  fread(mbuf.data(), sizeof(char), size, in_f);
  upload_data(xrnn_dev.handle, mbuf.data(), 0xc007a00000, size);
  fclose(in_f);

/*
  DEBUG_PRINT("[XRNN] RNNT REFLASH DDR");
  std::string file_path = "model/h_1_bin";
  FILE *in_f = fopen(file_path.c_str(), "rb");
  if(!in_f){
      printf("the file model/h_1_bin open error\r\n");
      xrnn_close(&xrnn_dev);
  }
  long size=0x160000;
  mbuf.resize(size);

  fread(mbuf.data(), sizeof(char), size, in_f);
  upload_data(xrnn_dev.handle, mbuf.data(), 0xc007900000, size);
  fclose(in_f);

  file_path = "model/h_7b0_bin";
  in_f = fopen(file_path.c_str(), "rb");
  if(!in_f){
      printf("the file model/h_7b0_bin open error\r\n");
      xrnn_close(&xrnn_dev);
  }
  size=0x2800;
  mbuf.resize(size);
  fread(mbuf.data(), sizeof(char), size, in_f);
  upload_data(xrnn_dev.handle, mbuf.data(), 0xc007b00000, size);
  fclose(in_f);

  file_path = "model/h_7c0_bin";
  in_f = fopen(file_path.c_str(), "rb");
  if(!in_f){
      printf("the file model/h_7c0_bin open error\r\n");
      xrnn_close(&xrnn_dev);
  }
  size=0x2800;
  mbuf.resize(size);
  fread(mbuf.data(), sizeof(char), size, in_f);
  upload_data(xrnn_dev.handle, mbuf.data(), 0xc007c00000, size);
  fclose(in_f);
*/
}
#ifdef __cplusplus
}
#endif
