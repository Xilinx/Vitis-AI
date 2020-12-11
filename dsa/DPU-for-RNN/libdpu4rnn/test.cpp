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
#include <chrono>
#include <iostream>
#include <fstream>
#include "./include/dpu4rnn.hpp"

using namespace std;

//#ifdef __cplusplus
//extern "C" {
//#endif

void split(char* s, int16_t* data, const char* ch, int pos) {
  if(ch == "") {
    printf("please input a sub-str which you want to split as the thrid parameter!\n");
    return;
  }
  char* pch;
  pch = strtok(s, " ");
  data = data + pos*224;
  while(pch != NULL) {
    *(data++) = atoi(pch);
    pch = strtok(NULL, " ");
  }
}

void readtxt(char* filename, int16_t* data) {
  FILE *f;
  if((f = fopen(filename, "r")) == NULL) {
    printf("fail to read!");
    exit(1);
  }
  char s[1400];
  int pos = 0;
  while(fgets(s, 1400, f) != NULL) {
    int len = strlen(s);
    s[len-1]='\0';
    split(s, data, " ", pos);
    printf("Now the pos is : %d ", pos);
    for(int i = 0; i < 10; ++i)
      printf("%d ", data[pos*224 + i]);
    printf("\n");
    pos++;
  }
//  fclose(f);
}

void writetxt(const char* filename, int16_t* data, int frame_num) {
  FILE *f;
  if((f = fopen(filename, "w")) == NULL) {
    printf("fail to read!");
    exit(1);
  }
  for (int i = 0; i<frame_num; ++i) {
    for (int j = 0; j < 128; ++j)
      fprintf(f, "%d ", data[i*128+j]);
    fputc('\n', f);
  }
//  fclose(f);
}

int main(int argc, char* argv[]) {
  auto t0 = std::chrono::system_clock::now();
  auto xxx = dpu4rnn::create("sentiment");
  auto t1 = std::chrono::system_clock::now();
  printf("111111111111\r\n");
  int frame_num = 500;
  char* input = (char*)malloc(sizeof(short)*32*frame_num);
  memset(input, 0, 2*32*frame_num);
  char* output = (char*)malloc(sizeof(short)*128);
  memset(output, 0, 2*128);
  //readtxt("fix_input.txt", (int16_t*)input);
  ifstream fin(argv[1], ios::binary);
  fin.read(input, frame_num*32*2);
  auto t2 = std::chrono::system_clock::now();
  printf("222222222222\r\n");
  xxx->run(input, frame_num*sizeof(short)*32, output, frame_num);
  printf("333333333333\r\n");
  auto t3 = std::chrono::system_clock::now();
  std::cout << "dpu init: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << std::endl;
  std::cout << "dpu run : " << std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << std::endl;
  //writetxt("result.bin", (int16_t*)output, frame_num);
  ofstream fout("result.bin");
  fout.write(output, 128*2);
  fout.close();
  fin.close();
  free(input);
  free(output);
  return 0;
}

/* satisfaction */
//int main(int argc, char* argv[]) {
//  auto t0 = std::chrono::system_clock::now();
//  //dpu4rnn xxx("sentiment");
//  auto xxx = dpu4rnn::create("satisfaction");
//  auto t1 = std::chrono::system_clock::now();
//  printf("111111111111\r\n");
//  //int frame_num = 500;
//  int frame_num = 25;
//  char* input = (char*)malloc(sizeof(short)*32*frame_num);
//  memset(input, 0, 2*32*frame_num);
//  char* output = (char*)malloc(sizeof(short)*128*frame_num);
//  memset(output, 0, 2*128*frame_num);
//  //readtxt("fix_input.txt", (int16_t*)input);
//  ifstream fin(argv[1], ios::binary);
//  fin.read(input, frame_num*32*2);
//  auto t2 = std::chrono::system_clock::now();
//  printf("222222222222\r\n");
//  xxx->run(input, frame_num*sizeof(short)*32, output, frame_num);
//  printf("333333333333\r\n");
//  auto t3 = std::chrono::system_clock::now();
//  std::cout << "dpu init: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << std::endl;
//  std::cout << "dpu run : " << std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << std::endl;
//  //writetxt("result.bin", (int16_t*)output, frame_num);
//  ofstream fout("result.bin");
//  fout.write(output, 128*frame_num*2);
//  fout.close();
//  fin.close();
//  free(input);
//  free(output);
//  return 0;
//}

/*  openie   */
//int main(int argc, char* argv[]) {
//  auto t0 = std::chrono::system_clock::now();
//  //dpu4rnn xxx("sentiment");
//  auto xxx = dpu4rnn::create("openie");
//  auto t1 = std::chrono::system_clock::now();
//  printf("111111111111\r\n");
//  //int frame_num = 500;
//  int frame_num = 36;
//  char* input = (char*)malloc(sizeof(short)*224*frame_num);
//  memset(input, 0, 2*200*frame_num);
//  char* output = (char*)malloc(sizeof(short)*300*frame_num);
//  memset(output, 0, 2*300*frame_num);
//  //readtxt("fix_input.txt", (int16_t*)input);
//  ifstream fin(argv[1], ios::binary);
//  fin.read(input, 36*224*2);
//  auto t2 = std::chrono::system_clock::now();
//  printf("222222222222\r\n");
//  xxx->run(input, frame_num*sizeof(short)*224, output, frame_num);
//  printf("333333333333\r\n");
//  auto t3 = std::chrono::system_clock::now();
//  std::cout << "dpu init: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << std::endl;
//  std::cout << "dpu run : " << std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << std::endl;
//  //writetxt("result.bin", (int16_t*)output, frame_num);
//  ofstream fout("result.bin");
//  fout.write(output, 300*frame_num*2);
//  fout.close();
//  fin.close();
//  free(input);
//  free(output);
//  return 0;
//}


//#ifdef __cplusplus
//}
//#endif
