/*
 * Copyright 2019 Xilinx, Inc.
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
#pragma once

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cassert>
using namespace std;

template <typename T>
void readBin(string filename, T* array, unsigned long long int size) {
    FILE* pFile = fopen(filename.c_str(), "rb");
    if (pFile == NULL) {
        cout << "Open File failed: " << filename << endl;
        exit(1);
    }
    fseek(pFile, 0, SEEK_END);
    unsigned long long int fsize = ftell(pFile);
    rewind(pFile);

    if (fsize != size) cout << "File " << filename << " size " << fsize << " arg size " << size << endl;
    assert(fsize == size);
    unsigned long long int rsize = fread(array, 1, fsize, pFile);
    if (rsize != fsize) {
        cout << "Read File failed: " << filename << endl;
        exit(1);
    }
    fclose(pFile);
    //  cout  << "Load file sucessfully: " << filename << endl;
}

template <typename T>
void writeBin(string filename, T* array, unsigned long long int size) {
    FILE* pFile = fopen(filename.c_str(), "wb");
    if (pFile == NULL) {
        cout << "Open File failed: " << filename << endl;
        exit(1);
    }

    unsigned long long int rsize = fwrite(array, 1, size, pFile);
    fclose(pFile);
    //   cout  << "Write file sucessfully: " << filename << endl;
}
