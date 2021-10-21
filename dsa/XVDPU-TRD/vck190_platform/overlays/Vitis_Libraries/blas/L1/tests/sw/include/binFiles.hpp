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
#ifndef __BINFILES_HPP_
#define __BINFILES_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>

using namespace std;

inline size_t getBinBytes(const string filename) {
    ifstream file(filename, ios::binary);
    file.unsetf(ios::skipws);
    streampos fileSize;
    file.seekg(0, ios::end);
    fileSize = file.tellg();
    file.close();
    return fileSize;
}

template <typename T>
vector<T> readBin(const string filename) {
    vector<T> vec;
    ifstream file(filename, ios::binary);
    file.unsetf(ios::skipws);
    streampos fileSize;
    file.seekg(0, ios::end);
    fileSize = file.tellg();
    file.seekg(0, ios::beg);

    const streampos vecSize = fileSize / sizeof(T);
    vec.resize(vecSize);

    file.read(reinterpret_cast<char*>(vec.data()), fileSize);
    file.close();
    return vec;
}

template <typename T, typename A>
bool readBin(const string filename, const streampos readSize, vector<T, A>& vec) {
    ifstream file(filename, ios::binary);
    file.unsetf(ios::skipws);
    streampos fileSize;
    file.seekg(0, ios::end);
    fileSize = file.tellg();
    file.seekg(0, ios::beg);
    if (readSize > 0 && fileSize != readSize) {
        cout << "WARNNING: file " << filename << " size " << fileSize << " doesn't match required size " << readSize
             << endl;
    }
    assert(fileSize >= readSize);

    const streampos vecSize = fileSize / sizeof(T);
    vec.resize(vecSize);

    file.read(reinterpret_cast<char*>(vec.data()), fileSize);
    file.close();
    if (file)
        return true;
    else
        return false;
}

template <typename T>
bool readBin(const string filename, const streampos readSize, T* vec) {
    ifstream file(filename, ios::binary);
    file.unsetf(ios::skipws);
    streampos fileSize;
    file.seekg(0, ios::end);
    fileSize = file.tellg();
    file.seekg(0, ios::beg);
    if (readSize > 0 && fileSize != readSize) {
        cout << "WARNNING: file " << filename << " size " << fileSize << " doesn't match required size " << readSize
             << endl;
    }
    assert(fileSize >= readSize);

    file.read(reinterpret_cast<char*>(vec), fileSize);
    file.close();
    if (file)
        return true;
    else
        return false;
}

template <typename T>
bool writeBin(const string filename, const streampos writeSize, const T* vec) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char*>(vec), writeSize);
    file.close();
    if (file)
        return true;
    else
        return false;
}

template <typename T, typename A>
bool writeBin(const string filename, const streampos writeSize, vector<T, A>& vec) {
    streampos fileSize = vec.size() * sizeof(T);
    if (writeSize > 0 && fileSize != writeSize) {
        cout << "WARNNING: vector size " << fileSize << " doesn't match required size " << writeSize << endl;
    }
    assert(fileSize >= writeSize);

    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<char*>(vec.data()), writeSize);
    file.close();
    if (file)
        return true;
    else
        return false;
}

#endif
