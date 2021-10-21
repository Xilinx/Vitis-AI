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
//================================== End Lic =================================================
#ifndef _SORTING_H_1028567
#define _SORTING_H_1028567
#include <algorithm> // std::sort
#include <vector>    // std::vector

template <typename T>
bool compare_nums(T i, T j) {
    return i < j;
}
template <typename T, int size>
void sort(T data[]) {
    std::vector<T> vectorData(data, data + size);
    std::sort(vectorData.begin(), vectorData.end(), compare_nums<T>);
    int i = 0;
    for (typename std::vector<T>::iterator iter = vectorData.begin(); iter != vectorData.end(); iter++) {
        data[i] = *iter;
        i++;
    }
}

#endif
