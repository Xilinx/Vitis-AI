/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
#ifndef LOGGER_H_
#define LOGGER_H_

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


#define ENABLE_LOG_TOFILE 1
#define ENABLE_LOG_TIME 1

//global logging
#define LogInfo(desc, ...) sda::LogWrapper(0, __FILE__, __LINE__, desc, ##__VA_ARGS__)
#define LogWarn(desc, ...) sda::LogWrapper(1, __FILE__, __LINE__, desc, ##__VA_ARGS__)
#define LogError(desc, ...) sda::LogWrapper(2, __FILE__, __LINE__, desc, ##__VA_ARGS__)

using namespace std;

namespace sda {

    enum LOGTYPE {etInfo, etWarning, etError};

    //string
    string& ltrim(string& s);
    string& rtrim(string& s);
    string& trim(string& s);
    string GetFileExt(const string& s);
    string GetFileTitleOnly(const string& s);

    string ToLower(const string& s);
    string ToUpper(const string& s);

    //time
    string GetTimeStamp();

    //paths
    string GetApplicationPath();


    //debug
    template<typename T>
    void PrintPOD(const vector<T>& pod, size_t display_count = 0, const int precision = 4) {

        size_t count = pod.size();
        if(display_count > 0)
            count = std::min<size_t>(pod.size(), display_count);

        for(size_t i = 0; i < count; i++) {
            cout << std::setprecision(precision) << pod[i] << ", ";
        }
        cout << endl;
    }

    //logging
    void LogWrapper(int etype, const char* file, int line, const char* desc, ...);

}



#endif /* LOGGER_H_ */
