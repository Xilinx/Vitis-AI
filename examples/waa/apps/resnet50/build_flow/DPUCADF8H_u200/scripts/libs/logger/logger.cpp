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
#include "logger.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <stdarg.h>
#include <time.h>
#ifdef WINDOWS
#include <direct.h>
#else
#include <unistd.h>
#endif

using namespace std;

namespace sda {

///////////////////////////////////////////////////////////////////////
string GetApplicationPath() {
#ifdef WINDOWS
#define GetCurrentDir _getcwd
#else
#define GetCurrentDir getcwd
#endif

    char strCurrentPath[FILENAME_MAX];

    if (!GetCurrentDir(strCurrentPath, sizeof(strCurrentPath))) {
        return string("");
    }

    /* not really required */
    strCurrentPath[sizeof(strCurrentPath) - 1] = '\0';
    return string(strCurrentPath);
}

string ToLower(const string &s) {
    string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

string ToUpper(const string &s) {
    string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

string GetTimeStamp() { return ""; }

// trim from start
string &ltrim(std::string &s) {
    s.erase(s.begin(),
            std::find_if(s.begin(),
                         s.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(),
                         s.rend(),
                         std::not1(std::ptr_fun<int, int>(std::isspace)))
                .base(),
            s.end());
    return s;
}

// trim from both ends
string &trim(std::string &s) { return ltrim(rtrim(s)); }

string GetFileExt(const string &s) {
    string strext = s.substr(s.find_last_of(".") + 1);
    return strext;
}

string GetFileTitleOnly(const string &s) {

    string temp = s;
    string::size_type d = temp.find_last_of("//");
    if (d == string::npos)
        d = temp.find_last_of("\\");
    if (d != string::npos)
        temp = temp.substr(d + 1);

    d = temp.find_last_of(".");
    if (d != string::npos)
        temp = temp.substr(0, d);

    return temp;
}

void LogWrapper(int etype, const char *file, int line, const char *desc, ...) {

    //crop file name from full path
    string strFileLoc(file);
    strFileLoc = strFileLoc.substr(strFileLoc.find_last_of("\\/") + 1);

    string strHeader = "";
    {
        char header[512];
        //source
        switch (etype) {
        case (sda::etError): {
            snprintf(header,
                     sizeof(header),
                     "ERROR: [%s:%d]",
                     strFileLoc.c_str(),
                     line);
            break;
        }
        case (sda::etInfo): {
            snprintf(header,
                     sizeof(header),
                     "INFO: [%s:%d]",
                     strFileLoc.c_str(),
                     line);
            break;
        }
        case (sda::etWarning): {
            snprintf(header,
                     sizeof(header),
                     "WARN: [%s:%d]",
                     strFileLoc.c_str(),
                     line);
            break;
        }
        }
        strHeader = string(header);
    }

    //time
    string strTime = "";
#ifdef ENABLE_LOG_TIME
    {
        time_t rawtime;
        time(&rawtime);
#ifdef ENABLE_SECURE_API
        char buffer[64];
        struct tm timeinfo;
        localtime_s(&timeinfo, &rawtime);
        asctime_s(timeinfo, buffer, sizeof(buffer))
            snprintf(buffer, sizeof(buffer), "TIME: [%s]", asctime(timeinfo));
        strTime = string(buffer);
#else
        char buffer[64];
        struct tm *timeinfo = localtime(&rawtime);
        string temp = string(asctime(timeinfo));
        temp = trim(temp);

        //        strftime(buffer, sizeof(buffer), "TIME: []")
        snprintf(buffer, sizeof(buffer), "TIME: [%s]", temp.c_str());
        strTime = string(buffer);
#endif
    }
#endif

    //format the message itself
    string strMsg = "";
    {
        char msg[512];
        va_list args;
        va_start(args, desc);
        vsnprintf(msg, sizeof(msg), desc, args);
        va_end(args);
        strMsg = string(msg);
    }

    //combine
    string strOut =
        strHeader + string(" ") + strTime + string(" ") + strMsg + string("\n");

    //display
    cout << strOut;

    //store
#ifdef ENABLE_LOG_TOFILE
    std::ofstream outfile;
    outfile.open("benchapp.log", std::ios_base::app);
    outfile << strOut;
#endif

    return;
}

} // namespace sda
