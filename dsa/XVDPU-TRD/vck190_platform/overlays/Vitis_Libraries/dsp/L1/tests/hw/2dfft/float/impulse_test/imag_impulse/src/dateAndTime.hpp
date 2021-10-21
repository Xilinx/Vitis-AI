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
#ifndef DATEANDTIME_H_
#define DATEANDTIME_H_
#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>
const std::string currentDateTime() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "Dated: %Y-%m-%d    Running Time : %X", &tstruct);
    return buf;
}

const std::string currentDateTimeText() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
    return buf;
}

const std::string currentDateTime4CSV() {
    time_t now = time(0);
    std::stringstream strs;
    strs << now;

    return strs.str();
}

#endif // DATEANDTIME_H_
